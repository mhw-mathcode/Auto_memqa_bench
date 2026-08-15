import json
import re
from typing import Any, Dict, List, Optional, Tuple
import os
import time
import shutil
from openai import OpenAI
import math
from src.mcq_scoring import normalize_answer_candidates, score_mcq_prediction, parse_mcq_gt_answers
from src.qa_only_response import QAOnlyRunner
from src.utils import count_qa_items, load_json_file, normalize_dataset_records, write_json_file
from src.pipeline_utils import log_event, log_subsection, print_log_section, print_kv

# --- 1. 题干与选项展示格式 ---

def _extract_core_question_text(question_text: str, unknown_placeholder: str = "") -> str:
    """
    兼容不同题目模板，稳定提取纯题干（不包含选项与作答提示）。

    提取策略：
    1. 优先截取首个选项行（A-F）之前的内容。
    2. 若未命中选项行，则尝试按常见提示语截断。
    3. 最后回退为首个非空行。
    """
    text = (question_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return unknown_placeholder

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return unknown_placeholder

    option_line_pattern = re.compile(r"^[A-Fa-f][\.．\)]\s+")

    first_option_idx = None
    for idx, line in enumerate(lines):
        if option_line_pattern.match(line):
            first_option_idx = idx
            break

    if first_option_idx is not None:
        stem = "\n".join(lines[:first_option_idx]).strip()
        if stem:
            # 兼容旧格式前缀："Please answer the question: ..."
            stem = re.sub(r"^Please\s+answer\s+the\s+question:\s*", "", stem, flags=re.IGNORECASE)
            return stem or unknown_placeholder

    for pattern in (
        r"\n\s*\n\s*You need to select",
        r"\n\s*Please provide the option corresponding to the only correct answer",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            stem = text[:match.start()].strip()
            stem = re.sub(r"^Please\s+answer\s+the\s+question:\s*", "", stem, flags=re.IGNORECASE)
            return stem or unknown_placeholder

    fallback = lines[0]
    fallback = re.sub(r"^Please\s+answer\s+the\s+question:\s*", "", fallback, flags=re.IGNORECASE)
    return fallback or unknown_placeholder


def _strip_option_prefix(value: str) -> str:
    """去掉选项字符串开头的 A./B. 等前缀，仅保留选项正文。"""
    text = str(value or "").strip()
    match = re.match(r"^[A-Fa-f][\.．\)]\s*(.*)$", text)
    return match.group(1).strip() if match else text


def _normalize_option_lines_preserve_order(option_value: Any) -> List[str]:
    """将 option 字段统一为 A-F 列表；保留原始 A-E 顺序，不做乱序。"""
    option_lines: List[str] = []

    if isinstance(option_value, dict):
        for letter in ["A", "B", "C", "D", "E", "F"]:
            matched_value = None
            for key, value in option_value.items():
                if str(key).strip().upper() == letter:
                    matched_value = value
                    break
            if matched_value is None:
                continue
            body = _strip_option_prefix(str(matched_value))
            if body:
                option_lines.append(f"{letter}. {body}")

    elif isinstance(option_value, list):
        next_letter_ord = ord("A")
        for raw_option in option_value:
            text = str(raw_option or "").strip()
            if not text:
                continue
            prefix_match = re.match(r"^([A-Fa-f])[\.．\)]\s*(.*)$", text)
            if prefix_match:
                letter = prefix_match.group(1).upper()
                body = prefix_match.group(2).strip()
                option_lines.append(f"{letter}. {body}")
                next_letter_ord = max(next_letter_ord, ord(letter) + 1)
            else:
                letter = chr(next_letter_ord)
                option_lines.append(f"{letter}. {text}")
                next_letter_ord += 1

    elif option_value not in (None, ""):
        option_lines.append(f"A. {_strip_option_prefix(str(option_value))}")

    has_f = any(re.match(r"^F[\.．\)]\s+", line, flags=re.IGNORECASE) for line in option_lines)
    if not has_f:
        option_lines.append("F. Cannot infer the answer based on the given information.")

    return option_lines


def _build_question_with_options(core_question: str, option_lines: List[str]) -> str:
    option_block = "\n".join(option_lines)
    return (
        f"{core_question.strip()}\n"
        f"{option_block}\n"
        "Please provide the option corresponding to the only correct answer, enclosed in parentheses, e.g., (X)."
    )


def format_questions_with_options(input_data: Any) -> Tuple[List[Dict[str, Any]], int]:
    """生成 question + options 展示版本；不改变选项顺序。"""
    import copy

    formatted_data = normalize_dataset_records(copy.deepcopy(input_data))
    formatted_count = 0

    for section in formatted_data:
        qa_list = section.get("qa", [])
        if not isinstance(qa_list, list):
            continue

        for qa_item in qa_list:
            if not isinstance(qa_item, dict):
                continue
            option_lines = _normalize_option_lines_preserve_order(qa_item.get("option", []))
            core_question = _extract_core_question_text(
                qa_item.get("question", ""),
                unknown_placeholder=str(qa_item.get("question", "")).strip(),
            )
            qa_item["option"] = option_lines
            qa_item["question"] = _build_question_with_options(core_question, option_lines)
            formatted_count += 1

    return formatted_data, formatted_count


# --- 1. response + eval ---

def run_qa_only(answer_llm_config, dataset_name, max_workers, output_file):
    runner = QAOnlyRunner(
        output_file,
        answer_llm_config=answer_llm_config,
    )
    runner.process_data_file(
        f"{dataset_name}", 
        max_workers=max_workers
    )
    runner.close()

# 规则评估
def run_eval(idx, file, output_file=None):
    """
    评估问答结果
    
    Args:
        idx: 轮次编号
        file: 输入文件路径
        output_file: 输出文件路径，如果为 None 则使用默认的 temp/result_{idx}.json
    """
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key, items in data.items():
        for item in items:
            answer_candidates = normalize_answer_candidates(item.get("answer_fixed"), item.get("answer", ""))

            # Scoring uses only the parsed answer; rationale is retained for audit.
            prediction_text = item.get("response") or item.get("response_option") or item.get("response", "")
            score_result = score_mcq_prediction(prediction_text, answer_candidates)

            item["score"] = 1 if score_result.get("is_correct", False) else 0
            item["prediction_malformed"] = score_result.get("prediction_malformed", False)
            item["predicted_options"] = score_result.get("predicted_options", [])
            item["ground_truth_options"] = score_result.get("ground_truth_options", [])

    # 使用指定的输出文件或默认路径
    if output_file is None:
        output_file = f"temp/result_{idx}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- 3. 统计结果，多次回答 ---
def aggregate_and_analyze_results(num_files: int, prefix: str, suffix: str, threshold: float, result_file: str, conversation_file):
    """
    聚合多个 JSON 文件的结果，统计每题的正确次数和正确率，并标记污染。
    """
    # 存储所有题目的统计数据:
    # Key: 提取出的核心问题文本 (e.g., "Regarding money, who did Ariel most habitually rely on?")
    # Value: { "correct_count": int, "total_count": int, "details": original_data, "responses": [str] }

    question_stats: Dict[str, Dict[str, Any]] = {}
    
    log_event("pollution_aggregate", status="start", files=num_files, threshold=threshold)

    for i in range(1, num_files + 1):
        file_name = f"{prefix}{i}{suffix}"
        if not os.path.exists(file_name):
            log_event("pollution_aggregate_file", status="skipped", file=file_name, reason="file_missing")
            continue

        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results_list = []
                if isinstance(data, dict):
                    for conv_items in data.values():
                        if isinstance(conv_items, list):
                            results_list.extend(conv_items)
        except Exception as e:
            log_event("pollution_aggregate_file", status="failed", file=file_name, reason="read_failed", error=e)
            continue

        log_event("pollution_aggregate_file", status="loaded", file=file_name, items=len(results_list))

        for item in results_list:
            full_question_text = item.get("question", "")
            score = item.get("score", 0.0)
            response = item.get("response") or item.get("response_option") or item.get("response", "NO_RESPONSE")
            response_raw = item.get("response_raw", response)
            response_reason = item.get("response_reason", "")
            response_time = item.get("response_time", 0.0)
            
            # 提取核心问题文本作为唯一键
            core_question_text = _extract_core_question_text(full_question_text, unknown_placeholder="UNKNOWN_QUESTION")

            if not core_question_text or core_question_text == "UNKNOWN_QUESTION":
                continue

            # 初始化或更新统计数据
            if core_question_text not in question_stats:
                question_stats[core_question_text] = {
                    "correct_count": 0,
                    "total_count": 0,
                    "responses_and_scores": [],  # 用于记录每次的 response 和 score
                }
            
            # 统计总次数
            question_stats[core_question_text]["total_count"] += 1
            
            # 统计答对次数
            if score == 1.0:
                question_stats[core_question_text]["correct_count"] += 1

            # 记录本次实验的 response 和 score
            question_stats[core_question_text]["responses_and_scores"].append({
                "response": response,
                "response_raw": response_raw,
                "reason": response_reason,
                "score": score,
                "response_time": response_time,
                "file_id": i
            })

    # ----------------------------------------------------
    # 2. 计算正确率和标记污染
    # ----------------------------------------------------
    
    log_subsection("Pollution score aggregation")
    pollution_by_core_question: Dict[str, Dict[str, Any]] = {}

    for core_question_text, stats in question_stats.items():
        correct_count = stats["correct_count"]
        total_count = stats["total_count"]
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        pollution_flag = "suspected" if accuracy >= threshold else "good"
        pollution_by_core_question[core_question_text] = {
            "result": pollution_flag,
            "correct_count": correct_count,
            "total_count": total_count,
            "accuracy": f"{accuracy:.4f}",
            "all_responses_and_scores": stats["responses_and_scores"],
        }

    # ----------------------------------------------------
    # 3. 输出结果
    # ----------------------------------------------------
    # 基于原始文件回填污染检测结果，保留每题对应的完整条目信息
    with open(conversation_file, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    if isinstance(output_data, dict):
        output_data = [output_data]

    for section in output_data:
        qa_list = section.get("qa", [])
        if not isinstance(qa_list, list):
            continue
        
        for qa_item in qa_list:
            core_question_text = _extract_core_question_text(
                qa_item.get("question", ""),
                unknown_placeholder="UNKNOWN_QUESTION"
            )
            pollution_result = pollution_by_core_question.get(core_question_text)
            if pollution_result:
                qa_item["pollution_check"] = pollution_result

    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        # Statistics reporting removed - only report pending questions at start
    except Exception as e:
        log_event("pollution_aggregate", status="failed", reason="write_failed", output=result_file, error=e)

# --- 4. membership_inference ---
def run_membership_inference_loss_api(
    candidate_path: str,
    output_path: str = "membership_results_api.json",
    *,
    api_key: Optional[str] = None,
    base_url: str = "https://api.siliconflow.cn/v1",
    model_name: str = "Qwen/Qwen3-14B",
    api_sleep: float = 0.0,
    top_print: int = 5,
    verbose: bool = True,
    return_sorted: bool = True,
    k_percent: float = 0.2, # Min-K% Prob 的比例参数
) -> List[Dict[str, Any]]:
    """
    改进版 MI 探测：支持 Min-K% Loss 算法，增强对长尾知识污染的识别能力。
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
    if not api_key:
        raise ValueError("api_key 为空：请通过参数 api_key 传入，或设置环境变量")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # =====================
    # 1. 鲁棒的数据加载逻辑
    # =====================
    def load_qas_from_json(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        samples = []
        # 兼容 list[dict] 结构，且处理内部有 "qa" 列表的情况
        if isinstance(raw, list):
            for idx, item in enumerate(raw):
                if "qa" in item and isinstance(item["qa"], list):
                    for j, qa in enumerate(item["qa"]):
                        samples.append({
                            "outer_id": str(item.get("id", f"idx_{idx}")),
                            "inner_id": j,
                            "question": qa["question"],
                            "answer": qa["answer"],
                            "pollution_check": qa.get("pollution_check"), # 10次问答数据在此
                            "category": qa.get("category")
                        })
                elif "question" in item:
                    samples.append({
                        "outer_id": str(item.get("outer_id", "na")),
                        "inner_id": item.get("inner_id", idx),
                        "question": item["question"],
                        "answer": item["answer"],
                        "pollution_check": item.get("pollution_check")
                    })
        return samples

    # =====================
    # 2. Token-level API 调用
    # =====================
    def call_llm(prompt: str) -> Tuple[List[str], List[float]]:
        # 使用 echo=True 获取全文本的 logprobs
        completion = client.completions.create(
            model=model_name,
            prompt=prompt,
            temperature=0.0,
            logprobs=1,
            echo=True,
            max_tokens=1 # 仅做探测，不生成新 token
        )
        choice = completion.choices[0]
        lp = choice.logprobs
        return list(lp.tokens), list(lp.token_logprobs)

    # =====================
    # 3. Min-K% Loss 核心计算
    # =====================
    def compute_enhanced_loss(question: str, answer: str) -> Dict[str, float]:
        tokens_q, _ = call_llm(question)
        tokens_qa, logprobs_qa = call_llm(question + answer)

        len_q = len(tokens_q)
        answer_logprobs = logprobs_qa[len_q:] # 截取答案部分的 logprobs
        
        if not answer_logprobs:
            return {"avg_loss": 999.0, "mink_loss": 999.0, "answer_len": 0}

        # 计算所有 token 的 Negative Log-Likelihood (NLL)
        nll_list = [-lp for lp in answer_logprobs]
        
        # 指标 A: 标准平均 Loss
        avg_loss = sum(nll_list) / len(nll_list)

        # 指标 B: Min-K% Loss (学术界防污染更推荐)
        # 选取 Loss 最高的 k% 个 token (即模型认为最难、概率最低的词)
        sorted_nll = sorted(nll_list, reverse=True)
        k_count = max(1, int(len(nll_list) * k_percent))
        mink_loss = sum(sorted_nll[:k_count]) / k_count

        return {
            "avg_loss": float(avg_loss),
            "mink_loss": float(mink_loss),
            "ppl": math.exp(min(avg_loss, 20)), # 防止溢出
            "answer_len": len(answer_logprobs),
        }

    # =====================
    # 4. 主执行流程
    # =====================
    samples = load_qas_from_json(candidate_path)
    if verbose:
        log_event("membership_inference", status="start", samples=len(samples), input=candidate_path)

    results = []
    for idx, sample in enumerate(samples):
        try:
            scores = compute_enhanced_loss(sample["question"], sample["answer"])
            
            record = {
                **sample, # 保留原始 question, answer, pollution_check
                "global_index": idx,
                "avg_loss": scores["avg_loss"],
                "mink_loss": scores["mink_loss"],
                "ppl": scores["ppl"],
                "answer_len": scores["answer_len"],
            }
            results.append(record)
            
            if verbose and (idx + 1) % 10 == 0:
                log_event(
                    "membership_inference",
                    status="progress",
                    processed=f"{idx + 1}/{len(samples)}",
                    avg_loss=f"{scores['avg_loss']:.4f}",
                    indent=4,
                )
            
            if api_sleep > 0:
                time.sleep(api_sleep)
        except Exception as e:
            log_event("membership_inference_sample", status="failed", sample=idx, error=e)

    # 按 avg_loss 升序排序（Loss 越低，污染嫌疑越大）
    if return_sorted:
        results.sort(key=lambda x: x["avg_loss"])
    
    # 给每条结果增加 rank 标签
    for rank, item in enumerate(results):
        item["rank_by_loss"] = rank

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

# --- 5. 运行配置 ---
CONTAMINATION_CHECK_ROUNDS = 3  # 多次独立采样，避免一次偶然猜中就删除题目
CONTAMINATION_THRESHOLD = 1.00  # 三轮全部答对才标记为可能被污染

def pollution_check_main(
    args,
    input_file_path: str,
    output_file_path: str,
    enable_contamination_check: bool = False,
    cleanup_temp_files: bool = True
) -> str:
    """
    步骤 3: 污染检查
    
    Args:
        args: 命令行参数
        input_file_path: 输入文件路径（v2b版本）
        output_file_path: 输出文件路径（v3版本）
        enable_contamination_check: 是否启用污染检测
        cleanup_temp_files: 污染检测完成后是否清理临时文件
    
    Returns:
        处理后的文件路径
    """
    print_log_section("STEP 3 | POLLUTION CHECK")
    
    def build_provider_config(model_value, base_url_value, api_key_value, optional_fields=None):
        config = {}
        if model_value:
            config["model"] = model_value
        if base_url_value:
            config["base_url"] = base_url_value
        if api_key_value:
            config["api_key"] = api_key_value
        if optional_fields:
            for key, value in optional_fields.items():
                if value not in (None, ""):
                    config[key] = value
        return config

    answer_llm_model = args.answer_llm_model
    answer_llm_base_url = args.answer_llm_base_url
    answer_llm_api_key = args.answer_llm_api_key
    answer_llm_config = build_provider_config(answer_llm_model, answer_llm_base_url, answer_llm_api_key)

    max_workers = args.max_workers

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # 计算待处理问题数
    input_data = load_json_file(input_file_path)
    total_questions = count_qa_items(input_data)
    log_event("pollution_check", status="start", total_questions=total_questions, input=input_file_path)

    # 保持选项顺序不变，但将 question 统一展开为「题干 + A-E/F 选项 + 作答提示」。
    formatted_data, formatted_count = format_questions_with_options(input_data)
    write_json_file(formatted_data, output_file_path, indent=4)
    log_event(
        "pollution_prepare",
        status="formatted_without_shuffle",
        formatted_questions=formatted_count,
        output=output_file_path,
    )
    
    def _is_abstain_item(qa_item: Dict[str, Any]) -> bool:
        """判断题目是否为弃权题（答案为 F 或标注为 Abstain）。"""
        label_text = str(qa_item.get("label", "") or "").strip().lower()
        if "abstain" in label_text:
            return True

        candidates = normalize_answer_candidates(
            qa_item.get("answer_fixed"),
            qa_item.get("answer", "")
        )
        parsed_candidates = [parse_mcq_gt_answers(candidate) for candidate in candidates]
        valid_candidates = [opt_set for opt_set in parsed_candidates if opt_set]

        return bool(valid_candidates) and all(opt_set == {"F"} for opt_set in valid_candidates)

    # 可选的污染检测
    def _run_contamination_check() -> None:
        """执行污染检测，临时文件放在本次运行的 temp/pollution_check/ 目录下。"""
        dataset_name = os.path.splitext(os.path.basename(output_file_path))[0]
        output_parent_dir = os.path.dirname(output_file_path) or "."
        pollution_temp_dir = os.path.join(output_parent_dir, "pollution_check", dataset_name)
        os.makedirs(pollution_temp_dir, exist_ok=True)
        
        log_subsection("Contamination check")
        print_kv("rounds", CONTAMINATION_CHECK_ROUNDS, indent=4)
        print_kv("temp_dir", pollution_temp_dir, indent=4)

        # 过滤掉弃权题（答案为F），不进入污染检测
        source_data = normalize_dataset_records(load_json_file(output_file_path))

        import copy
        filtered_data = copy.deepcopy(source_data)
        total_qa = 0
        skipped_abstain = 0

        for section in filtered_data:
            qa_list = section.get("qa", [])
            if not isinstance(qa_list, list):
                continue

            total_qa += len(qa_list)
            kept_questions = []
            for qa_item in qa_list:
                if _is_abstain_item(qa_item):
                    skipped_abstain += 1
                else:
                    kept_questions.append(qa_item)
            section["qa"] = kept_questions

        remain_for_check = total_qa - skipped_abstain
        log_event(
            "contamination_prepare",
            status="ready",
            total_qa=total_qa,
            skipped_abstain=skipped_abstain,
            remain_for_check=remain_for_check,
        )

        if remain_for_check <= 0:
            log_event("contamination_check", status="skipped", reason="no_non_abstain_questions")
            return

        filtered_input_file = os.path.join(pollution_temp_dir, "contamination_input_filtered.json")
        write_json_file(filtered_data, filtered_input_file, indent=4)
        
        # 临时文件路径
        temp_result_file = os.path.join(pollution_temp_dir, "current_round.json")
        
        # 运行多轮测试
        for idx in range(1, CONTAMINATION_CHECK_ROUNDS + 1):
            log_event("contamination_round", status="start", round=f"{idx}/{CONTAMINATION_CHECK_ROUNDS}")
            run_qa_only(answer_llm_config, filtered_input_file, max_workers, temp_result_file)
            
            # 保存本轮结果到专用目录
            round_result_file = os.path.join(pollution_temp_dir, f"round_{idx}.json")
            run_eval(idx, temp_result_file, output_file=round_result_file)
            log_event("contamination_round", status="success", round=f"{idx}/{CONTAMINATION_CHECK_ROUNDS}", output=round_result_file)
        
        # 聚合分析结果
        aggregate_and_analyze_results(
            CONTAMINATION_CHECK_ROUNDS,
            os.path.join(pollution_temp_dir, "round_"),
            ".json",
            CONTAMINATION_THRESHOLD,
            output_file_path,
            conversation_file=output_file_path
        )
        
        log_event("contamination_check", status="success", output=output_file_path)
        
        # 可选：清理临时文件
        if cleanup_temp_files:
            shutil.rmtree(pollution_temp_dir)
            log_event("contamination_temp", status="cleaned", temp_dir=pollution_temp_dir)
        else:
            log_event("contamination_temp", status="kept", temp_dir=pollution_temp_dir)
    
    if enable_contamination_check:
        _run_contamination_check()

    return output_file_path

