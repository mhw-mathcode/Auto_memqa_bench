import json
import re
from typing import Dict, List, Any, Tuple
import os
import time
import shutil
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from openai import OpenAI
import math
from src.qa_only_response import QAOnlyRunner

# --- 1. 打乱顺序 ---

def rename_and_shuffle_options(input_file_path: str, output_file_path: str) -> None:
    """
    读取 JSON 文件，进行人名替换，并打乱每道题目的选项顺序。

    :param input_file_path: 输入 JSON 文件的路径。
    :param output_file_path: 输出 JSON 文件的路径。
    """
    try:
        # 1. 读取输入文件
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data: List[Any] = json.load(f)

    except FileNotFoundError:
        print(f"❌ 错误: 未找到输入文件 {input_file_path}")
        return
    except json.JSONDecodeError:
        print(f"❌ 错误: 输入文件 {input_file_path} JSON 格式无效")
        return
    except Exception as e:
        print(f"❌ 读取文件时发生意外错误: {e}")
        return

    # 2. 遍历数据结构进行处理 (选项打乱)
    new_data = []
    
    # 外层结构通常是 List[Dict[str, List[Dict]]]
    if isinstance(data, dict):
        data = [data]
    for section in data:
        if 'qa' in section and isinstance(section['qa'], list):
            new_qa_list = []

            for item in section['qa']:
                # 使用深拷贝以保留所有嵌套字段
                import copy
                new_item = copy.deepcopy(item)

                # ===============================
                # 读取原始 option
                # ===============================
                orig_option_list = new_item.get("option", []) or []
                if not isinstance(orig_option_list, list):
                    orig_option_list = [str(orig_option_list)]

                # ===============================
                # 拆分 A–E 与 F
                # ===============================
                ae_options = []
                f_option = None

                for opt in orig_option_list:
                    opt = (opt or "").strip()
                    if opt.startswith("F.") or opt.startswith("F．"):
                        f_option = opt
                    else:
                        ae_options.append(opt)

                # ===============================
                # 读取原始答案
                # ===============================
                original_answer = (new_item.get("answer", "") or "").strip()

                # ===============================
                # 提取核心问题
                # ===============================
                question = new_item.get("question", "")
                pattern = r"(.*?)(?:\n\s*\n\s*You need to select)"
                match = re.search(pattern, question, re.DOTALL)

                core_question = match.group(1).strip() if match else question

                # ===============================
                # 工具函数：去掉 "A. "
                # ===============================
                def _strip_letter_prefix(s: str) -> str:
                    s = (s or "").strip()
                    parts = s.split(". ", 1)
                    return parts[1].strip() if len(parts) == 2 and len(parts[0]) == 1 else s

                # ===============================
                # 解析原正确答案正文
                # ===============================
                if len(original_answer) == 1 and original_answer.isalpha():
                    idx = ord(original_answer.upper()) - ord('A')
                    if 0 <= idx < len(ae_options):
                        answer_body = _strip_letter_prefix(ae_options[idx])
                    else:
                        answer_body = original_answer
                else:
                    answer_body = _strip_letter_prefix(original_answer)

                # ===============================
                # 只 shuffle A–E
                # ===============================
                option_bodies = [_strip_letter_prefix(opt) for opt in ae_options]

                rng = np.random.default_rng()
                rng.shuffle(option_bodies)

                # ===============================
                # 重新生成 A–E
                # ===============================
                new_option_list = []
                new_correct_answer = ""

                option_letters = ['A', 'B', 'C', 'D', 'E'][:len(option_bodies)]

                for letter, body in zip(option_letters, option_bodies):
                    new_opt_full = f"{letter}. {body}"
                    new_option_list.append(new_opt_full)
                    if body == answer_body and not new_correct_answer:
                        new_correct_answer = new_opt_full

                # ===============================
                # F 放回最后（不参与打乱）
                # ===============================
                if f_option:
                    new_option_list.append(f_option)
                    if original_answer.startswith("F"):
                        new_correct_answer = f_option
                else:
                    # 如果原本没有 F，可选择是否补一个
                    new_option_list.append(
                        "F. Cannot infer the answer based on the given information."
                    )

                # ===============================
                # 写回结果
                # ===============================
                new_item["option"] = new_option_list
                new_item["answer"] = new_correct_answer or original_answer

                new_option_txt = "\n".join(new_option_list)
                new_item["question"] = f"{core_question}\n{new_option_txt}\nPlease provide the option corresponding to the only correct answer, enclosed in parentheses, e.g., (X)."

                new_qa_list.append(new_item)

            # 深拷贝section以保留所有字段
            import copy
            new_section = copy.deepcopy(section)
            new_section["qa"] = new_qa_list
            new_data.append(new_section)
            
    # 3. 写入输出文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)

        print(f"✓ 选项打乱完成: {output_file_path}")

    except Exception as e:
        print(f"❌ 写入输出文件时发生错误: {e}")

# --- 2. response + eval ---

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
    def extract_option(s):
        if not isinstance(s, str):
            return None
        m = re.match(r"\s*([A-F])", s)
        return m.group(1) if m else None

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key, items in data.items():
        for item in items:
            ans = extract_option(item.get("answer", ""))
            resp = item.get("response_option", "")
            if ans and resp and ans == resp:
                item["score"] = 1
            else:
                item["score"] = 0

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

    def extract_core_question(full_question_text: str) -> str:
        """
        从完整的 'question' 字段中提取问题的核心文本。
        例如：从 'Please answer the question: ...\n\nYou need to select...' 中提取核心问题。
        """
        # 查找 "Please answer the question: " 和 "\n\nYou need to select" 之间的内容
        match = re.search(
            r"\s*(.*?)\s*\n\s*\n\s*You need to select",
            full_question_text,
            re.DOTALL
        )
        if match:
            # 提取核心问题，并移除首尾的空白和换行符
            core_text = match.group(1).split('\n\n')[0].strip()
            question_line = core_text.split('\n')[0].strip()
            return question_line
        
        # 如果正则匹配失败，返回首个非空行作为回退
        if not full_question_text:
            return "UNKNOWN_QUESTION"
        lines = [line.strip() for line in full_question_text.split("\n") if line.strip()]
        return lines[0] if lines else "UNKNOWN_QUESTION"
    
    question_stats: Dict[str, Dict[str, Any]] = {}
    
    print(f"--- 开始聚合来自 {num_files} 个文件的实验结果 ---")

    for i in range(1, num_files + 1):
        file_name = f"{prefix}{i}{suffix}"
        if not os.path.exists(file_name):
            print(f"⚠️ 警告: 文件 {file_name} 不存在，跳过。")
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
            print(f"❌ 错误: 读取文件 {file_name} 时发生异常: {e}")
            continue

        print(f"✅ 成功读取文件: {file_name}，包含 {len(results_list)} 个条目。")

        for item in results_list:
            full_question_text = item.get("question", "")
            score = item.get("score", 0.0)
            response_option_w_content = item.get("response_option_w_content", "NO_RESPONSE")
            response_time = item.get("response_time", 0.0)
            
            # 提取核心问题文本作为唯一键
            core_question_text = extract_core_question(full_question_text)

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
                "response_option_w_content": response_option_w_content,
                "score": score,
                "response_time": response_time,
                "file_id": i
            })

    # ----------------------------------------------------
    # 2. 计算正确率和标记污染
    # ----------------------------------------------------
    
    print("\n--- 计算正确率并进行污染标记 ---")
    pollution_by_core_question: Dict[str, Dict[str, Any]] = {}

    for core_question_text, stats in question_stats.items():
        correct_count = stats["correct_count"]
        total_count = stats["total_count"]
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        pollution_flag = "suspected" if accuracy > threshold else "good"
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
            core_question_text = extract_core_question(qa_item.get("question", ""))
            pollution_result = pollution_by_core_question.get(core_question_text)
            if pollution_result:
                qa_item["pollution_check"] = pollution_result

    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        # Statistics reporting removed - only report pending questions at start
    except Exception as e:
        print(f"❌ 错误: 写入输出文件时发生异常: {e}")

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
        print(f"[Info] Loaded {len(samples)} samples. Starting MI detection...")

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
                print(f"  Processed {idx+1}/{len(samples)} | Loss: {scores['avg_loss']:.4f}")
            
            if api_sleep > 0:
                time.sleep(api_sleep)
        except Exception as e:
            print(f"Error at sample {idx}: {e}")

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
CONTAMINATION_CHECK_ROUNDS = 10  # 污染检测的测试轮数
CONTAMINATION_THRESHOLD = 0.50  # 正确率 >= 50% 标记为可能被污染

def pollution_check_main(
    args,
    input_file_path: str,
    output_file_path: str,
    enable_contamination_check: bool = False,
    cleanup_temp_files: bool = True
) -> str:
    """
    步骤 4: 题目乱序和污染检查
    
    Args:
        args: 命令行参数
        input_file_path: 输入文件路径（v3版本）
        output_file_path: 输出文件路径（v4版本）
        enable_contamination_check: 是否启用污染检测
        cleanup_temp_files: 污染检测完成后是否清理临时文件
    
    Returns:
        处理后的文件路径
    """
    print("\n" + "="*60)
    print("🔄 步骤 4: 题目乱序和污染检查（打乱选项）")
    print("="*60)
    
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
    with open(input_file_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    total_questions = sum(len(item.get("qa", [])) for item in input_data)
    print(f"--- 步骤 4 开始处理：共 {total_questions} 个问题 ---")
    
    # 打乱选项顺序
    rename_and_shuffle_options(input_file_path, output_file_path)
    
    # 可选的污染检测
    def _run_contamination_check() -> None:
        """执行污染检测，所有临时文件统一放在 temp/pollution_check/ 目录下"""
        # 从输出文件路径提取数据集名称
        dataset_name = os.path.basename(output_file_path).replace("_v4.json", "")
        
        # 创建污染检测专用临时目录
        pollution_temp_dir = os.path.join("temp", "pollution_check", dataset_name)
        os.makedirs(pollution_temp_dir, exist_ok=True)
        
        print(f"\n🔍 开始污染检测 ({CONTAMINATION_CHECK_ROUNDS} 轮)")
        print(f"   临时文件目录: {pollution_temp_dir}")
        
        # 临时文件路径
        temp_result_file = os.path.join(pollution_temp_dir, "current_round.json")
        
        # 运行多轮测试
        for idx in range(1, CONTAMINATION_CHECK_ROUNDS + 1):
            print(f"   第 {idx}/{CONTAMINATION_CHECK_ROUNDS} 轮测试中...")
            run_qa_only(answer_llm_config, output_file_path, max_workers, temp_result_file)
            
            # 保存本轮结果到专用目录
            round_result_file = os.path.join(pollution_temp_dir, f"round_{idx}.json")
            run_eval(idx, temp_result_file, output_file=round_result_file)
        
        # 聚合分析结果
        aggregate_and_analyze_results(
            CONTAMINATION_CHECK_ROUNDS,
            os.path.join(pollution_temp_dir, "round_"),
            ".json",
            CONTAMINATION_THRESHOLD,
            output_file_path,
            conversation_file=output_file_path
        )
        
        print(f"✓ 污染检测完成，结果已更新到: {output_file_path}")
        
        # 可选：清理临时文件
        if cleanup_temp_files:
            shutil.rmtree(pollution_temp_dir)
            print(f"✓ 已清理临时文件: {pollution_temp_dir}")
        else:
            print(f"ℹ 临时文件保留在: {pollution_temp_dir}")
    
    if enable_contamination_check:
        _run_contamination_check()

    return output_file_path

