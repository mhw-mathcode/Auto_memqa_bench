#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
选项扰动工具 - 用于生成对抗性问题以测试模型鲁棒性
通常需要与完整的LLM集成和gap计算

注意：此工具需要OpenAI API，建议查看config.json配置
"""

import json
import time
import random
import os
from typing import List, Dict, Any
from openai import OpenAI
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config

# 注意：完整的option_perturbation功能已在src/tools.py中提供
# 此文件保留用于向后兼容性

# =========================
# 基础配置
# =========================
config_loader = get_config()
tool_config = config_loader.get_tool_config("option_perturbation")
GEN_MODEL, SCORE_MODEL = config_loader.get_option_perturbation_models()

tool_llm = tool_config.get("llm", {})
BASE_URL = tool_llm.get("base_url", "")
API_KEY = tool_llm.get("api_key", "")

gen_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
score_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# =========================
# 1. 数据加载
# =========================
def load_qas(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for i, item in enumerate(raw):
        if "qa" in item:
            for j, qa in enumerate(item["qa"]):
                samples.append({
                    "id": f"{i}_{j}",
                    "question": qa["question"],
                    "answer": qa["answer"]
                })
        else:
            samples.append({
                "id": str(i),
                "question": item["question"],
                "answer": item["answer"]
            })
    return samples

# =========================
# 2. 扰动算子（Qwen3）
# =========================
def gen_chat(prompt: str, temp=0.7) -> str:
    resp = gen_client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        extra_body={"enable_thinking": False},
    )
    return resp.choices[0].message.content.strip()

def perturb_paraphrase(q: str) -> str:
    """
    Deeply rewrites the question stem using sophisticated vocabulary and 
    altered syntax while keeping the options untouched.
    """
    prompt = f"""Task: Paraphrase the following multiple-choice question.
Requirements:
1. Reconstruct the question stem (the main question) using significantly different phrasing, sentence structure, and academic vocabulary.
2. Ensure the core semantic meaning and logical requirement remain identical to the original.
3. **Crucial**: Do not modify, reorder, or omit any part of the options list (A, B, C, etc.).
4. Maintain the professional tone of the assessment.

Original Question:
{q}

Output only the restructured question and the original options:"""
    return gen_chat(prompt)

def perturb_back_translation(q: str) -> str:
    """
    Cycles the text through different languages to introduce natural linguistic 
    variation, with a strict constraint on maintaining the MCQ format.
    """
    text = q
    # Using major languages known for distinct syntax (German/French) 
    # then returning to English for the final version.
    for lang in ["German", "French", "English"]:
        prompt = f"""Translate the following content into {lang}. 
If the content contains a list of options (e.g., ['A. ...', 'B. ...']), you MUST preserve the exact list structure and the identifiers (A, B, C). 
Maintain the technical accuracy of the question.

Content:
{text}"""
        text = gen_chat(prompt, temp=0.3)
    return text

def perturb_irrelevant_context(q: str, a: str) -> str:
    """
    Injects a realistic but useless narrative prefix to test the model's 
    ability to filter out noise, using the correct answer to avoid accidental conflicts.
    """
    prompt = f"""Task: Inject an irrelevant distractor context before the given multiple-choice question.

Original Question:
{q}

Correct Answer:
{a}

Requirements:
1. Write a 2-3 sentence background narrative (the "distractor") and place it exactly before the original question.
2. Thematic Alignment: The narrative should loosely match the tone or general topic of the question (e.g., daily life, workplace, decision-making) to appear natural.
3. Entity Interference: Explicitly introduce at least one new, irrelevant person, object, or location to test the QA model's entity confusion.
4. STRICT Neutrality: The distractor MUST NOT contradict the Correct Answer, nor provide any hints or logical shortcuts to it. It must be completely useless for solving the problem.
5. Seamless Transition: Ensure the text flows naturally from the distractor into the original question.
6. Keep Options Intact: Do not modify, reorder, or omit the original options (A, B, C, D) in the question.

Output ONLY the final combined text (distractor + original question + options):"""
    return gen_chat(prompt)

# =========================
# 3. 置信度计算（Min-K% Loss）
# =========================
def compute_confidence(question: str, answer: str, k_percent=0.2) -> float:
    # 1. 明确边界：加入换行和提示词，防止问题末尾的单词与答案首字母粘连成新的 Token
    # 去除 question 末尾的空白字符，确保拼接格式稳定
    prompt = question.rstrip() + "\nAnswer: " + answer

    resp = score_client.completions.create(
        model=SCORE_MODEL,
        prompt=prompt,
        temperature=0.0,
        logprobs=1,
        echo=True,
        max_tokens=1,
    )

    lp = resp.choices[0].logprobs
    token_logprobs = lp.token_logprobs
    text_offsets = lp.text_offset  # 获取每个 token 的字符偏移量

    # 2. 计算 answer 在完整 prompt 中的起始字符索引
    answer_start_idx = len(prompt) - len(answer)

    # 3. 根据 text_offset 提取属于 answer 的 token 概率
    ans_nll = []
    for i, offset in enumerate(text_offsets):
        # 如果该 token 的起始位置落在了 answer 的范围内
        if offset >= answer_start_idx:
            prob = token_logprobs[i]
            if prob is not None:
                ans_nll.append(-prob)  # 取负的 logprob 得到 NLL (Negative Log Likelihood)

    # 防御性编程：如果因为某种原因没提取到（极罕见），返回极小值
    if not ans_nll:
        return -999.0

    # 4. 计算 Min-K% Loss
    ans_nll.sort(reverse=True)
    k = max(1, int(len(ans_nll) * k_percent))
    mink_loss = sum(ans_nll[:k]) / k

    # 用负 loss 作为置信度（越大越自信）
    return -mink_loss

# =========================
# 4. 单样本 Gap 计算
# =========================
def process_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    Q, A = sample["question"], sample["answer"]

    conf_orig = compute_confidence(Q, A)

    Q_para = perturb_paraphrase(Q)
    conf_para = compute_confidence(Q_para, A)

    Q_bt = perturb_back_translation(Q)
    conf_bt = compute_confidence(Q_bt, A)

    # 【修改这里】传入答案 A
    Q_noise = perturb_irrelevant_context(Q, A)  
    conf_noise = compute_confidence(Q_noise, A)

    gaps = {
        "para": conf_orig - conf_para,
        "bt": conf_orig - conf_bt,
        "noise": conf_orig - conf_noise,
    }

    sample.update({
        "conf_orig": conf_orig,
        "conf_para": conf_para,
        "conf_bt": conf_bt,
        "conf_noise": conf_noise,
        "gap_para": gaps["para"],
        "gap_bt": gaps["bt"],
        "gap_noise": gaps["noise"],
        "gap": max(gaps.values()),
    })
    return sample

# =========================
# 5. 主流程
# =========================
def run_gap_pipeline(
    input_path: str,
    output_path: str = "final_challenge_set.json",
    processed_path: str | None = None,
    sleep: float = 0.5,
):
    # =====================================================
    # 1. 读取 RAW（全集）
    # =====================================================
    raw_id2qa = {}

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    for bi, block in enumerate(raw_data):
        for qi, qa in enumerate(block.get("qa", [])):
            qid = qa.get("id", f"{bi}_{qi}")
            raw_id2qa[qid] = qa

    print(f"[INFO] 原始问题总数: {len(raw_id2qa)}")

    # =====================================================
    # 2. 读取已有 gap 结果（如果有）
    # =====================================================
    processed_id2item = {}

    if processed_path and os.path.exists(processed_path):
        with open(processed_path, "r", encoding="utf-8") as f:
            processed = json.load(f)

        processed_id2item = {
            item["id"]: item for item in processed
        }

        print(f"[INFO] 读取已有 gap 结果: {len(processed_id2item)}")

    final_results = []

    # =====================================================
    # 3. 遍历【RAW 全集】
    # =====================================================
    for i, (qid, raw_qa) in enumerate(raw_id2qa.items()):
        Q = raw_qa["question"]

        # -------------------------------------------------
        # A. gap 已存在
        # -------------------------------------------------
        if qid in processed_id2item:
            item = processed_id2item[qid]

        # -------------------------------------------------
        # B. gap 不存在 → 补跑
        # -------------------------------------------------
        else:
            print(f"⚠️ [{i+1}] {qid} 未处理，开始补跑 gap")

            while True:
                try:
                    sample = {
                        "id": qid,
                        "question": Q,
                        "answer": raw_qa["answer"],
                    }
                    item = process_sample(sample)
                    processed_id2item[qid] = item
                    break
                except Exception as e:
                    print(f"❌ {qid} gap 失败，重试: {e}")
                    time.sleep(30)

            time.sleep(sleep)

        # -------------------------------------------------
        # 4. 选最优扰动类型
        # -------------------------------------------------
        gaps = {
            "para": item.get("gap_para", -1e9),
            "bt": item.get("gap_bt", -1e9),
            "noise": item.get("gap_noise", -1e9),
        }
        best_type = max(gaps, key=gaps.get)

        # -------------------------------------------------
        # 5. 重新生成 best_question（真正的扰动文本）
        # -------------------------------------------------
        while True:
            try:
                if best_type == "para":
                    best_q = perturb_paraphrase(Q)
                elif best_type == "bt":
                    best_q = perturb_back_translation(Q)
                elif best_type == "noise":
                    # 【修改这里】传入 raw_qa["answer"]
                    best_q = perturb_irrelevant_context(Q, raw_qa["answer"]) 
                else:
                    best_q = Q
                break
            except Exception as e:
                print(f"❌ {qid} 生成 {best_type} 失败，重试: {e}")
                time.sleep(30)

        # -------------------------------------------------
        # 6. 汇总
        # -------------------------------------------------
        final_results.append({
            "id": qid,
            "character": raw_qa.get("character"),
            "category": raw_qa.get("category"),
            "original_question": Q,
            "best_question": best_q,
            "best_type": best_type,
            "best_gap": item.get("gap"),
            "conf_orig": item.get("conf_orig"),
        })

        print(f"✅ [{i+1}/{len(raw_id2qa)}] {qid} best={best_type} gap={item.get('gap', 0):.4f}")

    # =====================================================
    # 7. 按 gap 排序 + 写文件
    # =====================================================
    final_results.sort(
        key=lambda x: (x["best_gap"] is not None, x["best_gap"]),
        reverse=True,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"\n🎯 最终挑战集生成完成：{output_path}")
    print(f"📦 总题目数: {len(final_results)}")

    return final_results


def replace_questions_by_text(raw_path, best_path, output_path):

    RAW_PATH = raw_path
    BEST_PATH = best_path
    OUTPUT_PATH = output_path
    # ========= 1. 读取 best question 映射 =========
    with open(BEST_PATH, "r", encoding="utf-8") as f:
        best_data = json.load(f)

    orig2best = {}
    for item in best_data:
        orig_q = item["original_question"].strip()
        best_q = item["best_question"]
        orig2best[orig_q] = best_q

    print(f"[INFO] Loaded mapping size: {len(orig2best)}")

    # ========= 2. 读取原始数据 =========
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    replaced = 0
    missing = 0

    # ========= 3. 遍历并替换 =========
    for block in raw_data:
        for qa in block.get("qa", []):
            q_text = qa["question"].strip()

            if q_text in orig2best:
                qa["question"] = orig2best[q_text]
                replaced += 1
            else:
                missing += 1

    # ========= 4. 写新文件 =========
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    print("\n✅ 替换完成")
    print(f"  - 成功替换: {replaced}")
    print(f"  - 未匹配到: {missing}")
    print(f"  - 输出文件: {OUTPUT_PATH}")

# =========================
# 6. CLI 入口
# =========================
if __name__ == "__main__":

    RAW_PATH = "dataset/friends_merged20_ds_shuffle.json"
    PROCESSED_PATH = "final_best_question.json"
    OUTPUT_PATH = "dataset/friends_merged20_ds_shuffle_adv.json"

    run_gap_pipeline(
        input_path=RAW_PATH,
        output_path=PROCESSED_PATH,
    )

    replace_questions_by_text(
        raw_path=RAW_PATH,
        best_path=PROCESSED_PATH,
        output_path=OUTPUT_PATH,
    )

