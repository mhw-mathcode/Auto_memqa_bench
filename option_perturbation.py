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

def perturb_irrelevant_context(q: str) -> str:
    """
    Injects a realistic but useless narrative prefix to test the model's 
    ability to filter out noise.
    """
    prompt = f"""Task: Insert a 'distractor context' before the question.
Requirements:
1. Write a 2-3 sentence background paragraph that is semantically natural and shares a general theme with the question (e.g., life choices, social expectations).
2. The added context must be **entirely irrelevant** to solving the problem; it should provide no hints or conflicting data.
3. Transition smoothly from the noise context to the original question.

Original Question:
{q}

Output the noise-injected version:"""
    return gen_chat(prompt)

# =========================
# 3. 置信度计算（Min-K% Loss）
# =========================
def compute_confidence(question: str, answer: str, k_percent=0.2) -> float:
    prompt = question + answer

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

    # 近似认为 answer 在末尾
    ans_len = len(answer)
    ans_lps = token_logprobs[-ans_len:]

    nll = [-x for x in ans_lps if x is not None]
    if not nll:
        return -999.0

    nll.sort(reverse=True)
    k = max(1, int(len(nll) * k_percent))
    mink_loss = sum(nll[:k]) / k

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

    Q_noise = perturb_irrelevant_context(Q)
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
                    best_q = perturb_irrelevant_context(Q)
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

