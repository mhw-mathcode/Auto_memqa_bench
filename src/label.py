import json
import time
import os
import re
from openai import OpenAI
import random

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_MODEL = os.getenv("LABEL_MODEL", "qwen3-14b")

LABELS = [
    "Fact Extraction (Single Dialogue)",
    "Fact Extraction (Multiple Dialogues)",
    "Memory Update",
    "Multi-hop",
]

PROMPT_TEMPLATE = """
You are a strict question type classifier and do not need to answer the question content.

Given:
1. A conversation
2. A question
3. A reference answer

Please determine which cognitive ability is required to answer the question.

[Allowed labels (choose exactly one, copy exactly)]
Fact Extraction (Single Dialogue)
Fact Extraction (Multiple Dialogues)
Memory Update
Multi-hop

[Critical output constraints]
- Output MUST be exactly one label from the allowed list above.
- Do NOT output explanations, punctuation, extra words, or multiple lines.
- If uncertain, still output the single most appropriate label from the allowed list.

【Judgment Criteria】
- If the answer can be provided based on a single scene and a single dialogue → Fact Extraction (Single Dialogue)
- If it is necessary to integrate facts from multiple dialogues → Fact Extraction (Multiple Dialogues)
- If it involves the correction of false memories and the discovery of the truth later → Memory Update
- If it is necessary to summarize the character's personality, motives, values, and long-term consistency → Multi-hop

【Dialogue】 
{conversation}

【Question】 
{question}

【Reference Answer】
{answer}

Output exactly one label:
"""


def _normalize_label(raw_label: str):
    """将模型输出归一化到四个合法标签之一。"""
    if not raw_label:
        return None

    first_line = next((line.strip() for line in raw_label.splitlines() if line.strip()), "")
    cleaned = first_line.strip().strip("`\"'").lstrip("-*•").strip()
    if cleaned in LABELS:
        return cleaned

    alias_map = {
        "fact extraction (single dialogue)": "Fact Extraction (Single Dialogue)",
        "fact extraction (single dialogues)": "Fact Extraction (Single Dialogue)",
        "fact extraction (single conversation)": "Fact Extraction (Single Dialogue)",
        "fact extraction (multiple dialogue)": "Fact Extraction (Multiple Dialogues)",
        "fact extraction (multiple dialogues)": "Fact Extraction (Multiple Dialogues)",
        "fact extraction (multiple conversations)": "Fact Extraction (Multiple Dialogues)",
        "memory update": "Memory Update",
        "multi hop": "Multi-hop",
        "multi-hop": "Multi-hop",
        "multihop": "Multi-hop",
    }

    normalized_key = re.sub(r"\s+", " ", cleaned.lower()).strip()
    if normalized_key in alias_map:
        return alias_map[normalized_key]

    if "memory" in normalized_key:
        return "Memory Update"
    if "multi" in normalized_key and "hop" in normalized_key:
        return "Multi-hop"
    if "fact" in normalized_key and "single" in normalized_key:
        return "Fact Extraction (Single Dialogue)"
    if "fact" in normalized_key and "multiple" in normalized_key:
        return "Fact Extraction (Multiple Dialogues)"

    return None


def classify_question(conversation, question, answer, client, model):
    prompt = PROMPT_TEMPLATE.format(
        conversation=conversation,
        question=question,
        answer=answer,
    )

    messages = [
        {"role": "system", "content": "你是一个严格的分类器。"},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(10):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=32,
                extra_body={"enable_thinking": False},
            )

            # 基本合法性检查
            if (
                resp
                and resp.choices
                and resp.choices[0].message
                and resp.choices[0].message.content
            ):
                raw_label = resp.choices[0].message.content.strip()
                normalized_label = _normalize_label(raw_label)
                if normalized_label:
                    return normalized_label

                raise ValueError(f"Invalid label output: {raw_label}")

            raise ValueError("Empty response")

        except Exception as e:
            sleep_time = 60 + random.uniform(0, 10)
            print(f"[Retry {attempt+1}/{10}] API failed: {e}")
            time.sleep(sleep_time)

    return "null"


def label_main(input_file_path: str, output_file_path: str,
               api_key: str = DEFAULT_API_KEY, base_url: str = DEFAULT_BASE_URL,
               model_name: str = DEFAULT_MODEL) -> str:
    """
    步骤 2: 题目标注
    
    Args:
        input_file_path: 输入文件路径（v1b 版本）
        output_file_path: 输出文件路径（v2版本）
        api_key: API密钥
        base_url: API基础URL
        model_name: 模型名称
    
    Returns:
        处理后的文件路径
    """
    print("\n" + "="*60)
    print("🔄 步骤 2: 题目标注")
    print("="*60)
    print(f"📥 输入文件: {input_file_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 读取输入文件
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 如果是单个对象，则转换为列表
    if isinstance(data, dict):
        data = [data]
    
    # 计算待处理问题数
    total_questions = sum(len(item.get("qa", [])) for item in data)
    print(f"--- 步骤 2 开始处理：共 {total_questions} 个问题 ---")
    
    # 处理数据 - 保持原始的列表结构
    final_results = []
    total_qa_count = 0
    skipped_count = 0
    
    for item in data:
        conversation = item.get("conversation", {})
        questions = item.get("qa", [])
        
        # 为每个问题添加标签
        labeled_questions = []
        for q in questions:
            # 筛选逻辑：
            # 1) only_evidence_check.result 必须为 right
            # 2) 若 iterative_evidence_ablation 第 3 轮仍非 wrong，则跳过
            should_skip = False

            only_evidence_check = q.get("only_evidence_check", {})
            only_result = only_evidence_check.get("result", "") if isinstance(only_evidence_check, dict) else ""
            if only_result != "right":
                should_skip = True

            iterative_ablation = q.get("iterative_evidence_ablation", [])
            if isinstance(iterative_ablation, list):
                for record in iterative_ablation:
                    if record.get("round") == 3 and record.get("result") != "wrong":
                        should_skip = True
                        break
                    
            if should_skip:
                # 未通过题目合理性验证：从后续流程中移除
                skipped_count += 1
                print(f"跳过标注（题目未通过合理性验证）: {q.get('question', '')[:50]}...")
            else:
                # 正常标注
                question_text = q.get("question", "")
                answer_text = q.get("answer", "")
                print(f"正在标注: {question_text[:50]}...")
                
                label = classify_question(conversation, question_text, answer_text, client, model_name)
                q["label"] = label
                labeled_questions.append(q)
                
                time.sleep(0.5)  # 防止限速
        
        # 深拷贝原始item以保留所有字段
        import copy
        result_item = copy.deepcopy(item)
        # 只更新 qa 字段
        result_item["qa"] = labeled_questions
        final_results.append(result_item)
        total_qa_count += len(labeled_questions)
    
    # 保存结果 - 保持列表格式
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    # Statistics reporting removed - only report pending questions at start
    return output_file_path

