import json
import time
import os
from openai import OpenAI
import random

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_MODEL = os.getenv("LABEL_MODEL", "qwen3-14b")

LABELS = [
    "事实提取（单对话）",
    "事实提取（多对话）",
    "记忆更新（失效的记忆，更新的记忆）",
    "多跳",
    "弃权"
]

PROMPT_TEMPLATE = """
你是一个问题类型标注器，不需要回答问题内容。

给定：
1. 一段对话（conversation）
2. 一个问题（question）

请判断：回答这个问题需要哪一种认知类型。

【可选标签（只能选一个，原样输出，不要解释）】
- 事实提取（单对话）
- 事实提取（多对话）
- 记忆更新（失效的记忆，更新的记忆）
- 多跳

【判定标准】
- 单一场景、单一对话即可回答 → 事实提取（单对话）
- 需要整合多段对话中的事实 → 事实提取（多对话）
- 涉及错误记忆被纠正、后来才发现真相 → 记忆更新（失效的记忆，更新的记忆）
- 需要总结人物性格、动机、价值观、长期一致性 → 多跳

【对话】
{conversation}

【问题】
{question}

请直接输出标签：
"""

def classify_question(conversation, question, client, model):
    prompt = PROMPT_TEMPLATE.format(
        conversation=conversation,
        question=question
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
                extra_body={"enable_thinking": False},
            )

            # 基本合法性检查
            if (
                resp
                and resp.choices
                and resp.choices[0].message
                and resp.choices[0].message.content
            ):
                label = resp.choices[0].message.content.strip()
                return label

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
    问题分类标注主函数
    
    Args:
        input_file_path: 输入文件路径（v2版本）
        output_file_path: 输出文件路径（v3版本）
        api_key: API密钥
        base_url: API基础URL
        model_name: 模型名称
    
    Returns:
        处理后的文件路径
    """
    print("\n" + "="*60)
    print("🔄 步骤 3: 问题分类标注")
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
    print(f"--- 步骤 3 开始处理：共 {total_questions} 个问题 ---")
    
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
            # 筛选逻辑：检查 iterative_evidence_ablation 中是否有 round=1 且 result="wrong"
            should_skip = False
            iterative_ablation = q.get("iterative_evidence_ablation", [])
            if iterative_ablation:
                for record in iterative_ablation:
                    if record.get("round") == 1 and record.get("result") == "wrong":
                        should_skip = True
                        break
            
            if should_skip:
                # 未通过题目合理性验证：从后续流程中移除
                skipped_count += 1
                print(f"跳过标注（未通过题目合理性验证）: {q.get('question', '')[:50]}...")
            else:
                # 正常标注
                question_text = q.get("question", "")
                print(f"正在标注: {question_text[:50]}...")
                
                label = classify_question(conversation, question_text, client, model_name)
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


def main():
    """原有的主函数，保留用于独立运行"""
    merged_conversation = {}
    all_qa = []
    session_counter = 1  # session计数器

    client = OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)
    for idx in range(1, 6):
        with open(f"An-Enemy-of-the-People/An-Enemy-of-the-People_{idx}.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            conversation = item["conversation"]
            questions = item.get("qa", [])
            
            # 将当前conversation中的所有session合并到merged_conversation中
            for session_key, session_content in conversation.items():
                new_session_key = f"session_{session_counter}"
                merged_conversation[new_session_key] = session_content
                session_counter += 1

            for q in questions:
                print(q.get("question"))
                # 注意：这里需要传入完整的conversation，而不仅仅是当前片段的对话
                label = classify_question(conversation, q.get("question"), client, DEFAULT_MODEL)
                q["label"] = label
                all_qa.append(q)
                time.sleep(0.5)  # 防止限速

    # 创建最终合并结果
    merged_result = {
        "id": "merged_version",
        "conversation": merged_conversation,
        "qa": all_qa
    }

    with open("An-Enemy-of-the-People_merged.json", "w", encoding="utf-8") as f:
        json.dump(merged_result, f, ensure_ascii=False, indent=4)

    print(f"合并完成！总共合并了 {session_counter-1} 个 session，{len(all_qa)} 个问答对。")


if __name__ == "__main__":
    main()

