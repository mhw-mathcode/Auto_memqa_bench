import os
import json
from typing import Any, Dict, Optional, List
from openai import OpenAI
import re
import time
import random
import argparse
from tqdm import tqdm

# 导入配置模块和各个处理模块
from config import PipelineConfig, VersionManager, build_llm_config
from src.pollution_check import pollution_check_main
from src.full_context import full_context_main
from src.new_qa import new_qa_main
from src.label import label_main

QA_GENERATE_PROMPT = """
{conversation}

Role:
You are a top-tier AI evaluation expert, specializing in designing extremely high-difficulty stress test datasets for evaluating large language models’ long-range, cross-conversation memory.

Task:
Based on the provided long text dialogue, please design 1 high-quality question-answer pair for each of the six users {user_list} for each of the four specific categories 1 to 4 (total of {total_question_num} pairs).

Part I: Core Objectives and Depth Requirements

1. Cross-Conversation Reasoning:
- It is strictly forbidden to generate questions that can be answered using only a single session or a single utterance.
- Each question must require the model to extract and integrate information from at least two (preferably three or more) distinct conversation fragments.
- Hard-case preference: prioritize fragmented information where a clue is planted in Session A, indirectly referenced in Session B, and only revealed or resolved in Session C.

2. Extreme Source Constraints:
- Absolutely no external knowledge, common sense assumptions, associative reasoning, or hallucinations are allowed.
- If a fact is not explicitly stated or logically necessitated by the dialogue, it must be treated as non-existent.

Part II: Strict Definitions of the Four Question Categories

1. Category 1 - Long-term Persona: Examines stable identity, underlying values, long-term preferences, or behavioral patterns. It must be a consistent characteristic exhibited across multiple sessions.

2. Category 2 - Short-term State: Examines immediate emotions, short-term needs, or temporary goals in specific situations. The focus is on capturing the specific triggers that generate this state.

3. Category 3 - Temporal: Examines the absolute/relative timing of events, sequential causal relationships, or the replacement of old and new information. Requires inferring logical chains through multiple timestamps.

4. Category 4 - Plot-driven Event (Event/Experience): Examines the cause, course, and outcome of a specific experience, as well as the subjective evaluation of the participants. It must include specific actions or decisions.

Part III: De-featureization and Strong Confusion Design

1. Question Stem Design (Natural & Implicit):
- Feature leakage is forbidden. Do not use phrases such as “based on their introverted personality” or “shows a stable coping pattern.”
- Questions must read like natural user inquiries.
  Incorrect: “Which option reflects Ariel’s stable breakup-coping style?”
  Correct: “Which statement best matches how Ariel dealt with the aftermath of the breakup?”
- Language should be direct, concrete, and non-rhetorical.

2. Hard Distractor Requirements:
- Length balance (critical): the correct answer must not be the longest or shortest among the five options.
  At least one distractor must be longer than the correct answer.
- Semantic proximity: distractors must be highly plausible and lie in a high-probability semantic neighborhood.
  Avoid extreme terms such as “always,” “never,” “completely,” or “absolutely.”
- Information confusion: distractors must include
  (1) outdated statements from the target user,
  (2) true information belonging to another character (e.g., Bennett),
  (3) statements that are logically similar but factually incorrect.
- Mutual independence: options must not overlap semantically.
  No option may partially contain another option’s content.

Part IV: Structured Proof (Necessary and Sufficient Condition Validation)

1. Atomic Extraction:
- Verbatim copying only. No paraphrasing or summarization is allowed.
- Semantic completeness: if an utterance contains pronouns (e.g., “he”),
  the immediately preceding utterance that resolves the reference must also be included.
- Single-ID constraint: each evidence item must correspond to exactly one dia_id.
  Merged IDs such as “1-2 1-3” are strictly forbidden.

2. The “Island” Self-Sufficiency Test:
- Logical closure: a third party reading only the evidence must be able to derive one and only one correct answer.
- No implicit knowledge: common sense or personality inference is forbidden.
  All reasoning must follow the form:
  E1 + E2 → Inference
- Textual traceability: every fact used in reasoning_steps must have a direct match in evidence_dialogues.
  Logical jumps are not allowed.

3. Self-Verification Metrics:
Before outputting the final JSON, both checks must be satisfied:
- Sufficiency: are the evidence items alone sufficient to 100% eliminate all four distractors?
- Necessity (minimality): if any single evidence item is removed, does the reasoning chain break?
  Ensure no redundancy or unnecessary information.

Part V: Output JSON Specification

{{
  "qa": [
    {{
      "character": "Ariel",
      "category": 1,
      "question": "[Direct, natural, focused on the character]",
      "option": [
        "A. ",
        "B. ",
        "C. ",
        "D. ",
        "E. "
      ],
      "answer": "C",
      "evidence_dialogues": [
        {{ "id": "E1", "speaker": "Ariel", "utterance": "...", "dia_id": "" }},
        {{ "id": "E2", "speaker": "Ariel", "utterance": "...", "dia_id": "" }}
      ],
      "reasoning_steps": [
        {{
          "step": 1,
          "inference": "[Intermediate logic]",
          "based_on": ["E1"]
        }},
        {{
          "step": 2,
          "inference": "[Cross-session conclusion]",
          "based_on": ["E1", "E2"]
        }}
      ]
    }}
  ]
}}
"""


QA_GENERATE_PROMPT_2 = """
{conversation}

Role:
You are a top-tier AI evaluation expert, specializing in designing extremely high-difficulty stress test datasets for evaluating large language models’ long-range, cross-conversation memory.

Task:
Based on the above conversation, construct questions that either focus on interpersonal relationships (such as social roles, intentions, power dynamics, or implicit emotional interactions) or require reasoning over fine-grained, specific data details within the provided context. The number of questions is flexible, but every question must be strongly aligned with its intended type and rely on inference or precise data reasoning rather than surface-level factual recall.

Part I: Core Objectives and Depth Requirements

1. Cross-Conversation Reasoning:
- It is strictly forbidden to generate questions that can be answered using only a single session or a single utterance.
- Each question must require the model to extract and integrate information from at least two (preferably three or more) distinct conversation fragments.
- Hard-case preference: prioritize fragmented information where a clue is planted in Session A, indirectly referenced in Session B, and only revealed or resolved in Session C.

2. Extreme Source Constraints:
- Absolutely no external knowledge, common sense assumptions, associative reasoning, or hallucinations are allowed.
- If a fact is not explicitly stated or logically necessitated by the dialogue, it must be treated as non-existent.

Part II: Strict Definitions of the Four Question Categories

Category 5 – Interpersonal Relationship Questions

Questions that focus on explicit interpersonal relationships between individuals, such as family ties (e.g., siblings), living arrangements (e.g., roommates), educational or professional relationships (e.g., classmates, colleagues), or other clearly stated relational roles. The question must be answerable only by correctly identifying or reasoning about the concrete relationship between people, not by interpreting emotions, personalities, or abstract social norms.

Category 6 – Fine-Grained Data Questions

Questions that require reasoning over explicit numerical information mentioned in the context, such as counts, dates, ages, durations, quantities, or other numeric values appearing in the narrative. All answer options should be numbers or numeric expressions, and the correct answer must depend on precise extraction, comparison, or calculation based strictly on the provided data.

Part III: De-featureization and Strong Confusion Design

1. Question Stem Design (Natural & Implicit):
- Feature leakage is forbidden. Do not use phrases such as “based on their introverted personality” or “shows a stable coping pattern.”
- Questions must read like natural user inquiries.
  Incorrect: “Which option reflects Ariel’s stable breakup-coping style?”
  Correct: “Which statement best matches how Ariel dealt with the aftermath of the breakup?”
- Language should be direct, concrete, and non-rhetorical.

2. Hard Distractor Requirements:
- Length balance (critical): the correct answer must not be the longest or shortest among the five options.
  At least one distractor must be longer than the correct answer.
- Semantic proximity: distractors must be highly plausible and lie in a high-probability semantic neighborhood.
  Avoid extreme terms such as “always,” “never,” “completely,” or “absolutely.”
- Information confusion: distractors must include
  (1) outdated statements from the target user,
  (2) true information belonging to another character (e.g., Bennett),
  (3) statements that are logically similar but factually incorrect.
- Mutual independence: options must not overlap semantically.
  No option may partially contain another option’s content.

Part IV: Structured Proof (Necessary and Sufficient Condition Validation)

1. Atomic Extraction:
- Verbatim copying only. No paraphrasing or summarization is allowed.
- Semantic completeness: if an utterance contains pronouns (e.g., “he”),
  the immediately preceding utterance that resolves the reference must also be included.
- Single-ID constraint: each evidence item must correspond to exactly one dia_id.
  Merged IDs such as “1-2 1-3” are strictly forbidden.

2. The “Island” Self-Sufficiency Test:
- Logical closure: a third party reading only the evidence must be able to derive one and only one correct answer.
- No implicit knowledge: common sense or personality inference is forbidden.
  All reasoning must follow the form:
  E1 + E2 → Inference
- Textual traceability: every fact used in reasoning_steps must have a direct match in evidence_dialogues.
  Logical jumps are not allowed.

3. Self-Verification Metrics:
Before outputting the final JSON, both checks must be satisfied:
- Sufficiency: are the evidence items alone sufficient to 100% eliminate all four distractors?
- Necessity (minimality): if any single evidence item is removed, does the reasoning chain break?
  Ensure no redundancy or unnecessary information.

Part V: Output JSON Specification

{{
  "qa": [
    {{
      "character": "Ariel",
      "category": 5/6,
      "question": "[Direct, natural, focused on the character]",
      "option": [
        "A. ",
        "B. ",
        "C. ",
        "D. ",
        "E. "
      ],
      "answer": "C",
      "evidence_dialogues": [
        {{ "id": "E1", "speaker": "Ariel", "utterance": "...", "dia_id": "" }},
        {{ "id": "E2", "speaker": "Ariel", "utterance": "...", "dia_id": "" }}
      ],
      "reasoning_steps": [
        {{
          "step": 1,
          "inference": "[Intermediate logic]",
          "based_on": ["E1"]
        }},
        {{
          "step": 2,
          "inference": "[Cross-session conclusion]",
          "based_on": ["E1", "E2"]
        }}
      ]
    }}
  ]
}}
"""

QA_GENERATE_PROMPT_3 = """
{conversation}

You are a top-tier AI evaluation expert specializing in designing stress test datasets to evaluate a model's ability to correctly refuse to answer when sufficient information is not available in the provided context. Your task is to construct questions that cannot be answered based solely on the provided long dialogue text. The model should be forced to 'abstain' or 'decline to answer' because the necessary information is missing, or requires external knowledge or inference beyond what is strictly present in the text.

Based on the provided long dialogue text, please design 1 question per user (Dr. Stockmann, Hovstad, Mrs. Stockmann, Peter Stockmann, Petra) for each of the four categories (1-4). This results in a total of 20 question-answer pairs.

Core Design Principle: Unanswerability
Every question must be impossible to answer definitively using only the information explicitly stated or logically entailed within the provided dialogue. Achieve this through:

Information Gap: The question asks about a fact, motive, state, or event that is never mentioned, described, or implied in any part of the dialogue.

Ambiguity/Contradiction: The dialogue contains conflicting information from different speakers or sessions about the key point of the question, making a single definitive answer impossible.

Requires External Knowledge: Answering correctly would require common sense, real-world knowledge, or associative reasoning that is not contained within the dialogue's text.

Temporal Impossibility: The question asks about an event clearly stated to happen after the dialogue's end or before its beginning, with no details given in the text.

Strict Adherence to Category Definitions (For Question Construction Only)
Construct questions that appear to fit these categories, but whose answers are unattainable.

Category 1 - Long-term Persona: Ask about a stable trait, value, or pattern that is not demonstrated or discussed across the sessions.

Category 2 - Short-term State: Ask about a specific emotion, need, or goal in a precise moment that is not revealed by the character's words or actions in the text.

Category 3 - Temporal: Ask about the timing, sequence, or duration of events that are not specified or are contradicted within the dialogue.

Category 4 - Plot-driven Event: Ask about the cause, detail, or outcome of a specific experience/action that is not described in the dialogue.

Question & Distractor Design for Unanswerability
Question Stem: Must sound natural and plausible, as if it could be answered by a knowledgeable reader.

Options: Include five options (A-E).

FIve Distractors: Each should be a plausible-sounding but incorrect statement. They should be constructed from out-of-context quotes, misattributed actions, or logical leaps not supported by the text.

One Correct Answer: F. Cannot infer the answer based on the given information.

Distractor Quality: Distractors should be semantically close to topics in the dialogue to create confusion, but must not be verifiably true based solely on the text.

Evidence & Reasoning Requirements (Proving Unanswerability)
evidence_dialogues: Provide 2-4 key dialogue snippets that are most relevant to the question's topic. These snippets should demonstrate that the information needed to answer is absent, vague, or contradictory. It is acceptable if the evidence shows a character discussing the topic without providing the asked-for information.

reasoning_steps: The logical steps must prove why no option can be confirmed as true and why the correct answer is to abstain.

Output Format

Output a valid JSON list named "qa" containing 20 objects.

Step 1: State what the question is asking for.

Step 2: Analyze the provided evidence to show the information gap, contradiction, or external knowledge requirement.

Step 3: Conclude that all distractors are unverifiable or false, and the only valid response is to decline to answer.

{{
  "qa": [
    {{
      "character": "Dr. Stockmann",
      "category": 1,
      "question": "[A natural-sounding question about the character that cannot be answered]",
      "options": [
        "A. [Plausible but unverifiable/distractor statement 1]",
        "B. [Plausible but unverifiable/distractor statement 2]",
        "C. [Plausible but unverifiable/distractor statement 3]",
        "D. [Plausible but unverifiable/distractor statement 4]",
        "E. [Plausible but unverifiable/distractor statement 5]",
        "F. Cannot infer the answer based on the given information."
      ],
      "answer": "F",
      "evidence_dialogues": [
        {{"id": "E1", "speaker": "X", "utterance": "...", "dia_id": "X-Y"}},
        {{"id": "E2", "speaker": "Y", "utterance": "...", "dia_id": "X-Z"}}
      ],
      "reasoning_steps": [
        {{
          "step": 1,
          "inference": "The question asks for [specific information X].",
          "based_on": []
        }},
        {{
          "step": 2,
          "inference": "The provided dialogues show characters discussing related topic Y, but never address or specify X. / The dialogues present conflicting views on X between Speaker A and Speaker B.",
          "based_on": ["E1", "E2"]
        }},
        {{
          "step": 3,
          "inference": "Options A-D make claims that are either contradicted by the text, attributed to the wrong character, or require assumptions beyond the text. Therefore, the only supportable conclusion is that the information is unavailable.",
          "based_on": ["E1", "E2"]
        }}
      ]
    }}
  ]
}}
"""

def call_openai_json(
    answer_prompt: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_s: int = 120
) -> Dict[str, Any]:
    """
    Invoke the large model to generate a JSON file
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL", None)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # -------- strict json helpers --------
    def _strip_code_fences(text: str) -> str:
        t = (text or "").strip()
        if t.startswith("```"):
            # remove opening fence: ``` or ```json
            t = re.sub(r"^\s*```(?:json)?\s*\n?", "", t, flags=re.IGNORECASE)
            # remove closing fence
            t = re.sub(r"\n?\s*```\s*$", "", t)
            t = t.strip()
        return t

    def _strict_json_loads(text: str) -> Dict[str, Any]:
        t = _strip_code_fences(text)
        obj = json.loads(t)  # strict: must parse directly
        if not isinstance(obj, dict):
            raise ValueError(f"Top-level JSON must be an object/dict, got {type(obj)}")
        return obj

    # -------- retry loop --------
    max_retries = 10  # json 生成失败重试次数
    last_content = ""

    for attempt in range(max_retries + 1):
        MAX_OTHER_ERROR_RETRIES = 10 # 其他请求失败重试次数
        llm_error_retries = 0
        other_error_retries = 0
        resp = None
        while True:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": answer_prompt}
                    ],
                    extra_body={
                        "enable_thinking": False
                    }
                )
                break
            except Exception as e:
                # --- 打印完整 Traceback ---
                # print(f"\n[ERROR] Attempt {attempt} failed for model {model}:")
                # traceback.print_exc() # 这行会打印完整的错误堆栈，包括错误发生的行号
                error_str = str(e).lower()
                print(error_str)
                if "rate limit" in error_str or "limit" in error_str or "overloaded" in error_str or "token" in error_str:
                    # 识别为 TPM 则一直重试
                    llm_error_retries += 1
                    other_error_retries = 0 # 重置其他错误计数
                    sleep_duration = random.uniform(2, 20) + 5 * llm_error_retries
                    error_message = f"LLM Rate Limit related Error. Retrying in {sleep_duration:.2f}s... Error: {e}"
                    print(error_message)
                    time.sleep(sleep_duration)
                else:
                    # 识别为其他错误
                    other_error_retries += 1
                    print("other_error_retries: ", other_error_retries)
                    if other_error_retries >= MAX_OTHER_ERROR_RETRIES:
                        response_content = "Error: Default response due to unrecoverable error." # 设置默认值
                        print(response_content)
                        break # 达到最大次数，跳出循环

        last_content = resp.choices[0].message.content or ""

        try:
            return _strict_json_loads(last_content)
        except Exception:
            # invalid json -> retry
            continue

    # if still invalid after retries
    raise ValueError(
        f"Judge did not return valid JSON after {max_retries + 1} attempts.\n"
        f"Last output:\n{last_content}"
    )


def main():
    """
    问答数据集处理主流程
    
    数据结构说明：
    - 所有版本都使用统一的列表格式：
      [
        {
          "filename": "原始文件名",
          "conversation": {...对话内容...},
          "qa": [...问答列表...]
        },
        ...
      ]
    
    流程说明：
    v0: 原始生成的问答对（每个文件对应一个列表元素）
    v1: 打乱选项和污染检查后的数据
    v2: 全上下文验证后的数据（保留答对的题目）
    v3: 标注分类后的数据
    v4: 精炼重构后的最终问答
    """
    parser = argparse.ArgumentParser(description="QA数据集完整处理流程")
    
    # 基础配置
    parser.add_argument("--dataset_name", required=True, help="数据集名称")
    parser.add_argument("--input_dir", default="dataset", help="输入数据目录")
    parser.add_argument("--output_dir", default="result", help="输出结果目录")
    parser.add_argument("--temp_dir", default="temp", help="临时文件目录")

    # QA生成模型配置
    parser.add_argument("--qa_llm_model", default="Qwen/Qwen3-14B")
    parser.add_argument("--qa_llm_base_url", default="")
    parser.add_argument("--qa_llm_api_key", default="")

    # 答案验证模型配置
    parser.add_argument("--answer_llm_model", default="Qwen/Qwen3-14B")
    parser.add_argument("--answer_llm_base_url", default="")
    parser.add_argument("--answer_llm_api_key", default="")
    
    # 并发配置
    parser.add_argument("--max_workers", type=int, default=4, help="最大工作线程数")
    
    # 流程控制（可选择性运行某些步骤）
    parser.add_argument("--skip_generation", action="store_true", help="跳过问答生成步骤")
    parser.add_argument("--skip_pollution", action="store_true", help="跳过污染检查步骤")
    parser.add_argument("--skip_full_context", action="store_true", help="跳过全上下文验证步骤")
    parser.add_argument("--skip_label", action="store_true", help="跳过标注分类步骤")
    parser.add_argument("--skip_refine", action="store_true", help="跳过精炼重构步骤")
    
    args = parser.parse_args()

    # 初始化配置
    config = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        qa_llm=build_llm_config(args.qa_llm_model, args.qa_llm_base_url, args.qa_llm_api_key),
        answer_llm=build_llm_config(args.answer_llm_model, args.answer_llm_base_url, args.answer_llm_api_key),
        max_workers=args.max_workers
    )
    
    version_manager = VersionManager(config)
    version_manager.print_version_info()
    
    print("\n" + "="*70)
    print(f"📚 开始处理数据集: {args.dataset_name}")
    print("="*70)
    
    # ============================================================
    # 步骤 0: 生成原始问答对 (v0)
    # ============================================================
    v0_path = version_manager.get_path("v0", args.dataset_name)
    
    if not args.skip_generation:
        print("\n" + "="*70)
        print("🔄 步骤 0: 生成原始问答对")
        print("="*70)
        
        all_data = []  # 改为存储包含 conversation 和 qa 的完整数据
        total_qa_count = 0
        
        dataset_dir = os.path.join(args.input_dir, args.dataset_name)

        json_files = sorted(
            f for f in os.listdir(dataset_dir)
            if f.endswith(".json")
        )
		
        for filename in tqdm(json_files, desc=f"处理 {args.dataset_name}"):
            file_path = os.path.join(dataset_dir, filename)

            if not os.path.exists(file_path):
                print(f"[SKIP] {file_path} 不存在")
                continue
            
            # 加载对话数据
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                conversation = data[0].get("conversation", []) if data else []
            elif isinstance(data, dict):
                conversation = data.get("conversation", [])
            else:
                raise TypeError(f"Unsupported data type: {type(data)}")

            if not conversation:
                print(f"[SKIP] {file_path} 对话为空")
                continue
            
            speakers = conversation.get("speakers", [])
            print(f"\n处理文件 {file_path}, 说话者: {speakers}")
            
            # 生成类别 1-4 的问题
            answer_prompt = QA_GENERATE_PROMPT.format(
                conversation=conversation,
                user_list=speakers,
                question_num=1,
                total_question_num=len(speakers) * 4
            )
            
            all_question = call_openai_json(
                answer_prompt=answer_prompt,
                model=args.qa_llm_model,
                api_key=args.qa_llm_api_key,
                base_url=args.qa_llm_base_url
            )
            
            # 生成类别 5-6 的问题
            answer_prompt_2 = QA_GENERATE_PROMPT_2.format(
                conversation=conversation,
            )
            
            all_question_2 = call_openai_json(
                answer_prompt=answer_prompt_2,
                model=args.qa_llm_model,
                api_key=args.qa_llm_api_key,
                base_url=args.qa_llm_base_url
            )
            
            # 汇总当前文件的问题
            current_qa = []
            if "qa" in all_question and isinstance(all_question["qa"], list):
                current_qa.extend(all_question["qa"])
                print(f"  生成类别1-4问题: {len(all_question['qa'])} 个")
            
            if "qa" in all_question_2 and isinstance(all_question_2["qa"], list):
                current_qa.extend(all_question_2["qa"])
                print(f"  生成类别5-6问题: {len(all_question_2['qa'])} 个")
            
            # 将 conversation 和 qa 打包成一个 dict
            file_data = {
                "filename": filename,
                "conversation": conversation,
                "qa": current_qa
            }
            all_data.append(file_data)
            total_qa_count += len(current_qa)
        
        # 保存 v0 版本 - 现在是一个 list
        with open(v0_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ v0 版本已保存: {v0_path}")
        print(f"   共生成 {len(all_data)} 个文件的数据")
        print(f"   共生成 {total_qa_count} 个问答对")
    else:
        print(f"\n⏩ 跳过问答生成步骤，使用现有文件: {v0_path}")
    
    # ============================================================
    # 步骤 1: 数据污染检查 (v0 -> v1)
    # ============================================================
    v1_path = version_manager.get_path("v1", args.dataset_name)
    
    if not args.skip_pollution:
        v1_path = pollution_check_main(args, v0_path, v1_path)
    else:
        print(f"\n⏩ 跳过污染检查步骤，使用现有文件: {v1_path}")
    
    # ============================================================
    # 步骤 2: 全上下文验证 - 两次检查 (v1 -> v2a -> v2b)
    # ============================================================
    v2a_path = version_manager.get_path("v2a", args.dataset_name)
    v2b_path = version_manager.get_path("v2b", args.dataset_name)
    
    if not args.skip_full_context:
        print("\n" + "="*70)
        print("🔄 步骤 2: 全上下文验证（两阶段筛选）")
        print("="*70)
        
        # 第一次检查：only_evidence=1，保留只看证据就能答对的题目
        print("\n📝 第一阶段：保留基于证据答对的题目")
        v2a_path, kept_count_a = full_context_main(
            args=args,
            input_file_path=v1_path,
            output_file_path=v2a_path,
            only_evidence=1,
            except_evidence=0
        )
        print(f"   ✓ 第一阶段完成，保留题目: {kept_count_a} 个")
        
        # 第二次检查：except_evidence=1，筛掉不看证据就能答对的题目
        print("\n📝 第二阶段：筛掉不看证据也能答对的题目")
        v2b_path, kept_count_b = full_context_main(
            args=args,
            input_file_path=v2a_path,
            output_file_path=v2b_path,
            only_evidence=0,
            except_evidence=1
        )
        print(f"   ✓ 第二阶段完成，最终保留题目: {kept_count_b} 个")
        
        # 使用第二阶段的输出作为最终的v2
        v2_path = v2b_path
        
        print("\n" + "="*70)
        print(f"📊 验证统计：初始 → {kept_count_a} → {kept_count_b}")
        print("="*70)
    else:
        print(f"\n⏩ 跳过全上下文验证步骤，使用现有文件: {v2b_path}")
        v2_path = v2b_path
    
    # ============================================================
    # 步骤 3: 问题分类标注 (v2 -> v3)
    # ============================================================
    v3_path = version_manager.get_path("v3", args.dataset_name)
    
    if not args.skip_label:
        v3_path = label_main(
            input_file_path=v2_path,
            output_file_path=v3_path,
            api_key=args.answer_llm_api_key,
            base_url=args.answer_llm_base_url,
            model_name=args.answer_llm_model
        )
    else:
        print(f"\n⏩ 跳过标注分类步骤，使用现有文件: {v3_path}")
    
    # ============================================================
    # 步骤 4: 问答精炼重构 (v3 -> v4)
    # ============================================================
    v4_path = version_manager.get_path("v4", args.dataset_name)
    final_output = os.path.join(args.output_dir, f"{args.dataset_name}_final.json")
    
    if not args.skip_refine:
        v4_path = new_qa_main(
            input_file_path=v3_path,
            output_file_path=v4_path,
            api_key=args.qa_llm_api_key,
            base_url=args.qa_llm_base_url,
            model=args.qa_llm_model
        )
        
        # 复制最终版本到输出目录
        import shutil
        shutil.copy(v4_path, final_output)
        print(f"\n📦 最终版本已复制到: {final_output}")
    else:
        print(f"\n⏩ 跳过精炼重构步骤，使用现有文件: {v4_path}")
    
    # ============================================================
    # 完成总结
    # ============================================================
    print("\n" + "="*70)
    print("🎉 数据处理流程全部完成！")
    print("="*70)
    print("\n📁 生成的文件：")
    print(f"  v0 (原始):     {v0_path}")
    print(f"  v1 (打乱):     {v1_path}")
    print(f"  v2 (验证):     {v2_path}")
    print(f"  v3 (标注):     {v3_path}")
    print(f"  v4 (重构):     {v4_path}")
    print(f"  最终输出:      {final_output}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()


# python qa_pipeline.py --dataset_name An-Enemy-of-the-People --skip_generation --skip_pollution --skip_full_context --skip_label --qa_llm_model qwen3-14b --qa_llm_base_url https://api.vveai.com/v1 --qa_llm_api_key UQ3rMy9zeMTzD4AAF83eB5F4EcE84d6d9170CcB56a43F8F3 --answer_llm_model qwen3-14b --answer_llm_base_url https://api.vveai.com/v1 --answer_llm_api_key UQ3rMy9zeMTzD4AAF83eB5F4EcE84d6d9170CcB56a43F8F3 --max_workers 4 > pipeline.log 2>&1

