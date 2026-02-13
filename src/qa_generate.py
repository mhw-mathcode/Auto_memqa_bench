import os
import json
from typing import Any, Dict, Optional, List
from openai import OpenAI
import re
import time
import random
from tqdm import tqdm

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

def generate_v0(dataset_name: str, input_dir: str, v0_path: str, llm_config) -> str:
    """
    生成 v0 原始问答对
    """
    print("\n" + "="*60)
    print("步骤 0: 生成原始问答对")
    print("="*60)

    dataset_dir = os.path.join(input_dir, dataset_name)
    if not os.path.isdir(dataset_dir):
      print(f"❌ 错误: 数据集目录不存在 {dataset_dir}")
      return ""

    json_files = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".json"))
    if not json_files:
      print(f"❌ 错误: 未找到任何 JSON 文件 {dataset_dir}")
      return ""

    all_data = []
    total_qa_count = 0
    skipped_count = 0
    generated_count = 0

    print(f"--- 步骤 0 开始处理：共 {len(json_files)} 个文件 ---")
    
    for filename in tqdm(json_files, desc=f"处理 {dataset_name}"):
      file_path = os.path.join(dataset_dir, filename)
      if not os.path.exists(file_path):
        print(f"[SKIP] {file_path} 不存在")
        continue

      with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

      # 提取 conversation 和 已有的 qa
      existing_qa = []
      if isinstance(data, list):
        conversation = data[0].get("conversation", {}) if data else {}
        existing_qa = data[0].get("qa", []) if data else []
      elif isinstance(data, dict):
        conversation = data.get("conversation", {})
        existing_qa = data.get("qa", [])
      else:
        print(f"[SKIP] {file_path} 数据类型不支持: {type(data)}")
        continue

      if not conversation:
        print(f"[SKIP] {file_path} 对话为空")
        continue

      speakers = conversation.get("speakers", [])
      
      # 检查是否已有问答对
      if existing_qa and isinstance(existing_qa, list) and len(existing_qa) > 0:
        # 已有问答对，直接使用
        print(f"\n✓ {filename} 已存在 {len(existing_qa)} 个问答对，跳过生成")
        current_qa = existing_qa
        skipped_count += 1
      else:
        # 没有问答对，调用 LLM 生成
        print(f"\n⚙ {filename} 未找到问答对，开始生成 (说话者: {speakers})")
        
        answer_prompt = QA_GENERATE_PROMPT.format(
          conversation=conversation,
          user_list=speakers,
          question_num=1,
          total_question_num=len(speakers) * 4
        )

        all_question = call_openai_json(
          answer_prompt=answer_prompt,
          model=llm_config.model,
          api_key=llm_config.api_key,
          base_url=llm_config.base_url
        )

        answer_prompt_2 = QA_GENERATE_PROMPT_2.format(
          conversation=conversation,
        )

        all_question_2 = call_openai_json(
          answer_prompt=answer_prompt_2,
          model=llm_config.model,
          api_key=llm_config.api_key,
          base_url=llm_config.base_url
        )

        current_qa = []
        if "qa" in all_question and isinstance(all_question["qa"], list):
          current_qa.extend(all_question["qa"])
          print(f"  生成类别1-4问题: {len(all_question['qa'])} 个")

        if "qa" in all_question_2 and isinstance(all_question_2["qa"], list):
          current_qa.extend(all_question_2["qa"])
          print(f"  生成类别5-6问题: {len(all_question_2['qa'])} 个")
        
        generated_count += 1

      file_data = {
        "filename": filename,
        "conversation": conversation,
        "qa": current_qa
      }
      all_data.append(file_data)
      total_qa_count += len(current_qa)

    with open(v0_path, "w", encoding="utf-8") as f:
      json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n--- 步骤 0 完成统计 ---")
    print(f"  总文件数: {len(json_files)}")
    print(f"  跳过生成 (已有QA): {skipped_count}")
    print(f"  LLM生成 (新QA): {generated_count}")
    print(f"  总问答对数: {total_qa_count}")
    print(f"  输出文件: {v0_path}")
    
    return v0_path

