import os
import json
from typing import Any, Dict, Optional, List, Tuple
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
Based on the provided long text dialogue, please design 1 high-quality question-answer pair for each user in {user_list} for each of the seven specific category 1-7 (total of {total_question_num} pairs). If a user speaks too little to support the creation of a sufficient number of questions, the limit on the number of questions that can be generated can be removed. You can try to explore as many questions as possible, but there are no requirements or restrictions on the number of generated questions.

Part I: Core Objectives and Depth Requirements

1. Cross-Conversation Reasoning:
- It is strictly forbidden to generate questions that can be answered using only a single session or a single utterance.
- Each question must require the model to extract and integrate information from at least two (preferably three or more) distinct conversation fragments.
- Hard-case preference: prioritize fragmented information where a clue is planted in Session A, indirectly referenced in Session B, and only revealed or resolved in Session C.

2. Extreme Source Constraints:
- Absolutely no external knowledge, common sense assumptions, associative reasoning, or hallucinations are allowed.
- If a fact is not explicitly stated or logically necessitated by the dialogue, it must be treated as non-existent.

Part II: Strict Definitions of the Two Question Categories

Category 1 - User Profile Category: Evaluates the model's ability to capture and maintain long-term, stable user attributes such as demographics, core values, and persistent habits to ensure persona consistency.

Category 2 - Event-based Category: Focuses on the precise storage and recall of discrete, structured facts and specific behaviors, characterized by the 5W1H (Who, What, Where, When, Why, How) framework.

Category 3 - Temporal Evolution Category: Assesses the model's capacity to track dynamic changes and state transitions over time, ensuring the latest information correctly updates or supersedes outdated memories.

Category 4 - Social Relationship & Interaction Category: Concentrates on mapping interpersonal networks and interaction patterns, including both explicit social links and implicit emotional nuances between multiple entities.

Category 5 - Fine-grained Data Category: Measures "pixel-level" memory precision for highly detailed micro-parameters, such as specific code snippets, numerical indicators, or character-specific strings.

Category 6 - Lessons Learned Category: Gauges the model's meta-cognitive ability to reflect on past feedback and apply error-correction experiences to optimize future strategies and interaction styles.

Category 7 - Plans & Commitments Category: Highlights the management of prospective memory, including the tracking of future tasks, scheduled events, and specific triggers for promised actions.

Part III: Construction Method Reference and Strong Confusion Design

When crafting each question, select one of the following five construction methods as your primary technique. The method shapes how the question is built and how distractors are arranged. Record the chosen method in the "label" field. Try to use each method at least once across all questions you generate for a given user.

- Fact Extraction (Single Dialogue): Ground the question in a single dialogue session. The correct answer is fully derivable from one scene; distractors are drawn from other sessions or other characters.
- Fact Extraction (Multiple Dialogues): Scatter the key clues across two or more sessions so that the answer can only be obtained by combining them. The cross-session dependency should be non-obvious.
- Memory Update: Exploit a fact that was stated differently at two points in time. The question rewards recognising the newer version; the outdated version must appear as a highly attractive distractor.
- Multi-hop: Require at least two inferential steps, each grounded in dialogue evidence, to reach the answer. No single utterance is sufficient; the chain must be traceable.
- Abstain: Construct a question for which none of the five options (A–E) is actually supported by the dialogue. Set "answer" to "F". All five options must look plausible but be factually wrong or unsupported. evidence_dialogues should demonstrate that none of the options can be derived from the text.

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

Critical Output Constraints:
- Return only one valid JSON object. Do not use markdown code fences.
- All keys and string values must use standard double quotes (").
- If a string contains inner double quotes, escape them as \\".

{{
  "qa": [
    {{
      "character": "Ariel",
      "category": 6,
      "question": "[Direct, natural, focused on the character]",
      "option": [
        "A. ",
        "B. ",
        "C. ",
        "D. ",
        "E. "
      ],
      "answer": "C",
      "label": "Fact Extraction (Multiple Dialogues)",
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
      base_url=base_url,
      timeout=timeout_s,
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

    def _normalize_common_quotes(text: str) -> str:
      # Normalize typographic quotes that often break strict JSON parsing.
      return (
        (text or "")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
      )

    def _extract_json_object_text(text: str) -> str:
      t = _strip_code_fences(_normalize_common_quotes(text)).strip()
      start = t.find("{")
      end = t.rfind("}")
      if start >= 0 and end > start:
        return t[start:end + 1].strip()
      return t

    def _strict_json_loads(text: str) -> Dict[str, Any]:
      t = _extract_json_object_text(text)
      obj = json.loads(t)  # strict: must parse directly
      if not isinstance(obj, dict):
        raise ValueError(f"Top-level JSON must be an object/dict, got {type(obj)}")
      return obj

    # -------- retry loop --------
    max_retries = 10  # json 生成失败重试次数
    max_other_error_retries = 10  # 其他请求失败重试次数
    last_content = ""
    last_finish_reason: Optional[str] = None
    use_enable_thinking = True  # 首次尝试带 enable_thinking，若不支持则自动降级
    use_json_object_mode = True  # 优先请求 JSON 对象模式，降低引号/格式错误

    for attempt in range(max_retries + 1):
      llm_error_retries = 0
      other_error_retries = 0
      resp = None

      while True:
        try:
          kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": answer_prompt}],
            "temperature": 0.0,
          }
          if use_enable_thinking:
            kwargs["extra_body"] = {"enable_thinking": False}
          if use_json_object_mode:
            kwargs["response_format"] = {"type": "json_object"}

          resp = client.chat.completions.create(**kwargs)
          break
        except Exception as e:
          error_str = str(e).lower()
          print(error_str)

          # 模型不支持 enable_thinking 参数 → 立即降级，不再传该参数
          if "enable_thinking" in error_str and "unknown parameter" in error_str:
            print("[INFO] 模型不支持 enable_thinking，已自动降级，不再传该参数")
            use_enable_thinking = False
            continue

          # 模型/网关不支持 response_format=json_object → 自动降级
          if "response_format" in error_str and (
            "unknown parameter" in error_str
            or "unsupported" in error_str
            or "not support" in error_str
          ):
            print("[INFO] 模型不支持 response_format=json_object，已自动降级")
            use_json_object_mode = False
            continue

          if (
            "rate limit" in error_str
            or "limit" in error_str
            or "overloaded" in error_str
            or "token" in error_str
          ):
            # 识别为 TPM 则一直重试
            llm_error_retries += 1
            other_error_retries = 0
            sleep_duration = random.uniform(2, 20) + 5 * llm_error_retries
            error_message = f"LLM Rate Limit related Error. Retrying in {sleep_duration:.2f}s... Error: {e}"
            print(error_message)
            time.sleep(sleep_duration)
            continue

          # 识别为其他错误
          other_error_retries += 1
          print("other_error_retries: ", other_error_retries)
          if other_error_retries >= max_other_error_retries:
            print("Error: Default response due to unrecoverable error.")
            break

      if resp is None:
        continue

      last_content = resp.choices[0].message.content or ""
      last_finish_reason = getattr(resp.choices[0], "finish_reason", None)
      if last_finish_reason == "length":
        print("[WARN] LLM finish_reason=length，输出可能被截断")

      try:
        return _strict_json_loads(last_content)
      except Exception as parse_error:
        print(f"[WARN] JSON parse failed on attempt {attempt + 1}/{max_retries + 1}: {parse_error}")
        continue

    # if still invalid after retries
    raise ValueError(
        f"Judge did not return valid JSON after {max_retries + 1} attempts.\n"
        f"Last finish_reason: {last_finish_reason}\n"
        f"Last output:\n{last_content}"
    )


def _build_qa_generate_prompt(conversation: Dict[str, Any], speakers: List[str]) -> str:
    return QA_GENERATE_PROMPT.format(
      conversation=conversation,
      user_list=speakers,
      question_num=1,
      total_question_num=len(speakers) * 7
    )


def _deduplicate_qa_items(qa_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_keys = set()

    for item in qa_items:
      if not isinstance(item, dict):
        continue
      key = (
        str(item.get("character", "")).strip(),
        str(item.get("question", "")).strip(),
      )
      if key in seen_keys:
        continue
      seen_keys.add(key)
      deduped.append(item)

    return deduped


def _generate_qa_for_speakers_with_split(
  conversation: Dict[str, Any],
  speakers: List[str],
  llm_config,
  filename: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    为 speaker 列表生成题目，失败时自动二分拆批重试。
    返回 (qa_list, failed_speakers)。
    """
    if not speakers:
      return [], []

    prompt = _build_qa_generate_prompt(conversation, speakers)

    try:
      all_users_qa = call_openai_json(
        answer_prompt=prompt,
        model=llm_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
      )

      qa_list = all_users_qa.get("qa", [])
      if not isinstance(qa_list, list):
        raise ValueError("字段 qa 缺失或不是 list")

      expected = set(speakers)
      filtered_qa: List[Dict[str, Any]] = []
      returned_characters = set()
      unexpected_characters = set()

      for item in qa_list:
        if not isinstance(item, dict):
          continue
        character = str(item.get("character", "")).strip()
        if character in expected:
          filtered_qa.append(item)
          returned_characters.add(character)
        elif character:
          unexpected_characters.add(character)

      filtered_qa = _deduplicate_qa_items(filtered_qa)

      if unexpected_characters:
        print(f"[WARN] {filename} 收到不在请求批次中的角色，已忽略: {sorted(unexpected_characters)}")

      if not filtered_qa:
        raise ValueError("模型返回 qa 为空，或角色与请求批次不匹配")

      missing_speakers = [speaker for speaker in speakers if speaker not in returned_characters]
      if missing_speakers:
        print(f"[WARN] {filename} 批次缺少角色，自动补生成: {missing_speakers}")
        recovered_qa, failed_missing = _generate_qa_for_speakers_with_split(
          conversation=conversation,
          speakers=missing_speakers,
          llm_config=llm_config,
          filename=filename,
        )
        merged_qa = filtered_qa + recovered_qa
        return _deduplicate_qa_items(merged_qa), failed_missing

      return filtered_qa, []

    except Exception as e:
      if len(speakers) == 1:
        print(f"⚠️ {filename} 角色 {speakers[0]} 生成失败: {e}")
        return [], list(speakers)

      split_idx = max(1, len(speakers) // 2)
      left = speakers[:split_idx]
      right = speakers[split_idx:]

      print(f"[WARN] {filename} 批次生成失败，拆分重试: {speakers} -> {left} | {right}")

      left_qa, left_failed = _generate_qa_for_speakers_with_split(
        conversation=conversation,
        speakers=left,
        llm_config=llm_config,
        filename=filename,
      )
      right_qa, right_failed = _generate_qa_for_speakers_with_split(
        conversation=conversation,
        speakers=right,
        llm_config=llm_config,
        filename=filename,
      )

      merged = left_qa + right_qa
      return _deduplicate_qa_items(merged), left_failed + right_failed

def generate_v0(
  dataset_name: str,
  input_dir: str,
  v0_path: str,
  llm_config,
  force_generate_new_qa: bool = False,
  initial_batch_size: int = 8,
) -> str:
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
    print(f"  生成模式: {'强制重建题目（忽略已有 qa）' if force_generate_new_qa else '复用已有 qa（默认）'}")
    
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

      
      # 适配 dataset\12_Angry_Men 格式 (speaker_1, speaker_2...)
      speakers = conversation.get("speakers", [])
      if not speakers:
          extracted_speakers = []
          # 提取 speaker_X 形式的键值
          for key, value in conversation.items():
              if key.startswith("speaker_") and key.split("_")[-1].isdigit():
                  extracted_speakers.append(value)
          if extracted_speakers:
              speakers = extracted_speakers

      # 检查是否已有问答对
      has_existing_qa = bool(existing_qa and isinstance(existing_qa, list) and len(existing_qa) > 0)
      if has_existing_qa and not force_generate_new_qa:
        # 已有问答对，直接使用
        print(f"\n✓ {filename} 已存在 {len(existing_qa)} 个问答对，跳过生成")
        current_qa = existing_qa
        skipped_count += 1
      else:
        if has_existing_qa and force_generate_new_qa:
          print(f"\n⚙ {filename} 检测到已有 {len(existing_qa)} 个问答对，但配置要求强制重建")
          print(f"\n⚙ {filename} 开始重新生成问答对 (说话者: {speakers})")
        else:
          # 没有问答对，调用 LLM 生成
          print(f"\n⚙ {filename} 未找到问答对，开始生成 (说话者: {speakers})")
        
        current_qa = []

        normalized_batch_size = max(1, int(initial_batch_size or 1))
        speaker_batches = [
          speakers[i:i + normalized_batch_size]
          for i in range(0, len(speakers), normalized_batch_size)
        ]

        failed_speakers_all: List[str] = []

        for batch in speaker_batches:
          print(f"  -> 生成批次: {batch}")
          batch_qa, failed_speakers = _generate_qa_for_speakers_with_split(
            conversation=conversation,
            speakers=batch,
            llm_config=llm_config,
            filename=filename,
          )
          current_qa.extend(batch_qa)
          failed_speakers_all.extend(failed_speakers)

        current_qa = _deduplicate_qa_items(current_qa)

        if failed_speakers_all:
          print(f"⚠️ {filename} 以下角色生成失败，已跳过: {sorted(set(failed_speakers_all))}")

        print(f"  生成用户 {speakers} 类别6问题: {len(current_qa)} 个")

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

