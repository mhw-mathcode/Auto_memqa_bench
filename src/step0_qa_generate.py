import os
import json
from typing import Any, Dict, Optional, List, Tuple
from openai import OpenAI
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.utils import load_json_file, write_json_file
from src.pipeline_utils import log_event, log_subsection, print_log_section, print_kv

DEFAULT_USER_LIST = [
  "Ariel",
  "Bennett",
  "Chloe",
  "Dexter",
  "Ethan",
  "Fiona"
]

TOTAL_CATEGORIES = 7
REQUIRED_CATEGORIES = set(range(1, TOTAL_CATEGORIES + 1))

QA_GENERATE_PROMPT = """
{conversation}

Role:
You are a top-tier AI evaluation expert, specializing in designing extremely high-difficulty stress test datasets for evaluating large language models’ long-range, cross-conversation memory.

Task:
Based on the provided long text dialogue, please design as much as possible high-quality question-answer pair for each of the seven specific categories 1-7.

Part I: Core Objectives and Depth Requirements

1. Cross-Conversation Reasoning:
- It is strictly forbidden to generate questions that can be answered using only a single session or a single utterance.
- Each question must require the model to extract and integrate information from at least two (preferably three or more) distinct conversation fragments.
- Hard-case preference: prioritize fragmented information where a clue is planted in Session A, indirectly referenced in Session B, and only revealed or resolved in Session C.

2. Extreme Source Constraints:
- Absolutely no external knowledge, common sense assumptions, associative reasoning, or hallucinations are allowed.
- If a fact is not explicitly stated or logically necessitated by the dialogue, it must be treated as non-existent.

Part II: Strict Definitions of the Seven Question Categories

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
- Abstain: If constructing an Abstain question, options A through E MUST ALL be distractors. All five options must look highly plausible but be factually wrong or unsupported by the dialogue. In this case, you must set the "answer" strictly to "F" by default. The evidence_dialogues must demonstrate exactly why none of the options (A-E) can be derived from the text.

1. Question Stem Design (Natural & Implicit):
- Feature leakage is forbidden. Do not use phrases such as “based on their introverted personality” or “shows a stable coping pattern.”
- Questions must read like natural user inquiries.
  Incorrect: “Which option reflects Ariel’s stable breakup-coping style?”
  Correct: “Which statement best matches how Ariel dealt with the aftermath of the breakup?”
- Language should be direct, concrete, and non-rhetorical.
- Follow the style of the trace benchmark questions: name the relevant person, object, event, relationship, or state transition in the stem, and ask what happened, how it changed, what relationship was established, or which exact detail remained true.
- For literary or narrative texts, do not write abstract template stems such as “Which option correctly combines two separate details?”, “Which statement preserves the paired details?”, “Which option matches the two passages?”, or “Which option correctly combines two separate moments?”. These are invalid.
- Do not make options into quote containers such as “One passage says ...; another says ...”. Options should be natural answer statements. Use the evidence_dialogues field for verbatim source text.

2. Hard Distractor Requirements:
- Options must be concise natural answer statements, not evidence dumps.
  Target each option at roughly 10-24 English words when possible; do not exceed 35 words unless the fact itself requires it.
- Do not join two copied evidence snippets with a semicolon. If two facts must appear together, write one natural sentence using "and", "while", or a short causal/temporal connector.
- Length balance (critical): the correct answer must not be the longest or shortest among the five options.
  At least one distractor must be longer than the correct answer.
- Enforce this numerically before returning JSON: every distractor length should be broadly comparable to the correct option, but do not pad options with copied evidence just to match length.
- Style balance: all five options must have comparable clause count, specificity, named-entity density, temporal precision, and grammatical structure. The correct option must not be the only complete multi-clause statement.
- Semantic proximity: distractors must be highly plausible and lie in a high-probability semantic neighborhood.
  Avoid extreme terms such as “always,” “never,” “completely,” or “absolutely.”
- Distractors should be naturally plausible, but they do not all need to be near-identical one-word perturbations. Mix confusion types: outdated state, wrong actor, wrong motive, wrong order, over-specific unsupported detail, and true detail attached to the wrong person.
- Avoid making all five options share the same sentence frame. The options should read like real candidate answers, not mechanically edited copies.
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
- No-context leakage: if the dialogue and evidence were hidden, would option length, detail, fluency, common sense, or wording style reveal the answer? If yes, rewrite all distractors before returning JSON.
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
      "question": "",
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
  timeout_s: int = 240
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

    def _best_effort_json_loads(text: str) -> Dict[str, Any]:
      """在严格解析失败后做一次轻量修复（如尾逗号）再解析。"""
      t = _extract_json_object_text(text)
      # 去除对象/数组结束前的尾逗号
      t = re.sub(r",\s*([}\]])", r"\1", t)
      obj = json.loads(t)
      if not isinstance(obj, dict):
        raise ValueError(f"Top-level JSON must be an object/dict, got {type(obj)}")
      return obj

    def _repair_json_with_llm(raw_text: str) -> str:
      """让模型仅做 JSON 语法修复，不改语义内容。"""
      repair_prompt = (
        "You are a JSON repair tool.\n"
        "Fix the JSON syntax only and preserve original semantics.\n"
        "Return ONLY one valid JSON object. No markdown, no explanation.\n"
        "If a value contains double quotes, escape them.\n\n"
        "Malformed JSON:\n"
        f"{_extract_json_object_text(raw_text)}"
      )
      repair_resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": repair_prompt}],
        temperature=0.0,
      )
      return repair_resp.choices[0].message.content or ""

    # -------- retry loop --------
    max_retries = 10  # json 生成失败重试次数
    max_other_error_retries = 10  # 其他请求失败重试次数
    max_timeout_retries = 10  # 请求超时重试次数
    last_content = ""
    last_finish_reason: Optional[str] = None
    use_json_object_mode = True  # 优先请求 JSON 对象模式，降低引号/格式错误
    json_retry_hint = ""

    for attempt in range(max_retries + 1):
      llm_error_retries = 0
      other_error_retries = 0
      timeout_retries = 0
      resp = None

      while True:
        try:
          kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": answer_prompt + json_retry_hint}],
            "temperature": 0.0,
          }
          if use_json_object_mode:
            kwargs["response_format"] = {"type": "json_object"}

          resp = client.chat.completions.create(**kwargs)
          break
        except Exception as e:
          error_str = str(e).lower()

          # 部分网关在携带 response_format 时会错误返回缺参，先降级再重试 chat。
          if "missing_required_parameter" in error_str or (
            "one of \"input\"" in error_str and "prompt" in error_str
          ):
            if use_json_object_mode:
              log_event("llm_json_call", status="fallback", reason="missing_required_parameter", action="disable_response_format")
              use_json_object_mode = False
              continue

          # 模型/网关不支持 response_format=json_object → 自动降级
          if "response_format" in error_str and (
            "unknown parameter" in error_str
            or "unsupported" in error_str
            or "not support" in error_str
          ):
            log_event("llm_json_call", status="fallback", reason="response_format_unsupported")
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
            log_event("llm_json_call", status="retry", reason="rate_limit_or_overload", attempt=llm_error_retries, wait=f"{sleep_duration:.2f}s")
            time.sleep(sleep_duration)
            continue

          if (
            "request timed out" in error_str
            or "timed out" in error_str
            or "timeout" in error_str
          ):
            timeout_retries += 1
            other_error_retries = 0
            sleep_duration = random.uniform(3, 10) + 4 * timeout_retries
            log_event("llm_json_call", status="retry", reason="timeout", attempt=f"{timeout_retries}/{max_timeout_retries}", wait=f"{sleep_duration:.2f}s")
            if timeout_retries >= max_timeout_retries:
              log_event("llm_json_call", status="failed", reason="timeout_retries_exceeded")
              break
            time.sleep(sleep_duration)
            continue

          # 识别为其他错误
          other_error_retries += 1
          log_event("llm_json_call", status="retry", reason="other_error", attempt=f"{other_error_retries}/{max_other_error_retries}", error=e)
          if other_error_retries >= max_other_error_retries:
            log_event("llm_json_call", status="failed", reason="unrecoverable_error")
            break
          time.sleep(random.uniform(1.0, 3.0))

      if resp is None:
        continue

      last_content = resp.choices[0].message.content or ""
      last_finish_reason = getattr(resp.choices[0], "finish_reason", None)
      if last_finish_reason == "length":
        log_event("llm_json_call", status="warning", reason="finish_reason_length")

      try:
        return _strict_json_loads(last_content)
      except Exception as parse_error:
        try:
          repaired = _best_effort_json_loads(last_content)
          log_event("json_repair", status="success", method="local", attempt=attempt + 1)
          return repaired
        except Exception:
          try:
            repaired_text = _repair_json_with_llm(last_content)
            repaired = _strict_json_loads(repaired_text)
            log_event("json_repair", status="success", method="llm", attempt=attempt + 1)
            return repaired
          except Exception as repair_error:
            log_event(
              "json_parse",
              status="retry",
              attempt=f"{attempt + 1}/{max_retries + 1}",
              parse_error=parse_error,
              repair_error=repair_error,
            )
          json_retry_hint = (
            "\n\nIMPORTANT JSON RETRY INSTRUCTION:\n"
            "Your previous output was invalid JSON. Return ONLY one valid JSON object.\n"
            "Do not include markdown fences.\n"
            "Do not use trailing commas.\n"
            "If a string contains double quotes, escape them as \\\".\n"
          )
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
      total_question_num=len(speakers) * TOTAL_CATEGORIES
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


def _speaker_category_coverage(qa_items: List[Dict[str, Any]], speakers: List[str]) -> Dict[str, set]:
    coverage: Dict[str, set] = {speaker: set() for speaker in speakers}
    for item in qa_items:
      if not isinstance(item, dict):
        continue
      character = str(item.get("character", "")).strip()
      if character not in coverage:
        continue
      try:
        category = int(item.get("category"))
      except (TypeError, ValueError):
        continue
      if 1 <= category <= TOTAL_CATEGORIES:
        coverage[character].add(category)
    return coverage


def _generate_qa_for_speakers_with_split(
  conversation: Dict[str, Any],
  speakers: List[str],
  llm_config,
  filename: str,
  repair_round: int = 0,
  max_repair_rounds: int = 3,
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
        log_event(
          "qa_generation_batch",
          status="warning",
          file=filename,
          unexpected_characters=sorted(unexpected_characters),
        )

      if not filtered_qa:
        raise ValueError("模型返回 qa 为空，或角色与请求批次不匹配")

      coverage = _speaker_category_coverage(filtered_qa, speakers)
      missing_speakers = [
        speaker for speaker in speakers
        if speaker not in returned_characters or coverage.get(speaker, set()) != REQUIRED_CATEGORIES
      ]

      for speaker in missing_speakers:
        have = sorted(coverage.get(speaker, set()))
        missing = sorted(REQUIRED_CATEGORIES - set(have))
        log_event(
          "qa_category_coverage",
          status="warning",
          file=filename,
          speaker=speaker,
          have=have,
          missing=missing,
        )

      if missing_speakers:
        if repair_round >= max_repair_rounds:
          log_event(
            "qa_category_repair",
            status="failed",
            file=filename,
            missing_speakers=missing_speakers,
            reason="max_repair_rounds",
          )
          return filtered_qa, missing_speakers

        log_event(
          "qa_category_repair",
          status="retry",
          file=filename,
          missing_speakers=missing_speakers,
          repair_round=f"{repair_round + 1}/{max_repair_rounds}",
        )
        recovered_qa, failed_missing = _generate_qa_for_speakers_with_split(
          conversation=conversation,
          speakers=missing_speakers,
          llm_config=llm_config,
          filename=filename,
          repair_round=repair_round + 1,
          max_repair_rounds=max_repair_rounds,
        )
        merged_qa = filtered_qa + recovered_qa
        merged_qa = _deduplicate_qa_items(merged_qa)

        merged_coverage = _speaker_category_coverage(merged_qa, speakers)
        still_missing = [
          speaker for speaker in speakers
          if merged_coverage.get(speaker, set()) != REQUIRED_CATEGORIES
        ]
        failed_union = sorted(set(failed_missing + still_missing))
        return merged_qa, failed_union

      return filtered_qa, []

    except Exception as e:
      if len(speakers) == 1:
        log_event(
          "qa_generation_speaker",
          status="failed",
          file=filename,
          speaker=speakers[0],
          error=e,
        )
        return [], list(speakers)

      split_idx = max(1, len(speakers) // 2)
      left = speakers[:split_idx]
      right = speakers[split_idx:]

      log_event(
        "qa_generation_batch",
        status="split_retry",
        file=filename,
        speakers=speakers,
        left=left,
        right=right,
        error=e,
      )

      left_qa, left_failed = _generate_qa_for_speakers_with_split(
        conversation=conversation,
        speakers=left,
        llm_config=llm_config,
        filename=filename,
        repair_round=repair_round,
        max_repair_rounds=max_repair_rounds,
      )
      right_qa, right_failed = _generate_qa_for_speakers_with_split(
        conversation=conversation,
        speakers=right,
        llm_config=llm_config,
        filename=filename,
        repair_round=repair_round,
        max_repair_rounds=max_repair_rounds,
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
  max_workers: int = 1,
  input_target: Optional[str] = None,
) -> str:
    """
    生成 v0 原始问答对
    """
    print_log_section("STEP 0 | GENERATE RAW QA")

    target_path = input_target or os.path.join(input_dir, dataset_name)
    if not os.path.isabs(target_path):
      target_path = os.path.abspath(target_path)

    if os.path.isfile(target_path):
      dataset_dir = os.path.dirname(target_path)
      json_files = [os.path.basename(target_path)]
    else:
      dataset_dir = target_path
      if not os.path.isdir(dataset_dir):
        log_event("generate_v0", status="failed", reason="dataset_input_missing", dataset_input=target_path)
        return ""
      json_files = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".json"))
      if not json_files:
        log_event("generate_v0", status="failed", reason="json_files_missing", dataset_dir=dataset_dir)
        return ""

    all_data = []
    total_qa_count = 0
    skipped_count = 0
    generated_count = 0
    invalid_count = 0

    log_subsection("Step 0 input summary")
    print_kv("dataset_input", target_path, indent=4)
    print_kv("dataset_dir", dataset_dir, indent=4)
    print_kv("json_files", len(json_files), indent=4)
    print_kv("mode", "force_generate" if force_generate_new_qa else "reuse_existing_qa", indent=4)
    normalized_workers = max(1, int(max_workers or 1))
    print_kv("max_workers", normalized_workers, indent=4)
    
    for file_idx, filename in enumerate(
      tqdm(json_files, desc=f"Step 0 files | {dataset_name}", unit="file"),
      start=1,
    ):
      file_path = os.path.join(dataset_dir, filename)
      if not os.path.exists(file_path):
        log_event(
          "generate_v0_file",
          status="skipped",
          file=file_path,
          reason="file_missing",
          progress=f"{file_idx}/{len(json_files)}",
        )
        continue

      try:
        data = load_json_file(file_path)
      except json.JSONDecodeError as exc:
        log_event(
          "generate_v0_file",
          status="skipped",
          file=file_path,
          reason="invalid_json",
          error=exc,
          progress=f"{file_idx}/{len(json_files)}",
        )
        invalid_count += 1
        continue
      except OSError as exc:
        log_event(
          "generate_v0_file",
          status="skipped",
          file=file_path,
          reason="read_failed",
          error=exc,
          progress=f"{file_idx}/{len(json_files)}",
        )
        invalid_count += 1
        continue

      # 提取 conversation 和 已有的 qa
      existing_qa = []
      if isinstance(data, list):
        conversation = data[0].get("conversation", {}) if data else {}
        existing_qa = data[0].get("qa", []) if data else []
      elif isinstance(data, dict):
        conversation = data.get("conversation", {})
        existing_qa = data.get("qa", [])
      else:
        log_event(
          "generate_v0_file",
          status="skipped",
          file=file_path,
          reason="unsupported_data_type",
          data_type=type(data),
          progress=f"{file_idx}/{len(json_files)}",
        )
        invalid_count += 1
        continue

      if not conversation:
        log_event(
          "generate_v0_file",
          status="skipped",
          file=file_path,
          reason="empty_conversation",
          progress=f"{file_idx}/{len(json_files)}",
        )
        invalid_count += 1
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

      # 默认 user_list 兜底为老友记六主角
      if not speakers:
        speakers = list(DEFAULT_USER_LIST)

      # 检查是否已有问答对
      has_existing_qa = bool(existing_qa and isinstance(existing_qa, list) and len(existing_qa) > 0)
      if has_existing_qa and not force_generate_new_qa:
        # 已有问答对，直接使用
        log_event(
          "generate_v0_file",
          status="reused_existing_qa",
          file=filename,
          existing_qa=len(existing_qa),
          progress=f"{file_idx}/{len(json_files)}",
        )
        current_qa = existing_qa
        skipped_count += 1
      else:
        if has_existing_qa and force_generate_new_qa:
          log_event(
            "generate_v0_file",
            status="force_regenerate",
            file=filename,
            existing_qa=len(existing_qa),
            speakers=speakers,
            progress=f"{file_idx}/{len(json_files)}",
          )
        else:
          # 没有问答对，调用 LLM 生成
          log_event(
            "generate_v0_file",
            status="generating",
            file=filename,
            speakers=speakers,
            progress=f"{file_idx}/{len(json_files)}",
          )
        
        current_qa = []

        normalized_batch_size = max(1, int(initial_batch_size or 1))
        speaker_batches = [
          speakers[i:i + normalized_batch_size]
          for i in range(0, len(speakers), normalized_batch_size)
        ]

        failed_speakers_all: List[str] = []

        effective_workers = min(normalized_workers, len(speaker_batches))
        log_event(
          "generate_v0_workers",
          status="configured",
          file=filename,
          requested=normalized_workers,
          effective=effective_workers,
          batches=len(speaker_batches),
        )

        def generate_batch(batch):
          log_event("generate_v0_batch", status="start", file=filename, speakers=batch, indent=4)
          return _generate_qa_for_speakers_with_split(
            conversation=conversation,
            speakers=batch,
            llm_config=llm_config,
            filename=filename,
          )

        if effective_workers == 1:
          batch_results = [generate_batch(batch) for batch in speaker_batches]
        else:
          with ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix="generate-v0",
          ) as executor:
            batch_results = list(executor.map(generate_batch, speaker_batches))

        for batch_qa, failed_speakers in batch_results:
          current_qa.extend(batch_qa)
          failed_speakers_all.extend(failed_speakers)

        current_qa = _deduplicate_qa_items(current_qa)

        if failed_speakers_all:
          log_event(
            "generate_v0_file",
            status="warning",
            file=filename,
            failed_speakers=sorted(set(failed_speakers_all)),
          )

        log_event(
          "generate_v0_file",
          status="generated",
          file=filename,
          speakers=speakers,
          qa_count=len(current_qa),
        )

        generated_count += 1

      file_data = {
        "filename": filename,
        "conversation": conversation,
        "qa": current_qa
      }
      all_data.append(file_data)
      total_qa_count += len(current_qa)

    write_json_file(all_data, v0_path, indent=2)

    log_subsection("Step 0 output summary")
    print_kv("total_files", len(json_files), indent=4)
    print_kv("reused_existing_qa_files", skipped_count, indent=4)
    print_kv("generated_files", generated_count, indent=4)
    print_kv("invalid_or_skipped_files", invalid_count, indent=4)
    print_kv("total_qa", total_qa_count, indent=4)
    print_kv("output", v0_path, indent=4)
    
    return v0_path

