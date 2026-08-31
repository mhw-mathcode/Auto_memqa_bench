import copy
import ast
import json
import logging
import os
import random
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

from src.ablation_chunking import (
    AblationChunk,
    OversizedDialogueTurnError,
    bisect_ablation_chunk,
    chunk_conversation,
    estimate_tokens,
    rank_chunks,
)
from src.mcq_scoring import (
    MULTIPLE_SELECT,
    ORDERING,
    SINGLE_CHOICE,
    answer_evidence_covers_selection,
    count_question_types,
    extract_answer_fragment_from_json_text,
    get_answer_instruction,
    normalize_answer_candidates,
    normalize_question_type,
    parse_question_answer,
    propagate_candidate_evidence_provenance,
    score_mcq_prediction,
)
from src.utils import (
    align_evidence_dialogues,
    build_dialogue_index,
    count_qa_items,
    get_dialogue_utterance,
    load_json_file,
    normalize_dataset_records,
    normalize_reasoning_steps,
    write_json_file_atomic,
)
from src.pipeline_utils import log_event, log_subsection, print_log_section, print_kv
from src.question_formatting import strip_answer_instruction_suffix


class EvidenceProcessingError(RuntimeError):
    """A retryable infrastructure or model-response failure for one question."""


def clean_json_response(response: str) -> str:
    """清理 LLM 返回的 JSON 字符串，修复常见格式问题。"""
    if not isinstance(response, str):
        return response

    cleaned = response.replace(r"\'", "'")
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        cleaned = cleaned[start_idx:end_idx + 1]

    return cleaned.strip()


def _strip_trailing_commas(text: str) -> str:
    """删除 JSON 对象/数组闭合前的尾逗号。"""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _quote_unquoted_object_keys(text: str) -> str:
    """给常见的未加引号对象 key 补双引号。"""
    return re.sub(
        r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:',
        r'\1"\2":',
        text,
    )


def _truncate_to_balanced_json_object(text: str) -> str:
    """截断到第一个括号平衡的 JSON 对象，避免模型在末尾追加残片。"""
    start = text.find("{")
    if start < 0:
        return text

    in_string = False
    escaped = False
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    return text[start:]


def parse_json_response(response: str) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """解析并保守修复 LLM JSON 响应。

    返回 (parsed, cleaned_text, repair_status)。只做结构性修复，不猜测答案内容。
    """
    cleaned = clean_json_response(response)
    candidates = [cleaned]

    balanced = _truncate_to_balanced_json_object(cleaned)
    if balanced != cleaned:
        candidates.append(balanced)

    repaired = _strip_trailing_commas(_quote_unquoted_object_keys(balanced))
    if repaired not in candidates:
        candidates.append(repaired)

    for idx, candidate in enumerate(candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, candidate, "strict" if idx == 0 else "repaired_json"
        except json.JSONDecodeError:
            pass

    # Python literal parsing can recover single-quoted dicts and True/False/None-like outputs.
    literal_candidate = _strip_trailing_commas(balanced)
    try:
        literal = ast.literal_eval(literal_candidate)
        if isinstance(literal, dict):
            normalized = json.loads(json.dumps(literal, ensure_ascii=False))
            return normalized, json.dumps(normalized, ensure_ascii=False), "literal_eval"
    except (SyntaxError, ValueError, TypeError):
        pass

    salvaged_answer = extract_answer_fragment_from_json_text(cleaned)
    dia_ids = re.findall(r'"dia_id"\s*:\s*"([^"]+)"', cleaned)
    speakers = re.findall(r'"speaker"\s*:\s*"([^"]*)"', cleaned)
    if salvaged_answer and dia_ids:
        evidence_dialogues = []
        for idx, dia_id in enumerate(dia_ids, start=1):
            speaker = speakers[idx - 1] if idx - 1 < len(speakers) else ""
            evidence_dialogues.append(
                {
                    "id": f"E{idx}",
                    "dia_id": dia_id,
                    "speaker": speaker,
                    "utterance": "",
                }
            )
        salvaged = {
            "question": "",
            "answer": salvaged_answer,
            "evidence_dialogues": evidence_dialogues,
        }
        return salvaged, json.dumps(salvaged, ensure_ascii=False), "salvaged_references"

    return None, cleaned, "failed"


def build_provider_config(
    model_value: Optional[str],
    base_url_value: Optional[str],
    api_key_value: Optional[str],
    optional_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构造统一的 LLM 配置。"""
    config: Dict[str, Any] = {}
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


ANSWER_PROMPT_WITH_HISTORY_EXTRACT = """
You are a strict JSON evidence selector for a multiple-choice QA ablation test.

Your job has two steps, in this exact order:
1. Select dialogue evidence from the CURRENT JSONL conversation.
2. Answer the question using only the selected evidence.

You must not answer from memory, story knowledge, common sense, or earlier removed evidence.

======================
EVIDENCE PROTOCOL
======================

The conversation is JSONL. Each line has exactly:
- "dia_id"
- "speaker"
- "utterance"

For every answer other than standalone "(F)", `evidence_dialogues` MUST contain supporting evidence objects.
If F appears alongside other selected options or inside an ordering sequence, treat it as a normal option and cite evidence for it.
Each evidence object MUST point to one JSONL line that is still present in the CURRENT conversation.

Required evidence fields:
- `id`: local evidence label only, such as "E1", "E2". Never put a dialogue id here.
- `dia_id`: copied exactly from a JSONL line, such as "D1:29".
- `speaker`: copied exactly from the SAME JSONL line as `dia_id`.
- `utterance`: prefer "" unless the source text is short and easy to JSON-escape.

Important:
- It is VALID to set `"utterance": ""` when `dia_id` and `speaker` are exact.
- The validator will recover the original utterance from `dia_id`.
- This is the preferred format for long, quoted, or multi-line source text.
- If you do include `utterance`, it must be copied exactly from the same JSONL line or be an exact contiguous substring of that line.

======================
STRICT FAILURE RULES
======================

Choose standalone (F) if any of these are true:
- You cannot find a supporting JSONL line in the CURRENT conversation.
- The supporting line was removed in a previous ablation round and is no longer present.
- You know the answer but cannot provide exact `dia_id` and `speaker`.
- The evidence would require paraphrase, summary, inference without a cited line, or external knowledge.

Do NOT do any of these:
- Do not return an answer other than standalone (F) with empty `evidence_dialogues`.
- Do not reuse evidence that is no longer in the CURRENT conversation.
- Do not invent or approximate `dia_id`.
- Do not put E1/E2 or option letters in `dia_id`.
- Do not copy the whole JSONL object into `utterance`.
- Do not put raw line breaks inside JSON string values.
- Do not output Markdown, comments, explanations, or extra text.

======================
DECISION RULE
======================

Return a supported answer other than standalone (F) only when:
1. You have selected current JSONL lines supporting every option in the answer.
2. Their exact `dia_id` and `speaker` are in `evidence_dialogues`.
3. Every evidence item has an `option` field naming the answer option it supports.
4. For an Ordering question, every evidence item also has `sequence_position`, starting at 1, and the positions jointly establish the entire answer sequence.

Otherwise choose standalone (F).

{{retry_instruction}}

======================
OUTPUT FORMAT
======================

Return exactly one valid JSON object:

{
    "question": "<copy the question text>",
    "answer": "<answer in the required format>",
    "evidence_dialogues": [
        {
            "id": "E1",
            "option": "A",
            "sequence_position": 1,
            "dia_id": "<exact dia_id from current JSONL>",
            "speaker": "<exact speaker from the same JSONL line>",
            "utterance": ""
        }
    ]
}

For standalone fallback answer "(F)", return:

{
    "question": "<copy the question text>",
    "answer": "(F)",
    "evidence_dialogues": []
}

======================
CONVERSATION HISTORY (CURRENT JSONL ONLY)
======================
{{conversation_history}}

======================
QUESTION
======================
{{question}}

ANSWER FORMAT:
{{answer_instruction}}
"""


ANSWER_PROMPT_ONLY_EVIDENCE = """
You are a rigorous intelligent assistant. Your task is to answer questions by synthesizing provided evidence and pre-defined reasoning logic.

# CONTEXT:
You will be provided with two types of information:

Evidence Fragments: Raw data or facts extracted from materials.

Reasoning Steps: Specific logical paths or intermediate deductions that must be followed.

# INSTRUCTIONS:

STRICT SCOPE: Your answer must be derived exclusively from the "EVIDENCE" and "REASONING STEPS" sections below. Do not use external knowledge or introduce original reasoning that contradicts or exceeds the provided steps.

--- REFERENCE MATERIAL ---
[EVIDENCE]
{{evidence}}
--- END OF MATERIAL ---

Question: {{question}}

ANSWER FORMAT:
{{answer_instruction}}
"""


ABLATION_CHUNK_SCAN_PROMPT = """
You are scanning ONE remaining conversation chunk during an evidence-ablation
test. Identify only text in this chunk that could support or conflict with an
answer option. Do not choose the final global answer and do not use story
knowledge, memory, or common sense.

The conversation is JSONL. Every non-none candidate must cite one or more exact
lines from this chunk using exact `dia_id` and `speaker`. Prefer an empty
`utterance`; the validator will recover the source text. Use only support types
`full`, `partial`, `conflict`, or `none`.

Return exactly one JSON object:
{
  "chunk_id": "{{chunk_id}}",
  "candidate_evidence": [
    {
      "option": "A",
      "sequence_position": 1,
      "support_type": "partial",
      "evidence_dialogues": [
        {
          "id": "E1",
          "option": "A",
          "sequence_position": 1,
          "dia_id": "D1:1",
          "speaker": "narrator",
          "utterance": ""
        }
      ]
    }
  ]
}

Omit irrelevant options. If this chunk contains no candidate evidence, return
an empty `candidate_evidence` list.

CURRENT CHUNK:
{{conversation_history}}

QUESTION:
{{question}}

FINAL OUTPUT REQUIREMENT:
Do not answer the multiple-choice question directly. Return only the JSON
object with `chunk_id` and `candidate_evidence`; never return a bare option such
as "(A)" or "(F)".
"""


ABLATION_CANDIDATE_REDUCE_PROMPT = """
You are making the GLOBAL multiple-choice decision for an evidence-ablation
test. You may use only the validated candidate evidence below. Do not use
outside knowledge, memory of the story, or facts not explicitly present in the
candidates.

Return exactly one JSON object:
{
  "answer": "{{answer_example}}",
  "evidence_dialogues": [
    {
      "option": "A",
      "sequence_position": 1,
      "dia_id": "D1:1",
      "speaker": "narrator",
      "utterance": ""
    }
  ]
}

For every answer other than standalone "(F)", cite the exact `dia_id` and
`speaker` from the candidate evidence that support every selected option. If F
appears alongside other options or in an ordering sequence, treat it as a normal
option and cite evidence for it. If the candidate evidence cannot establish a
supported answer, return standalone "(F)" with an empty evidence list.
For Ordering questions, every evidence item must include `sequence_position`,
equal to that option's one-based position in the returned answer sequence.

VALIDATED CANDIDATE EVIDENCE:
{{candidate_evidence}}

QUESTION:
{{question}}

FINAL OUTPUT REQUIREMENT:
Return only the JSON object described above. Even for standalone F, return
{"answer":"(F)","evidence_dialogues":[]}; never return bare "(F)".
"""


ABLATION_CANDIDATE_SELECT_PROMPT = """
You are reducing an oversized pool of already validated evidence candidates.
Select candidates that may be needed to decide the question globally,
including partial and conflicting evidence. Do not answer the question and do
not rewrite evidence. Retain at most 8 candidates.

Return exactly one JSON object:
{
  "candidate_indexes": [0, 2]
}

CANDIDATES:
{{candidate_evidence}}

QUESTION:
{{question}}

FINAL OUTPUT REQUIREMENT:
Do not answer the multiple-choice question directly. Return only the JSON
object with `candidate_indexes`; never return a bare option such as "(A)" or
"(F)".
"""


DEFAULT_LLM_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
ITERATIVE_ABLATION_MAX_ROUNDS = 5
DEFAULT_ABLATION_CONFIG = {
    "context_limit": 32768,
    "prompt_safety_tokens": 4096,
    "chunk_tokens": 8192,
    "retrieval_chunks": 6,
    "chunk_max_workers": 4,
}


def normalize_ablation_config(
    raw_config: Optional[Dict[str, Any]],
    logger=None,
) -> Dict[str, int]:
    """Validate optional iterative-ablation limits and apply safe defaults."""

    event_logger = logger or logging.getLogger(__name__)
    raw_config = raw_config if isinstance(raw_config, dict) else {}
    normalized = dict(DEFAULT_ABLATION_CONFIG)

    def use_integer(key: str, minimum: int) -> None:
        raw_value = raw_config.get(key, DEFAULT_ABLATION_CONFIG[key])
        try:
            if isinstance(raw_value, bool):
                raise ValueError("boolean is not an integer setting")
            if isinstance(raw_value, int):
                value = raw_value
            elif (
                isinstance(raw_value, str)
                and re.fullmatch(r"[+-]?\d+", raw_value.strip())
            ):
                value = int(raw_value.strip())
            else:
                raise ValueError("value must be an integer")
            if value < minimum:
                raise ValueError(f"value must be at least {minimum}")
        except (TypeError, ValueError) as exc:
            event_logger.warning(
                "EVENT | ablation_config | status=fallback | key=%s | value=%r | default=%d | error=%s",
                key,
                raw_value,
                DEFAULT_ABLATION_CONFIG[key],
                exc,
            )
            return
        normalized[key] = value

    use_integer("context_limit", 1)
    use_integer("prompt_safety_tokens", 0)
    use_integer("chunk_tokens", 1)
    use_integer("retrieval_chunks", 1)
    use_integer("chunk_max_workers", 1)

    context_limit = normalized["context_limit"]
    if normalized["prompt_safety_tokens"] >= context_limit:
        fallback_safety = min(
            DEFAULT_ABLATION_CONFIG["prompt_safety_tokens"],
            max(0, context_limit // 8),
        )
        event_logger.warning(
            "EVENT | ablation_config | status=fallback | key=prompt_safety_tokens | value=%r | default=%d | reason=must_be_below_context_limit",
            normalized["prompt_safety_tokens"],
            fallback_safety,
        )
        normalized["prompt_safety_tokens"] = fallback_safety

    available_context = max(
        2,
        context_limit - normalized["prompt_safety_tokens"],
    )
    if normalized["chunk_tokens"] >= available_context:
        fallback_chunk_tokens = min(
            DEFAULT_ABLATION_CONFIG["chunk_tokens"],
            available_context - 1,
        )
        event_logger.warning(
            "EVENT | ablation_config | status=fallback | key=chunk_tokens | value=%r | default=%d | reason=must_be_below_available_context",
            normalized["chunk_tokens"],
            fallback_chunk_tokens,
        )
        normalized["chunk_tokens"] = max(1, fallback_chunk_tokens)

    return normalized


def derive_evidence_stage_paths(output_file_path: str) -> Tuple[str, str]:
    """根据输出路径推导 v2a/v2b 两阶段检测路径。"""
    base, ext = os.path.splitext(output_file_path)

    if base.endswith("_v2a"):
        prefix = base[:-4]
        return f"{prefix}_v2a{ext}", f"{prefix}_v2b{ext}"
    if base.endswith("_v2b"):
        prefix = base[:-4]
        return f"{prefix}_v2a{ext}", f"{prefix}_v2b{ext}"
    if base.endswith("_v2"):
        prefix = base[:-3]
        return f"{prefix}_v2a{ext}", f"{prefix}_v2b{ext}"

    return f"{base}_v2a{ext}", f"{base}_v2b{ext}"


class FullContextManager:
    """步骤 1 题目合理性检测管理器。"""

    def __init__(
        self,
        output_path: str,
        logger=None,
        figure_view: bool = False,
        llm_config=None,
        ablation_config=None,
    ):
        load_dotenv()
        self.output_path = output_path
        llm_config = llm_config or {}

        llm_model = llm_config.get("model") or DEFAULT_LLM_MODEL
        llm_base_url = llm_config.get("base_url") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        llm_api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")

        client_kwargs: Dict[str, Any] = {}
        if llm_base_url:
            client_kwargs["base_url"] = llm_base_url
        if llm_api_key:
            client_kwargs["api_key"] = llm_api_key

        self.openai_client = OpenAI(**client_kwargs)
        self.model_name = llm_model
        self.logger = logger if logger else logging.getLogger(__name__)
        self.figure_view = figure_view
        self.ablation_config = normalize_ablation_config(
            ablation_config,
            logger=self.logger,
        )

        self.results: List[List[Dict[str, Any]]] = []
        self.lock = threading.Lock()
        self.original_data: List[Dict[str, Any]] = []
        self.failed_questions = 0

    def _resolve_speaker_name(self, raw_speaker: Any, speaker_map: Dict[str, str]) -> str:
        if isinstance(raw_speaker, str):
            if raw_speaker in speaker_map:
                return speaker_map[raw_speaker]
            return raw_speaker

        if isinstance(raw_speaker, int):
            key = f"speaker_{raw_speaker}"
            return speaker_map.get(key, str(raw_speaker))

        return str(raw_speaker)

    def _format_conversation(self, conversation_item: Dict[str, Any]) -> str:
        """将 conversation 字典格式化为 JSONL，显式暴露 dia_id 供模型复制。"""
        if not isinstance(conversation_item, dict):
            return ""

        history: List[str] = []
        speaker_map: Dict[str, str] = {}

        for key, value in conversation_item.items():
            if re.fullmatch(r"speaker_\d+", str(key)) and isinstance(value, str):
                speaker_map[str(key)] = value

        for alias in ("speaker_a", "speaker_b"):
            alias_value = conversation_item.get(alias)
            if isinstance(alias_value, str):
                speaker_map[alias] = alias_value

        session_keys = [
            str(key)
            for key in conversation_item.keys()
            if re.fullmatch(r"session_\d+", str(key))
        ]

        session_keys.sort(key=lambda key: int(re.match(r"session_(\d+)", key).group(1)))

        for session_key in session_keys:
            chats = conversation_item.get(session_key, [])
            if not isinstance(chats, list):
                continue

            for chat in chats:
                if not isinstance(chat, dict):
                    continue
                if "speaker" not in chat:
                    continue

                speaker_name = self._resolve_speaker_name(chat.get("speaker"), speaker_map)
                text_content = chat.get("utterance", chat.get("text", ""))
                dia_id = str(chat.get("dia_id", "")).strip()
                if not dia_id or not str(text_content or "").strip():
                    continue

                if self.figure_view and "img_url" in chat and "blip_caption" in chat:
                    text_content += f" [Image: {chat.get('img_url')}] with caption: {chat.get('blip_caption')}"

                history.append(
                    json.dumps(
                        {
                            "dia_id": dia_id,
                            "speaker": speaker_name,
                            "utterance": text_content,
                        },
                        ensure_ascii=False,
                    )
                )

        return "\n".join(history).strip()

    def _validate_chunk_candidates(
        self,
        raw_candidates: Any,
        chunk: AblationChunk,
    ) -> List[Dict[str, Any]]:
        """Keep only option evidence that aligns to the scanned chunk."""

        if not isinstance(raw_candidates, list):
            return []

        validated: List[Dict[str, Any]] = []
        allowed_support_types = {"full", "partial", "conflict"}
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict):
                continue

            option = str(raw_candidate.get("option", "")).strip().upper()
            support_type = str(
                raw_candidate.get("support_type", "")
            ).strip().lower()
            if option not in {"A", "B", "C", "D", "E", "F"}:
                continue
            if support_type not in allowed_support_types:
                continue

            raw_evidence = propagate_candidate_evidence_provenance(raw_candidate)
            if not isinstance(raw_evidence, list) or not raw_evidence:
                continue

            aligned_evidence: List[Dict[str, Any]] = []
            every_evidence_item_aligned = True
            for raw_evidence_item in raw_evidence:
                normalized_item, _ = self._normalize_model_evidence_output(
                    [raw_evidence_item],
                    chunk.conversation,
                )
                if len(normalized_item) != 1:
                    every_evidence_item_aligned = False
                    break
                aligned_item, _ = align_evidence_dialogues(
                    normalized_item,
                    chunk.conversation,
                )
                if len(aligned_item) != 1:
                    every_evidence_item_aligned = False
                    break
                aligned_evidence.extend(aligned_item)
            if not every_evidence_item_aligned or not aligned_evidence:
                continue

            validated.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "option": option,
                    "support_type": support_type,
                    "evidence_dialogues": aligned_evidence,
                }
            )

        return validated

    def _scan_ablation_chunk(
        self,
        chunk: AblationChunk,
        question: str,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        """Extract option-specific evidence candidates from one chunk."""

        prompt = self._build_ablation_chunk_scan_prompt(
            chunk,
            question,
        )
        last_response = ""
        last_error = "empty_or_invalid_model_response"
        for attempt in range(1, max(1, int(max_attempts)) + 1):
            raw_response, _, max_context_exceeded = self._call_llm(
                prompt,
                max_retries=1,
            )
            last_response = raw_response
            if max_context_exceeded:
                return {
                    "status": "context_exceeded",
                    "chunk_id": chunk.chunk_id,
                    "candidate_evidence": [],
                    "attempts": attempt,
                    "response": raw_response,
                    "error": "chunk_context_exceeded",
                }
            parsed, _, repair_status = parse_json_response(raw_response)
            if parsed is None:
                last_error = f"invalid_json:{repair_status}"
                continue

            if "candidate_evidence" not in parsed:
                last_error = "candidate_evidence_missing"
                continue
            raw_candidates = parsed.get("candidate_evidence")
            if not isinstance(raw_candidates, list):
                last_error = "candidate_evidence_not_list"
                continue
            validated_candidates: List[Dict[str, Any]] = []
            invalid_candidate = False
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, dict):
                    invalid_candidate = True
                    break
                option = str(
                    raw_candidate.get("option", "")
                ).strip().upper()
                support_type = str(
                    raw_candidate.get("support_type", "")
                ).strip().lower()
                raw_evidence = raw_candidate.get(
                    "evidence_dialogues",
                    [],
                )
                if (
                    option in {"A", "B", "C", "D", "E", "F"}
                    and support_type == "none"
                    and isinstance(raw_evidence, list)
                    and not raw_evidence
                ):
                    continue
                validated_one = self._validate_chunk_candidates(
                    [raw_candidate],
                    chunk,
                )
                if len(validated_one) != 1:
                    invalid_candidate = True
                    break
                validated_candidates.extend(validated_one)
            if invalid_candidate:
                last_error = "invalid_or_unaligned_non_none_candidate"
                continue

            return {
                "status": "ok",
                "chunk_id": chunk.chunk_id,
                "candidate_evidence": validated_candidates,
                "attempts": attempt,
                "response": raw_response,
                "error": "",
            }

        return {
            "status": "failed",
            "chunk_id": chunk.chunk_id,
            "candidate_evidence": [],
            "attempts": max(1, int(max_attempts)),
            "response": last_response,
            "error": last_error,
        }

    def _build_ablation_chunk_scan_prompt(
        self,
        chunk: AblationChunk,
        question: str,
    ) -> str:
        return Template(ABLATION_CHUNK_SCAN_PROMPT).render(
            {
                "chunk_id": chunk.chunk_id,
                "conversation_history": self._format_conversation(
                    chunk.conversation
                ),
                "question": strip_answer_instruction_suffix(question),
            }
        )

    def _deduplicate_ablation_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Deduplicate exact option-support-evidence candidates."""

        deduplicated: List[Dict[str, Any]] = []
        seen = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            evidence = candidate.get("evidence_dialogues", [])
            if not isinstance(evidence, list) or not evidence:
                continue
            evidence_key = tuple(
                (
                    str(item.get("dia_id", "")).strip(),
                    str(item.get("speaker", "")).strip(),
                    str(item.get("utterance", "")).strip(),
                )
                for item in evidence
                if isinstance(item, dict)
            )
            key = (
                str(candidate.get("option", "")).strip().upper(),
                str(candidate.get("support_type", "")).strip().lower(),
                evidence_key,
            )
            if not evidence_key or key in seen:
                continue
            seen.add(key)
            deduplicated.append(candidate)
        return deduplicated

    def _sort_ablation_candidates(
        self,
        candidates: List[Dict[str, Any]],
        remaining_conversation: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Order exact evidence and candidates by global conversation position."""

        order_by_dia_id: Dict[str, int] = {}
        global_order = 0
        session_keys = [
            str(key)
            for key, value in remaining_conversation.items()
            if re.fullmatch(r"session_\d+", str(key))
            and isinstance(value, list)
        ]
        session_keys.sort(
            key=lambda key: int(key.split("_")[1])
        )
        for session_key in session_keys:
            for turn in remaining_conversation.get(session_key, []):
                if not isinstance(turn, dict):
                    continue
                dia_id = str(turn.get("dia_id", "")).strip().casefold()
                if dia_id and dia_id not in order_by_dia_id:
                    order_by_dia_id[dia_id] = global_order
                global_order += 1

        unknown_order = global_order + 1
        ordered_candidates: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            ordered_candidate = copy.deepcopy(candidate)
            evidence = ordered_candidate.get(
                "evidence_dialogues",
                [],
            )
            if isinstance(evidence, list):
                evidence.sort(
                    key=lambda item: order_by_dia_id.get(
                        str(item.get("dia_id", "")).strip().casefold(),
                        unknown_order,
                    )
                    if isinstance(item, dict)
                    else unknown_order
                )
            ordered_candidates.append(ordered_candidate)

        ordered_candidates.sort(
            key=lambda candidate: (
                min(
                    (
                        order_by_dia_id.get(
                            str(evidence.get("dia_id", ""))
                            .strip()
                            .casefold(),
                            unknown_order,
                        )
                        for evidence in candidate.get(
                            "evidence_dialogues",
                            [],
                        )
                        if isinstance(evidence, dict)
                    ),
                    default=unknown_order,
                ),
                str(candidate.get("option", "")).strip().upper(),
            )
        )
        return ordered_candidates

    def _reduce_ablation_candidates(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        remaining_conversation: Dict[str, Any],
        question_type: Any = None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """Choose the global answer using only validated exact evidence."""

        candidates = self._sort_ablation_candidates(
            candidates,
            remaining_conversation,
        )
        candidates = self._deduplicate_ablation_candidates(candidates)
        original_candidate_count = len(candidates)
        diagnostics = {
            "candidate_count": original_candidate_count,
            "candidate_dia_id_count": 0,
            "candidate_batching_used": False,
            "retained_candidate_count": original_candidate_count,
        }
        if not candidates:
            return {
                "answer": "(F)",
                "evidence_dialogues": [],
            }, "", diagnostics

        prompt = self._build_ablation_reducer_prompt(question, candidates, question_type)
        prompt_budget = self._ablation_prompt_budget()
        if estimate_tokens(prompt, self.model_name) > prompt_budget:
            diagnostics["candidate_batching_used"] = True
            candidates, batching_diagnostics = (
                self._compress_ablation_candidates_for_reducer(
                    question,
                    candidates,
                    question_type,
                )
            )
            diagnostics["candidate_batching"] = batching_diagnostics
            diagnostics["retained_candidate_count"] = len(candidates)
            if not candidates:
                diagnostics["error"] = batching_diagnostics.get(
                    "error",
                    "candidate_batching_failed",
                )
                return {}, "", diagnostics
            candidates = self._sort_ablation_candidates(
                candidates,
                remaining_conversation,
            )
            prompt = self._build_ablation_reducer_prompt(question, candidates, question_type)

        allowed_dia_ids = {
            str(evidence.get("dia_id", "")).strip().casefold()
            for candidate in candidates
            for evidence in candidate.get("evidence_dialogues", [])
            if isinstance(evidence, dict) and evidence.get("dia_id")
        }
        diagnostics["candidate_dia_id_count"] = len(allowed_dia_ids)
        raw_response, _, _ = self._call_llm(prompt, max_retries=3)
        parsed, _, repair_status = parse_json_response(raw_response)
        diagnostics["parse_status"] = repair_status
        if not isinstance(parsed, dict):
            if re.fullmatch(r"\(\s*F\s*\)", str(raw_response).strip(), re.IGNORECASE):
                diagnostics["parse_status"] = "standalone_f_fallback"
                return {
                    "answer": "(F)",
                    "evidence_dialogues": [],
                }, raw_response, diagnostics
            diagnostics["error"] = "invalid_reducer_response"
            return {}, raw_response, diagnostics

        answer = str(parsed.get("answer", "")).strip()
        predicted_sequence, prediction_malformed = parse_question_answer(answer, question_type)
        predicted = set(predicted_sequence)
        if prediction_malformed or not predicted_sequence:
            diagnostics["error"] = "invalid_reducer_answer"
            return {}, raw_response, diagnostics
        if predicted == {"F"}:
            return {
                "answer": answer or "(F)",
                "evidence_dialogues": [],
            }, raw_response, diagnostics

        normalized, normalization_report = self._normalize_model_evidence_output(
            parsed.get("evidence_dialogues", []),
            remaining_conversation,
        )
        aligned, alignment_report = align_evidence_dialogues(
            normalized,
            remaining_conversation,
        )
        aligned = [
            evidence
            for evidence in aligned
            if str(evidence.get("dia_id", "")).strip().casefold()
            in allowed_dia_ids
        ]
        if normalize_question_type(question_type) == ORDERING:
            position_by_option = {
                option: position
                for position, option in enumerate(predicted_sequence, start=1)
            }
            for evidence in aligned:
                option = str(evidence.get("option", "")).strip().upper()
                if option in position_by_option:
                    evidence["sequence_position"] = position_by_option[option]
        diagnostics["normalization"] = normalization_report
        diagnostics["alignment"] = alignment_report
        if not aligned:
            diagnostics["error"] = "non_f_answer_without_candidate_evidence"
            return {}, raw_response, diagnostics
        if not answer_evidence_covers_selection(
            question_type,
            predicted_sequence,
            aligned,
        ):
            diagnostics["error"] = "incomplete_answer_evidence_coverage"
            return {}, raw_response, diagnostics

        return {
            "answer": answer,
            "evidence_dialogues": aligned,
        }, raw_response, diagnostics

    def _ablation_prompt_budget(self) -> int:
        return max(
            1,
            int(self.ablation_config["context_limit"])
            - int(self.ablation_config["prompt_safety_tokens"]),
        )

    def _build_ablation_reducer_prompt(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        question_type: Any = None,
    ) -> str:
        normalized_type = normalize_question_type(question_type)
        answer_example = {
            MULTIPLE_SELECT: "(A,C)",
            ORDERING: "(B,A,D,C)",
        }.get(normalized_type, "(A)")
        return Template(ABLATION_CANDIDATE_REDUCE_PROMPT).render(
            {
                "candidate_evidence": json.dumps(
                    candidates,
                    ensure_ascii=False,
                ),
                "question": strip_answer_instruction_suffix(question),
                "answer_example": answer_example,
            }
        )

    def _partition_ablation_candidate_batches(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        question_type: Any = None,
    ) -> List[List[Dict[str, Any]]]:
        """Greedily form candidate batches that fit the reducer budget."""

        budget = self._ablation_prompt_budget()
        batches: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for candidate in candidates:
            proposed = current + [candidate]
            prompt = self._build_ablation_reducer_prompt(
                question,
                proposed,
                question_type,
            )
            if estimate_tokens(prompt, self.model_name) <= budget:
                current = proposed
                continue
            if current:
                batches.append(current)
                current = []
            single_prompt = self._build_ablation_reducer_prompt(
                question,
                [candidate],
                question_type,
            )
            if estimate_tokens(single_prompt, self.model_name) > budget:
                return []
            current = [candidate]
        if current:
            batches.append(current)
        return batches

    def _select_relevant_ablation_candidates(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Select a bounded exact subset from one oversized reducer batch."""

        indexed_candidates = [
            {
                "candidate_index": idx,
                **candidate,
            }
            for idx, candidate in enumerate(candidates)
        ]
        prompt = Template(ABLATION_CANDIDATE_SELECT_PROMPT).render(
            {
                "candidate_evidence": json.dumps(
                    indexed_candidates,
                    ensure_ascii=False,
                ),
                "question": strip_answer_instruction_suffix(question),
            }
        )
        raw_response, _, context_exceeded = self._call_llm(
            prompt,
            max_retries=3,
        )
        parsed, _, repair_status = parse_json_response(raw_response)
        diagnostics = {
            "input_count": len(candidates),
            "retained_count": len(candidates),
            "parse_status": repair_status,
        }
        if context_exceeded or not isinstance(parsed, dict):
            diagnostics["error"] = (
                "selector_context_exceeded"
                if context_exceeded
                else "invalid_selector_response"
            )
            return list(candidates), diagnostics

        raw_indexes = parsed.get("candidate_indexes", [])
        if not isinstance(raw_indexes, list):
            diagnostics["error"] = "candidate_indexes_not_list"
            return list(candidates), diagnostics
        indexes = []
        for raw_index in raw_indexes:
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(candidates) and index not in indexes:
                indexes.append(index)
        retained = [candidates[index] for index in indexes[:8]]
        diagnostics["retained_count"] = len(retained)
        return retained, diagnostics

    def _compress_ablation_candidates_for_reducer(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        question_type: Any = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Recursively batch and shrink an oversized exact candidate pool."""

        current = list(candidates)
        passes: List[Dict[str, Any]] = []
        for pass_id in range(1, 7):
            batches = self._partition_ablation_candidate_batches(
                question,
                current,
                question_type,
            )
            if not batches:
                return [], {
                    "passes": passes,
                    "error": "single_candidate_exceeds_reducer_budget",
                }

            retained: List[Dict[str, Any]] = []
            batch_diagnostics = []
            for batch in batches:
                selected, diagnostics = (
                    self._select_relevant_ablation_candidates(
                        question,
                        batch,
                    )
                )
                retained.extend(selected)
                batch_diagnostics.append(diagnostics)
            retained = self._deduplicate_ablation_candidates(retained)
            passes.append(
                {
                    "pass": pass_id,
                    "input_count": len(current),
                    "batch_count": len(batches),
                    "retained_count": len(retained),
                    "batches": batch_diagnostics,
                }
            )
            if not retained:
                return [], {
                    "passes": passes,
                    "error": "selector_retained_no_candidates",
                }

            prompt = self._build_ablation_reducer_prompt(
                question,
                retained,
                question_type,
            )
            if (
                estimate_tokens(prompt, self.model_name)
                <= self._ablation_prompt_budget()
            ):
                return retained, {"passes": passes}
            if len(retained) >= len(current):
                return [], {
                    "passes": passes,
                    "error": "candidate_batching_made_no_progress",
                }
            current = retained

        return [], {
            "passes": passes,
            "error": "candidate_batching_pass_limit_reached",
        }

    def _scan_ablation_chunks(
        self,
        chunks: List[AblationChunk],
        question: str,
    ) -> List[Dict[str, Any]]:
        """Scan chunks with bounded concurrency while returning source order."""

        if not chunks:
            return []
        max_workers = max(
            1,
            min(
                int(self.ablation_config["chunk_max_workers"]),
                len(chunks),
            ),
        )
        if max_workers == 1:
            serial_results = []
            for chunk in chunks:
                try:
                    result = self._scan_ablation_chunk_with_split(chunk, question)
                except Exception as exc:  # noqa: PERF203
                    result = {
                        "status": "failed",
                        "chunk_id": chunk.chunk_id,
                        "candidate_evidence": [],
                        "attempts": 1,
                        "response": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                serial_results.append(result)
            return serial_results

        results_by_id: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(
                    self._scan_ablation_chunk_with_split,
                    chunk,
                    question,
                ): chunk
                for chunk in chunks
            }
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    results_by_id[chunk.chunk_id] = future.result()
                except Exception as exc:  # noqa: PERF203
                    results_by_id[chunk.chunk_id] = {
                        "status": "failed",
                        "chunk_id": chunk.chunk_id,
                        "candidate_evidence": [],
                        "attempts": 1,
                        "response": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        return [results_by_id[chunk.chunk_id] for chunk in chunks]

    def _scan_ablation_chunk_with_split(
        self,
        chunk: AblationChunk,
        question: str,
    ) -> Dict[str, Any]:
        """Retry a context-failed multi-turn chunk as smaller complete-turn chunks."""

        prompt = self._build_ablation_chunk_scan_prompt(
            chunk,
            question,
        )
        if (
            estimate_tokens(prompt, self.model_name)
            > self._ablation_prompt_budget()
        ):
            result = {
                "status": "context_exceeded",
                "chunk_id": chunk.chunk_id,
                "candidate_evidence": [],
                "attempts": 0,
                "response": "",
                "error": "scanner_prompt_preflight_exceeded",
            }
        else:
            result = self._scan_ablation_chunk(chunk, question)
        if result.get("status") != "context_exceeded":
            return result
        if len(chunk.dia_ids) <= 1:
            return {
                **result,
                "status": "failed",
                "error": "single_turn_chunk_context_exceeded",
            }

        raw_children = bisect_ablation_chunk(
            chunk,
            self.model_name,
        )
        if len(raw_children) != 2:
            return {
                **result,
                "status": "failed",
                "error": "chunk_split_made_no_progress",
            }

        child_results = [
            self._scan_ablation_chunk_with_split(child, question)
            for child in raw_children
        ]
        failed_children = [
            child_result
            for child_result in child_results
            if child_result.get("status") != "ok"
        ]
        return {
            "status": "failed" if failed_children else "ok",
            "chunk_id": chunk.chunk_id,
            "candidate_evidence": [
                candidate
                for child_result in child_results
                if child_result.get("status") == "ok"
                for candidate in child_result.get("candidate_evidence", [])
            ],
            "attempts": result.get("attempts", 0)
            + sum(
                child_result.get("attempts", 0)
                for child_result in child_results
            ),
            "response": result.get("response", ""),
            "error": (
                "one_or_more_split_children_failed"
                if failed_children
                else ""
            ),
            "split_children": child_results,
        }

    def _has_correct_new_ablation_evidence(
        self,
        data: Dict[str, Any],
        answer_candidates: List[str],
        cumulative_removed_evidence: List[Dict[str, Any]],
        question_type: Any = None,
    ) -> bool:
        if not data:
            return False
        score_result = score_mcq_prediction(
            data.get("answer", ""),
            answer_candidates,
            question_type,
        )
        if not score_result.get("is_correct", False):
            return False
        if not answer_evidence_covers_selection(
            question_type,
            score_result.get("predicted_options", []),
            data.get("evidence_dialogues", []),
        ):
            return False

        removed_ids = {
            str(item.get("dia_id", "")).strip().casefold()
            for item in cumulative_removed_evidence
            if isinstance(item, dict) and item.get("dia_id")
        }
        removed_text = {
            str(item.get("utterance", "")).strip().casefold()
            for item in cumulative_removed_evidence
            if isinstance(item, dict) and item.get("utterance")
        }
        for evidence in data.get("evidence_dialogues", []):
            if not isinstance(evidence, dict):
                continue
            dia_id = str(evidence.get("dia_id", "")).strip().casefold()
            utterance = str(evidence.get("utterance", "")).strip().casefold()
            if dia_id:
                if dia_id not in removed_ids:
                    return True
                continue
            if utterance and utterance not in removed_text:
                return True
        return False

    def _answer_after_ablation(
        self,
        conversation_item: Dict[str, Any],
        question: str,
        answer_candidates: List[str],
        cumulative_removed_evidence: List[Dict[str, Any]],
        question_type: Any = None,
        retry_instruction: str = "",
        chunk_result_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """Answer one ablation round using full context or chunked coverage."""

        remaining_conversation = self._conversation_without_evidence(
            conversation_item,
            cumulative_removed_evidence,
        )
        remaining_text = self._format_conversation(remaining_conversation)
        remaining_tokens = estimate_tokens(remaining_text, self.model_name)
        prompt = self._build_prompt(
            conversation_item=conversation_item,
            question=question,
            evidence_blocks=cumulative_removed_evidence,
            only_evidence=0,
            except_evidence=1,
            retry_instruction=retry_instruction,
            question_type=question_type,
        )
        prompt_tokens = estimate_tokens(prompt, self.model_name)
        prompt_budget = max(
            1,
            int(self.ablation_config["context_limit"])
            - int(self.ablation_config["prompt_safety_tokens"]),
        )
        audit = {
            "ablation_mode": "full_context",
            "remaining_estimated_tokens": remaining_tokens,
            "chunk_count": 1 if remaining_text else 0,
            "retrieved_chunk_count": 0,
            "exhaustive_scan_used": False,
            "scanned_chunk_count": 0,
            "failed_chunk_count": 0,
            "coverage_complete": True,
        }
        if prompt_tokens <= prompt_budget:
            (
                data,
                raw_response,
                _,
                _,
                max_context_exceeded,
            ) = self._request_json_answer(
                conversation_item=conversation_item,
                question=question,
                evidence_blocks=cumulative_removed_evidence,
                only_evidence=0,
                except_evidence=1,
                max_json_retries=3,
                retry_instruction=retry_instruction,
                question_type=question_type,
            )
            if not max_context_exceeded:
                return data, raw_response, audit
            audit["full_context_fallback"] = "api_context_exceeded"

        audit["ablation_mode"] = "chunked"
        audit["coverage_complete"] = False
        try:
            chunks = chunk_conversation(
                remaining_conversation,
                max_source_tokens=max(
                    1,
                    int(self.ablation_config["chunk_tokens"]),
                ),
                model_name=self.model_name,
            )
        except (OversizedDialogueTurnError, ValueError) as exc:
            audit["chunking_error"] = str(exc)
            return {}, "", audit

        audit["chunk_count"] = len(chunks)
        if not chunks:
            audit["coverage_complete"] = True
            return {
                "answer": "(F)",
                "evidence_dialogues": [],
            }, "", audit

        if chunk_result_cache is None:
            chunk_result_cache = {}
        current_chunk_ids = {chunk.chunk_id for chunk in chunks}
        result_by_chunk = {
            chunk_id: result
            for chunk_id, result in chunk_result_cache.items()
            if chunk_id in current_chunk_ids and result.get("status") == "ok"
        }

        retrieval_query = "\n".join(
            [
                question,
                *[
                    str(item.get("utterance", ""))
                    for item in cumulative_removed_evidence
                    if isinstance(item, dict)
                ],
            ]
        )
        retrieved_chunks = rank_chunks(
            chunks,
            retrieval_query,
            max(1, int(self.ablation_config["retrieval_chunks"])),
        )
        audit["retrieved_chunk_count"] = len(retrieved_chunks)
        retrieved_to_scan = [
            chunk
            for chunk in retrieved_chunks
            if chunk.chunk_id not in result_by_chunk
        ]
        if retrieved_to_scan:
            for result in self._scan_ablation_chunks(retrieved_to_scan, question):
                result_by_chunk[result["chunk_id"]] = result
                if result.get("status") == "ok":
                    chunk_result_cache[result["chunk_id"]] = result
        retrieved_chunk_ids = {chunk.chunk_id for chunk in retrieved_chunks}
        retrieved_candidates = [
            candidate
            for chunk_id, result in result_by_chunk.items()
            if chunk_id in retrieved_chunk_ids
            if result.get("status") == "ok"
            for candidate in result.get("candidate_evidence", [])
        ]
        data, raw_response, reducer_diagnostics = self._reduce_ablation_candidates(
            question,
            retrieved_candidates,
            remaining_conversation,
            question_type,
        )
        audit["retrieval_reducer"] = reducer_diagnostics
        audit["scanned_chunk_count"] = len(
            [chunk_id for chunk_id in result_by_chunk if chunk_id in retrieved_chunk_ids]
        )
        audit["failed_chunk_count"] = sum(
            result.get("status") != "ok"
            for chunk_id, result in result_by_chunk.items()
            if chunk_id in retrieved_chunk_ids
        )
        audit["coverage_complete"] = (
            len(result_by_chunk) == len(chunks)
            and all(
                result.get("status") == "ok"
                for result in result_by_chunk.values()
            )
        )
        if audit["coverage_complete"] and self._has_correct_new_ablation_evidence(
            data,
            answer_candidates,
            cumulative_removed_evidence,
            question_type,
        ):
            return data, raw_response, audit

        audit["exhaustive_scan_used"] = True
        successful_chunk_ids = {
            chunk_id
            for chunk_id, result in result_by_chunk.items()
            if result.get("status") == "ok"
        }
        remaining_chunks = [
            chunk
            for chunk in chunks
            if chunk.chunk_id not in successful_chunk_ids
        ]
        if remaining_chunks:
            for result in self._scan_ablation_chunks(remaining_chunks, question):
                result_by_chunk[result["chunk_id"]] = result
                if result.get("status") == "ok":
                    chunk_result_cache[result["chunk_id"]] = result

        failed_results = [
            result
            for result in result_by_chunk.values()
            if result.get("status") != "ok"
        ]
        audit["scanned_chunk_count"] = len(result_by_chunk)
        audit["failed_chunk_count"] = len(failed_results)
        audit["coverage_complete"] = (
            len(result_by_chunk) == len(chunks)
            and not failed_results
        )
        if not audit["coverage_complete"]:
            audit["coverage_error"] = "incomplete_exhaustive_chunk_scan"
            audit["failed_chunks"] = [
                {
                    "chunk_id": result.get("chunk_id", ""),
                    "error": result.get("error", "unknown_chunk_error"),
                    "attempts": result.get("attempts", 0),
                    "status": result.get("status", "failed"),
                    "response_preview": " ".join(
                        str(result.get("response", "")).split()
                    )[:300],
                }
                for result in failed_results
            ]
            return {}, raw_response, audit

        all_candidates = [
            candidate
            for chunk in chunks
            for candidate in result_by_chunk[chunk.chunk_id].get(
                "candidate_evidence",
                [],
            )
        ]
        data, raw_response, reducer_diagnostics = self._reduce_ablation_candidates(
            question,
            all_candidates,
            remaining_conversation,
            question_type,
        )
        audit["exhaustive_reducer"] = reducer_diagnostics
        return data, raw_response, audit

    def _conversation_without_evidence(
        self,
        conversation_item: Dict[str, Any],
        evidence_blocks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """从 conversation 中删除已知证据对应的对话。"""
        conv = copy.deepcopy(conversation_item)

        evidence_dia_ids = set()
        evidence_sessions_no_time = set()
        evidence_targets: List[Dict[str, Any]] = []

        def _normalize_dia_id(dia_id: Any) -> Optional[str]:
            normalized = str(dia_id or "").strip()
            if not normalized or normalized == "N/A":
                return None
            return normalized.casefold()

        def _normalize_match_text(text: Any) -> str:
            normalized = str(text or "")
            normalized = (
                normalized.replace("\u2019", "'")
                .replace("\u2018", "'")
                .replace("\u201c", '"')
                .replace("\u201d", '"')
                .replace("\u2026", "...")
            )
            normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
            return normalized

        def _compact_text(text: str) -> str:
            return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text or "")

        def _cleanup_remaining(text: str) -> str:
            cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
            # 清理仅剩说话人前缀（如 "Harry:" / "John: John:"）的伪残留
            cleaned = re.sub(r"^(?:[a-z][a-z0-9_\-']*:\s*)+", "", cleaned, flags=re.IGNORECASE).strip()
            if not re.search(r"[a-z0-9\u4e00-\u9fff]", cleaned, flags=re.IGNORECASE):
                return ""
            return cleaned

        def _subtract_fragment(remaining: str, fragment: str) -> str:
            if not remaining or not fragment:
                return remaining

            remaining = _cleanup_remaining(remaining)
            fragment = _cleanup_remaining(fragment)
            if not remaining or not fragment:
                return remaining

            remaining_fold = remaining.casefold()
            fragment_fold = fragment.casefold()

            # 1) 完全一致直接删空
            if remaining_fold == fragment_fold:
                return ""

            # 2) 双向包含（AB 证据、A 对话 或证据是对话的子句）
            if fragment_fold in remaining_fold and len(fragment_fold) >= 8:
                return _cleanup_remaining(re.sub(re.escape(fragment), " ", remaining, flags=re.IGNORECASE))
            if remaining_fold in fragment_fold and len(remaining_fold) >= 8:
                return ""

            # 3) 去标点后的包含匹配（处理 ... / 标点差异）
            remaining_compact = _compact_text(remaining)
            fragment_compact = _compact_text(fragment)
            if remaining_compact and fragment_compact:
                if remaining_compact == fragment_compact:
                    return ""
                if fragment_compact in remaining_compact and len(fragment_compact) >= 8:
                    remaining_compact = remaining_compact.replace(fragment_compact, "")
                    return _cleanup_remaining(remaining_compact)
                if remaining_compact in fragment_compact and len(remaining_compact) >= 8:
                    return ""

            # 4) 高相似度兜底（处理轻微字符差异）
            if remaining_compact and fragment_compact and min(len(remaining_compact), len(fragment_compact)) >= 20:
                import difflib

                ratio = difflib.SequenceMatcher(None, remaining_compact, fragment_compact).ratio()
                if ratio >= 0.92:
                    return ""

            return remaining

        def _is_low_information_residual(
            remaining: str,
            original_utterance: str,
            matched_fragments: List[str],
        ) -> bool:
            remaining_clean = _cleanup_remaining(remaining)
            if not remaining_clean:
                return True

            remaining_compact = _compact_text(remaining_clean)
            if not remaining_compact:
                return True

            # 非常短的残留通常是噪声。
            if len(remaining_compact) <= 5:
                return True

            original_compact = _compact_text(_normalize_match_text(original_utterance))
            if original_compact:
                ratio = len(remaining_compact) / max(len(original_compact), 1)
                # 若残留只占原证据很小比例，通常是被省略号/标点切分导致的尾部噪声。
                if len(remaining_compact) <= 12 and ratio <= 0.15:
                    return True

            # 若残留本身已包含在已命中的对话片段里，多半是文本清洗导致的伪残留。
            for fragment in matched_fragments or []:
                fragment_compact = _compact_text(_normalize_match_text(fragment))
                if fragment_compact and remaining_compact in fragment_compact and len(remaining_compact) <= 24:
                    return True

            filler_tokens = {
                "uh", "um", "hmm", "oh", "ah", "yeah", "yes", "no", "ok", "okay", "well", "hey",
                "to", "the", "a", "an", "and", "or", "but", "i", "you", "he", "she", "it", "we", "they",
            }
            tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", remaining_clean.lower())
            if tokens and all(token in filler_tokens for token in tokens):
                return True

            return False

        for evidence in evidence_blocks or []:
            if not isinstance(evidence, dict):
                continue

            dia_id = evidence.get("dia_id")
            utterance = evidence.get("utterance", "")
            utterance_norm = _normalize_match_text(utterance)
            normalized_dia_id = _normalize_dia_id(dia_id)

            if utterance_norm:
                evidence_targets.append(
                    {
                        "evidence": evidence,
                        "utterance_norm": utterance_norm,
                        "remaining_norm": utterance_norm,
                        "matched_fragments": [],
                        "dia_id": normalized_dia_id,
                    }
                )

            if normalized_dia_id:
                evidence_dia_ids.add(normalized_dia_id)

            if dia_id == "N/A":
                match = re.search(r"(session_\d+)_date_time", utterance)
                if match:
                    evidence_sessions_no_time.add(match.group(1))

        for session_key in evidence_sessions_no_time:
            conv.pop(f"{session_key}_date_time", None)

        for key, value in list(conv.items()):
            if not re.fullmatch(r"session_\d+", str(key)):
                continue
            if not isinstance(value, list):
                continue

            filtered_chats = []
            for chat in value:
                if not isinstance(chat, dict):
                    filtered_chats.append(chat)
                    continue

                chat_text_raw = str(get_dialogue_utterance(chat)).strip()
                chat_text_norm = _normalize_match_text(chat_text_raw)
                chat_dia_id = _normalize_dia_id(chat.get("dia_id"))
                match_by_dia = chat_dia_id in evidence_dia_ids
                match_by_text = False

                # 已通过 evidence alignment 的证据，删除语义以 dia_id 为准：
                # 命中 dia_id 就删除整条原始 dialogue turn，并直接标记对应证据已删除。
                # 这样可以正确处理 evidence 是原文子片段、省略号片段或范围 dia_id 展开后的情况。
                if match_by_dia and chat_text_norm and chat_dia_id:
                    for target in evidence_targets:
                        if target.get("dia_id") != chat_dia_id:
                            continue
                        target["remaining_norm"] = ""
                        if chat_text_raw:
                            target["matched_fragments"].append(chat_text_raw)

                if chat_text_norm:
                    for target in evidence_targets:
                        # 对有 dia_id 的证据，只允许 dia_id 命中触发删除。
                        # 避免短文本（如 "THE GUARD"）误删其他包含同名实体的对话。
                        if target.get("dia_id"):
                            continue
                        before_remaining = str(target.get("remaining_norm", ""))
                        after_remaining = _subtract_fragment(before_remaining, chat_text_norm)
                        if after_remaining != before_remaining:
                            match_by_text = True
                            target["remaining_norm"] = after_remaining
                            if chat_text_raw:
                                target["matched_fragments"].append(chat_text_raw)

                if match_by_dia or match_by_text:
                    continue
                filtered_chats.append(chat)

            conv[key] = filtered_chats

        remaining_evidence_fragments = []
        for target in evidence_targets:
            remaining_norm = _cleanup_remaining(str(target.get("remaining_norm", "")))
            if not remaining_norm:
                continue
            evidence_obj = target.get("evidence", {}) if isinstance(target.get("evidence"), dict) else {}
            matched_fragments = target.get("matched_fragments", [])
            if not isinstance(matched_fragments, list):
                matched_fragments = []

            if _is_low_information_residual(
                remaining=remaining_norm,
                original_utterance=str(evidence_obj.get("utterance", "")),
                matched_fragments=matched_fragments,
            ):
                continue

            remaining_evidence_fragments.append(
                {
                    "evidence_id": evidence_obj.get("id", ""),
                    "original_utterance": evidence_obj.get("utterance", ""),
                    "remaining_after_deletion": remaining_norm,
                    "matched_dialogue_fragments": matched_fragments,
                }
            )

        if remaining_evidence_fragments:
            self.logger.warning(
                "EVENT | evidence_deletion | status=warning | remaining_count=%d | remaining=%s",
                len(remaining_evidence_fragments),
                json.dumps(remaining_evidence_fragments, ensure_ascii=False),
            )

        return conv

    def _build_prompt(
        self,
        conversation_item: Dict[str, Any],
        question: str,
        evidence_blocks: List[Dict[str, Any]],
        only_evidence: int,
        except_evidence: int,
        allow_cannot_infer: bool = False,
        retry_instruction: Optional[str] = None,
        question_type: Any = None,
    ) -> str:
        if only_evidence == 1:
            template = Template(ANSWER_PROMPT_ONLY_EVIDENCE)
            evidence_text = json.dumps(evidence_blocks, ensure_ascii=False)
            normalized_type = normalize_question_type(question_type)
            return template.render(
                {
                    "evidence": evidence_text,
                    "question": question,
                    "answer_instruction": get_answer_instruction(question_type),
                }
            )

        if except_evidence == 1:
            conversation_obj = self._conversation_without_evidence(conversation_item, evidence_blocks)
        else:
            conversation_obj = conversation_item

        conversation_history = self._format_conversation(conversation_obj)
        template = Template(ANSWER_PROMPT_WITH_HISTORY_EXTRACT)
        return template.render(
            {
                "conversation_history": conversation_history,
                "question": question,
                "retry_instruction": retry_instruction or "",
                "answer_instruction": get_answer_instruction(question_type),
            }
        )

    def _call_llm(self, prompt: str, max_retries: int = 5) -> Tuple[str, float, int]:
        """调用 LLM，返回 (response, response_time, max_context_exceeded)。"""
        request_id = f"evidence-check-{uuid.uuid4()}"
        start_time = time.time()
        max_context_exceeded = 0
        last_error: Optional[Exception] = None
        last_error_traceback = ""

        for attempt in range(1, max_retries + 1):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = response.choices[0].message.content or ""
                return content, time.time() - start_time, max_context_exceeded

            except Exception as exc:  # noqa: PERF203
                last_error = exc
                error_text = str(exc)
                last_error_traceback = traceback.format_exc()
                if "maximum context length" in error_text.lower():
                    max_context_exceeded = 1
                    self.logger.warning(
                        "EVENT | evidence_llm_call | status=context_exceeded | attempt=%d/%d | request_id=%s | error=%s",
                        attempt,
                        max_retries,
                        request_id,
                        error_text,
                    )
                    return (
                        "Error: maximum context length exceeded.",
                        time.time() - start_time,
                        max_context_exceeded,
                    )

                # 某些 OpenAI 兼容网关会把仅 system 消息识别为缺少 input/prompt。
                # 这里兜底使用 Responses API，避免持续 400。
                if "missing_required_parameter" in error_text or (
                    "one of \"input\"" in error_text.lower()
                    and "prompt" in error_text.lower()
                ):
                    try:
                        response2 = self.openai_client.responses.create(
                            model=self.model_name,
                            input=prompt,
                            temperature=0.0,
                        )
                        content2 = getattr(response2, "output_text", "") or ""
                        return content2, time.time() - start_time, max_context_exceeded
                    except Exception as exc2:  # noqa: PERF203
                        last_error = exc2
                        error_text = str(exc2)
                        last_error_traceback = traceback.format_exc()
                        if "maximum context length" in error_text.lower():
                            max_context_exceeded = 1
                            return (
                                "Error: maximum context length exceeded.",
                                time.time() - start_time,
                                max_context_exceeded,
                            )

                if attempt >= max_retries:
                    break

                sleep_seconds = random.uniform(45, 75)
                self.logger.warning(
                    (
                        "EVENT | evidence_llm_call | status=retry | attempt=%d/%d | wait=%.2fs | "
                        "request_id=%s | error_type=%s | error=%s | error_repr=%r\n%s"
                    ),
                    attempt,
                    max_retries,
                    sleep_seconds,
                    request_id,
                    type(last_error).__name__ if last_error else "",
                    error_text,
                    last_error,
                    last_error_traceback,
                )
                time.sleep(sleep_seconds)

        self.logger.error(
            (
                "EVENT | evidence_llm_call | status=failed | attempts=%d | request_id=%s | "
                "error_type=%s | error=%s | error_repr=%r\n%s"
            ),
            max_retries,
            request_id,
            type(last_error).__name__ if last_error else "",
            last_error,
            last_error,
            last_error_traceback,
        )
        raise EvidenceProcessingError(
            f"LLM request failed after {max_retries} attempts: {last_error}"
        ) from last_error

    def _request_json_answer(
        self,
        conversation_item: Dict[str, Any],
        question: str,
        evidence_blocks: List[Dict[str, Any]],
        only_evidence: int,
        except_evidence: int,
        allow_cannot_infer: bool = False,
        max_json_retries: int = 10,
        retry_instruction: Optional[str] = None,
        question_type: Any = None,
    ) -> Tuple[Dict[str, Any], str, float, str, int]:
        """
        获取 JSON 响应。
        返回: (parsed_json, raw_response, total_response_time, prompt, max_context_exceeded)
        """
        last_response = ""
        last_prompt = ""
        total_response_time = 0.0
        max_context_exceeded = 0

        for retry_idx in range(max_json_retries):
            prompt = self._build_prompt(
                conversation_item=conversation_item,
                question=question,
                evidence_blocks=evidence_blocks,
                only_evidence=only_evidence,
                except_evidence=except_evidence,
                allow_cannot_infer=allow_cannot_infer,
                retry_instruction=retry_instruction,
                question_type=question_type,
            )

            response, response_time, context_flag = self._call_llm(prompt)
            total_response_time += response_time
            max_context_exceeded = max(max_context_exceeded, context_flag)
            last_response = response
            last_prompt = prompt
            if context_flag:
                self.logger.warning(
                    "EVENT | evidence_json_parse | status=context_exceeded | attempt=%d/%d",
                    retry_idx + 1,
                    max_json_retries,
                )
                return (
                    {},
                    response,
                    total_response_time,
                    prompt,
                    max_context_exceeded,
                )

            parsed, _, repair_status = parse_json_response(response)
            if isinstance(parsed, dict):
                if repair_status != "strict":
                    self.logger.info(
                        "EVENT | evidence_json_parse | status=repaired | attempt=%d/%d | repair=%s",
                        retry_idx + 1,
                        max_json_retries,
                        repair_status,
                    )
                return parsed, response, total_response_time, prompt, max_context_exceeded

            try:
                json.loads(clean_json_response(response))
            except json.JSONDecodeError as exc:
                self.logger.warning(
                    "EVENT | evidence_json_parse | status=retry | attempt=%d/%d | error=%s",
                    retry_idx + 1,
                    max_json_retries,
                    exc,
                )
            else:
                self.logger.warning(
                    "EVENT | evidence_json_parse | status=retry | attempt=%d/%d | error=parsed_non_object_json",
                    retry_idx + 1,
                    max_json_retries,
                )

        self.logger.error("EVENT | evidence_json_parse | status=failed | attempts=%d", max_json_retries)
        raise EvidenceProcessingError(
            f"JSON response parsing failed after {max_json_retries} attempts"
        )

    def _request_text_answer(
        self,
        conversation_item: Dict[str, Any],
        question: str,
        evidence_blocks: List[Dict[str, Any]],
        only_evidence: int,
        except_evidence: int,
        allow_cannot_infer: bool = False,
        question_type: Any = None,
    ) -> Tuple[str, float, str, int]:
        """获取自由格式文本响应，不做 JSON 解析。"""
        prompt = self._build_prompt(
            conversation_item=conversation_item,
            question=question,
            evidence_blocks=evidence_blocks,
            only_evidence=only_evidence,
            except_evidence=except_evidence,
            allow_cannot_infer=allow_cannot_infer,
            question_type=question_type,
        )
        response, response_time, max_context_exceeded = self._call_llm(prompt)
        return response, response_time, prompt, max_context_exceeded

    def _answer_allows_cannot_infer(self, answer_candidates: List[str]) -> bool:
        """Return whether a standalone F is one of the accepted answers."""
        for candidate in answer_candidates:
            sequence, malformed = parse_question_answer(candidate, SINGLE_CHOICE)
            if not malformed and sequence == ["F"]:
                return True
        return False

    def _get_aligned_dialogue_evidence(
        self,
        question_item: Dict[str, Any],
        conversation_item: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """校验题目 evidence_dialogues 是否能与原始 conversation 严格对齐。"""
        return align_evidence_dialogues(
            question_item.get("evidence_dialogues", []),
            conversation_item,
        )

    def _build_only_evidence_blocks(
        self,
        question_item: Dict[str, Any],
        aligned_dialogue_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """only_evidence 阶段保留有效对话证据和结构化 reasoning。"""
        evidence_ids = [
            evidence.get("id")
            for evidence in aligned_dialogue_evidence
            if isinstance(evidence, dict) and evidence.get("id")
        ]
        reasoning_steps = normalize_reasoning_steps(
            question_item.get("reasoning_steps", []),
            evidence_ids,
        )
        return list(aligned_dialogue_evidence) + reasoning_steps

    def _normalize_model_evidence_output(
        self,
        raw_evidence: Any,
        remaining_conversation: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """修复模型证据输出里的常见 schema 错误。

        修复只基于当前剩余原对话：字段别名、id/dia_id 错位、只给 dia_id
        时补回原文。这样可以减少格式导致的无效尝试，但不会引入原对话外证据。
        """
        if not isinstance(raw_evidence, list):
            if isinstance(raw_evidence, dict):
                raw_evidence = [raw_evidence]
            elif isinstance(raw_evidence, str) and raw_evidence.strip():
                raw_evidence = [{"utterance": raw_evidence.strip()}]
            else:
                return [], {
                    "moved_id_to_dia_id": 0,
                    "recovered_alias_dia_id": 0,
                    "filled_utterance_from_dia_id": 0,
                    "string_evidence_items": 0,
                }

        dialogue_index = build_dialogue_index(remaining_conversation)
        normalized: List[Dict[str, Any]] = []
        moved_count = 0
        alias_dia_id_count = 0
        filled_utterance_count = 0
        string_item_count = 0

        dia_id_aliases = (
            "dia_id",
            "dialogue_id",
            "dialog_id",
            "turn_id",
            "source_dia_id",
            "source_id",
            "line_id",
        )
        utterance_aliases = (
            "utterance",
            "text",
            "quote",
            "evidence",
            "content",
            "source_utterance",
        )

        def _first_non_empty(mapping: Dict[str, Any], keys: Tuple[str, ...]) -> str:
            for key in keys:
                value = mapping.get(key)
                if value not in (None, ""):
                    return str(value).strip()
            return ""

        for idx, item in enumerate(raw_evidence, start=1):
            if isinstance(item, str) and item.strip():
                item = {"utterance": item.strip()}
                string_item_count += 1
            elif not isinstance(item, dict):
                continue

            evidence = dict(item)
            raw_id = str(evidence.get("id", "") or "").strip()
            raw_dia_id = _first_non_empty(evidence, dia_id_aliases)
            raw_utterance = _first_non_empty(evidence, utterance_aliases)
            raw_id_looks_like_dia_id = bool(re.fullmatch(r"[A-Za-z]+\d+:\d+(?:-\d+)?", raw_id))

            if (
                (not raw_dia_id or raw_dia_id.upper() == "N/A")
                and raw_id_looks_like_dia_id
                and raw_id.casefold() in dialogue_index
            ):
                evidence["dia_id"] = dialogue_index[raw_id.casefold()].get("dia_id", raw_id)
                raw_dia_id = str(evidence.get("dia_id", "") or "").strip()
                moved_count += 1

            if raw_dia_id and raw_dia_id.upper() != "N/A":
                source_turn = dialogue_index.get(raw_dia_id.casefold())
                if source_turn:
                    canonical_dia_id = str(source_turn.get("dia_id") or raw_dia_id).strip()
                    if evidence.get("dia_id") != canonical_dia_id:
                        evidence["dia_id"] = canonical_dia_id
                        if str(item.get("dia_id") or "").strip() != canonical_dia_id:
                            alias_dia_id_count += 1

                    if not raw_utterance:
                        evidence["utterance"] = str(source_turn.get("utterance") or "")
                        raw_utterance = str(evidence.get("utterance") or "").strip()
                        filled_utterance_count += 1
                    elif "utterance" not in evidence:
                        evidence["utterance"] = raw_utterance

                    if evidence.get("speaker") in (None, ""):
                        evidence["speaker"] = source_turn.get("speaker")
                elif raw_utterance and "utterance" not in evidence:
                    evidence["utterance"] = raw_utterance
            elif raw_utterance and "utterance" not in evidence:
                evidence["utterance"] = raw_utterance

            # id 是本轮 evidence 的局部编号，统一重写，避免 D1:29 继续污染下游。
            evidence["id"] = f"E{len(normalized) + 1}"
            normalized.append(evidence)

        return normalized, {
            "moved_id_to_dia_id": moved_count,
            "recovered_alias_dia_id": alias_dia_id_count,
            "filled_utterance_from_dia_id": filled_utterance_count,
            "string_evidence_items": string_item_count,
        }

    def _build_evidence_retry_instruction(
        self,
        attempt_id: int,
        used_alignment_report: Dict[str, Any],
    ) -> str:
        """生成下一次消融回答的证据格式纠错提示。"""
        invalid_items = used_alignment_report.get("invalid_items", [])
        if not isinstance(invalid_items, list):
            invalid_items = []

        reasons = sorted(
            {
                str(item.get("reason") or "")
                for item in invalid_items
                if isinstance(item, dict) and item.get("reason")
            }
        )
        reason_text = ", ".join(reasons) if reasons else used_alignment_report.get(
            "reason",
            "missing_or_unaligned_evidence",
        )

        return (
            "PREVIOUS ATTEMPT FAILED EVIDENCE VALIDATION.\n"
            f"- Failed attempt: {attempt_id}\n"
            f"- Validation issue: {reason_text}\n"
            "- Your previous answer is invalid even if the option letter was correct.\n"
            "- Retry from scratch using only JSONL lines still present in the CURRENT conversation.\n"
            "- Prefer evidence objects with exact `dia_id`, exact `speaker`, and `utterance` set to \"\".\n"
            "- Do not copy long dialogue text into `utterance`; the validator will recover it from `dia_id`.\n"
            "- Do not cite removed evidence or any dia_id that is absent from the CURRENT conversation.\n"
            "- Every answer other than standalone (F) is valid only when `evidence_dialogues` contains current JSONL line references covering every selected option, including F when it appears with other options or in an ordering sequence.\n"
            "- If you cannot provide exact current `dia_id` and `speaker`, answer standalone (F)."
        )

    def _run_iterative_ablation(
        self,
        conversation_item: Dict[str, Any],
        answer: str,
        question: str,
        base_evidence_blocks: List[Dict[str, Any]],
        question_type: Any = None,
    ) -> List[Dict[str, Any]]:
        """五轮迭代证据删除。

        流程：
        1. 先删除题目原始证据，询问模型。
        2. 如果模型答错，立即中断并保留题目。
        3. 如果模型答对，要求模型返回支撑其答对的新对话证据。
        4. 将新证据叠加到删除列表里，再问一次。
        5. 最多重复五轮；五轮后仍答对，说明题目不够依赖证据，应过滤。
        """
        iterative_records: List[Dict[str, Any]] = []
        cumulative_removed_evidence = list(base_evidence_blocks)
        answer_candidates = normalize_answer_candidates(None, answer)
        gold_allows_cannot_infer = self._answer_allows_cannot_infer(answer_candidates)
        seen_evidence_keys = {
            (
                str(evidence.get("dia_id", "")).strip(),
                str(evidence.get("utterance", "")).strip(),
            )
            for evidence in cumulative_removed_evidence
            if isinstance(evidence, dict)
        }

        max_round_attempts = 5
        for round_id in range(1, ITERATIVE_ABLATION_MAX_ROUNDS + 1):
            accepted_record: Optional[Dict[str, Any]] = None
            invalid_attempts: List[Dict[str, Any]] = []
            retry_instruction = ""
            terminal_ablation_audit: Optional[Dict[str, Any]] = None
            round_chunk_result_cache: Dict[str, Dict[str, Any]] = {}
            remaining_conversation = self._conversation_without_evidence(
                conversation_item,
                cumulative_removed_evidence,
            )

            for attempt_id in range(1, max_round_attempts + 1):
                data, raw_response, ablation_audit = self._answer_after_ablation(
                    conversation_item,
                    question=question,
                    answer_candidates=answer_candidates,
                    cumulative_removed_evidence=cumulative_removed_evidence,
                    question_type=question_type,
                    retry_instruction=retry_instruction,
                    chunk_result_cache=round_chunk_result_cache,
                )

                if ablation_audit.get("coverage_complete", True):
                    terminal_ablation_audit = None
                if not ablation_audit.get("coverage_complete", True):
                    terminal_ablation_audit = ablation_audit
                    invalid_attempts.append(
                        {
                            "attempt": attempt_id,
                            "reason": "incomplete_chunk_coverage",
                            "response": raw_response,
                            "ablation_audit": ablation_audit,
                        }
                    )
                    continue

                if not data:
                    invalid_attempts.append(
                        {
                            "attempt": attempt_id,
                            "reason": "empty_or_invalid_model_response",
                            "response": raw_response,
                            "ablation_audit": ablation_audit,
                        }
                    )
                    continue

                score_result = score_mcq_prediction(
                    data.get("answer", ""),
                    answer_candidates,
                    question_type,
                )
                is_right = bool(score_result.get("is_correct", False))
                predicted_options = set(score_result.get("predicted_options", []))
                correct_abstain_without_evidence = (
                    is_right
                    and gold_allows_cannot_infer
                    and predicted_options == {"F"}
                )
                used_evidence = data.get("evidence_dialogues", [])
                if not isinstance(used_evidence, list):
                    used_evidence = []
                raw_used_evidence = used_evidence
                used_evidence, evidence_output_normalization = self._normalize_model_evidence_output(
                    raw_used_evidence,
                    remaining_conversation,
                )

                aligned_used_evidence, used_alignment_report = align_evidence_dialogues(
                    used_evidence,
                    remaining_conversation,
                )
                if any(evidence_output_normalization.values()):
                    used_alignment_report = {
                        **used_alignment_report,
                        "model_output_normalization": evidence_output_normalization,
                    }
                new_aligned_evidence: List[Dict[str, Any]] = []
                for evidence in aligned_used_evidence:
                    key = (
                        str(evidence.get("dia_id", "")).strip(),
                        str(evidence.get("utterance", "")).strip(),
                    )
                    if key in seen_evidence_keys:
                        continue
                    new_aligned_evidence.append(evidence)

                evidence_coverage_complete = answer_evidence_covers_selection(
                    question_type,
                    score_result.get("predicted_options", []),
                    aligned_used_evidence,
                )
                if is_right and not evidence_coverage_complete:
                    invalid_attempts.append(
                        {
                            "attempt": attempt_id,
                            "answer": data.get("answer", ""),
                            "used_evidence": used_evidence,
                            "used_evidence_alignment_check": used_alignment_report,
                            "reason": "incomplete_answer_evidence_coverage",
                        }
                    )
                    retry_instruction = (
                        "PREVIOUS ATTEMPT DID NOT CITE OPTION-LEVEL EVIDENCE FOR EVERY "
                        "SELECTED OPTION. Return an `option` field on each evidence item "
                        "and cover the complete answer."
                    )
                    continue

                if is_right and not new_aligned_evidence and not correct_abstain_without_evidence:
                    invalid_attempts.append(
                        {
                            "attempt": attempt_id,
                            "answer": data.get("answer", ""),
                            "raw_used_evidence": raw_used_evidence,
                            "used_evidence": used_evidence,
                            "used_evidence_alignment_check": used_alignment_report,
                            "model_output_normalization": evidence_output_normalization,
                            "reason": "correct_non_f_answer_without_verifiable_new_evidence",
                        }
                    )
                    retry_instruction = self._build_evidence_retry_instruction(
                        attempt_id,
                        used_alignment_report,
                    )
                    self.logger.warning(
                        "EVENT | iterative_ablation_attempt | status=retry | round=%d | attempt=%d/%d | reason=correct_non_f_answer_without_verifiable_new_evidence",
                        round_id,
                        attempt_id,
                        max_round_attempts,
                    )
                    continue

                if not is_right:
                    stop_reason = "model_answered_wrong_after_evidence_removed"
                elif correct_abstain_without_evidence and not new_aligned_evidence:
                    stop_reason = "correct_abstain_without_remaining_evidence"
                elif round_id >= ITERATIVE_ABLATION_MAX_ROUNDS:
                    stop_reason = "max_rounds_reached_while_still_right"
                else:
                    stop_reason = "continue"

                accepted_record = {
                    "round": round_id,
                    "attempt": attempt_id,
                    "answer": data.get("answer", ""),
                    "raw_used_evidence": raw_used_evidence,
                    "used_evidence": used_evidence,
                    "validated_used_evidence": aligned_used_evidence,
                    "newly_removed_evidence": new_aligned_evidence,
                    "used_evidence_alignment_check": used_alignment_report,
                    "invalid_attempts": invalid_attempts,
                    "cumulative_removed_evidence": list(cumulative_removed_evidence),
                    "all_evidence": list(cumulative_removed_evidence),
                    "result": "right" if is_right else "wrong",
                    "stop_reason": stop_reason,
                    "correct_abstain_without_evidence": correct_abstain_without_evidence,
                    "should_continue": stop_reason == "continue",
                    "ablation_audit": ablation_audit,
                }
                break

            if accepted_record is None:
                extraction_failure_reason = (
                    "incomplete_chunk_coverage"
                    if terminal_ablation_audit is not None
                    else "evidence_extraction_failed_after_retries"
                )
                iterative_records.append(
                    {
                        "round": round_id,
                        "attempts": len(invalid_attempts),
                        "answer": "",
                        "used_evidence": [],
                        "validated_used_evidence": [],
                        "newly_removed_evidence": [],
                        "used_evidence_alignment_check": {
                            "result": "fail",
                            "reason": extraction_failure_reason,
                        },
                        "invalid_attempts": invalid_attempts,
                        "cumulative_removed_evidence": list(cumulative_removed_evidence),
                        "all_evidence": list(cumulative_removed_evidence),
                        "result": "invalid_evidence",
                        "stop_reason": extraction_failure_reason,
                        "should_continue": False,
                        "should_filter": False,
                        "needs_rerun": True,
                        "ablation_audit": terminal_ablation_audit or {},
                        "failed_chunks": (
                            terminal_ablation_audit or {}
                        ).get("failed_chunks", []),
                    }
                )
                break

            iterative_records.append(accepted_record)

            if accepted_record.get("result") != "right":
                break
            if accepted_record.get("stop_reason") != "continue":
                break

            for evidence in accepted_record.get("newly_removed_evidence", []):
                key = (
                    str(evidence.get("dia_id", "")).strip(),
                    str(evidence.get("utterance", "")).strip(),
                )
                seen_evidence_keys.add(key)
            cumulative_removed_evidence.extend(accepted_record.get("newly_removed_evidence", []))

        return iterative_records

    def _summarize_iterative_ablation(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总消融检测结果，供后续过滤和人工复核使用。"""
        if not isinstance(records, list) or not records:
            return {
                "result": "passed",
                "passed": True,
                "should_filter": False,
                "reason": "no_ablation_records",
                "rounds": 0,
            }

        for record in records:
            if isinstance(record, dict) and (
                record.get("needs_rerun") is True
                or record.get("result") == "invalid_evidence"
            ):
                return {
                    "result": "needs_rerun",
                    "passed": False,
                    "should_filter": False,
                    "needs_rerun": True,
                    "reason": record.get("stop_reason", "evidence_extraction_failed_after_retries"),
                    "rounds": len(records),
                    "stop_round": record.get("round"),
                }
            if isinstance(record, dict) and record.get("should_filter") is True:
                return {
                    "result": "failed",
                    "passed": False,
                    "should_filter": True,
                    "reason": record.get("stop_reason", "iterative_ablation_failed"),
                    "rounds": len(records),
                    "stop_round": record.get("round"),
                }
            if isinstance(record, dict) and record.get("result") == "wrong":
                return {
                    "result": "passed",
                    "passed": True,
                    "should_filter": False,
                    "reason": "model_answered_wrong_after_evidence_removed",
                    "rounds": len(records),
                    "stop_round": record.get("round"),
                }
            if (
                isinstance(record, dict)
                and record.get("stop_reason") == "correct_abstain_without_remaining_evidence"
            ):
                return {
                    "result": "passed",
                    "passed": True,
                    "should_filter": False,
                    "reason": "correct_abstain_without_remaining_evidence",
                    "rounds": len(records),
                    "stop_round": record.get("round"),
                }

        last_record = records[-1] if isinstance(records[-1], dict) else {}
        last_round = last_record.get("round")
        last_result = last_record.get("result")
        if last_round == ITERATIVE_ABLATION_MAX_ROUNDS and last_result == "right":
            return {
                "result": "failed",
                "passed": False,
                "should_filter": True,
                "reason": last_record.get("stop_reason", "max_rounds_reached_while_still_right"),
                "rounds": len(records),
                "stop_round": last_round,
            }

        return {
            "result": "passed",
            "passed": True,
            "should_filter": False,
            "reason": last_record.get("stop_reason", "model_kept_answering_correctly"),
            "rounds": len(records),
            "stop_round": last_round,
        }

    def _process_single_question(
        self,
        conversation_item: Dict[str, Any],
        question_item: Dict[str, Any],
        record_idx: int,
        qa_idx: int,
        pbar,
        only_evidence: int,
        except_evidence: int,
    ) -> Dict[str, Any]:
        question = question_item.get("question", "")
        answer = question_item.get("answer", "")
        answer_candidates = normalize_answer_candidates(question_item.get("answer_fixed"), answer)
        question_type = normalize_question_type(question_item.get("question_type"))
        allow_cannot_infer = self._answer_allows_cannot_infer(answer_candidates)
        conversation = conversation_item.get("conversation", {})

        aligned_dialogue_evidence, alignment_report = self._get_aligned_dialogue_evidence(
            question_item,
            conversation,
        )
        evidence_blocks = self._build_only_evidence_blocks(question_item, aligned_dialogue_evidence)

        if alignment_report.get("result") != "pass":
            result = copy.deepcopy(question_item)
            result["evidence_alignment_check"] = alignment_report
            if only_evidence == 1:
                result["only_evidence_check"] = {
                    "result": "maybe_wrong",
                    "response": "",
                    "answer_prompt": "",
                    "response_time": 0.0,
                    "max_context_exceeded": 0,
                    "prediction_malformed": True,
                    "predicted_options": [],
                    "ground_truth_options": answer_candidates,
                    "error": "evidence_alignment_failed",
                }
            elif except_evidence == 1:
                result["iterative_evidence_ablation"] = []
                result["iterative_evidence_ablation_summary"] = {
                    "result": "failed",
                    "passed": False,
                    "reason": "evidence_alignment_failed",
                    "rounds": 0,
                }
            else:
                result["fullcontext_check"] = {
                    "result": "maybe_wrong",
                    "error": "evidence_alignment_failed",
                }

            return result

        if allow_cannot_infer and (only_evidence == 1 or except_evidence == 1):
            result = copy.deepcopy(question_item)
            result["evidence_alignment_check"] = alignment_report
            result["evidence_dialogues"] = aligned_dialogue_evidence

            if only_evidence == 1:
                result["only_evidence_check"] = {
                    "result": "skipped_abstain",
                    "passed": True,
                    "skipped": True,
                    "reason": "answer_is_f_no_falsification_stage",
                    "response": "",
                    "answer_prompt": "",
                    "response_time": 0.0,
                    "max_context_exceeded": 0,
                    "prediction_malformed": False,
                    "predicted_options": [],
                    "ground_truth_options": answer_candidates,
                }
            else:
                result["iterative_evidence_ablation"] = []
                result["iterative_evidence_ablation_summary"] = {
                    "result": "passed",
                    "passed": True,
                    "should_filter": False,
                    "skipped": True,
                    "reason": "answer_is_f_no_falsification_stage",
                    "rounds": 0,
                }

            return result

        data: Dict[str, Any] = {}
        if except_evidence == 1:
            result = copy.deepcopy(question_item)
            result["evidence_alignment_check"] = alignment_report
            result["evidence_dialogues"] = aligned_dialogue_evidence
            ablation_records = self._run_iterative_ablation(
                conversation_item=conversation,
                answer=answer,
                question=question,
                base_evidence_blocks=aligned_dialogue_evidence,
                question_type=question_type,
            )
            ablation_summary = self._summarize_iterative_ablation(ablation_records)
            result["iterative_evidence_ablation"] = ablation_records
            result["iterative_evidence_ablation_summary"] = ablation_summary

            if (
                ablation_summary.get("needs_rerun") is True
                or ablation_summary.get("result") == "needs_rerun"
            ):
                last_record = (
                    ablation_records[-1]
                    if ablation_records and isinstance(ablation_records[-1], dict)
                    else {}
                )
                self.logger.error(
                    "EVENT | iterative_ablation | status=needs_rerun | "
                    "record=%d | qa=%d | question_type=%s | reason=%s | "
                    "stop_round=%s | failed_chunks=%s",
                    record_idx,
                    qa_idx,
                    question_type,
                    ablation_summary.get("reason", "unknown"),
                    ablation_summary.get("stop_round", ""),
                    json.dumps(
                        last_record.get("failed_chunks", []),
                        ensure_ascii=False,
                    ),
                )

            return result

        score_result = None
        if only_evidence == 1:
            for parse_attempt in range(1, 4):
                response, response_time, answer_prompt, max_context_flag = self._request_text_answer(
                    conversation_item=conversation,
                    question=question,
                    evidence_blocks=evidence_blocks,
                    only_evidence=only_evidence,
                    except_evidence=except_evidence,
                    allow_cannot_infer=allow_cannot_infer,
                    question_type=question_type,
                )
                score_result = score_mcq_prediction(response, answer_candidates, question_type)
                parse_failed = (
                    score_result.get("prediction_malformed", False)
                    or not score_result.get("predicted_options")
                )
                if not parse_failed:
                    break
                self.logger.warning(
                    "EVENT | evidence_answer_parse | status=retry | attempt=%d/3",
                    parse_attempt,
                )
            if (
                score_result.get("prediction_malformed", False)
                or not score_result.get("predicted_options")
            ):
                raise EvidenceProcessingError(
                    "v2a answer parsing failed after 3 attempts"
                )
        else:
            data, response, response_time, answer_prompt, max_context_flag = self._request_json_answer(
                conversation_item=conversation,
                question=question,
                evidence_blocks=aligned_dialogue_evidence,
                only_evidence=only_evidence,
                except_evidence=except_evidence,
                allow_cannot_infer=allow_cannot_infer,
                max_json_retries=10,
                question_type=question_type,
            )

        if score_result is None:
            score_result = score_mcq_prediction(response, answer_candidates, question_type)
        check_result = "right" if score_result.get("is_correct", False) else "maybe_wrong"

        result = copy.deepcopy(question_item)
        context_data = {
            "result": check_result,
            "response": response,
            "answer_prompt": answer_prompt,
            "response_time": response_time,
            "max_context_exceeded": max_context_flag,
            "prediction_malformed": score_result.get("prediction_malformed", False),
            "predicted_options": score_result.get("predicted_options", []),
            "ground_truth_options": score_result.get("ground_truth_options", []),
        }
        result["evidence_alignment_check"] = alignment_report
        result["evidence_dialogues"] = aligned_dialogue_evidence

        if only_evidence == 1:
            result["only_evidence_check"] = context_data
        else:
            result["fullcontext_check"] = context_data

        return result

    @staticmethod
    def _question_identity(question_item: Dict[str, Any]) -> Dict[str, Any]:
        """Return fields that must remain stable between source and checkpoint."""
        identity_fields = (
            "id",
            "character",
            "category",
            "question_type",
            "question",
            "option",
            "answer",
            "answer_fixed",
            "label",
            "reasoning_steps",
        )
        return {
            key: copy.deepcopy(question_item.get(key))
            for key in identity_fields
            if key in question_item
        }

    def _validate_checkpoint_snapshot(
        self,
        source_data: List[Dict[str, Any]],
        snapshot_data: List[Dict[str, Any]],
    ) -> None:
        if len(source_data) != len(snapshot_data):
            raise ValueError(
                "Step 2 检查点记录数量不匹配: "
                f"source={len(source_data)}, checkpoint={len(snapshot_data)}"
            )

        for record_idx, (source_record, snapshot_record) in enumerate(
            zip(source_data, snapshot_data)
        ):
            source_qa = source_record.get("qa", [])
            snapshot_qa = snapshot_record.get("qa", [])
            if len(source_qa) != len(snapshot_qa):
                raise ValueError(
                    "Step 2 检查点题目数量不匹配: "
                    f"record={record_idx}, source={len(source_qa)}, "
                    f"checkpoint={len(snapshot_qa)}"
                )
            for qa_idx, (source_question, snapshot_question) in enumerate(
                zip(source_qa, snapshot_qa)
            ):
                if self._question_identity(source_question) != self._question_identity(
                    snapshot_question
                ):
                    raise ValueError(
                        "Step 2 检查点题目不匹配: "
                        f"record={record_idx}, qa={qa_idx}"
                    )

    @staticmethod
    def _is_question_terminal(
        question_item: Dict[str, Any],
        only_evidence: int,
        except_evidence: int,
    ) -> bool:
        if only_evidence == 1:
            result = question_item.get("only_evidence_check")
            return isinstance(result, dict) and "result" in result
        if except_evidence == 1:
            only_check = question_item.get("only_evidence_check")
            if (
                isinstance(only_check, dict)
                and "result" in only_check
                and only_check.get("result") != "right"
            ):
                return True
            summary = question_item.get("iterative_evidence_ablation_summary")
            return (
                isinstance(summary, dict)
                and "result" in summary
                and summary.get("result") != "needs_rerun"
                and summary.get("needs_rerun") is not True
            )
        result = question_item.get("fullcontext_check")
        return isinstance(result, dict) and "result" in result

    def _build_checkpoint_snapshot(
        self,
        snapshot_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        final_results = copy.deepcopy(snapshot_records)
        for record_idx, record in enumerate(final_results):
            record["qa"] = copy.deepcopy(self.results[record_idx])
        return final_results

    def process_data_file(
        self,
        file_path: str,
        only_evidence: int,
        except_evidence: int,
        max_workers: int = 10,
        checkpoint_every_questions: int = 1,
    ) -> int:
        """处理数据文件，逐批原子保存，并从已有输出跳过已完成题目。"""
        raw_data = load_json_file(file_path)
        data = normalize_dataset_records(raw_data)
        for item in data:
            for question_item in item.get("qa", []):
                if isinstance(question_item, dict):
                    question_item["question_type"] = normalize_question_type(
                        question_item.get("question_type")
                    )
        self.original_data = data
        max_workers = max(1, int(max_workers or 1))
        checkpoint_every_questions = max(1, int(checkpoint_every_questions or 1))

        if os.path.exists(self.output_path):
            snapshot_data = normalize_dataset_records(load_json_file(self.output_path))
            for item in snapshot_data:
                for question_item in item.get("qa", []):
                    if isinstance(question_item, dict):
                        question_item["question_type"] = normalize_question_type(
                            question_item.get("question_type")
                        )
            self._validate_checkpoint_snapshot(data, snapshot_data)
        else:
            snapshot_data = copy.deepcopy(data)
            write_json_file_atomic(snapshot_data, self.output_path, indent=4)

        self.results = [
            copy.deepcopy(item.get("qa", []))
            for item in snapshot_data
        ]

        total_questions = count_qa_items(data)
        if total_questions == 0:
            self.failed_questions = 0
            log_event("evidence_process", status="skipped", reason="no_questions")
            write_json_file_atomic(data, self.output_path, indent=4)
            return 0

        for question_type, total in count_question_types(data).items():
            log_event(
                "evidence_question_type",
                status="input",
                question_type=question_type,
                total=total,
            )

        alignment_total = 0
        alignment_pass = 0
        alignment_failed = 0
        for item in data:
            conversation = item.get("conversation", {})
            for question_item in item.get("qa", []):
                alignment_total += 1
                _, alignment_report = self._get_aligned_dialogue_evidence(question_item, conversation)
                if alignment_report.get("result") == "pass":
                    alignment_pass += 1
                else:
                    alignment_failed += 1
        log_event(
            "evidence_alignment_precheck",
            status="success" if alignment_failed == 0 else "warning",
            total=alignment_total,
            passed=alignment_pass,
            failed=alignment_failed,
        )
        if alignment_failed:
            log_event(
                "evidence_alignment_precheck",
                status="skipped_before_llm",
                failed_questions=alignment_failed,
                reason="evidence_alignment_failed",
            )

        pending_positions = []
        completed_questions = 0
        for record_idx, item in enumerate(data):
            for qa_idx, question_item in enumerate(item.get("qa", [])):
                checkpoint_question = self.results[record_idx][qa_idx]
                if self._is_question_terminal(
                    checkpoint_question,
                    only_evidence,
                    except_evidence,
                ):
                    completed_questions += 1
                else:
                    pending_positions.append((record_idx, qa_idx, item, question_item))

        pending_questions = len(pending_positions)
        if except_evidence == 1:
            log_event(
                "evidence_process",
                status="start",
                total_questions=total_questions,
                pending_questions=pending_questions,
                max_workers=max_workers,
                mode="iterative_ablation",
            )
        else:
            log_event(
                "evidence_process",
                status="start",
                total_questions=total_questions,
                pending_questions=pending_questions,
                max_workers=max_workers,
                mode="only_evidence" if only_evidence == 1 else "full_context",
            )

        dirty_questions = 0
        failed_questions = 0

        def flush_checkpoint(force: bool = False) -> None:
            nonlocal dirty_questions
            with self.lock:
                if dirty_questions == 0 or (
                    not force and dirty_questions < checkpoint_every_questions
                ):
                    return
                checkpoint = self._build_checkpoint_snapshot(snapshot_data)
                flushed = dirty_questions
                dirty_questions = 0
            write_json_file_atomic(checkpoint, self.output_path, indent=4)
            log_event(
                "evidence_checkpoint",
                status="saved",
                phase="v2b" if except_evidence == 1 else "v2a",
                completed_in_batch=flushed,
                output=self.output_path,
            )

        with tqdm(total=total_questions, desc="Step 2 evidence check", unit="question") as pbar:
            if completed_questions:
                pbar.update(completed_questions)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_single_question,
                        item,
                        question_item,
                        record_idx,
                        qa_idx,
                        pbar,
                        only_evidence,
                        except_evidence,
                    ): (record_idx, qa_idx)
                    for record_idx, qa_idx, item, question_item in pending_positions
                }

                for future in as_completed(futures):
                    record_idx, qa_idx = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        failed_questions += 1
                        self.logger.exception("EVENT | evidence_task | status=failed | error=%s", exc)
                    else:
                        needs_rerun = False
                        if except_evidence == 1 and isinstance(result, dict):
                            summary = result.get(
                                "iterative_evidence_ablation_summary",
                                {},
                            )
                            needs_rerun = isinstance(summary, dict) and (
                                summary.get("needs_rerun") is True
                                or summary.get("result") == "needs_rerun"
                            )
                        if needs_rerun:
                            failed_questions += 1
                        with self.lock:
                            self.results[record_idx][qa_idx] = result
                            dirty_questions += 1
                        flush_checkpoint()
                    finally:
                        pbar.update(1)

        flush_checkpoint(force=True)

        log_event("evidence_process", status="saving", output=self.output_path)
        final_results = self._build_checkpoint_snapshot(snapshot_data)
        write_json_file_atomic(final_results, self.output_path, indent=4)
        log_event(
            "evidence_resume_summary",
            status="warning" if failed_questions else "success",
            completed_before_run=completed_questions,
            attempted=pending_questions,
            failed=failed_questions,
        )
        self.failed_questions = failed_questions

        type_stats = {
            question_type: {"processed": 0, "passed": 0, "failed": 0}
            for question_type in (SINGLE_CHOICE, MULTIPLE_SELECT, ORDERING)
        }
        for item in final_results:
            for question_item in item.get("qa", []):
                question_type = normalize_question_type(question_item.get("question_type"))
                stats = type_stats[question_type]
                stats["processed"] += 1
                if except_evidence == 1:
                    passed = bool(
                        (question_item.get("iterative_evidence_ablation_summary") or {}).get("passed")
                    )
                elif only_evidence == 1:
                    passed = (question_item.get("only_evidence_check") or {}).get("result") in {
                        "right",
                        "skipped_abstain",
                    }
                else:
                    passed = (question_item.get("fullcontext_check") or {}).get("result") == "right"
                stats["passed" if passed else "failed"] += 1
        for question_type, stats in type_stats.items():
            log_event(
                "evidence_question_type",
                status="output",
                question_type=question_type,
                **stats,
            )

        return sum(len(item.get("qa", [])) for item in final_results)


def evidence_check_main(
    args,
    input_file_path: str,
    output_file_path: str,
    only_evidence: int = 0,
    except_evidence: int = 0,
) -> tuple:
    """
    步骤 2: 题目合理性检测。

    默认模式会执行两阶段检测：only_evidence -> iterative ablation。
    """
    default_mode = only_evidence == 0 and except_evidence == 0
    if default_mode:
        mode_desc = "两阶段筛选（only_evidence -> iterative ablation）"
    elif only_evidence == 1:
        mode_desc = "只使用证据"
    else:
        mode_desc = "排除证据"

    print_log_section(f"STEP 2 | EVIDENCE VALIDITY CHECK | {mode_desc}")
    print_kv("input", input_file_path, indent=2)

    answer_llm_config = build_provider_config(
        getattr(args, "answer_llm_model", None),
        getattr(args, "answer_llm_base_url", None),
        getattr(args, "answer_llm_api_key", None),
    )
    ablation_config = normalize_ablation_config(
        {
            "context_limit": getattr(
                args,
                "ablation_context_limit",
                DEFAULT_ABLATION_CONFIG["context_limit"],
            ),
            "prompt_safety_tokens": getattr(
                args,
                "ablation_prompt_safety_tokens",
                DEFAULT_ABLATION_CONFIG["prompt_safety_tokens"],
            ),
            "chunk_tokens": getattr(
                args,
                "ablation_chunk_tokens",
                DEFAULT_ABLATION_CONFIG["chunk_tokens"],
            ),
            "retrieval_chunks": getattr(
                args,
                "ablation_retrieval_chunks",
                DEFAULT_ABLATION_CONFIG["retrieval_chunks"],
            ),
            "chunk_max_workers": getattr(
                args,
                "ablation_chunk_max_workers",
                DEFAULT_ABLATION_CONFIG["chunk_max_workers"],
            ),
        }
    )

    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    checkpoint_every_questions = max(
        1,
        int(getattr(args, "checkpoint_every_questions", 1) or 1),
    )

    if default_mode:
        evidence_only_path, evidence_ablation_path = derive_evidence_stage_paths(output_file_path)
        only_evidence_workers = max(
            1,
            int(getattr(args, "only_evidence_max_workers", args.max_workers) or 1),
        )
        iterative_ablation_workers = max(
            1,
            int(getattr(args, "iterative_ablation_max_workers", args.max_workers) or 1),
        )

        log_subsection("Phase 1 | only evidence")
        manager_evidence_only = FullContextManager(
            output_path=evidence_only_path,
            llm_config=answer_llm_config,
            figure_view=False,
        )
        kept_count_evidence_only = manager_evidence_only.process_data_file(
            file_path=input_file_path,
            only_evidence=1,
            except_evidence=0,
            max_workers=only_evidence_workers,
            checkpoint_every_questions=checkpoint_every_questions,
        )
        if manager_evidence_only.failed_questions:
            raise EvidenceProcessingError(
                "v2a has retryable failed questions; resume this run to continue"
            )

        log_subsection("Phase 2 | iterative evidence ablation")
        manager_evidence_ablation = FullContextManager(
            output_path=evidence_ablation_path,
            llm_config=answer_llm_config,
            figure_view=False,
            ablation_config=ablation_config,
        )
        kept_count_evidence_ablation = manager_evidence_ablation.process_data_file(
            file_path=evidence_only_path,
            only_evidence=0,
            except_evidence=1,
            max_workers=iterative_ablation_workers,
            checkpoint_every_questions=checkpoint_every_questions,
        )
        if manager_evidence_ablation.failed_questions:
            raise EvidenceProcessingError(
                "v2b has retryable failed questions; resume this run to continue"
            )

        log_event(
            "evidence_check_summary",
            status="success",
            only_evidence=kept_count_evidence_only,
            iterative_ablation=kept_count_evidence_ablation,
        )
        return evidence_ablation_path, kept_count_evidence_ablation

    manager = FullContextManager(
        output_path=output_file_path,
        llm_config=answer_llm_config,
        figure_view=False,
        ablation_config=ablation_config if except_evidence == 1 else None,
    )

    if only_evidence == 1:
        mode_workers = getattr(args, "only_evidence_max_workers", args.max_workers)
    elif except_evidence == 1:
        mode_workers = getattr(args, "iterative_ablation_max_workers", args.max_workers)
    else:
        mode_workers = args.max_workers

    kept_count = manager.process_data_file(
        file_path=input_file_path,
        only_evidence=only_evidence,
        except_evidence=except_evidence,
        max_workers=mode_workers,
        checkpoint_every_questions=checkpoint_every_questions,
    )
    if manager.failed_questions:
        phase = "v2a" if only_evidence == 1 else "v2b" if except_evidence == 1 else "full_context"
        raise EvidenceProcessingError(
            f"{phase} has retryable failed questions; resume this run to continue"
        )
    return output_file_path, kept_count
