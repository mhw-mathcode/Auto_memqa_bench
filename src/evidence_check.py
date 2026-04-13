import copy
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

from src.mcq_scoring import normalize_answer_candidates, score_mcq_prediction
from src.utils import normalize_dataset_records


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
You are an information extraction assistant.

Your task is NOT to explain or summarize.
Your task is to EXTRACT all dialogue turns that are USED to answer the question.

======================
CRITICAL RULES (MUST FOLLOW)
======================

1. Evidence MUST be copied word-for-word from the conversation history.
   - Copy-Paste only.
   - Do NOT paraphrase, shorten, merge, or reformat.
   - Even minor edits (punctuation, tense, spacing) are forbidden.

2. Each evidence item MUST correspond to exactly ONE dialogue turn in the history.

3. For each evidence item, you MUST:
   - Provide the exact "utterance" text.
   - Provide the corresponding "dia_id" from the SAME dialogue turn.
   - Ensure the utterance and dia_id come from the SAME original message.

4. Do NOT invent dia_id.
   - If you cannot find an exact dia_id for an utterance, DO NOT include that utterance. Also, do not rely on this utterance to answer the questions.

5. Include ALL dialogue turns that are necessary to answer the question.
   - If multiple turns are required, include all of them.
   - If only one turn is sufficient, include only that one.

6. Do NOT include irrelevant dialogue turns.
   - Only include evidence that is directly used to determine the answer.

======================
OUTPUT FORMAT (JSON ONLY)
======================

Return ONLY the following JSON structure.
Do NOT include explanations, comments, or extra fields.

{
    "question": "[Direct, natural, focused on the character]",
    "answer": "(A)",
    "evidence_dialogues": [
        {
            "id": "E1",
            "speaker": "<speaker_name>",
            "utterance": "<exact copied utterance>",
            "dia_id": "<exact dia_id from the conversation>"
        }
    ]
}

======================
CONVERSATION HISTORY
======================
{{conversation_history}}

======================
QUESTION
======================
{{question}}
"""


ANSWER_PROMPT_ONLY_EVIDENCE = """
You are a rigorous intelligent assistant. Your task is to answer questions by synthesizing provided evidence and pre-defined reasoning logic.

# CONTEXT:
You will be provided with two types of information:

Evidence Fragments: Raw data or facts extracted from materials.

Reasoning Steps: Specific logical paths or intermediate deductions that must be followed.

# INSTRUCTIONS:

STRICT SCOPE: Your answer must be derived exclusively from the "EVIDENCE" and "REASONING STEPS" sections below. Do not use external knowledge or introduce original reasoning that contradicts or exceeds the provided steps.

NO AMBIGUITY: {{cannot_infer_instruction}}

THOUGHT PROCESS: Before providing the final answer, perform a "Internal Chain of Thought" to verify that every part of your conclusion is anchored in either a piece of evidence or a provided reasoning step. Provide your reasoning steps, and then answer this question.

--- REFERENCE MATERIAL ---
[EVIDENCE]
{{evidence}}
--- END OF MATERIAL ---

Question: {{question}}
"""


DEFAULT_LLM_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"


def derive_v1_stage_paths(output_file_path: str) -> Tuple[str, str]:
    """根据输出路径推导 v1a / v1b 路径。"""
    base, ext = os.path.splitext(output_file_path)

    if base.endswith("_v1a"):
        prefix = base[:-4]
    elif base.endswith("_v1b"):
        prefix = base[:-4]
    elif base.endswith("_v1"):
        prefix = base[:-3]
    else:
        prefix = base

    return f"{prefix}_v1a{ext}", f"{prefix}_v1b{ext}"


class FullContextManager:
    """步骤 1 题目合理性检测管理器。"""

    def __init__(self, output_path: str, logger=None, figure_view: bool = False, llm_config=None):
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

        self.results: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.lock = threading.Lock()
        self.original_data: List[Dict[str, Any]] = []

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
        """将 conversation 字典格式化为可读文本。"""
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
            timestamp = conversation_item.get(f"{session_key}_date_time", "Unknown time")
            history.append(f"\n--- Turn started at {timestamp} ---")

            chats = conversation_item.get(session_key, [])
            if not isinstance(chats, list):
                continue

            for chat in chats:
                if not isinstance(chat, dict):
                    continue
                if "speaker" not in chat or "text" not in chat:
                    continue

                speaker_name = self._resolve_speaker_name(chat.get("speaker"), speaker_map)
                text_content = chat.get("text", "")

                if self.figure_view and "img_url" in chat and "blip_caption" in chat:
                    text_content += f" [Image: {chat.get('img_url')}] with caption: {chat.get('blip_caption')}"

                history.append(f"{speaker_name}: {text_content}")

        return "\n".join(history).strip()

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

                chat_text_norm = _normalize_match_text(chat.get("text", ""))
                chat_text_raw = str(chat.get("text", "")).strip()
                chat_dia_id = _normalize_dia_id(chat.get("dia_id"))
                match_by_dia = chat_dia_id in evidence_dia_ids
                match_by_text = False

                # 若按 dia_id 删除，也同步扣减对应证据剩余文本，避免“已删对话但残留仍显示完整”
                if match_by_dia and chat_text_norm and chat_dia_id:
                    for target in evidence_targets:
                        if target.get("dia_id") != chat_dia_id:
                            continue
                        before_remaining = str(target.get("remaining_norm", ""))
                        after_remaining = _subtract_fragment(before_remaining, chat_text_norm)
                        if after_remaining != before_remaining:
                            target["remaining_norm"] = after_remaining
                            if chat_text_raw:
                                target["matched_fragments"].append(chat_text_raw)

                if chat_text_norm:
                    for target in evidence_targets:
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
                "以下证据文本未能完全从对话中删除，残留片段如下（共 %d 条）:\n%s",
                len(remaining_evidence_fragments),
                json.dumps(remaining_evidence_fragments, ensure_ascii=False, indent=2),
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
    ) -> str:
        if only_evidence == 1:
            template = Template(ANSWER_PROMPT_ONLY_EVIDENCE)
            evidence_text = json.dumps(evidence_blocks, ensure_ascii=False)
            cannot_infer_instruction = (
                "You must provide a definitive answer. Option F (Cannot infer the answer based on the given information) is allowed only when the provided evidence truly cannot support any non-F option."
                if allow_cannot_infer
                else "You must provide a definitive answer and you are forbidden to choose option F (Cannot infer the answer based on the given information)."
            )
            return template.render(
                {
                    "evidence": evidence_text,
                    "question": question,
                    "cannot_infer_instruction": cannot_infer_instruction,
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
            }
        )

    def _call_llm(self, prompt: str, max_retries: int = 5) -> Tuple[str, float, int]:
        """调用 LLM，返回 (response, response_time, max_context_exceeded)。"""
        request_id = f"evidence-check-{uuid.uuid4()}"
        start_time = time.time()
        max_context_exceeded = 0
        last_error: Optional[Exception] = None

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
                if "maximum context length" in error_text.lower():
                    max_context_exceeded = 1

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

                if attempt >= max_retries:
                    break

                sleep_seconds = random.uniform(45, 75)
                self.logger.warning(
                    "LLM call failed (%s), retry %d/%d in %.2fs, request_id=%s",
                    error_text,
                    attempt,
                    max_retries,
                    sleep_seconds,
                    request_id,
                )
                time.sleep(sleep_seconds)

        self.logger.error(
            "LLM call failed permanently after %d attempts, request_id=%s, error=%s",
            max_retries,
            request_id,
            last_error,
        )
        return "Error: Failed to get response from LLM.", time.time() - start_time, max_context_exceeded

    def _request_json_answer(
        self,
        conversation_item: Dict[str, Any],
        question: str,
        evidence_blocks: List[Dict[str, Any]],
        only_evidence: int,
        except_evidence: int,
        allow_cannot_infer: bool = False,
        max_json_retries: int = 10,
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
            )

            response, response_time, context_flag = self._call_llm(prompt)
            total_response_time += response_time
            max_context_exceeded = max(max_context_exceeded, context_flag)
            last_response = response
            last_prompt = prompt

            try:
                parsed = json.loads(clean_json_response(response))
                if isinstance(parsed, dict):
                    return parsed, response, total_response_time, prompt, max_context_exceeded
            except json.JSONDecodeError as exc:
                self.logger.warning(
                    "JSON parse failed (%d/%d): %s",
                    retry_idx + 1,
                    max_json_retries,
                    exc,
                )

        self.logger.error("JSON parse failed after %d retries", max_json_retries)
        return {}, last_response, total_response_time, last_prompt, max_context_exceeded

    def _request_text_answer(
        self,
        conversation_item: Dict[str, Any],
        question: str,
        evidence_blocks: List[Dict[str, Any]],
        only_evidence: int,
        except_evidence: int,
        allow_cannot_infer: bool = False,
    ) -> Tuple[str, float, str, int]:
        """获取自由格式文本响应，不做 JSON 解析。"""
        prompt = self._build_prompt(
            conversation_item=conversation_item,
            question=question,
            evidence_blocks=evidence_blocks,
            only_evidence=only_evidence,
            except_evidence=except_evidence,
            allow_cannot_infer=allow_cannot_infer,
        )
        response, response_time, max_context_exceeded = self._call_llm(prompt)
        return response, response_time, prompt, max_context_exceeded

    def _answer_allows_cannot_infer(self, answer_candidates: List[str]) -> bool:
        """判断标准答案是否允许选择 F。"""
        for candidate in answer_candidates:
            text = str(candidate or "").strip()
            if re.match(r"^[Ff](?:[\s\)\]\.:,，、\-]|$)", text):
                return True
        return False

    def _normalize_evidence_field(self, value: Any, field_name: str) -> List[Dict[str, Any]]:
        """将任意格式的证据字段统一转换为列表，避免非 list 数据被丢弃。"""
        normalized: List[Dict[str, Any]] = []

        if value in (None, ""):
            return normalized

        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    normalized.append(item)
                elif item not in (None, ""):
                    normalized.append({"source_field": field_name, "content": item})
            return normalized

        if isinstance(value, dict):
            return [value]

        return [{"source_field": field_name, "content": value}]

    def _get_evidence_blocks(
        self,
        question_item: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """统一返回传给模型的证据块: evidence_dialogues + reasoning_steps。"""
        evidence_dialogues = question_item.get("evidence_dialogues", [])
        reasoning_steps = question_item.get("reasoning_steps", [])

        dialogues_list = self._normalize_evidence_field(evidence_dialogues, "evidence_dialogues")
        reasoning_list = self._normalize_evidence_field(reasoning_steps, "reasoning_steps")

        return dialogues_list + reasoning_list

    def _run_iterative_ablation(
        self,
        conversation_item: Dict[str, Any],
        answer: str,
        question: str,
        base_evidence_blocks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """兼容保留：排除证据模式下的迭代记录。"""
        iterative_records: List[Dict[str, Any]] = []
        remaining_evidence = list(base_evidence_blocks)
        answer_candidates = normalize_answer_candidates(None, answer)

        for round_id in range(1, 4):
            data, _, _, _, _ = self._request_json_answer(
                conversation_item=conversation_item,
                question=question,
                evidence_blocks=remaining_evidence,
                only_evidence=0,
                except_evidence=1,
                max_json_retries=10,
            )

            if not data:
                break

            score_result = score_mcq_prediction(data.get("answer", ""), answer_candidates)
            is_right = bool(score_result.get("is_correct", False))
            used_evidence = data.get("evidence_dialogues", [])

            iterative_records.append(
                {
                    "round": round_id,
                    "answer": data.get("answer", ""),
                    "used_evidence": used_evidence,
                    "all_evidence": remaining_evidence,
                    "result": "right" if is_right else "wrong",
                }
            )

            if not is_right:
                break
            if not isinstance(used_evidence, list) or not used_evidence:
                break

            remaining_evidence.extend(used_evidence)

        return iterative_records

    def _process_single_question(
        self,
        conversation_item: Dict[str, Any],
        question_item: Dict[str, Any],
        idx: int,
        pbar,
        only_evidence: int,
        except_evidence: int,
    ) -> Dict[str, Any]:
        question = question_item.get("question", "")
        answer = question_item.get("answer", "")
        answer_candidates = normalize_answer_candidates(question_item.get("answer_fixed"), answer)
        allow_cannot_infer = self._answer_allows_cannot_infer(answer_candidates)

        evidence_blocks = self._get_evidence_blocks(question_item)

        data: Dict[str, Any] = {}
        if only_evidence == 1:
            response, response_time, answer_prompt, max_context_flag = self._request_text_answer(
                conversation_item=conversation_item.get("conversation", {}),
                question=question,
                evidence_blocks=evidence_blocks,
                only_evidence=only_evidence,
                except_evidence=except_evidence,
                allow_cannot_infer=allow_cannot_infer,
            )
        else:
            data, response, response_time, answer_prompt, max_context_flag = self._request_json_answer(
                conversation_item=conversation_item.get("conversation", {}),
                question=question,
                evidence_blocks=evidence_blocks,
                only_evidence=only_evidence,
                except_evidence=except_evidence,
                allow_cannot_infer=allow_cannot_infer,
                max_json_retries=10,
            )

        score_result = score_mcq_prediction(response, answer_candidates)
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

        if only_evidence == 1:
            result["only_evidence_check"] = context_data
        elif except_evidence == 1:
            result["iterative_evidence_ablation"] = self._run_iterative_ablation(
                conversation_item=conversation_item.get("conversation", {}),
                answer=answer,
                question=question,
                base_evidence_blocks=evidence_blocks,
            )
        else:
            result["fullcontext_check"] = context_data

        with self.lock:
            self.results[idx].append(result)

        pbar.update(1)
        return result

    def process_data_file(self, file_path: str, only_evidence: int, except_evidence: int, max_workers: int = 10) -> int:
        """处理数据文件并写出结果。"""
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        data = normalize_dataset_records(raw_data)
        self.original_data = data
        self.results = defaultdict(list)

        total_questions = sum(len(item.get("qa", [])) for item in data)
        if total_questions == 0:
            print("No questions found to process.")
            return 0

        if except_evidence == 1:
            pending_questions = 0
            for item in data:
                for question_item in item.get("qa", []):
                    only_check = question_item.get("only_evidence_check", {})
                    # v1b 逐题判定：仅当 v1a 明确给出非 right 结果时才跳过。
                    if not isinstance(only_check, dict) or "result" not in only_check:
                        pending_questions += 1
                    elif only_check.get("result") == "right":
                        pending_questions += 1
            print(
                f"--- Starting Full Context Evaluation: {total_questions} total questions, "
                f"{pending_questions} pending ---"
            )
        else:
            print(f"--- Starting Full Context Evaluation: {total_questions} total questions to process ---")

        with tqdm(total=total_questions, desc="💡 Full Context Progress") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                for idx, item in enumerate(data):
                    for question_item in item.get("qa", []):
                        if except_evidence == 1:
                            only_check = question_item.get("only_evidence_check", {})
                            if isinstance(only_check, dict) and "result" in only_check and only_check.get("result") != "right":
                                pbar.update(1)
                                with self.lock:
                                    self.results[idx].append(copy.deepcopy(question_item))
                                continue

                        future = executor.submit(
                            self._process_single_question,
                            item,
                            question_item,
                            idx,
                            pbar,
                            only_evidence,
                            except_evidence,
                        )
                        futures.append(future)

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        self.logger.exception("A task failed in the thread pool: %s", exc)

        print("\nAll threads finished. Saving final results to disk...")

        final_results: List[Dict[str, Any]] = []
        for idx, item in enumerate(self.original_data):
            result_item = copy.deepcopy(item)
            result_item["qa"] = self.results.get(idx, item.get("qa", []))
            final_results.append(result_item)

        with open(self.output_path, "w", encoding="utf-8") as file:
            json.dump(final_results, file, indent=4, ensure_ascii=False)

        return sum(len(item.get("qa", [])) for item in final_results)


def evidence_check_main(
    args,
    input_file_path: str,
    output_file_path: str,
    only_evidence: int = 0,
    except_evidence: int = 0,
) -> tuple:
    """
    步骤 1: 题目合理性检测。

    默认模式会执行两阶段检测：v1a(only_evidence) -> v1b(iterative ablation)。
    """
    default_mode = only_evidence == 0 and except_evidence == 0
    if default_mode:
        mode_desc = "两阶段筛选（v1a -> v1b）"
    elif only_evidence == 1:
        mode_desc = "只使用证据"
    else:
        mode_desc = "排除证据"

    print("\n" + "=" * 60)
    print(f"🔄 题目合理性检测 - {mode_desc}")
    print("=" * 60)
    print(f"📥 输入文件: {input_file_path}")

    answer_llm_config = build_provider_config(
        getattr(args, "answer_llm_model", None),
        getattr(args, "answer_llm_base_url", None),
        getattr(args, "answer_llm_api_key", None),
    )

    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if default_mode:
        v1a_path, v1b_path = derive_v1_stage_paths(output_file_path)

        print("\n📝 第一阶段：只使用证据 (v1a)")
        manager_v1a = FullContextManager(
            output_path=v1a_path,
            llm_config=answer_llm_config,
            figure_view=False,
        )
        kept_count_v1a = manager_v1a.process_data_file(
            file_path=input_file_path,
            only_evidence=1,
            except_evidence=0,
            max_workers=args.max_workers,
        )

        print("\n📝 第二阶段：迭代删除证据 (v1b)")
        manager_v1b = FullContextManager(
            output_path=v1b_path,
            llm_config=answer_llm_config,
            figure_view=False,
        )
        kept_count_v1b = manager_v1b.process_data_file(
            file_path=v1a_path,
            only_evidence=0,
            except_evidence=1,
            max_workers=args.max_workers,
        )

        print(f" 验证统计：v1a {kept_count_v1a} -> v1b {kept_count_v1b}")
        return v1b_path, kept_count_v1b

    manager = FullContextManager(
        output_path=output_file_path,
        llm_config=answer_llm_config,
        figure_view=False,
    )

    kept_count = manager.process_data_file(
        file_path=input_file_path,
        only_evidence=only_evidence,
        except_evidence=except_evidence,
        max_workers=args.max_workers,
    )
    return output_file_path, kept_count
