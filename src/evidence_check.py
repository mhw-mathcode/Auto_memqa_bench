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


def extract_option_letter(text: Any) -> Optional[str]:
    """从答案文本中提取 A-F 选项字母。"""
    if not isinstance(text, str):
        return None
    match = re.match(r"\s*([A-F])", text)
    return match.group(1) if match else None


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
    "answer": "C. the full option text content",
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
You are a rigorous intelligent assistant. Your task is to answer questions based solely on the provided evidence fragments (Evidence and reasoning steps).

# CONTEXT:
You will receive a set of evidence fragments extracted from the original materials. These fragments contain all the information necessary to answer the question.

# INSTRUCTIONS:
1. **EVIDENCE ONLY**: Your answer must be derived entirely from the text provided in the "EVIDENCE" section. The use of external knowledge, reasoning, or information not mentioned in the passage is strictly prohibited.
2. **Conflict of evidence**: If there is a conflict between different fragments, please refer to the fragment with the most complete logic or the most recent one.
3. **Cannot answer**: If the provided evidence is insufficient to answer the question, please indicate directly (or select a specific option according to the specific testing requirements).

# OUTPUT
You are required to answer in JSON format only.
Return the answer strictly in the following structure:
{
    "question": "[Direct, natural, focused on the character]",
    "answer": "Letter. Full option text",
}

--- EVIDENCE ---
{{evidence}}
--- END OF EVIDENCE ---

Question: {{question}}
"""


DEFAULT_LLM_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"


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
        evidence_dialogues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """从 conversation 中删除已知证据对应的对话。"""
        conv = copy.deepcopy(conversation_item)

        evidence_dia_ids = set()
        evidence_sessions_no_time = set()

        for evidence in evidence_dialogues or []:
            if not isinstance(evidence, dict):
                continue

            dia_id = evidence.get("dia_id")
            utterance = evidence.get("utterance", "")

            if dia_id and dia_id != "N/A":
                evidence_dia_ids.add(dia_id)

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
                if chat.get("dia_id") in evidence_dia_ids:
                    continue
                filtered_chats.append(chat)

            conv[key] = filtered_chats

        return conv

    def _build_prompt(
        self,
        conversation_item: Dict[str, Any],
        question: str,
        evidence_blocks: List[Dict[str, Any]],
        only_evidence: int,
        except_evidence: int,
        evidence_dialogues: List[Dict[str, Any]],
    ) -> str:
        if only_evidence == 1:
            template = Template(ANSWER_PROMPT_ONLY_EVIDENCE)
            evidence_text = json.dumps(evidence_blocks, ensure_ascii=False)
            return template.render({"evidence": evidence_text, "question": question})

        if except_evidence == 1:
            conversation_obj = self._conversation_without_evidence(conversation_item, evidence_dialogues)
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
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.0,
                    extra_body={"enable_thinking": False},
                )
                content = response.choices[0].message.content or ""
                return content, time.time() - start_time, max_context_exceeded

            except Exception as exc:  # noqa: PERF203
                last_error = exc
                error_text = str(exc)
                if "maximum context length" in error_text.lower():
                    max_context_exceeded = 1

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
        evidence_dialogues: List[Dict[str, Any]],
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
                evidence_dialogues=evidence_dialogues,
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

    def _get_evidence_blocks(
        self,
        question_item: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """合并 evidence_dialogues 和 reasoning_steps。"""
        evidence_dialogues = question_item.get("evidence_dialogues", [])
        reasoning_steps = question_item.get("reasoning_steps", [])

        dialogues_list = evidence_dialogues if isinstance(evidence_dialogues, list) else []
        reasoning_list = reasoning_steps if isinstance(reasoning_steps, list) else []

        return dialogues_list + reasoning_list, dialogues_list

    def _run_iterative_ablation(
        self,
        conversation_item: Dict[str, Any],
        answer: str,
        question: str,
        base_evidence_dialogues: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """兼容保留：排除证据模式下的迭代记录。"""
        iterative_records: List[Dict[str, Any]] = []
        remaining_evidence = list(base_evidence_dialogues)
        standard_option = extract_option_letter(answer)

        for round_id in range(1, 6):
            data, _, _, _, _ = self._request_json_answer(
                conversation_item=conversation_item,
                question=question,
                evidence_blocks=remaining_evidence,
                only_evidence=0,
                except_evidence=1,
                evidence_dialogues=remaining_evidence,
                max_json_retries=10,
            )

            if not data:
                break

            response_option = extract_option_letter(data.get("answer", ""))
            is_right = bool(standard_option and response_option == standard_option)
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
        answer_option = extract_option_letter(answer)

        evidence_blocks, evidence_dialogues = self._get_evidence_blocks(question_item)

        data, response, response_time, answer_prompt, max_context_flag = self._request_json_answer(
            conversation_item=conversation_item.get("conversation", {}),
            question=question,
            evidence_blocks=evidence_blocks,
            only_evidence=only_evidence,
            except_evidence=except_evidence,
            evidence_dialogues=evidence_dialogues,
            max_json_retries=10,
        )

        response_option = extract_option_letter(data.get("answer", "")) if data else extract_option_letter(response)
        check_result = "right" if (answer_option and response_option == answer_option) else "maybe_wrong"

        result = copy.deepcopy(question_item)
        context_data = {
            "result": check_result,
            "response": response,
            "answer_prompt": answer_prompt,
            "response_time": response_time,
            "max_context_exceeded": max_context_flag,
        }

        if only_evidence == 1:
            result["only_evidence_check"] = context_data
        elif except_evidence == 1:
            result["iterative_evidence_ablation"] = self._run_iterative_ablation(
                conversation_item=conversation_item.get("conversation", {}),
                answer=answer,
                question=question,
                base_evidence_dialogues=evidence_dialogues,
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
                    if isinstance(only_check, dict) and only_check.get("result") == "right":
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
                            if not isinstance(only_check, dict) or only_check.get("result") != "right":
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

    默认模式会执行 v1 的 only_evidence 检测。
    """
    default_mode = only_evidence == 0 and except_evidence == 0
    mode_desc = "只使用证据" if (only_evidence == 1 or default_mode) else "排除证据"

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

    manager = FullContextManager(
        output_path=output_file_path,
        llm_config=answer_llm_config,
        figure_view=False,
    )

    if default_mode:
        print("\n📝 执行合理性检测 (v1)")
        kept_count = manager.process_data_file(
            file_path=input_file_path,
            only_evidence=1,
            except_evidence=0,
            max_workers=args.max_workers,
        )
        return output_file_path, kept_count

    kept_count = manager.process_data_file(
        file_path=input_file_path,
        only_evidence=only_evidence,
        except_evidence=except_evidence,
        max_workers=args.max_workers,
    )
    return output_file_path, kept_count
