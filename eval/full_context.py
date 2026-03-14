import argparse
import json
import logging
import os
import random
import sys
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

from src.utils import normalize_dataset_records


ANSWER_PROMPT_FULL_CONTEXT = """
You are an intelligent assistant. Your task is to answer a multiple-choice question based only on the provided conversation history.

# CONTEXT
You have access to the complete conversation history between two speakers. This history contains all the information needed to answer the question.

# RULES
1. Use only the conversation history. Do not use outside knowledge.
2. Choose exactly one option.
3. Pay attention to timestamps and message order. Prefer the latest explicit information when conflicts exist.
4. If the answer cannot be determined from the history, choose the dedicated "Cannot infer" option if one exists.
5. Do not explain your reasoning.
6. Your final output must be only the option letter in parentheses, for example: (B)

--- CONVERSATION HISTORY ---
{{conversation_history}}
--- END OF HISTORY ---

Question: {{question}}

Options:
{{options_text}}

Final answer:
"""

DEFAULT_LLM_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"


def _coerce_numeric_key(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _sort_key(value: Any) -> Tuple[int, Any]:
    coerced = _coerce_numeric_key(value)
    if isinstance(coerced, int):
        return (0, coerced)
    return (1, str(coerced))


def resolve_output_path(input_path: Path, provided_path: Optional[str], suffix: str) -> Path:
    if provided_path:
        return Path(provided_path).resolve()

    output_dir = PROJECT_ROOT / "result"
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir / f"{input_path.stem}{suffix}").resolve()


def normalize_option_lines(options: Sequence[Any]) -> List[str]:
    normalized: List[str] = []
    for index, option in enumerate(options or []):
        letter = chr(ord("A") + index)
        option_text = str(option).strip()
        if option_text[:3].startswith(f"{letter}.") or (len(option_text) >= 3 and option_text[1:3] == ". "):
            normalized.append(option_text)
        else:
            normalized.append(f"{letter}. {option_text}")
    return normalized


def extract_question_stem(question: str, option_lines: Sequence[str]) -> str:
    question_text = str(question or "").strip()
    if not question_text:
        return ""

    for marker in (
        "Please provide the option corresponding to the only correct answer",
        "You need to select the correct answer from the following options:",
    ):
        if marker in question_text:
            question_text = question_text.split(marker, 1)[0].strip()

    if option_lines:
        first_option = option_lines[0]
        option_index = question_text.find(f"\n{first_option}")
        if option_index >= 0:
            question_text = question_text[:option_index].strip()

    return question_text


def serialize_results(result_map: Dict[int, Dict[int, Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    serialized: Dict[str, List[Dict[str, Any]]] = {}
    for conv_idx in sorted(result_map.keys(), key=_sort_key):
        question_map = result_map[conv_idx]
        serialized[str(conv_idx)] = [
            question_map[q_idx] for q_idx in sorted(question_map.keys(), key=_sort_key)
        ]
    return serialized


class FullContextRunner:
    def __init__(
        self,
        output_path: Path,
        logger: Optional[logging.Logger] = None,
        figure_view: bool = False,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        load_dotenv()
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
        self.output_path = output_path
        self.logger = logger if logger else logging.getLogger(__name__)
        self.figure_view = figure_view
        self.template = Template(ANSWER_PROMPT_FULL_CONTEXT)
        self.lock = threading.Lock()
        self.results: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)

    def _format_conversation(self, conversation_item: Dict[str, Any]) -> str:
        history: List[str] = []
        speaker_a = conversation_item.get("speaker_a", "speaker_a")
        speaker_b = conversation_item.get("speaker_b", "speaker_b")

        conversation_keys = [
            key
            for key in conversation_item
            if key not in ["speaker_a", "speaker_b"] and not key.endswith(("_date_time", "_timestamp"))
        ]

        for key in conversation_keys:
            timestamp = conversation_item.get(f"{key}_date_time", "Unknown time")
            history.append(f"\n--- Turn started at {timestamp} ---")
            chats = conversation_item.get(key, [])
            if not isinstance(chats, list):
                continue

            for chat in chats:
                if not isinstance(chat, dict) or "speaker" not in chat or "text" not in chat:
                    continue

                speaker_name = speaker_a if chat.get("speaker") == speaker_a else speaker_b
                text_content = str(chat.get("text", ""))
                if self.figure_view and "img_url" in chat and "blip_caption" in chat:
                    text_content += (
                        f" [Image: {chat.get('img_url')}] with caption: {chat.get('blip_caption')}"
                    )
                history.append(f"{speaker_name}: {text_content}")

        return "\n".join(history)

    def _build_prompt(
        self,
        conversation_history: str,
        question_item: Dict[str, Any],
        history_length: int,
    ) -> Tuple[Dict[str, str], str]:
        conversation_lines = conversation_history.splitlines()
        history_slice = "\n".join(conversation_lines[:history_length]) if history_length else ""
        option_lines = normalize_option_lines(question_item.get("option") or [])
        question_stem = extract_question_stem(question_item.get("question", ""), option_lines)
        prompt_components = {
            "conversation_history": history_slice,
            "question": question_stem or str(question_item.get("question", "")).strip(),
            "options_text": "\n".join(option_lines),
        }
        return prompt_components, self.template.render(prompt_components)

    def _answer_question_with_full_context(
        self,
        conversation_history: str,
        question_item: Dict[str, Any],
        max_retries: int = 5,
    ) -> Tuple[str, float, str, int]:
        request_id = f"full-context-{uuid.uuid4()}"
        response_content = ""
        start_time = time.time()
        max_context_exceeded = 0
        trim_attempts = 0
        max_trim_attempts = 5

        conversation_lines = conversation_history.splitlines()
        current_length = len(conversation_lines)
        _prompt_components, answer_prompt = self._build_prompt(conversation_history, question_item, current_length)

        attempt = 0
        while attempt < max_retries:
            attempt += 1
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": answer_prompt}],
                    temperature=0.0,
                    extra_body={"enable_thinking": False},
                )
                response_content = response.choices[0].message.content or ""
                break
            except Exception as exc:
                error_message = str(exc)
                error_lower = error_message.lower()
                is_context_error = "maximum context length" in error_lower
                if is_context_error:
                    max_context_exceeded = 1

                if is_context_error and trim_attempts < max_trim_attempts and current_length > 0:
                    trim_attempts += 1
                    trim_size = max(1, int(current_length * 0.1))
                    current_length = max(0, current_length - trim_size)
                    _prompt_components, answer_prompt = self._build_prompt(
                        conversation_history,
                        question_item,
                        current_length,
                    )
                    continue

                if attempt >= max_retries:
                    response_content = f"Error: Failed to get response from LLM. [{request_id}]"
                    break

                sleep_time = random.uniform(10, 20)
                self.logger.warning(
                    "LLM call failed for %s on attempt %s/%s: %s. Retrying in %.2f seconds.",
                    request_id,
                    attempt,
                    max_retries,
                    error_message,
                    sleep_time,
                )
                time.sleep(sleep_time)

        response_time = time.time() - start_time
        return response_content, response_time, answer_prompt, max_context_exceeded

    def _build_result(
        self,
        question_item: Dict[str, Any],
        response: str,
        response_time: float,
        answer_prompt: str,
        max_context_flag: int,
    ) -> Dict[str, Any]:
        return {
            "question": question_item.get("question", ""),
            "answer": question_item.get("answer", ""),
            "answer_fixed": question_item.get("answer_fixed"),
            "category": question_item.get("category", -1),
            "option": question_item.get("option", []),
            "character": question_item.get("character"),
            "response": response,
            "response_time": response_time,
            "answer_prompt": answer_prompt,
            "max_context_exceeded": max_context_flag,
        }

    def _process_single_question(
        self,
        conversation_item: Dict[str, Any],
        question_item: Dict[str, Any],
        conv_idx: int,
        question_idx: int,
        pbar: tqdm,
    ) -> Dict[str, Any]:
        conversation_history = self._format_conversation(conversation_item.get("conversation", {}))
        response, response_time, answer_prompt, max_context_flag = self._answer_question_with_full_context(
            conversation_history,
            question_item,
        )

        result = self._build_result(
            question_item,
            response,
            response_time,
            answer_prompt,
            max_context_flag,
        )

        with self.lock:
            self.results[conv_idx][question_idx] = result

        pbar.update(1)
        return result

    def process_data_file(self, file_path: str, max_workers: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        data = normalize_dataset_records(raw_data)
        total_questions = sum(len(item.get("qa", [])) for item in data)
        if total_questions == 0:
            raise ValueError("No questions found in the input dataset.")

        self.results = defaultdict(dict)

        print(f"--- Starting Full Context Inference: {total_questions} total questions ---")
        with tqdm(total=total_questions, desc="Full Context") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="full-context") as executor:
                futures = []
                for conv_idx, item in enumerate(data):
                    for question_idx, question_item in enumerate(item.get("qa", [])):
                        futures.append(
                            executor.submit(
                                self._process_single_question,
                                item,
                                question_item,
                                conv_idx,
                                question_idx,
                                pbar,
                            )
                        )

                for future in as_completed(futures):
                    future.result()

        results_dict = serialize_results(self.results)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as file:
            json.dump(results_dict, file, indent=4, ensure_ascii=False)

        print(f"Raw responses saved to: {self.output_path}")
        return results_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-context inference only.")
    parser.add_argument("--input_file", required=True, help="Path to the dataset JSON file.")
    parser.add_argument("--output_file", default=None, help="Optional path for raw full-context outputs.")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of worker threads.")
    parser.add_argument("--model", default=None, help="Model name for the answer LLM.")
    parser.add_argument("--base_url", default=None, help="Base URL for the answer LLM API.")
    parser.add_argument("--api_key", default=None, help="API key for the answer LLM.")
    parser.add_argument(
        "--figure_view",
        action="store_true",
        help="Include image captions in the serialized conversation when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file).resolve()
    output_path = resolve_output_path(input_path, args.output_file, "_full_context_raw.json")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runner = FullContextRunner(
        output_path=output_path,
        figure_view=args.figure_view,
        llm_config={
            "model": args.model,
            "base_url": args.base_url,
            "api_key": args.api_key,
        },
    )
    runner.process_data_file(str(input_path), max_workers=args.max_workers)


if __name__ == "__main__":
    main()