import json
import logging
import os
import random
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.mcq_scoring import (
    MULTIPLE_SELECT,
    ORDERING,
    get_answer_instruction,
    normalize_question_type,
    strip_prediction_text,
)
from src.utils import compute_dataset_stats, stream_normalized_dataset
from src.pipeline_utils import log_event

load_dotenv()

DEFAULT_LLM_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")

ANSWER_PROMPT_QA_ONLY = """
You are an expert knowledge retrieval and logical deduction system tasked with testing the limits of your internal parametric memory and analytical reasoning.

# INSTRUCTIONS:
1. Answer the provided choice question using ONLY your pre-trained world knowledge, common sense, and logical deduction. Do not expect any external context or memory banks to be provided.
2. {{selection_rule}} {{f_rule}}
3. Evaluate Option Plausibility: Carefully analyze the provided options. Eliminate options that are logically absurd, contradict common sense, or feel out of place for natural human dialogue/behavior. Select the option that makes the most logical or real-world sense, even if you do not know the exact source material.
4. If the question contains specific character names or recognizable scenarios, leverage your broad knowledge of popular culture and human interaction to deduce the most likely answer.
5. Explain WHY you selected that option. Your reason must name the decisive basis: pre-trained factual knowledge, a recognizable scenario, logical elimination, common sense, or linguistic/behavioral plausibility. Do not invent dialogue, events, relationships, or other context that is absent from the question and options.
6. Return exactly two lines in this format. {{answer_instruction}} The reason must be 1-2 concise sentences:
Answer: <answer>
Reason: <why this option is more likely than the alternatives>

Question: {{question}}
"""


def parse_qa_only_response(text):
    """Extract the option and rationale while retaining tolerant legacy parsing."""
    raw = str(text or "").strip()
    cleaned = raw.split("</think>", 1)[-1].strip() if "</think>" in raw else raw

    json_text = cleaned
    if json_text.startswith("```") and json_text.endswith("```"):
        json_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", json_text, flags=re.IGNORECASE)
    try:
        payload = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        payload = None

    if isinstance(payload, dict):
        answer = ""
        for key in ("answer", "final_answer", "response"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                answer = value.strip()
                break
        reason = str(payload.get("reason") or payload.get("rationale") or "").strip()
        return answer or strip_prediction_text(cleaned), reason

    answer_match = re.search(
        r"(?im)^\s*(?:answer|final answer)\s*[:：]\s*[\(\[]?\s*([A-F](?:\s*[,，]\s*[A-F])*)\s*[\)\]]?(?:\s|$|[.，。])",
        cleaned,
    )
    reason_match = re.search(r"(?ims)^\s*(?:reason|rationale|理由)\s*[:：]\s*(.+?)\s*$", cleaned)
    answer = f"({answer_match.group(1).upper()})" if answer_match else strip_prediction_text(cleaned)
    reason = reason_match.group(1).strip() if reason_match else ""
    return answer, reason

class IncrementalResultsWriter:
    """
    Append-only writer that buffers conversation results and materializes the
    final JSON payload once, avoiding repeated full rewrites under concurrency.
    """
    def __init__(self, output_path: str, flush_every: int = 8):
        self._output_path = Path(output_path)
        self._temp_path = self._output_path.with_name(self._output_path.name + ".partial")
        self._flush_every = max(1, flush_every)
        self._buffer = []
        self._lock = threading.Lock()

        parent = self._output_path.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        temp_parent = self._temp_path.parent
        if temp_parent and not temp_parent.exists():
            temp_parent.mkdir(parents=True, exist_ok=True)
        if self._temp_path.exists():
            self._temp_path.unlink()

    def append(self, conversation_idx: int, records):
        payload = {"idx": conversation_idx, "results": records}
        with self._lock:
            self._buffer.append(payload)
            if len(self._buffer) >= self._flush_every:
                self._flush_locked()

    def _flush_locked(self):
        if not self._buffer:
            return
        with self._temp_path.open("a", encoding="utf-8") as handle:
            for item in self._buffer:
                json.dump(item, handle, ensure_ascii=False)
                handle.write("\n")
        self._buffer.clear()

    def flush(self):
        with self._lock:
            self._flush_locked()

    def finalize(self):
        self.flush()
        aggregated = {}
        if self._temp_path.exists():
            with self._temp_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    key = str(payload.get("idx"))
                    aggregated.setdefault(key, []).extend(payload.get("results", []))

        tmp_path = self._output_path.with_name(self._output_path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(aggregated, handle, indent=4, ensure_ascii=False)
        tmp_path.replace(self._output_path)

        if self._temp_path.exists():
            self._temp_path.unlink()


class QAOnlyRunner:
    """
    不使用 Memory / 向量库 / 检索，只用 LLM 直接回答问题。
    """

    def __init__(
        self,
        output_path="results.json",
        logger=None,
        llm_config=None,
        answer_llm_config=None,
    ):
        
        answer_llm_config = answer_llm_config or llm_config
        self.logger = logger if logger else logging.getLogger(__name__)

        self.answer_llm_model = answer_llm_config.get("model") or DEFAULT_LLM_MODEL
        answer_base_url = answer_llm_config.get("base_url") or DEFAULT_BASE_URL
        answer_api_key = answer_llm_config.get("api_key") or DEFAULT_API_KEY

        self.answer_client = OpenAI(api_key=answer_api_key, base_url=answer_base_url)

        self.output_path = output_path
        self._results_state_lock = threading.Lock()
        self._results_buffer = {}
        self._expected_results_per_conversation = []
        self._results_writer = IncrementalResultsWriter(output_path)

    def _record_result(self, conversation_idx: int, result):
        payload = None
        with self._results_state_lock:
            bucket = self._results_buffer.setdefault(conversation_idx, [])
            bucket.append(result)
            expected = 0
            if self._expected_results_per_conversation and conversation_idx < len(self._expected_results_per_conversation):
                expected = self._expected_results_per_conversation[conversation_idx]
            if expected and len(bucket) >= expected:
                payload = (conversation_idx, list(bucket))
                self._results_buffer.pop(conversation_idx, None)
        if payload:
            idx, records = payload
            self._results_writer.append(idx, records)

    def _resolve_max_workers(self, requested: int) -> int:
        try:
            resolved = int(requested)
        except (TypeError, ValueError):
            resolved = 0
        if resolved <= 0:
            self.logger.warning(
                "EVENT | qa_only_workers | status=fallback | requested=%s | resolved=1",
                requested,
            )
            return 1
        return resolved

    def safe_chat(self, model, messages, temperature=0.0, max_tokens=64, sleep_time=10):
        """
        简单的限流重试封装（429/TPM/rate limit）。
        """
        while True:
            try:
                return self.answer_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                s = str(e).lower()
                if "missing_required_parameter" in s or (
                    "one of \"input\"" in s and "prompt" in s
                ) or "contents is required" in s:
                    prompt_parts = [
                        str(msg.get("content", "")).strip()
                        for msg in messages
                        if isinstance(msg, dict) and str(msg.get("content", "")).strip()
                    ]
                    prompt_text = "\n".join(prompt_parts).strip()
                    if not prompt_text:
                        raise ValueError("qa_only prompt is empty") from e
                    # Some OpenAI-compatible Gemini gateways reject chat messages
                    # with provider-side `contents is required`; Responses API is
                    # often accepted by those gateways.
                    try:
                        resp2 = self.answer_client.responses.create(
                            model=model,
                            input=prompt_text,
                            temperature=temperature,
                        )
                        content2 = getattr(resp2, "output_text", "") or ""
                        return SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content=content2))]
                        )
                    except Exception as e2:
                        if "contents is required" in str(e2).lower():
                            raise ValueError(
                                "provider rejected qa_only request: contents is required"
                            ) from e2
                        s = str(e2).lower()
                        e = e2

                if ("429" in s) or ("tpm" in s) or ("rate limit" in s):
                    wait_s = sleep_time + random.uniform(0, 3)
                    log_event("qa_only_answer", status="retry", reason="rate_limit", wait=f"{wait_s:.1f}s")
                    time.sleep(wait_s)
                    continue
                raise

    def answer_question(self, question: str, question_type=None, max_retries=20):
        """
        只回答问题，不做任何检索。
        """
        request_id = f"qa-only-{uuid.uuid4()}"
        normalized_type = normalize_question_type(question_type)
        if normalized_type == MULTIPLE_SELECT:
            selection_rule = "You MUST select every correct option; incomplete or extra selections are incorrect."
            f_rule = "Do not refuse to answer; if F is explicitly provided as a normal option and is correct, include it like any other option."
        elif normalized_type == ORDERING:
            selection_rule = "You MUST return all options in the correct sequence; any ordering error is incorrect."
            f_rule = "Do not refuse to answer; if F is explicitly provided as a normal option, place it in the sequence like any other option."
        else:
            selection_rule = "You MUST select the single most likely correct option."
            f_rule = 'Under no circumstances should you refuse to answer, state that there is insufficient context, or choose/output "F" (Insufficient evidence/Refusal).'
        prompt = (
            ANSWER_PROMPT_QA_ONLY
            .replace("{{question}}", question)
            .replace("{{selection_rule}}", selection_rule)
            .replace("{{f_rule}}", f_rule)
            .replace("{{answer_instruction}}", get_answer_instruction(normalized_type))
        )

        attempts = 0
        sleep_penalty = 0.0
        start = time.time()
        last_err = None

        while attempts < max_retries:
            attempts += 1
            try:
                resp = self.safe_chat(
                    model=self.answer_llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    # Pollution filtering aggregates three independent samples.
                    # A non-zero temperature prevents one deterministic guess from
                    # being counted as three independent confirmations.
                    temperature=0.7,
                    max_tokens=192,
                    sleep_time=min(30, 2 + attempts),
                )
                content = resp.choices[0].message.content or ""
                elapsed = max(0.0, time.time() - start - sleep_penalty)
                self.logger.info(
                    "EVENT | qa_only_answer | status=success | request_id=%s | elapsed=%.2fs | attempts=%d",
                    request_id,
                    elapsed,
                    attempts,
                )
                return content.strip(), elapsed, prompt
            except Exception as e:
                last_err = e
                error_text = str(e).lower()
                if (
                    "provider rejected qa_only request" in error_text
                    or "qa_only prompt is empty" in error_text
                ):
                    self.logger.error(
                        "EVENT | qa_only_answer | status=failed | request_id=%s | non_retryable=True | error=%s",
                        request_id,
                        e,
                    )
                    break
                backoff = min(20.0, 0.8 * (2 ** (attempts - 1))) + random.uniform(0.1, 0.6)
                sleep_penalty += backoff
                self.logger.warning(
                    "EVENT | qa_only_answer | status=retry | request_id=%s | attempt=%d/%d | backoff=%.2fs | error=%s | trace=%s",
                    request_id, attempts, max_retries, backoff, str(e), traceback.format_exc()
                )
                time.sleep(backoff)

        # 兜底
        self.logger.error(
            "EVENT | qa_only_answer | status=failed | request_id=%s | error=%s",
            request_id,
            last_err,
        )
        return "Error", 0.0, prompt

    def process_question(self, val, idx, pbar=None):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])

        question_type = normalize_question_type(val.get("question_type"))
        response, response_time, pollution_check_prompt = self.answer_question(
            question,
            question_type,
        )
        response_option, response_reason = parse_qa_only_response(response)

        result = {
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "category": category,
            "evidence": evidence,
            "response": response_option,
            "response_raw": response,
            "response_reason": response_reason,
            "response_time": response_time,
            "pollution_check_prompt": pollution_check_prompt,
        }

        self._record_result(idx, result)

        if pbar:
            pbar.update(1)
        return result

    def process_data_file(self, file_path, max_workers=5):
        dataset_path = Path(file_path)
        stats = compute_dataset_stats(dataset_path)
        total_questions = stats.get("total_questions", 0)
        if total_questions == 0:
            log_event("qa_only_process", status="skipped", reason="no_questions")
            self._results_writer = IncrementalResultsWriter(self.output_path)
            self._results_writer.finalize()
            return

        resolved_workers = self._resolve_max_workers(max_workers)
        log_event(
            "qa_only_process",
            status="start",
            total_questions=total_questions,
            max_workers=resolved_workers,
            input=str(dataset_path),
        )

        self._expected_results_per_conversation = stats.get("qa_per_conversation", [])
        self._results_buffer = {}
        self._results_writer = IncrementalResultsWriter(self.output_path)

        futures = {}
        drain_threshold = max(resolved_workers, 1) * 4
        successful_count = 0
        failed_count = 0
        question_type_counts = {
            "single_choice": 0,
            "multiple_choice": 0,
            "ordering": 0,
        }

        def consume_one(pbar):
            nonlocal successful_count, failed_count
            if not futures:
                return
            done, _ = wait(tuple(futures.keys()), return_when=FIRST_COMPLETED)
            for finished in done:
                conv_idx, task_label = futures.pop(finished)
                try:
                    finished.result()
                    successful_count += 1
                except Exception as exc:
                    failed_count += 1
                    error_details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    log_event(
                        "qa_only_task",
                        status="failed",
                        task=task_label,
                        error=exc,
                        trace=error_details,
                    )
                    pbar.update(1)

        with tqdm(total=total_questions, desc="Step 3 no-context QA", unit="question") as pbar:
            try:
                with ThreadPoolExecutor(max_workers=resolved_workers, thread_name_prefix="qa-only-main") as executor:
                    for conv_idx, item in enumerate(stream_normalized_dataset(dataset_path)):
                        qa_list = item.get("qa", [])
                        for question_item in qa_list:
                            question_type_counts[
                                normalize_question_type(question_item.get("question_type"))
                            ] += 1
                            future = executor.submit(self.process_question, question_item, conv_idx, pbar)
                            question_preview = (question_item.get("question") or "").strip().replace("\n", " ")
                            if len(question_preview) > 40:
                                question_preview = question_preview[:37] + "..."
                            futures[future] = (conv_idx, f"Conv {conv_idx} - {question_preview}")

                            if len(futures) >= drain_threshold:
                                consume_one(pbar)

                    while futures:
                        consume_one(pbar)

            finally:
                # flush remaining buffered
                pending_flush = []
                with self._results_state_lock:
                    for conv_idx, bucket in self._results_buffer.items():
                        if bucket:
                            pending_flush.append((conv_idx, list(bucket)))
                    self._results_buffer.clear()
                for conv_idx, bucket in pending_flush:
                    self._results_writer.append(conv_idx, bucket)
                self._results_writer.finalize()

        log_event(
            "qa_only_process",
            status="completed",
            success=successful_count,
            failed=failed_count,
            output=self.output_path,
        )
        for question_type, total in question_type_counts.items():
            log_event(
                "qa_only_question_type",
                status="completed",
                question_type=question_type,
                total=total,
            )

    def close(self):
        pass
