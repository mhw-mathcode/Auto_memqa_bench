import json
import logging
import os
import random
import threading
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.mcq_scoring import strip_prediction_text
from src.utils import compute_dataset_stats, stream_normalized_dataset

load_dotenv()

DEFAULT_LLM_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")

ANSWER_PROMPT_QA_ONLY = """
You are an expert knowledge retrieval and logical deduction system tasked with testing the limits of your internal parametric memory and analytical reasoning.

# INSTRUCTIONS:
1. Answer the provided choice question using ONLY your pre-trained world knowledge, common sense, and logical deduction. Do not expect any external context or memory banks to be provided.
2. You MUST select the single most likely correct option. Under no circumstances should you refuse to answer, state that there is insufficient context, or choose/output "F" (Insufficient evidence/Refusal).
3. **Evaluate Option Plausibility:** Carefully analyze the provided options. Eliminate options that are logically absurd, contradict common sense, or feel out of place for natural human dialogue/behavior. Select the option that makes the most logical or real-world sense, even if you do not know the exact source material.
4. If the question contains specific character names or recognizable scenarios, leverage your broad knowledge of popular culture and human interaction to deduce the most likely answer.
5. The final answer must be strictly the selected option letter or the exact text of the chosen option (under 5-6 words).

Question: {{question}}
"""

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
        self._max_parallelism_cap = max(1, min(os.cpu_count() * 2 or 8, 18))

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
        if requested is None or requested <= 0:
            self.logger.warning("Received invalid max_workers=%s. Falling back to 1.", requested)
            return 1
        resolved = min(requested, self._max_parallelism_cap)
        if resolved != requested:
            self.logger.info("Capping max_workers from %s to %s.", requested, resolved)
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
                    extra_body={
                        "enable_thinking": False
                    }
                )
            except Exception as e:
                s = str(e).lower()
                if ("429" in s) or ("tpm" in s) or ("rate limit" in s):
                    wait_s = sleep_time + random.uniform(0, 3)
                    print(f"⚠️ 触发速率限制，等待 {wait_s:.1f}s 后重试...")
                    time.sleep(wait_s)
                    continue
                raise

    def answer_question(self, question: str, max_retries=20):
        """
        只回答问题，不做任何检索。
        """
        request_id = f"qa-only-{uuid.uuid4()}"
        prompt = ANSWER_PROMPT_QA_ONLY.replace("{{question}}", question)

        attempts = 0
        sleep_penalty = 0.0
        start = time.time()
        last_err = None

        while attempts < max_retries:
            attempts += 1
            try:
                resp = self.safe_chat(
                    model=self.answer_llm_model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.0,
                    max_tokens=64,
                    sleep_time=min(30, 2 + attempts),
                )
                content = resp.choices[0].message.content or ""
                elapsed = max(0.0, time.time() - start - sleep_penalty)
                self.logger.info("Answer success %s in %.2fs (attempts=%d)", request_id, elapsed, attempts)
                return content.strip(), elapsed, prompt
            except Exception as e:
                last_err = e
                backoff = min(20.0, 0.8 * (2 ** (attempts - 1))) + random.uniform(0.1, 0.6)
                sleep_penalty += backoff
                self.logger.warning(
                    "Answer retry %s attempt %d/%d backoff=%.2fs err=%s\n%s",
                    request_id, attempts, max_retries, backoff, str(e), traceback.format_exc()
                )
                time.sleep(backoff)

        # 兜底
        self.logger.error("Answer failed permanently %s: %s", request_id, last_err)
        return "Error", 0.0, prompt

    def process_question(self, val, idx, pbar=None):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])

        response, response_time, pollution_check_prompt = self.answer_question(question)
        response_option = strip_prediction_text(response)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response_option,
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
            print("No questions found to process.")
            self._results_writer = IncrementalResultsWriter(self.output_path)
            self._results_writer.finalize()
            return

        print(f"--- 预计总共需要处理 {total_questions} 个问题 ---")

        resolved_workers = self._resolve_max_workers(max_workers)
        print(f"⚙️ 使用 max_workers = {resolved_workers}")

        self._expected_results_per_conversation = stats.get("qa_per_conversation", [])
        self._results_buffer = {}
        self._results_writer = IncrementalResultsWriter(self.output_path)

        futures = {}
        drain_threshold = max(resolved_workers, 1) * 4
        successful_count = 0
        failed_count = 0

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
                    pbar.write(f"\n--- ❌ Error processing task '{task_label}' ---")
                    pbar.write(f"{exc}\n{error_details}\n")
                    pbar.update(1)

        with tqdm(total=total_questions, desc="💡Total Questions Progress") as pbar:
            try:
                with ThreadPoolExecutor(max_workers=resolved_workers, thread_name_prefix="qa-only-main") as executor:
                    for conv_idx, item in enumerate(stream_normalized_dataset(dataset_path)):
                        qa_list = item.get("qa", [])
                        for question_item in qa_list:
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

        print(f"\n✅ All questions processed. Success: {successful_count}, Failed: {failed_count}")

    def close(self):
        pass
