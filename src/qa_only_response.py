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

from src.utils import compute_dataset_stats, stream_normalized_dataset

load_dotenv()

DEFAULT_LLM_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")

ANSWER_PROMPT_QA_ONLY = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from multiple speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question. You also have access to knowledge graph relations for each user, showing connections between entities, concepts, and events relevant to that user.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from all speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from all speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.
9. Use the knowledge graph relations to understand the user's knowledge network and identify important relationships between entities in the user's world.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Analyze the knowledge graph relations to understand the user's knowledge context
6. Formulate a precise, concise answer based solely on the evidence in the memories
7. Double-check that your answer directly addresses the question asked
8. Ensure your final answer is specific and avoids vague time references

# OUTPUT
You are required to answer in JSON format only.
Return the answer strictly in the following structure:

{
    "option": "",
    "option_w_content": ""
}

# Rules you must follow:
1. The value of "option" must be exactly one single uppercase letter from this set only: A, B, C, D, E, F
2. The value of "option_w_content" must contain: the option letter + a dot + a space + the full option text content (for example: "C. Deep learning is a type of machine learning")
3. Do NOT output anything except the JSON. No explanations, No extra text, No markdown, No comments

Question: {{question}}
"""

ANSWER_PROMPT_QA_ONLY_V1 = """
# ROLE:
You are a High-Precision Logic and Universal Knowledge Engine. Your objective is to solve complex multiple-choice questions by synthesizing linguistic nuances, world facts, and logical consistency without relying on external data.

# CORE REASONING PRINCIPLES:
1. **Semantic Deconstruction**: Break down the question into its core intent, identifying hidden assumptions and key constraints.
2. **Internal Fact-Mapping**: Search your vast internal database for historical, scientific, cultural, and logical truths related to the query.
3. **Plausibility Assessment**: For each option, simulate a "world state" where it is true. If that state violates common sense or physical laws, discard it.
4. **Linguistic Cues**: Analyze the wording of options. Extreme qualifiers (e.g., "always", "never", "only") often indicate incorrect answers, while nuanced language often points to the truth.
5. **Relational Logic**: Compare options against each other. If two options are mutually exclusive, the answer is likely one of them. If two options imply each other, both may be incorrect.

# EXECUTION STEPS (Chain of Thought):
1. **Targeting**: Precisely define what the question is asking for (e.g., a date, a cause, a location, or a concept).
2. **Context Recovery**: Reconstruct the most likely context or "missing memory" based on the entities mentioned in the question using general knowledge.
3. **Option Filtering**: 
    - Eliminate options that are factually impossible.
    - Eliminate options that are logically inconsistent with the question's premise.
4. **Probabilistic Selection**: From the remaining candidates, select the option with the highest statistical probability of being correct in a real-world scenario.
5. **Conciseness Check**: Ensure the final answer content is distilled to its most essential 5-6 words.

# OUTPUT PROTOCOL:
- You must respond ONLY with a JSON object.
- No preamble, no postscript, no explanation.
- Adhere strictly to the A-F option range.

{
    "option": "[Letter]",
    "option_w_content": "[Letter]. [Brief Content]"
}

# CONSTRAINTS:
- The content in "option_w_content" must be a direct answer, not a sentence explaining why.
- If the question contains relative time, use 2026 as the "Current Year" for reference if needed.
- Trust your internal logic above all else.

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

        MAX_RETRY = 5 
        for _ in range(MAX_RETRY):
            response, response_time, pollution_check_prompt = self.answer_question(question)
            try:
                # 检查是不是合法 JSON
                data = json.loads(response)
                # 若必须包含字段，也可以继续判断
                if "option" in data and "option_w_content" in data:
                    option = data.get("option")
                    option_w_content = data.get("option_w_content")

            except json.JSONDecodeError:
                # 不是 JSON，继续重试
                pass

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response_option": option,
            "response_option_w_content": option_w_content,
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
