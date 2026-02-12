import json
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import uuid

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm
from src.utils import normalize_dataset_records
import re
import copy

# --- Step 1: Define the Prompt Template for the Full Context approach ---
# This prompt is designed to take the entire conversation history directly.
ANSWER_PROMPT_FULL_CONTEXT = """
You are an intelligent assistant. Your task is to answer a question based on the provided conversation history.

# CONTEXT:
You have access to the complete conversation history between some speakers. This history contains all the information needed to answer the question.

# INSTRUCTIONS:
1.  Carefully read the entire conversation history to understand the context.
2.  Analyze the user's question and locate the relevant parts of the conversation that contain the answer.
3.  Pay close attention to timestamps and the order of messages to resolve any conflicting information, prioritizing the most recent messages if necessary.
4.  Your answer must be derived directly from the text in the history. Do not infer information or use external knowledge.
5.  Keep your answer concise and to the point, ideally less than 5-6 words, unless more detail is explicitly required by the question.
6.  If the question involves relative time references (e.g., "last year"), use the timestamps in the conversation to calculate the specific date or year.

# APPROACH (Think step by step):
1.  Identify the key entities and concepts in the question.
2.  Scan the full conversation history to find all mentions related to these keys.
3.  Synthesize the information from the relevant messages to formulate a precise answer.
4.  Double-check that your answer directly addresses the question and is supported by the provided text.

# OUTPUT
You are required to answer in JSON format only.
Return the answer strictly in the following structure:
{
    "question": "[Direct, natural, focused on the character]",
    "answer": "Letter. Full option text",
}

--- CONVERSATION HISTORY ---
{{conversation_history}}
--- END OF HISTORY ---

Question: {{question}}
"""

ANSWER_PROMPT_FULL_CONTEXT_ADD_EVID = """
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

FORMAT = """
# OUTPUT
You are required to answer in JSON format only.
Return the answer strictly in the following structure:
{
    "reasoning": "Your step-by-step logical deduction...",
    "evidence": "Verbatim quotes from the history...",
    "option": "A-F",
    "option_w_content": "Letter. Full option text"
}

# Rules you must follow:
1. The value of "option" must be exactly one single uppercase letter from this set only: A, B, C, D, E, F
2. The value of "option_w_content" must contain: the option letter + a dot + a space + the full option text content (for example: "C. Deep learning is a type of machine learning")
3. Do NOT output anything except the JSON. No explanations, No extra text, No markdown, No comments
"""

DEFAULT_LLM_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-14B")
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

class FullContextManager:
    """
    Handles the logic for answering questions using the full conversation context.
    This class reads a dataset, processes each question against its full conversation,
    and saves the LLM-generated answers.
    """
    def __init__(self, output_path, logger=None, figure_view=False, llm_config=None):
        load_dotenv()
        self.output_path = output_path
        llm_config = llm_config or {}
        llm_model = llm_config.get("model") or DEFAULT_LLM_MODEL
        llm_base_url = llm_config.get("base_url") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        llm_api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")

        openai_client_kwargs = {}
        if llm_base_url:
            openai_client_kwargs["base_url"] = llm_base_url
        if llm_api_key:
            openai_client_kwargs["api_key"] = llm_api_key
        self.openai_client = OpenAI(**openai_client_kwargs)
        os.environ["MODEL"] = llm_model
        self.model_name = llm_model
        self.logger = logger if logger else logging.getLogger(__name__)
        self.results = defaultdict(list)
        self.lock = threading.Lock()
        
        self.figure_view = figure_view

    def _log_llm_call(self, request_id, attempt, max_retries, prompt_components, full_prompt, response_content, status):
        """Formats and logs the complete LLM interaction."""
        log_message = f"""
========================= LLM Call Start (Full Context) =========================
--------------------------- INPUT ----------------------------
{full_prompt}
--------------------------- OUTPUT ---------------------------
{response_content if response_content else 'N/A'}
========================== SUMMARY ==========================
Request ID: {request_id}
Attempt: {attempt}/{max_retries}
Status: {status}
========================== LLM Call End ==========================
"""
        if "Success" in status:
            self.logger.info(log_message)
        else:
            self.logger.error(log_message)

    def _format_conversation(self, conversation_item: dict) -> str:
        """
        Formats the conversation dictionary into a readable string.
        Supports multi-speaker format:
          - speaker_0, speaker_1, ...
          - session_1, session_2, ... with session_1_date_time etc.
        Also keeps backward compatibility with speaker_a/speaker_b style if present.
        """

        history = []

        # -------- 1) Build speaker map --------
        speaker_map = {}

        # New style: speaker_0, speaker_1, ...
        for k, v in conversation_item.items():
            if re.fullmatch(r"speaker_\d+", str(k)) and isinstance(v, str):
                speaker_map[k] = v

        # Backward compatibility: speaker_a / speaker_b
        # (如果你的 normalize_dataset_records 还会产生这种字段，也能兼容)
        if "speaker_a" in conversation_item and isinstance(conversation_item["speaker_a"], str):
            speaker_map[conversation_item["speaker_a"]] = conversation_item["speaker_a"]
        if "speaker_b" in conversation_item and isinstance(conversation_item["speaker_b"], str):
            speaker_map[conversation_item["speaker_b"]] = conversation_item["speaker_b"]

        # 额外：如果 speaker_map 为空，后面也能 fallback，不会崩

        # -------- 2) Collect and sort sessions --------
        # session_1, session_2, ... (skip *_date_time / *_timestamp)
        session_keys = []
        for k in conversation_item.keys():
            k = str(k)
            if re.fullmatch(r"session_\d+", k):
                session_keys.append(k)

        def session_index(sk: str) -> int:
            m = re.match(r"session_(\d+)", sk)
            return int(m.group(1)) if m else 10**9

        session_keys.sort(key=session_index)

        # -------- 3) Render each session --------
        for sk in session_keys:
            timestamp = conversation_item.get(f"{sk}_date_time", "Unknown time")
            history.append(f"\n--- Turn started at {timestamp} ---")

            chats = conversation_item.get(sk, [])
            if not isinstance(chats, list):
                continue

            for chat in chats:
                if not isinstance(chat, dict):
                    continue
                if "speaker" not in chat or "text" not in chat:
                    continue

                raw_speaker = chat.get("speaker")

                # case A: chat["speaker"] == "speaker_0" -> Monica
                if isinstance(raw_speaker, str) and raw_speaker in speaker_map:
                    speaker_name = speaker_map[raw_speaker]

                # case B: chat["speaker"] == "Monica" already
                elif isinstance(raw_speaker, str):
                    # 如果 speaker_map 里是值（Monica）也允许识别一下
                    if raw_speaker in speaker_map.values():
                        speaker_name = raw_speaker
                    else:
                        speaker_name = raw_speaker

                # case C: speaker is int like 0 -> map to speaker_0 if exists
                elif isinstance(raw_speaker, int):
                    key = f"speaker_{raw_speaker}"
                    speaker_name = speaker_map.get(key, str(raw_speaker))

                else:
                    speaker_name = str(raw_speaker)

                text_content = chat.get("text", "")
                if self.figure_view:
                    if "img_url" in chat and "blip_caption" in chat:
                        text_content += f" [Image: {chat.get('img_url')}] with caption: {chat.get('blip_caption')}"

                history.append(f"{speaker_name}: {text_content}")

        return "\n".join(history)

    def _format_conversation_wo_evidence(self, conversation_item: dict, evidence_dialogues):
        """
        返回一个新的 conversation_item：
        - 删除 evidence 中 dia_id != N/A 的对话
        - dia_id == N/A 时，删除对应 session_x_date_time
        - 不改任何 text 内容
        - 不做格式化
        """

        conv = copy.deepcopy(conversation_item)

        # ============================================================
        # 1) 解析 evidence
        # ============================================================
        evidence_dia_ids = set()
        evidence_sessions_no_time = set()

        for ev in evidence_dialogues or []:
            dia_id = ev.get("dia_id")
            utterance = ev.get("utterance", "")

            # 1.1 删除具体对话
            if dia_id and dia_id != "N/A":
                evidence_dia_ids.add(dia_id)

            # 1.2 删除 session_x_date_time
            if dia_id == "N/A":
                m = re.search(r"(session_\d+)_date_time", utterance)
                if m:
                    evidence_sessions_no_time.add(m.group(1))

        # ============================================================
        # 2) 删除 session 时间信息
        # ============================================================
        for sk in evidence_sessions_no_time:
            dt_key = f"{sk}_date_time"
            if dt_key in conv:
                del conv[dt_key]

        # ============================================================
        # 3) 删除 evidence 对话（通过 dia_id）
        # ============================================================
        for k, v in conv.items():
            if not re.fullmatch(r"session_\d+", str(k)):
                continue
            if not isinstance(v, list):
                continue

            filtered_chats = []
            for chat in v:
                if not isinstance(chat, dict):
                    filtered_chats.append(chat)
                    continue

                if chat.get("dia_id") in evidence_dia_ids:
                    continue  # 直接丢弃证据对话

                filtered_chats.append(chat)

            conv[k] = filtered_chats

        return conv

    
    def _answer_question_with_full_context(self, conversation, evidence, only_evidence, except_evidence, question, evidence_dialogues, max_retries=5):
        """Calls the LLM with the full context to get an answer."""
        request_id = f"full-context-q-{uuid.uuid4()}"
        response_content = None
        start_time = time.time()
        max_context_exceeded = 0
        max_trim_attempts = 5
        trim_attempts = 0

        if except_evidence == 0: # full
            conversation_history = self._format_conversation(conversation)
        else: # full without evidence
            conversation_history = self._format_conversation_wo_evidence(conversation, evidence_dialogues)

        # modify
        # conversation_lines = conversation_history.splitlines()
        # total_lines = len(conversation_lines)
        # current_length = total_lines if total_lines > 0 else 0

        conversation_lines = conversation_history
        current_length = len([
            k for k in conversation_history
            if k.startswith("session_") and not k.endswith("_date_time")
        ])
        context = conversation_history

        def build_prompt(length):
            # modify
            # context = "\n".join(conversation_lines[:length]) if length else ""
            context = conversation_history
            if only_evidence == 1:
                context = evidence

            prompt_components_local = {
                "conversation_history" if only_evidence == 0 else "evidence": context,
                "question": question,
            }

            self.template = Template(ANSWER_PROMPT_FULL_CONTEXT_ADD_EVID if only_evidence == 0 else ANSWER_PROMPT_ONLY_EVIDENCE)
            rendered_prompt = self.template.render(prompt_components_local)
            return prompt_components_local, rendered_prompt

        prompt_components, answer_prompt = build_prompt(current_length)

        attempt = 0
        while attempt < max_retries:
            attempt += 1
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": answer_prompt}],
                    temperature=0.0,
                    extra_body={
                        "enable_thinking": False
                    }
                )
                response_content = response.choices[0].message.content
                self._log_llm_call(
                    request_id,
                    attempt,
                    max_retries,
                    prompt_components,
                    answer_prompt,
                    response_content,
                    "Success",
                )
                break  # Success, exit loop
            except Exception as e:
                error_message = f"LLM API call failed. Error: {e}"
                error_lower = str(e).lower()
                is_context_error = "maximum context length" in error_lower
                status_c = "Failed Attempt"

                if is_context_error:
                    max_context_exceeded = 1
                if (
                    is_context_error
                    and trim_attempts < max_trim_attempts
                    and current_length > 0
                ):
                    trim_attempts += 1
                    trim_size = max(1, int(current_length * 0.1))
                    current_length = max(0, current_length - trim_size)
                    current_history_str, prompt_components, answer_prompt = build_prompt(current_length)
                    trim_message = (
                        f"{error_message} | Reducing conversation history by 10% "
                        f"(trim attempt {trim_attempts}/{max_trim_attempts})."
                    )
                    self._log_llm_call(
                        request_id,
                        attempt,
                        max_retries,
                        prompt_components,
                        "",  # Avoid logging large prompt repeatedly
                        trim_message,
                        status_c,
                    )
                    # Retry immediately without additional wait.
                    continue

                if attempt >= max_retries:
                    status_c = "Failed Attempt (END)"
                    error_message = (
                        f"Request ID [{request_id}] - LLM call failed permanently after {max_retries} attempts."
                    )
                    response_content = "Error: Failed to get response from LLM."
                    self._log_llm_call(
                        request_id,
                        attempt,
                        max_retries,
                        prompt_components,
                        "",
                        error_message,
                        status_c,
                    )
                    break

                sleep_time = random.uniform(45, 75)
                status_c += f", retrying in {sleep_time:.2f} seconds..."
                self._log_llm_call(
                    request_id,
                    attempt,
                    max_retries,
                    prompt_components,
                    "",
                    error_message,
                    status_c,
                )
                time.sleep(sleep_time)

        response_time = time.time() - start_time
        return response_content, response_time, answer_prompt, max_context_exceeded

    def _process_single_question(self, conversation_item, question_item, idx, pbar, only_evidence, except_evidence):
        """Worker function to process one question using its conversation context."""
        question = question_item.get("question", "")
        answer = question_item.get("answer", "")
        category = question_item.get("category", -1)
        pollution_check = question_item.get("pollution_check", "")

        evidence = question_item.get("evidence", "")

        evidence_dialogues = question_item.get("evidence_dialogues", "")
        reasoning_steps = question_item.get("reasoning_steps", "")

        evidence =  evidence_dialogues + reasoning_steps

        # conversation_history = self._format_conversation(conversation_item["conversation"])

        def extract_option(s):
            if not isinstance(s, str):
                return None
            m = re.match(r"\s*([A-F])", s)
            return m.group(1) if m else None
        ans = extract_option(answer)

        response, response_time, answer_prompt, max_context_flag = self._answer_question_with_full_context(
            conversation_item["conversation"], evidence, only_evidence, except_evidence, question, evidence_dialogues
        )

        if "# OUTPUT" in answer_prompt:
            MAX_RETRY = 5 
            for _ in range(MAX_RETRY):
                response, response_time, answer_prompt, max_context_flag = self._answer_question_with_full_context(
                    conversation_item["conversation"], evidence, only_evidence, except_evidence, question, evidence_dialogues
                )
                try:
                    # 检查是不是合法 JSON
                    data = json.loads(response)
                    break
                except json.JSONDecodeError:
                    continue

            if answer == data.get("answer"):
                check_result = "right"
            else: check_result = "maybe_wrong"

        else:
            if answer in response:
                check_result = "right"
            else: check_result = "maybe_wrong"
        
        # 1. 保留原始question_item的所有字段（深拷贝）
        import copy
        result = copy.deepcopy(question_item)

        # 2. 准备内部的数据块
        context_data = {
            "result": check_result,
            "response": response,
            "answer_prompt": answer_prompt,
            "response_time": response_time,
            "max_context_exceeded": max_context_flag,
        }

        # 3. 根据条件动态决定键名
        if only_evidence == 1:
            result["only_evidence_check"] = context_data
        elif except_evidence == 1:
            iterative_records = []
            remaining_evidence = evidence_dialogues

            round_id = 1
            for i in range(1):
                response, response_time, answer_prompt, max_context_flag = self._answer_question_with_full_context(
                    conversation_item["conversation"],
                    evidence,
                    only_evidence,
                    except_evidence,
                    question=question,
                    evidence_dialogues=remaining_evidence
                )

                data = json.loads(response)
                is_right = (data["answer"] == answer)

                iterative_records.append({
                    "round": round_id,
                    "answer": data["answer"],
                    "used_evidence": data["evidence_dialogues"],
                    "all_evidence": remaining_evidence,
                    "result": "right" if is_right else "wrong"
                })

                if not is_right or not remaining_evidence:
                    break

                remaining_evidence = remaining_evidence + data["evidence_dialogues"]
                round_id += 1

            result["iterative_evidence_ablation"] = iterative_records
            # result["except_evidence_check"] = context_data
        else:
            result["fullcontext_check"] = context_data

        # The lock is now only held for a very short time to do a quick memory update.
        with self.lock:
            self.results[idx].append(result)

        pbar.update(1)
        return result

    def process_data_file(self, file_path, only_evidence, except_evidence, max_workers=10):
        """
        Main method to orchestrate the processing of the dataset.
        It uses a thread pool to handle questions in parallel.
        """
        with open(file_path, "r") as f:
            raw_data = json.load(f)
        data = normalize_dataset_records(raw_data)
        
        # 保存原始数据的 conversation 信息
        self.original_data = data
        
        # 用于统计跳过的题目数量
        self.skipped_count = 0

        total_questions = sum(len(item.get("qa", [])) for item in data)
        if total_questions == 0:
            print("No questions found to process.")
            return

        if except_evidence == 1:
            pending_questions = 0
            for item in data:
                for question_item in item.get("qa", []):
                    only_check = question_item.get("only_evidence_check", {})
                    if only_check.get("result") != "wrong":
                        pending_questions += 1
            print(
                f"--- Starting Full Context Evaluation: {total_questions} total questions, {pending_questions} pending ---"
            )
        else:
            print(f"--- Starting Full Context Evaluation: {total_questions} total questions to process ---")

        with tqdm(total=total_questions, desc="💡 Full Context Progress") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for idx, item in enumerate(data):
                    for question_item in item.get("qa", []):
                        # 筛选逻辑：如果是 except_evidence 阶段（v2b），检查上一步的 only_evidence_check
                        if except_evidence == 1:
                            only_check = question_item.get("only_evidence_check", {})
                            if only_check.get("result") == "wrong":
                                # 跳过此题目，但更新进度条
                                pbar.update(1)
                                # 仍然将原题目保留在结果中
                                with self.lock:
                                    self.results[idx].append(question_item)
                                    self.skipped_count += 1
                                continue
                        
                        future = executor.submit(
                            self._process_single_question,
                            item, question_item, idx, pbar, only_evidence, except_evidence
                        )
                        futures.append(future)

                # Wait for all futures to complete and handle potential exceptions
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        self.logger.error(f"A task failed in the thread pool: {e}")
                        self.logger.error(f"Full error traceback: {error_details}")

        # --- THIS IS THE CORRECT PLACE TO SAVE THE FILE ---
        # All threads are done, now write the final result to the file once.
        print("\nAll threads finished. Saving final results to disk...")
        
        # 重新组织结果，保留原始item的所有字段
        final_results = []
        for idx, item in enumerate(self.original_data):
            if idx in self.results and len(self.results[idx]) > 0:
                # 深拷贝原始item以保留所有字段
                import copy
                result_item = copy.deepcopy(item)
                # 只更新qa字段
                result_item["qa"] = self.results[idx]
                final_results.append(result_item)
        
        with open(self.output_path, "w") as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)

        # 统计最终保留的题目数量
        total_kept = sum(len(v) for v in self.results.values())
        return total_kept

def full_context_main(args, input_file_path: str, output_file_path: str,
                      only_evidence: int = 0, except_evidence: int = 0) -> tuple:
    """
    题目合理性验证主函数
    
    Args:
        args: 命令行参数
        input_file_path: 输入文件路径（v1版本）
        output_file_path: 输出文件路径（v2版本）
        only_evidence: 是否只使用证据（1=只看证据答对的保留）
        except_evidence: 是否排除证据（1=不看证据答对的筛掉）
    
    Returns:
        (处理后的文件路径, 保留的题目数量)
    """
    mode_desc = "只使用证据" if only_evidence == 1 else ("排除证据" if except_evidence == 1 else "两阶段筛选")
    print("\n" + "="*60)
    print(f"🔄 题目合理性验证 - {mode_desc}")
    print("="*60)
    print(f"📥 输入文件: {input_file_path}")

    def build_provider_config(model_value, base_url_value, api_key_value, optional_fields=None):
        config = {}
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
    
    answer_llm_model = args.answer_llm_model
    answer_llm_base_url = args.answer_llm_base_url
    answer_llm_api_key = args.answer_llm_api_key
    answer_llm_config = build_provider_config(answer_llm_model, answer_llm_base_url, answer_llm_api_key)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # 两阶段筛选：v2a(only_evidence=1) -> v2b(except_evidence=1)
    if only_evidence == 0 and except_evidence == 0:
        with open(input_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        total_questions = sum(len(item.get("qa", [])) for item in raw_data) if isinstance(raw_data, list) else 0

        base_path, ext = os.path.splitext(output_file_path)
        if base_path.endswith("_v2"):
            v2a_path = base_path + "a" + ext
        else:
            v2a_path = base_path + "_v2a" + ext

        print("\n📝 第一阶段：只使用证据 (v2a)")
        full_context_manager_a = FullContextManager(
            output_path=v2a_path,
            llm_config=answer_llm_config,
            figure_view=False
        )
        kept_count_a = full_context_manager_a.process_data_file(
            file_path=input_file_path,
            only_evidence=1,
            except_evidence=0,
            max_workers=args.max_workers
        )

        print("\n📝 第二阶段：排除证据 (v2b)")
        full_context_manager_b = FullContextManager(
            output_path=output_file_path,
            llm_config=answer_llm_config,
            figure_view=False
        )
        kept_count_b = full_context_manager_b.process_data_file(
            file_path=v2a_path,
            only_evidence=0,
            except_evidence=1,
            max_workers=args.max_workers
        )
        print(f" 验证统计：初始 {total_questions} → v2a {kept_count_a} → v2b {kept_count_b}")

        return output_file_path, kept_count_b

    # 单阶段执行
    full_context_manager = FullContextManager(
        output_path=output_file_path,
        llm_config=answer_llm_config,
        figure_view=False
    )

    kept_count = full_context_manager.process_data_file(
        file_path=input_file_path,
        only_evidence=only_evidence,
        except_evidence=except_evidence,
        max_workers=args.max_workers
    )

    return output_file_path, kept_count

