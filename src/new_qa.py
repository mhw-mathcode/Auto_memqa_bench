import json
import os
import re
import tiktoken
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.mcq_scoring import normalize_answer_candidates, score_mcq_prediction

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_MODEL = os.getenv("NEW_QA_MODEL", "Qwen/Qwen3-14B")

# 弃用 MAX_CLUSTER_JSON_CHARS，保留 Token 上限用于日志预警
DEFAULT_MAX_CLUSTER_TOKENS = int(os.getenv("MAX_CLUSTER_TOKENS", "8000"))
MAX_CLUSTER_TOKENS = DEFAULT_MAX_CLUSTER_TOKENS

BASE_URL = DEFAULT_BASE_URL
API_KEY = DEFAULT_API_KEY
MODEL = DEFAULT_MODEL

# 初始化 Tokenizer (cl100k_base 是 OpenAI 兼容接口的通用极速估算器)
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    print(f"!!! Tokenizer 加载失败: {e}")
    tokenizer = None
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

ABSTAIN_OPTION_TEXT = "F. Cannot infer the answer based on the given information."
ABSTAIN_QUESTIONS_PER_CATEGORY = 2
ABSTAIN_GENERATION_RETRIES = 6
ABSTAIN_VALIDATION_RETRIES = 3

CATEGORY_DESCRIPTIONS = {
    1: "Long-term Persona: stable identity, values, long-term preferences, or recurring behavioral patterns.",
    2: "Short-term State: immediate emotions, short-term needs, temporary goals, or situational mental states.",
    3: "Temporal: time order, sequence changes, timing relations, or replacement of old and new information.",
    4: "Plot-driven Event: specific events, experiences, actions, decisions, outcomes, and participant evaluations.",
    5: "Interpersonal Relationship: explicit concrete relationships between people such as family, colleagues, classmates, roommates, and similar roles.",
    6: "Fine-Grained Data: explicit numerical information such as counts, dates, ages, durations, quantities, or comparisons/calculations.",
}

ABSTAIN_VALIDATE_PROMPT = """
You are an intelligent assistant. Your task is to answer a multiple-choice question based only on the provided conversation history.

# RULES
1. Use only the conversation history. Do not use outside knowledge.
2. Choose exactly one option.
3. If the answer cannot be determined from the history, choose the dedicated Cannot infer option.
4. Do not explain your reasoning.
5. Your final output must be only the option letter in parentheses, for example: (B)

--- CONVERSATION HISTORY ---
{conversation_history}
--- END OF HISTORY ---

Question: {question}

Options:
{options_text}

Final answer:
"""

def gen_chat(prompt: str, temp=0.7) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            extra_body={"enable_thinking": False},
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"!!! 调用模型时发生错误: {e}")
        return ""

# ================= 核心处理类 =================

class UltimateMemoryRefiner:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.raw_data = []
        self.original_data = []  # 保存原始的完整数据结构

    def load_data(self):
        """加载数据并标记 episode_index"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果是单个 dict，转换为列表
        if isinstance(data, dict):
            data = [data]
        
        # 保存原始数据结构
        self.original_data = data
        
        # 提取所有 QA 并添加 qid（全局唯一ID）、episode_index 和 source_index
        qid = 1
        for idx, item in enumerate(data):
            if "qa" in item:
                for qa_item in item["qa"]:
                    # 如果原来没有 qid，生成一个
                    if "qid" not in qa_item:
                        qa_item["qid"] = qid
                        qid += 1
                    qa_item["episode_index"] = idx
                    qa_item["source_index"] = idx  # 标记来源item
                    self.raw_data.append(qa_item)
        
        print(f"--- [Step 1] 数据加载完成，共计 {len(self.raw_data)} 条原始 QA，来自 {len(data)} 个数据源 ---")

    def compact_cluster(self, cluster):
        """压缩字段以适应 LLM 输入"""
        sorted_cluster = sorted(cluster, key=lambda x: x.get("episode_index", 0))
        compact = []

        for item in sorted_cluster:
            compact_item = {
                "qid": item.get("qid"),
                "episode_index": item.get("episode_index"),
                "character": item.get("character"),
                "question": item.get("question"),
                "option": item.get("option"),
                "answer": item.get("answer")
            }
            compact.append(compact_item)

        return compact

    def estimate_cluster_size(self, cluster):
        """使用 Tokenizer 精确计算 Context 占用"""
        compact_json = json.dumps(self.compact_cluster(cluster), ensure_ascii=False)
        if tokenizer:
            return len(tokenizer.encode(compact_json))
        else:
            # 降级方案：如果是纯中文+JSON结构，简单按 2 字符 = 1 Token 估算
            return len(compact_json) // 2

    def format_category_counts(self, cluster):
        counts = {i: 0 for i in range(1, 7)}
        for item in cluster:
            try:
                category = int(item.get("category", 0))
            except (TypeError, ValueError):
                category = 0
            if category in counts:
                counts[category] += 1
        return " ".join(f"{i}:{counts[i]}" for i in range(1, 7))

    def _resolve_speaker_name(self, raw_speaker: Any, speaker_map: Dict[str, str]) -> str:
        if isinstance(raw_speaker, str):
            if raw_speaker in speaker_map:
                return speaker_map[raw_speaker]
            return raw_speaker

        if isinstance(raw_speaker, int):
            key = f"speaker_{raw_speaker}"
            return speaker_map.get(key, str(raw_speaker))

        return str(raw_speaker)

    def _format_conversation_for_prompt(self, conversation_item: Dict[str, Any]) -> str:
        if not isinstance(conversation_item, dict):
            return ""

        history: List[str] = []
        speaker_map: Dict[str, str] = {}

        raw_speakers = conversation_item.get("speakers")
        if isinstance(raw_speakers, list):
            for idx, speaker_name in enumerate(raw_speakers, start=1):
                if isinstance(speaker_name, str) and speaker_name.strip():
                    speaker_map[f"speaker_{idx}"] = speaker_name.strip()

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
            timestamp = conversation_item.get(f"{session_key}_date_time") or conversation_item.get(
                f"{session_key}_time",
                "Unknown time",
            )
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
                history.append(f"{speaker_name}: {chat.get('text', '')}")

        return "\n".join(history).strip()

    def _normalize_option_lines(self, options: List[Any]) -> List[str]:
        normalized: List[str] = []
        for index, option in enumerate(options or []):
            letter = chr(ord("A") + index)
            option_text = str(option).strip()
            if option_text[:3].startswith(f"{letter}.") or (len(option_text) >= 3 and option_text[1:3] == ". "):
                normalized.append(option_text)
            else:
                normalized.append(f"{letter}. {option_text}")
        return normalized

    def _extract_question_stem(self, question: str, option_lines: List[str]) -> str:
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

    def _build_abstain_generation_prompt(
        self,
        conversation_item: Dict[str, Any],
        source_questions: List[Dict[str, Any]],
    ) -> str:
        category_text = "\n".join(
            f"- Category {category}: {description}"
            for category, description in CATEGORY_DESCRIPTIONS.items()
        )
        conversation_history = self._format_conversation_for_prompt(conversation_item)

        return f"""
You are an expert dataset construction system generating adversarial "abstain" (unanswerable) questions for long-context LLM memory evaluation.

You will be given a standalone conversation transcript and a list of existing positive QA items derived from that same text.

Your task is to generate exactly {ABSTAIN_QUESTIONS_PER_CATEGORY * 6} new multiple-choice questions that are highly plausible but intentionally unanswerable based strictly on the provided conversation.

### Perturbation Strategies for Abstain Questions:
To make the questions challenging and deceptive, use a diverse mix of the following strategies, heavily drawing inspiration from the `source_questions`:
1. **Entity Swapping**: Take a real event from the text/source_questions, but swap the character/subject. (e.g., If Person A did X, ask why Person B did X).
2. **Out-of-Scope Detail**: Take a real event, but ask for a hyper-specific detail not mentioned. (e.g., They drove a car -> What was the license plate?).
3. **False Premise**: Embed a fabricated assumption into the question stem. (e.g., "Why was Person A crying when they left?" - when they actually left calmly).
4. **The Phantom Entity**: Introduce a plausible person, object, or concept that fits the conversational universe but is completely absent from this specific text chunk.
5. **Temporal/Causal Extension**: Ask about the aftermath, preceding events, or deeper motivations that are not explicitly stated in this exact transcript.

### Strict Requirements:
1. Generate exactly {ABSTAIN_QUESTIONS_PER_CATEGORY} questions for each category 1-6.
2. The questions MUST sound highly realistic and native to the conversation's context. Do not ask absurd or obviously random questions.
3. Every question must remain genuinely unanswerable using ONLY the provided conversation. The correct answer must always be the abstain option.
4. Options A-E must be plausible distractor answers. They should sound like things that *could* be true in this context, making it tempting for a hallucinating LLM to choose them.
5. Option F must be exactly: {ABSTAIN_OPTION_TEXT}
6. The answer field must be exactly: {ABSTAIN_OPTION_TEXT}
7. The label field must be exactly: abstain
8. Keep the question stem completely separate from the option list.
9. Return ONLY a valid JSON array. Do not wrap it in Markdown formatting (no ```json).

Category definitions:
{category_text}

Reference QA (Mutate these using the Perturbation Strategies to create paired abstain questions):
{json.dumps(source_questions, ensure_ascii=False, indent=2)}

Conversation Context:
{conversation_history}

Output JSON array schema:
[
  {{
    "question": "Question stem only, containing a deceptive premise or swapped entity",
    "option": [
      "A. [Plausible but unsupported distractor]",
      "B. [Plausible but unsupported distractor]",
      "C. [Plausible but unsupported distractor]",
      "D. [Plausible but unsupported distractor]",
      "E. [Plausible but unsupported distractor]",
      "F. Cannot infer the answer based on the given information."
    ],
    "answer": "F",
    "reasoning": "Identify which Perturbation Strategy was used, and explicitly explain why the conversation lacks the required evidence.",
    "label": "abstain",
    "category": 1
  }}
]
"""
    
    def _normalize_generated_abstain_questions(
        self,
        generated_questions: Any,
        source_index: int,
    ) -> List[Dict[str, Any]]:
        if not isinstance(generated_questions, list):
            return []

        normalized_questions: List[Dict[str, Any]] = []
        category_counts = {category: 0 for category in range(1, 7)}

        for item in generated_questions:
            if not isinstance(item, dict):
                continue

            try:
                category = int(item.get("category", 0))
            except (TypeError, ValueError):
                continue

            if category not in category_counts:
                continue
            if category_counts[category] >= ABSTAIN_QUESTIONS_PER_CATEGORY:
                continue

            question_text = str(item.get("question", "")).strip()
            if not question_text:
                continue

            raw_options = item.get("option", [])
            if not isinstance(raw_options, list):
                continue

            non_abstain_options: List[str] = []
            for option in raw_options:
                option_text = str(option or "").strip()
                if not option_text:
                    continue
                if option_text.startswith("F."):
                    continue
                non_abstain_options.append(option_text)

            if len(non_abstain_options) < 5:
                continue

            normalized_questions.append(
                {
                    "question": question_text,
                    "option": non_abstain_options[:5] + [ABSTAIN_OPTION_TEXT],
                    "answer": ABSTAIN_OPTION_TEXT,
                    "reasoning": str(item.get("reasoning", "")).strip(),
                    "label": "abstain",
                    "category": category,
                    "character": str(item.get("character", "Unspecified") or "Unspecified"),
                    "original_qa": item.get("original_qa", []) if isinstance(item.get("original_qa"), list) else [],
                    "source_index": source_index,
                }
            )
            category_counts[category] += 1

        return normalized_questions

    def _build_abstain_validation_prompt(
        self,
        conversation_item: Dict[str, Any],
        question_item: Dict[str, Any],
    ) -> str:
        option_lines = self._normalize_option_lines(question_item.get("option", []))
        question_stem = self._extract_question_stem(question_item.get("question", ""), option_lines)
        conversation_history = self._format_conversation_for_prompt(conversation_item)
        return ABSTAIN_VALIDATE_PROMPT.format(
            conversation_history=conversation_history,
            question=question_stem or str(question_item.get("question", "")).strip(),
            options_text="\n".join(option_lines),
        )

    def _validate_abstain_question_on_conversation(
        self,
        question_item: Dict[str, Any],
        conversation_item: Dict[str, Any],
        target_name: str,
    ) -> bool:
        prompt = self._build_abstain_validation_prompt(conversation_item, question_item)
        response = ""
        for _ in range(ABSTAIN_VALIDATION_RETRIES):
            response = gen_chat(prompt, temp=0.0)
            if response:
                break

        score_result = score_mcq_prediction(
            response,
            normalize_answer_candidates(None, question_item.get("answer", ABSTAIN_OPTION_TEXT)),
        )

        if not score_result.get("is_correct", False):
            print(
                f"      × 弃权题验证失败 [{target_name}]: {question_item.get('question', '')[:60]}... -> {response[:120]}"
            )
            return False
        return True

    def _build_merged_conversation(self):
        merged_conversation = OrderedDict()
        global_session_idx = 1

        speaker_list = []
        speaker_seen = set()
        sessions = []

        speaker_key_pattern = re.compile(r"^speaker_(\d+)$")
        speaker_value_pattern = re.compile(r"^speaker_(\d+)$")
        session_key_pattern = re.compile(r"^session_(\d+)$")
        session_time_pattern = re.compile(r"^session_(\d+)_(date_time|time)$")

        def add_speaker(name):
            if not isinstance(name, str):
                return
            clean_name = name.strip()
            if not clean_name or clean_name in speaker_seen:
                return
            speaker_seen.add(clean_name)
            speaker_list.append(clean_name)

        for item in self.original_data:
            conversation = item.get("conversation", {})
            if not isinstance(conversation, dict):
                continue

            indexed_speakers = []
            session_contents = {}
            session_meta = defaultdict(dict)

            raw_speakers = conversation.get("speakers")
            if isinstance(raw_speakers, list):
                for idx, speaker_name in enumerate(raw_speakers, start=1):
                    if isinstance(speaker_name, str) and speaker_name.strip():
                        indexed_speakers.append((idx, speaker_name.strip()))

            for key, content in conversation.items():
                key_str = str(key)

                key_match = speaker_key_pattern.fullmatch(key_str)
                if key_match:
                    if isinstance(content, str) and content.strip():
                        indexed_speakers.append((int(key_match.group(1)), content.strip()))
                    continue

                if isinstance(content, str):
                    value_match = speaker_value_pattern.fullmatch(content)
                    if value_match and isinstance(key, str) and key.strip():
                        indexed_speakers.append((int(value_match.group(1)), key.strip()))
                        continue

                session_match = session_key_pattern.fullmatch(key_str)
                if session_match:
                    session_idx = int(session_match.group(1))
                    session_contents[session_idx] = content
                    continue

                session_time_match = session_time_pattern.fullmatch(key_str)
                if session_time_match:
                    session_idx = int(session_time_match.group(1))
                    time_suffix = session_time_match.group(2)
                    session_meta[session_idx][time_suffix] = content

            for _, speaker_name in sorted(indexed_speakers, key=lambda x: x[0]):
                add_speaker(speaker_name)

            for session_idx in sorted(session_contents.keys()):
                sessions.append(
                    {
                        "content": session_contents[session_idx],
                        "meta": session_meta.get(session_idx, {}),
                    }
                )

        if speaker_list:
            merged_conversation["speakers"] = speaker_list

        for session_item in sessions:
            new_session_key = f"session_{global_session_idx}"
            meta = session_item.get("meta", {})

            if "date_time" in meta:
                merged_conversation[f"{new_session_key}_date_time"] = meta["date_time"]
            if "time" in meta:
                merged_conversation[f"{new_session_key}_time"] = meta["time"]

            merged_conversation[new_session_key] = session_item.get("content", [])
            global_session_idx += 1

        return merged_conversation

    def _generate_validated_abstain_questions(self) -> List[Dict[str, Any]]:
        print(f"\n>>> 开始生成弃权问题并进行跨对话验证")
        valid_questions: List[Dict[str, Any]] = []
        merged_conversation = self._build_merged_conversation()

        for source_index, item in enumerate(self.original_data):
            conversation = item.get("conversation", {})
            if not isinstance(conversation, dict) or not conversation:
                continue

            source_questions = []
            for qa in item.get("qa", []):
                if not isinstance(qa, dict):
                    continue
                source_questions.append(
                    {
                        "question": qa.get("question"),
                        "category": qa.get("category"),
                        "character": qa.get("character"),
                    }
                )

            print(
                f"\n>>> 正在为原始对话 {source_index + 1}/{len(self.original_data)} 生成弃权题..."
            )

            candidates: Optional[List[Dict[str, Any]]] = None
            for attempt in range(ABSTAIN_GENERATION_RETRIES):
                prompt = self._build_abstain_generation_prompt(conversation, source_questions)
                if attempt > 0:
                    prompt += "\n\nReturn JSON array only. Do not include markdown fences or any extra commentary."
                response = gen_chat(prompt, temp=0.4)
                parsed = self.extract_json(response)
                normalized = self._normalize_generated_abstain_questions(parsed, source_index)
                if normalized:
                    candidates = normalized
                    break

            if not candidates:
                print(f"      × 原始对话 {source_index + 1} 的弃权题生成失败，跳过")
                continue

            print(f"      - 初始生成 {len(candidates)} 道弃权题候选")

            for candidate in candidates:
                is_valid = True

                if not self._validate_abstain_question_on_conversation(
                    candidate,
                    conversation,
                    f"source_{source_index + 1}",
                ):
                    continue

                for target_index, target_item in enumerate(self.original_data):
                    if target_index == source_index:
                        continue
                    target_conversation = target_item.get("conversation", {})
                    if not isinstance(target_conversation, dict) or not target_conversation:
                        continue
                    if not self._validate_abstain_question_on_conversation(
                        candidate,
                        target_conversation,
                        f"other_{target_index + 1}",
                    ):
                        is_valid = False
                        break

                if is_valid and not self._validate_abstain_question_on_conversation(
                    candidate,
                    merged_conversation,
                    "merged_all",
                ):
                    is_valid = False

                if is_valid:
                    valid_questions.append(candidate)
                    print(f"      √ 弃权题通过验证: {candidate.get('question', '')[:60]}...")

        print(f"\n>>> 弃权题验证完成，保留 {len(valid_questions)} 道题")
        return valid_questions

    def build_refine_prompt(self, subject, final_chunk):
        """最终生成 Prompt"""
        compact_chunk = self.compact_cluster(final_chunk)
        return f"""
# Task Description
You are a high-difficulty long-context logical evaluation and question generation system.

You will be given a set of original QA data about a specific entity {subject}.
All data is ordered temporally. 

Your task is to generate a set of challenging logical evaluation questions that assess:
- Long-term memory update ability
- Cross-chunk logical integration ability

All questions must be derived strictly from the provided data.
They must require temporal reasoning, state comparison, conflict resolution, and multi-evidence integration, rather than surface-level paraphrasing.

# Dimension 1: Memory Update

When an entity’s state S is A at time t1 and is later explicitly or implicitly updated to B at time t2, you must construct questions around this state transition, including but not limited to:

1. Causal (Why-based)
   - Ask why state A became invalid
   - Ask which explicitly mentioned events, decisions, or conditions caused or enabled the transition to state B

2. Boundary / Timing
   - Ask for the specific point at which the old state was irreversibly overturned
   - This point is not necessarily the first anomaly, but when the update became final

3. Final-State Verification
   - Explicitly include early state A as a strong distractor in the question
   - Ask about the entity’s final state at the end of the full data sequence
   - Designed to detect reliance on outdated memory

For each subtype above:
If multiple updates, reversals, or influencing factors exist in the data, you should generate multiple questions from different analytical perspectives, not just a single question.

# Dimension 2: Fact Extraction (Multiple Conversations)

You must actively identify related facts or patterns distributed across multiple non-adjacent semantic chunks and construct questions that require joint reasoning across them, including but not limited to:

1. Set Construction / Inductive Aggregation
   - Ask the model to enumerate all moments or behaviors matching an abstract property
   - The property must not be explicitly summarized in any single chunk

2. Trend / Frequency Analysis
   - Ask whether a behavior, attitude, or decision pattern changes over time
   - Changes may involve escalation, attenuation, or structural shifts

3. Multi-Chunk Dependency (Fragment Assembly)
   - The correct answer must depend on information from multiple events
   - Missing any chunk should lead to an incomplete or incorrect answer

The same cross-chunk pattern may be queried from multiple angles, and multiple questions should be generated when appropriate.

# Mandatory Constraints

1. No External Knowledge
   - All questions, options, and answers must be based exclusively on the provided data
   - No background knowledge, common sense completion, or assumptions allowed

2. No Meta-Context References
   - Do not mention “episodes”, “chapters”, “earlier text”, or similar notions
   - Treat the input strictly as a complete and standalone data sequence

3. Implicit Reasoning Requirement
   - Questions must implicitly require deep reasoning
   - The model is NOT required to expose reasoning in its answer

# Option Construction Constraints

1. Each question must provide multiple options (e.g., A / B / C / D / E).

2. Incorrect options must be plausible but unambiguously wrong:
   - They must be partially supported by the text
   - But invalidated by later updates or cross-chunk evidence

3. Common sources of incorrect options include:
   - Reliance on early states while ignoring updates
   - Use of a single chunk while ignoring others
   - Confusing correlation with causation

4. The correct option must not be obtainable via keyword matching alone;
   it must require temporal ordering, state comparison, or evidence integration.

# Answer Field Constraints

1. The `answer` field must contain ONLY the final conclusion:
   - e.g., the correct option letter, final state, key timestamp, or final set
   - No explanation, justification, or reasoning text

2. The answer must be unique and deterministic.
   Ambiguous or multi-valid answers are not allowed.

# Explanation / Reasoning Field Constraints

1. The `reasoning` (or `explanation`) field must document:
   - The logical basis for the correct answer
   - How memory updates occurred
   - How information from multiple chunks was integrated
   - How conflicts were identified and resolved

2. This field exists for:
   - Annotation quality control
   - Debugging and error analysis
   - Benchmark interpretability

3. Every key claim in the reasoning must be traceable
   to a specific semantic chunk or time point.

4. "category": The question type label, which must remain consistent with the category of the selected original question(s) from which this item is constructed.

5. "original_qa": A list of question corresponding to the original QA items that were selected, referenced, or integrated to construct the current question.

# Input Semantic Cluster
{json.dumps(compact_chunk, ensure_ascii=False, indent=4)}

# Output Format (JSON)
[
    {{
        "question": "Complete question text with necessary distractors",
        "option": ["A ...", "B ...", "C ...", "D ...", "E ..."],
        "answer": "Final conclusion or correct option label",
        "reasoning": "Detailed explanation of how the answer follows from the full data",
        "label": "It must be filled in with either 'Memory Update' or 'Fact Extraction (Multiple Conversations)', and absolutely not any other value.",
        "category": 1,
        "is_conflict": true / false,
        "original_qa": ["Based on her interactions...", "What is Sandy's immediate reaction..."]
    }}
]
"""

    def process(self):
        """
        处理流程：对所有问题进行全局分析和重构（取消聚类，直接传入全量数据）
        """
        self.load_data()
        
        print(f"\n>>> 开始全局逻辑分析和重构处理")
        
        # 按角色分组（全局）
        subject_buckets = defaultdict(list)
        for qa in self.raw_data:
            subject_buckets[qa.get("character", "Unknown")].append(qa)
        
        # 存储所有重构后的新问题
        all_refined_qa = []
        
        for subject, subject_qa_list in subject_buckets.items():
            if subject == "Unknown": 
                continue
            
            print(f"\n>>> 正在处理角色【{subject}】(共 {len(subject_qa_list)} 个问题)...")
            
            # 直接将该角色的所有问题作为一个大 Group
            group = subject_qa_list
            
            if len(group) < 2:
                print(f"      - 角色 {subject} 问题数少于 2，跳过重构")
                continue
            
            # 评估 Token 占用（用于日志预警）
            token_usage = self.estimate_cluster_size(group)
            print(f"      - 整体作为 1 个数据簇传入模型。预估 Token 占用: ~{token_usage}")
            if token_usage > MAX_CLUSTER_TOKENS:
                print(f"      ! 提醒：当前角色问题总 Token (~{token_usage}) 已超过设定的参考阈值 ({MAX_CLUSTER_TOKENS})。")
            
            category_summary = self.format_category_counts(group)
            print(f"      - 类别统计(1-6): {category_summary}")

            # 生成重构后的问题
            refined = None
            max_retries = 10

            for attempt in range(max_retries):
                current_prompt = self.build_refine_prompt(subject, group)
                if attempt > 0:
                    current_prompt += "\n\n**重要修正**：请直接输出 JSON 数组格式（以 [ 开头，以 ] 结束），严禁包含任何 Markdown 代码块标签、前言、解释或结尾总结。"
                
                response = gen_chat(current_prompt)
                refined = self.extract_json(response)
                
                if refined:
                    all_refined_qa.extend(refined)
                    print(f"      √ 角色 {subject} 重构成功 (尝试 {attempt+1} 次): 生成了 {len(refined)} 道新题")
                    break
                else:
                    print(f"      ! 角色 {subject} 第 {attempt+1} 次解析失败，正在重试...")
            
            if not refined:
                print(f"      × 角色 {subject} 在 {max_retries} 次尝试后全数失败，跳过")
        
        # 弃权题生成已移除：仅保留重构生成的新题。
        self.save(list(all_refined_qa))

    def extract_json(self, text):
        try:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            return json.loads(match.group()) if match else None
        except: return None

    def save(self, refined_qa_list):
        """
        保存重构后的数据，合并新问题和原始问题
        """
        print(f"\n>>> 开始合并新问题和原始问题...")
        
        # 1. 处理新问题：标签映射，收集要删除的原始问题文本
        remove_questions = set()
        processed_new_qa = []
        
        for new_q in refined_qa_list:
            if "session" in new_q.get("question", ""):
                continue
            
            if new_q.get("label") == "Memory Update":
                print(f"  [记忆更新] {new_q.get('question', '')[:50]}...")
                for q_text in new_q.get("original_qa", []):
                    if isinstance(q_text, str) and q_text:
                        remove_questions.add(q_text)
            elif new_q.get("label") == "Fact Extraction (Multiple Conversations)":
                print(f"  [事实提取（多对话）] {new_q.get('question', '')[:50]}...")
            
            processed_new_qa.append(new_q)
        
        # 2. 收集所有保留的原始问题（深拷贝）
        import copy
        all_qa = []
        for original_q in self.raw_data:
            if "session" in original_q.get("question", ""):
                continue
            
            original_question = original_q.get("question", "")
            if original_question and any(q_text in original_question for q_text in remove_questions):
                print(f"  [删除原题] {original_question[:50]}...")
                continue
            
            q_copy = copy.deepcopy(original_q)
            q_copy.pop("episode_index", None)
            q_copy.pop("source_index", None)
            all_qa.append(q_copy)
        
        # 3. 标记所有新生成的题目，以便后续步骤识别
        for new_q in processed_new_qa:
            new_q["is_generated_qa"] = True
        
        # 4. 合并新问题和原始问题
        all_qa.extend(processed_new_qa)
        
        # 5. 重新分配 qid
        for idx, q in enumerate(all_qa, start=1):
            q["qid"] = idx
        
        # 6. 合并所有 conversation
        merged_conversation = self._build_merged_conversation()
        
        # 7. 构建最终输出结构
        final_data = [
            {
                "qa": all_qa,
                "conversation": merged_conversation
            }
        ]
        
        # 8. 保存文件
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)


def new_qa_main(input_file_path: str, output_file_path: str,
                api_key: str = DEFAULT_API_KEY, base_url: str = DEFAULT_BASE_URL,
                model: str = DEFAULT_MODEL) -> str:
    """
    步骤 3: 问答精炼重构（全量长上下文处理模式）
    """
    print("\n" + "="*60)
    print("🔄 步骤 3: 问答精炼重构（全量上下文逻辑增强）")
    print("="*60)
    print(f"📥 输入文件: {input_file_path}")
    
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    total_questions = sum(len(item.get("qa", [])) for item in input_data)
    print(f"--- 步骤 3 开始处理：共 {total_questions} 个问题 ---")
    
    global client, MODEL, BASE_URL, API_KEY
    API_KEY = api_key
    BASE_URL = base_url
    MODEL = model
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    processor = UltimateMemoryRefiner(
        input_file=input_file_path, 
        output_file=output_file_path
    )
    processor.process()
    
    return output_file_path

if __name__ == "__main__":
    # 示例用法
    processor = UltimateMemoryRefiner(
        input_file="./An-Enemy-of-the-People_merged.json", 
        output_file="An-Enemy-of-the-People_new2.json"
    )
    processor.process()

