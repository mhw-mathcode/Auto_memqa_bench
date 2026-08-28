import json
import os
import re
import threading
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm
from src.utils import (
    align_evidence_dialogues,
    build_dialogue_index,
    count_qa_items,
    load_json_file,
    normalize_dataset_records,
    normalize_evidence_dialogues,
    normalize_reasoning_steps,
    write_json_file,
)
from src.pipeline_utils import log_event, log_subsection, print_log_section, print_kv
from src.mcq_scoring import (
    MULTIPLE_SELECT,
    ORDERING,
    SINGLE_CHOICE,
    normalize_answer_candidates,
    normalize_question_type,
    parse_question_answer,
)

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_MODEL = os.getenv("NEW_QA_MODEL", "Qwen/Qwen3-14B")

BASE_URL = DEFAULT_BASE_URL
API_KEY = DEFAULT_API_KEY
MODEL = DEFAULT_MODEL

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
_client_local = threading.local()

VALID_LABELS = {
    "Fact Extraction (Single Dialogue)",
    "Fact Extraction (Multiple Dialogues)",
    "Memory Update",
    "Multi-hop",
    "Abstain",
}


def normalize_question_reference(value: Any) -> str:
    """Normalize a question string for matching generated `original_qa` references."""
    text = str(value or "").strip()
    if not text:
        return ""
    # Generated/final questions may contain appended options and answer instructions.
    text = re.split(r"\n\s*A[\.\)]\s+", text, maxsplit=1)[0]
    text = re.split(r"\n\s*Please provide\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return re.sub(r"\s+", " ", text).strip().casefold()


def remove_legacy_number_fields(item: Dict[str, Any]) -> None:
    """移除历史版本遗留的题目编号字段，避免继续写入新版本数据。"""
    legacy_key = "q" + "id"
    item.pop(legacy_key, None)


def gen_chat(prompt: str, temp=0.7) -> str:
    client_key = (API_KEY, BASE_URL)
    if getattr(_client_local, "client_key", None) != client_key:
        _client_local.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        _client_local.client_key = client_key
    active_client = _client_local.client
    try:
        resp = active_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        error_text = str(e).lower()
        if "missing_required_parameter" in error_text or (
            "one of \"input\"" in error_text and "prompt" in error_text
        ):
            try:
                resp2 = active_client.responses.create(
                    model=MODEL,
                    input=prompt,
                    temperature=temp,
                )
                return (getattr(resp2, "output_text", "") or "").strip()
            except Exception as e2:
                log_event("refine_llm_call", status="failed", error=e2)
                return ""

        log_event("refine_llm_call", status="failed", error=e)
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
        data = normalize_dataset_records(load_json_file(self.input_file))
        
        # 保存原始数据结构
        self.original_data = data
        
        # 提取所有 QA，并记录来源位置供本阶段内部排序/合并使用。
        for idx, item in enumerate(data):
            if "qa" in item:
                for qa_item in item["qa"]:
                    qa_item["episode_index"] = idx
                    qa_item["source_index"] = idx  # 标记来源item
                    self.raw_data.append(qa_item)
        
        log_event(
            "refine_load_data",
            status="success",
            raw_qa=len(self.raw_data),
            records=len(data),
        )

    def compact_cluster(self, cluster):
        """压缩字段以适应 LLM 输入，同时保留可复用的原始证据。"""
        sorted_cluster = sorted(cluster, key=lambda x: x.get("episode_index", 0))
        compact = []

        for item in sorted_cluster:
            evidence_dialogues = normalize_evidence_dialogues(item.get("evidence_dialogues", []))
            evidence_ids = [
                str(ev.get("id"))
                for ev in evidence_dialogues
                if isinstance(ev, dict) and ev.get("id")
            ]
            compact_item = {
                "episode_index": item.get("episode_index"),
                "character": item.get("character"),
                "category": item.get("category"),
                "label": item.get("label"),
                "question": item.get("question"),
                "option": item.get("option"),
                "answer": item.get("answer"),
                "evidence_dialogues": evidence_dialogues,
                "reasoning_steps": normalize_reasoning_steps(item.get("reasoning_steps", []), evidence_ids),
            }
            compact.append(compact_item)

        return compact

    def format_category_counts(self, cluster):
        counts = {i: 0 for i in range(1, 8)}
        for item in cluster:
            try:
                category = int(item.get("category", 0))
            except (TypeError, ValueError):
                category = 0
            if category in counts:
                counts[category] += 1
        return " ".join(f"{i}:{counts[i]}" for i in range(1, 8))

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

# Dimension 2: Fact Extraction (Multiple Dialogues)

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

4. Natural Trace-Style Question Design
   - Write questions in the same natural style as the trace datasets: mention the concrete person, object, event, relationship, or state transition being tested.
   - Do NOT use abstract template stems such as "Which option correctly combines two separate details?", "Which statement preserves the paired details?", "Which option matches the two passages?", or "Which option correctly combines two separate moments?".
   - For literary or narrative data, the question should sound like it is asking about the story itself, not about the dataset construction method.
   - Good stems look like:
     * "How did Jonathan's first impression of the old man connect with the later description of his face?"
     * "What changed in Mina's understanding after the earlier warning was confirmed by later events?"
     * "Which statement best matches how Van Helsing and Arthur handled Lucy's condition across the two scenes?"
     * "What promise or plan was still unresolved after the group left the house?"
   - Options should be natural answer statements. Do not write options as quote containers such as "One passage says ...; another says ..."; verbatim source text belongs in evidence_dialogues.

# Option Construction Constraints

1. Each question must provide multiple options (e.g., A / B / C / D / E).

2. Incorrect options must be plausible but unambiguously wrong:
   - They must be partially supported by the text
   - But invalidated by later updates or cross-chunk evidence

3. Concision, length, and style balance are hard constraints:
   - Options must be concise natural answer statements, not evidence dumps.
   - Target each option at roughly 10-24 English words when possible; do not exceed 35 words unless the fact itself requires it.
   - Do not join two copied evidence snippets with a semicolon. If two facts must appear together, write one natural sentence using "and", "while", or a short causal/temporal connector.
   - Do not pad options with copied evidence just to match length.
   - At least one distractor must be slightly longer than the correct option and at least one must be slightly shorter.
   - The correct option must be neither the longest nor the shortest.
   - All options must have comparable clause count, specificity, named-entity density, temporal precision, and grammatical structure.
   - The correct option must not be the only complete, detailed, or multi-condition statement.

4. Common sources of incorrect options include:
   - Reliance on early states while ignoring updates
   - Use of a single chunk while ignoring others
   - Confusing correlation with causation
   - A minimal factual perturbation that changes exactly one decisive actor, order, location, quantity, motive, trigger, ownership, or final state
   - A true detail attached to the wrong person or event
   - An over-specific detail that is plausible but unsupported by the evidence

5. Do not make all five options near-identical copies with only one changed word. Distractors should be close enough to be plausible, but varied enough to read like real competing answers.

6. Never use absurd, irrelevant, generically reckless, or extreme behavior merely to make an option wrong. Avoid giveaway words such as "always", "never", "completely", "all people", or "immediately discard" unless the correct option uses the same style.

7. The correct option must not be obtainable via keyword matching alone;
   it must require temporal ordering, state comparison, or evidence integration.

8. Perform a no-context leakage check before returning JSON: hide the source QA and evidence, then inspect only the question and options. If length, detail, fluency, common sense, or wording style makes the correct option stand out, rewrite all distractors.

# Answer Field Constraints

1. Include `question_type` using exactly one of `single_choice`, `multiple_choice`, or `ordering`.
2. For `single_choice`, `answer` must be one option letter such as `A`.
3. For `multiple_choice`, `answer` must contain every correct letter in parentheses, such as `(A,E)`.
4. For `ordering`, `answer` must contain every option in the correct sequence, such as `(B,A,D,C)`.
5. For `multiple_choice` and `ordering`, option F is not added automatically. If F is explicitly provided as a normal option, it may be included with other correct options.
6. The answer must be unique and deterministic, with no explanation in the `answer` field.

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

# Source QA / Evidence / Reasoning Steps Constraints (Mandatory)

1. Every generated item MUST include `evidence_dialogues` and `reasoning_steps` fields.
2. `original_qa` is the binding source list for the reconstructed question:
   - It MUST contain the exact question text of every source QA item you used.
   - Do not summarize or rename source questions.
   - Downstream code will automatically concatenate the evidence_dialogues from these referenced source QA items.
3. `evidence_dialogues` in your output is only a draft and will be overwritten by deterministic source-evidence merging.
   Still, it must NOT be invented. You may only copy evidence items from the referenced source QA items.
4. `reasoning_steps` should be newly written for the reconstructed question, but every step must be supportable by the referenced source QA evidence.
5. Each copied evidence item must preserve the exact original `utterance` and `dia_id`; do not paraphrase, shorten, merge, or rewrite evidence text.
6. `evidence_dialogues` must be a non-empty JSON array, and each item should follow:
    - "id": "E1", "E2", ...
    - "speaker": speaker name or null
    - "utterance": exact evidence text
    - "dia_id": exact dialogue id copied from the source evidence item
7. `reasoning_steps` must be a non-empty JSON array, and each item should follow:
    - "step": integer starting from 1
    - "inference": one atomic reasoning statement
    - "based_on": list of evidence ids, e.g. ["E1", "E2"]
8. Do NOT output `reasoning_steps` as a plain string.
9. Keep all fields JSON-serializable and valid.
10. If you cannot support a reconstructed question by referencing exact source QA items in `original_qa`, do not generate that question.

# Input Semantic Cluster
{json.dumps(compact_chunk, ensure_ascii=False, indent=4)}

# Output Format (JSON)
[
    {{
        "character": "",
        "question_type": "single_choice",
        "question": "Complete question text with necessary distractors",
        "option": ["A ...", "B ...", "C ...", "D ...", "E ..."],
        "answer": "A",
        "evidence_dialogues": [
            {{"id": "E1", "speaker": "Name or null", "utterance": "Exact supporting text", "dia_id": "D1:23"}}
        ],
        "reasoning_steps": [
            {{"step": 1, "inference": "Atomic inference here", "based_on": ["E1"]}}
        ],
        "label": "Use exactly one of: Fact Extraction (Single Dialogue), Fact Extraction (Multiple Dialogues), Memory Update, Multi-hop, Abstain.",
        "category": 1,
        "original_qa": ["Based on her interactions...", "What is Sandy's immediate reaction..."]
    }}
]
"""

    def _normalize_option_field(self, option_value: Any) -> List[str]:
        """将 option 字段统一成 A./B./... 列表。"""
        if isinstance(option_value, dict):
            options: List[str] = []
            for letter in ["A", "B", "C", "D", "E", "F"]:
                for key, value in option_value.items():
                    if str(key).strip().upper() == letter:
                        body = str(value or "").strip()
                        options.append(f"{letter}. {body}" if body and not body.upper().startswith(f"{letter}.") else body)
                        break
            return options

        if isinstance(option_value, list):
            options = [str(option or "").strip() for option in option_value if str(option or "").strip()]
        elif option_value not in (None, ""):
            options = [str(option_value).strip()]
        else:
            options = []

        normalized: List[str] = []
        for idx, option in enumerate(options):
            letter = chr(ord("A") + idx)
            if re.match(r"^[A-Fa-f][\.．\)]\s*", option):
                normalized.append(option)
            else:
                normalized.append(f"{letter}. {option}")
        return normalized

    def _normalize_answer_letter(
        self,
        answer_value: Any,
        options: List[str],
        question_type: Any = None,
    ) -> str:
        """Normalize answers without discarding multi-select or ordering data."""
        normalized_type = normalize_question_type(question_type)
        answer_candidate = normalize_answer_candidates(None, answer_value)[0]
        sequence, malformed = parse_question_answer(answer_candidate, normalized_type)
        if sequence and not malformed:
            if normalized_type == SINGLE_CHOICE:
                return sequence[0]
            if normalized_type == MULTIPLE_SELECT:
                sequence = sorted(sequence)
            return f"({','.join(sequence)})"

        answer = str(answer_candidate or "").strip()
        match = re.match(r"^\(?\s*([A-Fa-f])\s*\)?(?:[\.．\)]\s*)?$", answer)
        if match:
            return match.group(1).upper()
        prefix_match = re.match(r"^\(?\s*([A-Fa-f])\s*\)?[\.．\)]\s+", answer)
        if prefix_match:
            return prefix_match.group(1).upper()

        clean_answer = re.sub(r"^[A-Fa-f][\.．\)]\s*", "", answer).strip()
        for idx, option in enumerate(options):
            option_body = re.sub(r"^[A-Fa-f][\.．\)]\s*", "", str(option or "")).strip()
            if clean_answer and option_body and clean_answer == option_body:
                return chr(ord("A") + idx)
        return answer

    def _normalize_label(self, label_value: Any) -> str:
        """统一 label 命名，避免历史 prompt 写法污染新版本数据。"""
        label = str(label_value or "").strip()
        if label == "Fact Extraction (Multiple Conversations)":
            return "Fact Extraction (Multiple Dialogues)"
        return label

    def _normalize_refined_questions(self, refined: Any) -> List[Dict[str, Any]]:
        """兜底补齐重构题字段，生成统一 schema。"""
        if not isinstance(refined, list):
            return []

        normalized_items: List[Dict[str, Any]] = []
        for item in refined:
            if not isinstance(item, dict):
                continue

            normalized = dict(item)
            question_type = normalize_question_type(normalized.get("question_type"))
            normalized["question_type"] = question_type
            normalized["option"] = self._normalize_option_field(normalized.get("option", []))
            normalized["answer"] = self._normalize_answer_letter(
                normalized.get("answer", ""),
                normalized["option"],
                question_type,
            )
            normalized["label"] = self._normalize_label(normalized.get("label", ""))
            normalized.pop("options", None)
            evidence_dialogues = normalize_evidence_dialogues(normalized.get("evidence_dialogues"))
            for evidence_idx, evidence in enumerate(evidence_dialogues, start=1):
                evidence["id"] = f"E{evidence_idx}"
                evidence.setdefault("speaker", None)
                evidence.setdefault("utterance", "")
                evidence.setdefault("dia_id", "N/A")
            normalized["evidence_dialogues"] = evidence_dialogues

            evidence_ids = [ev["id"] for ev in evidence_dialogues if ev.get("id")]
            reasoning_source = normalized.get("reasoning_steps")
            if not reasoning_source:
                reasoning_source = normalized.get("reasoning") or normalized.get("explanation")
            normalized["reasoning_steps"] = normalize_reasoning_steps(reasoning_source, evidence_ids)

            normalized.setdefault("character", "")
            normalized.setdefault("question", "")
            normalized.setdefault("option", [])
            normalized.setdefault("answer", "")
            normalized.setdefault("label", "")
            normalized.setdefault("category", None)
            normalized.setdefault("original_qa", [])
            if not isinstance(normalized.get("original_qa"), list):
                normalized["original_qa"] = [str(normalized.get("original_qa"))]

            normalized_items.append(normalized)

        return normalized_items

    def _build_original_question_lookup(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build a lookup table from original question text to original QA items."""
        lookup: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for original_q in self.raw_data:
            key = normalize_question_reference(original_q.get("question", ""))
            if key:
                lookup[key].append(original_q)
        return lookup

    def _resolve_referenced_original_questions(
        self,
        generated_q: Dict[str, Any],
        original_lookup: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Resolve `original_qa` strings emitted by the model back to source QA items."""
        references = generated_q.get("original_qa", [])
        if isinstance(references, str):
            references = [references]
        if not isinstance(references, list):
            references = []

        resolved: List[Dict[str, Any]] = []
        seen_questions = set()
        lookup_items = list(original_lookup.items())

        for reference in references:
            ref_key = normalize_question_reference(reference)
            if not ref_key:
                continue

            candidates = original_lookup.get(ref_key, [])
            if not candidates:
                # Be tolerant to harmless truncation in model-emitted references,
                # but only when there is a clear substring relationship.
                substring_matches = [
                    item
                    for question_key, items in lookup_items
                    if ref_key in question_key or question_key in ref_key
                    for item in items
                ]
                if len(substring_matches) == 1:
                    candidates = substring_matches

            for item in candidates:
                question_key = normalize_question_reference(item.get("question", ""))
                if question_key and question_key not in seen_questions:
                    seen_questions.add(question_key)
                    resolved.append(item)

        return resolved

    def _canonical_evidence_from_referenced_questions(
        self,
        source_questions: List[Dict[str, Any]],
        merged_conversation: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Concatenate and canonicalize evidence from referenced source questions."""
        dialogue_index = build_dialogue_index(merged_conversation)
        merged_evidence: List[Dict[str, Any]] = []
        seen_dia_ids = set()

        for source_q in source_questions:
            for evidence in normalize_evidence_dialogues(source_q.get("evidence_dialogues", [])):
                dia_id = str(evidence.get("dia_id") or "").strip()
                if not dia_id or dia_id.upper() == "N/A":
                    continue
                if dia_id.casefold() in seen_dia_ids:
                    continue

                source_turn = dialogue_index.get(dia_id.casefold())
                if not source_turn:
                    continue

                seen_dia_ids.add(dia_id.casefold())
                merged_evidence.append(
                    {
                        "id": f"E{len(merged_evidence) + 1}",
                        "speaker": source_turn.get("speaker"),
                        "utterance": source_turn.get("utterance", ""),
                        "dia_id": source_turn.get("dia_id"),
                    }
                )

        return merged_evidence

    def _rewrite_reasoning_for_merged_evidence(
        self,
        generated_q: Dict[str, Any],
        evidence_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Keep model-written reasoning text but remap references to merged evidence ids."""
        reasoning_source = generated_q.get("reasoning_steps")
        if not reasoning_source:
            reasoning_source = generated_q.get("reasoning") or generated_q.get("explanation")

        reasoning_steps = normalize_reasoning_steps(reasoning_source, evidence_ids)
        if not reasoning_steps and evidence_ids:
            reasoning_steps = [
                {
                    "step": 1,
                    "inference": "The reconstructed question is supported by the merged evidence from the referenced source questions.",
                    "based_on": list(evidence_ids),
                }
            ]

        valid_ids = set(evidence_ids)
        for idx, step in enumerate(reasoning_steps, start=1):
            step["step"] = idx
            based_on = step.get("based_on", [])
            if not isinstance(based_on, list):
                based_on = [based_on]
            based_on = [str(item) for item in based_on if str(item) in valid_ids]
            step["based_on"] = based_on or list(evidence_ids)

        return reasoning_steps

    def _attach_referenced_source_evidence(
        self,
        generated_q: Dict[str, Any],
        original_lookup: Dict[str, List[Dict[str, Any]]],
        merged_conversation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Overwrite generated evidence with concatenated evidence from referenced originals."""
        normalized = dict(generated_q)
        source_questions = self._resolve_referenced_original_questions(
            normalized,
            original_lookup,
        )
        source_evidence = self._canonical_evidence_from_referenced_questions(
            source_questions,
            merged_conversation,
        )
        evidence_ids = [evidence["id"] for evidence in source_evidence]

        normalized["evidence_dialogues"] = source_evidence
        normalized["reasoning_steps"] = self._rewrite_reasoning_for_merged_evidence(
            normalized,
            evidence_ids,
        )
        normalized["source_evidence_merge"] = {
            "policy": "concat_evidence_from_referenced_original_qa",
            "referenced_original_count": len(source_questions),
            "merged_evidence_count": len(source_evidence),
            "referenced_questions": [
                source_q.get("question", "")
                for source_q in source_questions
            ],
        }
        return normalized

    def _normalize_existing_question(self, question_item: Dict[str, Any]) -> Dict[str, Any]:
        """补齐保留原题的统一字段，避免不同来源题目 schema 不一致。"""
        normalized = dict(question_item)
        normalized["question_type"] = normalize_question_type(
            normalized.get("question_type")
        )
        normalized["option"] = self._normalize_option_field(
            normalized.get("option", normalized.get("options", []))
        )
        normalized.pop("options", None)
        normalized["answer"] = self._normalize_answer_letter(
            normalized.get("answer", ""),
            normalized["option"],
            normalized["question_type"],
        )
        normalized["label"] = self._normalize_label(normalized.get("label", ""))

        evidence_dialogues = normalize_evidence_dialogues(normalized.get("evidence_dialogues", []))
        for evidence_idx, evidence in enumerate(evidence_dialogues, start=1):
            evidence["id"] = str(evidence.get("id") or f"E{evidence_idx}")
            evidence.setdefault("speaker", None)
            evidence.setdefault("utterance", "")
            evidence.setdefault("dia_id", "N/A")
        normalized["evidence_dialogues"] = evidence_dialogues

        evidence_ids = [ev.get("id") for ev in evidence_dialogues if ev.get("id")]
        normalized["reasoning_steps"] = normalize_reasoning_steps(
            normalized.get("reasoning_steps", []),
            evidence_ids,
        )

        normalized.setdefault("character", "")
        normalized.setdefault("question", "")
        normalized.setdefault("label", "")
        normalized.setdefault("category", None)
        return normalized

    def _validate_refined_question(
        self,
        question_item: Dict[str, Any],
        merged_conversation: Dict[str, Any],
        *,
        require_source_evidence_merge: bool = False,
        strict_label_rules: bool = False,
    ) -> Dict[str, Any]:
        """校验重构题 schema 与 evidence-dia_id 对齐。"""
        errors: List[str] = []

        required_fields = [
            "character",
            "question",
            "option",
            "answer",
            "evidence_dialogues",
            "reasoning_steps",
            "label",
            "category",
        ]
        for field in required_fields:
            if field not in question_item:
                errors.append(f"missing_field:{field}")

        normalized_type = normalize_question_type(question_item.get("question_type"))
        minimum_options = 5 if normalized_type == SINGLE_CHOICE else 2
        if (
            not isinstance(question_item.get("option"), list)
            or len(question_item.get("option", [])) < minimum_options
        ):
            errors.append("invalid_option")
        if not str(question_item.get("question", "")).strip():
            errors.append("empty_question")
        answer_candidate = normalize_answer_candidates(
            None,
            question_item.get("answer", ""),
        )[0]
        answer_sequence, answer_malformed = parse_question_answer(
            answer_candidate,
            normalized_type,
        )
        option_letters = {
            match.group(1).upper()
            for option in question_item.get("option", [])
            for match in [re.match(r"^([A-Fa-f])[\.．\)]", str(option or "").strip())]
            if match
        }
        if not answer_candidate.strip():
            errors.append("empty_answer")
        elif answer_malformed or not answer_sequence:
            errors.append("invalid_answer_format")
        elif any(letter not in option_letters for letter in answer_sequence):
            errors.append("answer_option_out_of_range")
        elif normalized_type == ORDERING and set(answer_sequence) != option_letters:
            errors.append("ordering_requires_all_options")
        if question_item.get("label") not in VALID_LABELS:
            errors.append("invalid_label")
        try:
            category_value = int(question_item.get("category"))
        except (TypeError, ValueError):
            category_value = 0
        if category_value < 1 or category_value > 7:
            errors.append("invalid_category")

        aligned_evidence, evidence_report = align_evidence_dialogues(
            question_item.get("evidence_dialogues", []),
            merged_conversation,
        )
        if evidence_report.get("result") != "pass":
            errors.append(f"evidence_alignment:{evidence_report.get('result')}")

        if require_source_evidence_merge:
            merge_meta = question_item.get("source_evidence_merge", {})
            if not isinstance(merge_meta, dict):
                merge_meta = {}
            if int(merge_meta.get("referenced_original_count") or 0) <= 0:
                errors.append("missing_resolved_original_qa_reference")
            if int(merge_meta.get("merged_evidence_count") or 0) <= 0:
                errors.append("empty_merged_source_evidence")

        if strict_label_rules:
            answer_letter = answer_sequence[0] if len(answer_sequence) == 1 else ""
            label = str(question_item.get("label", "")).strip()
            evidence_count = len(aligned_evidence)

            if answer_letter == "F" and label != "Abstain":
                errors.append("answer_f_requires_abstain_label")
            if label == "Abstain" and answer_letter != "F":
                errors.append("abstain_label_requires_answer_f")
            if label == "Fact Extraction (Single Dialogue)" and evidence_count != 1:
                errors.append(f"single_dialogue_requires_one_evidence:{evidence_count}")
            if label == "Fact Extraction (Multiple Dialogues)" and evidence_count < 2:
                errors.append(f"multiple_dialogues_requires_at_least_two_evidence:{evidence_count}")
            if label == "Memory Update" and evidence_count < 2:
                errors.append(f"memory_update_requires_at_least_two_evidence:{evidence_count}")
            if label == "Multi-hop" and evidence_count < 2:
                errors.append(f"multihop_requires_at_least_two_evidence:{evidence_count}")

        evidence_ids = [ev.get("id") for ev in aligned_evidence if ev.get("id")]
        reasoning_steps = normalize_reasoning_steps(question_item.get("reasoning_steps"), evidence_ids)
        if not reasoning_steps:
            errors.append("empty_reasoning_steps")

        validated = dict(question_item)
        validated["evidence_dialogues"] = aligned_evidence
        validated["reasoning_steps"] = reasoning_steps
        validated["evidence_alignment_check"] = evidence_report
        validated["schema_check"] = {
            "result": "pass" if not errors else "fail",
            "errors": errors,
        }
        return validated

    def _refine_subject(self, task) -> List[Dict[str, Any]]:
        subject_idx, total_subjects, subject, group = task
        log_subsection(f"Subject {subject_idx}/{total_subjects} | {subject}", indent=2)

        if len(group) < 2:
            log_event(
                "refine_subject",
                status="skipped",
                subject=subject,
                reason="too_few_questions",
                input_q=len(group),
                indent=4,
            )
            return []

        category_summary = self.format_category_counts(group)
        log_event(
            "refine_subject",
            status="start",
            subject=subject,
            input_q=len(group),
            category_counts=category_summary,
            indent=4,
        )

        max_retries = 10
        for attempt in range(max_retries):
            current_prompt = self.build_refine_prompt(subject, group)
            if attempt > 0:
                current_prompt += "\n\n**重要修正**：请直接输出 JSON 数组格式（以 [ 开头，以 ] 结束），严禁包含任何 Markdown 代码块标签、前言、解释或结尾总结。"

            response = gen_chat(current_prompt)
            refined = self.extract_json(response)
            if refined:
                normalized_refined = self._normalize_refined_questions(refined)
                log_event(
                    "refine_subject",
                    status="generated",
                    subject=subject,
                    attempts=attempt + 1,
                    generated=len(normalized_refined),
                    indent=4,
                )
                return normalized_refined

            log_event(
                "refine_subject",
                status="retry",
                subject=subject,
                reason="json_parse_failed",
                attempt=f"{attempt + 1}/{max_retries}",
                indent=4,
            )

        log_event(
            "refine_subject",
            status="failed",
            subject=subject,
            attempts=max_retries,
            generated=0,
            indent=4,
        )
        return []

    def process(self, max_workers: int = 1):
        """
        处理流程：对所有问题进行全局分析和重构（取消聚类，直接传入全量数据）
        """
        self.load_data()
        
        log_subsection("Step 1 refinement plan")
        input_type_counts = {
            SINGLE_CHOICE: 0,
            MULTIPLE_SELECT: 0,
            ORDERING: 0,
        }
        for qa in self.raw_data:
            input_type_counts[normalize_question_type(qa.get("question_type"))] += 1
        log_event("refine_question_types", status="input", **input_type_counts)
        
        # 按角色分组（全局）
        subject_buckets = defaultdict(list)
        for qa in self.raw_data:
            subject_buckets[qa.get("character", "Unknown")].append(qa)

        valid_subjects = [
            (subject, qa_list)
            for subject, qa_list in subject_buckets.items()
            if subject != "Unknown"
        ]
        print_kv("subjects", len(valid_subjects), indent=4)
        print_kv(
            "strategy",
            "evidence-first reconstruction; generated evidence must reuse source QA evidence exactly",
            indent=4,
        )
        
        all_refined_qa: List[Dict[str, Any]] = []
        normalized_workers = max(1, int(max_workers or 1))
        effective_workers = min(normalized_workers, max(1, len(valid_subjects)))
        log_event(
            "refine_workers",
            status="configured",
            requested=normalized_workers,
            effective=effective_workers,
            subjects=len(valid_subjects),
        )
        tasks = [
            (subject_idx, len(valid_subjects), subject, subject_qa_list)
            for subject_idx, (subject, subject_qa_list) in enumerate(valid_subjects, start=1)
        ]

        if effective_workers == 1:
            results = map(self._refine_subject, tasks)
            with tqdm(total=len(tasks), desc="Step 1 subjects", unit="subject") as pbar:
                for refined_items in results:
                    all_refined_qa.extend(refined_items)
                    pbar.update(1)
        else:
            with ThreadPoolExecutor(
                max_workers=effective_workers,
                thread_name_prefix="refine-subject",
            ) as executor:
                with tqdm(total=len(tasks), desc="Step 1 subjects", unit="subject") as pbar:
                    for refined_items in executor.map(self._refine_subject, tasks):
                        all_refined_qa.extend(refined_items)
                        pbar.update(1)
        
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
        log_subsection("Step 1 merge and validation")
        merged_conversation = self._build_merged_conversation()
        original_lookup = self._build_original_question_lookup()
        
        # 1. 处理新问题：标签映射，收集要删除的原始问题文本
        remove_questions = set()
        processed_new_qa = []
        dropped_new_qa = []
        
        for new_q in refined_qa_list:
            if "session" in new_q.get("question", ""):
                dropped_new_qa.append({"question": new_q.get("question", ""), "reason": "contains_meta_session"})
                continue

            evidence_merged_q = self._attach_referenced_source_evidence(
                new_q,
                original_lookup,
                merged_conversation,
            )
            validated_q = self._validate_refined_question(
                evidence_merged_q,
                merged_conversation,
                require_source_evidence_merge=True,
                strict_label_rules=True,
            )
            if validated_q.get("schema_check", {}).get("result") != "pass":
                dropped_new_qa.append(
                    {
                        "question": new_q.get("question", ""),
                        "reason": "schema_or_evidence_failed",
                        "schema_check": validated_q.get("schema_check"),
                        "evidence_alignment_check": validated_q.get("evidence_alignment_check"),
                        "source_evidence_merge": validated_q.get("source_evidence_merge"),
                    }
                )
                continue
            
            if validated_q.get("label") == "Memory Update":
                for q_text in validated_q.get("original_qa", []):
                    if isinstance(q_text, str) and q_text:
                        remove_questions.add(q_text)
            
            processed_new_qa.append(validated_q)

        log_event(
            "refine_merge",
            status="validated",
            generated_candidates=len(refined_qa_list),
            kept_generated=len(processed_new_qa),
            dropped_generated=len(dropped_new_qa),
            source_evidence_merged=sum(
                int((item.get("source_evidence_merge") or {}).get("merged_evidence_count") or 0)
                for item in processed_new_qa
            ),
            source_questions_referenced=sum(
                int((item.get("source_evidence_merge") or {}).get("referenced_original_count") or 0)
                for item in processed_new_qa
            ),
        )
        if dropped_new_qa:
            log_subsection("Dropped generated examples", indent=4)
            for item in dropped_new_qa[:5]:
                question_preview = str(item.get("question", ""))[:80].replace("\n", " ")
                log_event(
                    "refine_drop_example",
                    status="dropped",
                    reason=item.get("reason"),
                    schema_errors=(item.get("schema_check") or {}).get("errors"),
                    source_evidence_merge=item.get("source_evidence_merge"),
                    question=question_preview,
                    indent=6,
                )
        
        # 2. 收集所有保留的原始问题（深拷贝）
        import copy
        all_qa = []
        removed_original_count = 0
        dropped_original_count = 0
        dropped_original_examples: List[Dict[str, Any]] = []
        for original_q in self.raw_data:
            if "session" in original_q.get("question", ""):
                continue
            
            original_question = original_q.get("question", "")
            if original_question and any(q_text in original_question for q_text in remove_questions):
                removed_original_count += 1
                continue
            
            q_copy = copy.deepcopy(original_q)
            remove_legacy_number_fields(q_copy)
            q_copy.pop("episode_index", None)
            q_copy.pop("source_index", None)
            normalized_original = self._normalize_existing_question(q_copy)
            validated_original = self._validate_refined_question(
                normalized_original,
                merged_conversation,
            )
            if validated_original.get("schema_check", {}).get("result") != "pass":
                dropped_original_count += 1
                if len(dropped_original_examples) < 5:
                    dropped_original_examples.append(
                        {
                            "question": normalized_original.get("question", ""),
                            "schema_check": validated_original.get("schema_check", {}),
                        }
                    )
                continue
            all_qa.append(validated_original)

        if dropped_original_count:
            log_event(
                "refine_merge",
                status="dropped_original_invalid_schema",
                dropped=dropped_original_count,
            )
            for item in dropped_original_examples:
                question_preview = str(item.get("question", ""))[:80].replace("\n", " ")
                log_event(
                    "refine_drop_original_example",
                    status="dropped",
                    question=question_preview,
                    schema_check=item.get("schema_check"),
                    indent=4,
                )
        
        # 3. 标记所有新生成的题目，以便后续步骤识别
        for new_q in processed_new_qa:
            remove_legacy_number_fields(new_q)
            new_q["is_generated_qa"] = True
        
        # 4. 合并新问题和原始问题
        all_qa.extend(processed_new_qa)
        
        # 7. 构建最终输出结构
        final_data = [
            {
                "qa": all_qa,
                "conversation": merged_conversation
            }
        ]
        
        # 8. 保存文件
        write_json_file(final_data, self.output_file, indent=2)
        output_type_counts = {
            SINGLE_CHOICE: 0,
            MULTIPLE_SELECT: 0,
            ORDERING: 0,
        }
        for qa in all_qa:
            output_type_counts[normalize_question_type(qa.get("question_type"))] += 1
        log_event("refine_question_types", status="output", **output_type_counts)
        log_event(
            "refine_save",
            status="success",
            original_kept=len(all_qa) - len(processed_new_qa),
            original_removed=removed_original_count,
            original_dropped_invalid=dropped_original_count,
            final_qa=len(all_qa),
            output=self.output_file,
        )


def new_qa_main(input_file_path: str, output_file_path: str,
                api_key: str = DEFAULT_API_KEY, base_url: str = DEFAULT_BASE_URL,
                model: str = DEFAULT_MODEL, max_workers: int = 1) -> str:
    """
    步骤 1: 问题精炼与重构（全量长上下文处理模式）
    """
    print_log_section("STEP 1 | REFINE AND REBUILD QA")
    print_kv("input", input_file_path, indent=2)
    
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    input_data = load_json_file(input_file_path)
    total_questions = count_qa_items(input_data)
    log_event("refine_process", status="start", total_questions=total_questions)
    
    global client, MODEL, BASE_URL, API_KEY
    API_KEY = api_key
    BASE_URL = base_url
    MODEL = model
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    processor = UltimateMemoryRefiner(
        input_file=input_file_path, 
        output_file=output_file_path
    )
    processor.process(max_workers=max_workers)
    
    return output_file_path

if __name__ == "__main__":
    # 示例用法
    processor = UltimateMemoryRefiner(
        input_file="./An-Enemy-of-the-People_merged.json", 
        output_file="An-Enemy-of-the-People_new2.json"
    )
    processor.process()
