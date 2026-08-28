import json
import re
from typing import Any, Dict, List, Sequence, Set, Tuple


SINGLE_CHOICE = "single_choice"
MULTIPLE_SELECT = "multiple_choice"
ORDERING = "ordering"
QUESTION_TYPES = (SINGLE_CHOICE, MULTIPLE_SELECT, ORDERING)

SINGLE_CHOICE_INSTRUCTION = (
    "Please provide the option corresponding to the only correct answer, "
    "enclosed in parentheses, e.g., (X)."
)
MULTIPLE_SELECT_INSTRUCTION = (
    "Please provide all correct options enclosed in parentheses, separated by commas, "
    "e.g., (X, Y, Z). No points will be awarded for incomplete or incorrect selections."
)
ORDERING_INSTRUCTION = (
    "Please provide the options in the correct order enclosed in parentheses, separated "
    "by commas, e.g., (Y, Z, X, W). No points will be awarded for an incorrect sequence."
)


def normalize_question_type(value: Any) -> str:
    """Map external question type labels to stable internal names.

    Missing and unknown values intentionally fall back to single choice for
    compatibility with datasets produced before ``question_type`` existed.
    """
    normalized = re.sub(r"[\s_-]+", " ", str(value or "").strip().casefold())
    if normalized in {
        "multiple choice",
        "multiple select",
        "multiple selection",
        "multi select",
    }:
        return MULTIPLE_SELECT
    if normalized in {"ordering", "order", "ranking", "sequence"}:
        return ORDERING
    return SINGLE_CHOICE


def get_answer_instruction(question_type: Any) -> str:
    normalized_type = normalize_question_type(question_type)
    if normalized_type == MULTIPLE_SELECT:
        return MULTIPLE_SELECT_INSTRUCTION
    if normalized_type == ORDERING:
        return ORDERING_INSTRUCTION
    return SINGLE_CHOICE_INSTRUCTION


def count_question_types(dataset: Any) -> Dict[str, int]:
    """Count QA items by normalized type, including legacy single-choice data."""
    counts = {question_type: 0 for question_type in QUESTION_TYPES}
    if isinstance(dataset, dict):
        records = dataset.values()
    elif isinstance(dataset, list):
        records = dataset
    else:
        records = []

    for record in records:
        if not isinstance(record, dict):
            continue
        qa_items = record.get("qa", [])
        if not isinstance(qa_items, list):
            continue
        for item in qa_items:
            if isinstance(item, dict):
                counts[normalize_question_type(item.get("question_type"))] += 1
    return counts


def extract_answer_fragment_from_json_text(text: Any) -> str:
    """Recover only an explicit answer field from otherwise malformed JSON text."""
    match = re.search(
        r'["\']answer["\']\s*:\s*["\']?\s*'
        r'(\(?\s*[A-Fa-f](?:\s*[,，]\s*[A-Fa-f])*\s*\)?)',
        str(text or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    sequence = re.findall(r"[A-Fa-f]", match.group(1).upper())
    if not sequence:
        return ""
    return f"({','.join(sequence)})"


def answer_evidence_covers_selection(
    question_type: Any,
    selected_options: Sequence[Any],
    evidence_dialogues: Any,
) -> bool:
    """Require option-level evidence coverage for multi-select and ordering answers."""
    normalized_type = normalize_question_type(question_type)
    selected_sequence = [
        str(option).strip().upper()
        for option in selected_options
        if str(option).strip()
    ]
    selected = set(selected_sequence)
    if normalized_type == SINGLE_CHOICE or not selected_sequence:
        return True
    if not isinstance(evidence_dialogues, list):
        return False
    supported = {
        str(evidence.get("option", "")).strip().upper()
        for evidence in evidence_dialogues
        if isinstance(evidence, dict)
        and str(evidence.get("option", "")).strip().upper() in set("ABCDEF")
    }
    if not selected.issubset(supported):
        return False
    if normalized_type != ORDERING:
        return True

    position_options: Dict[int, Set[str]] = {}
    for evidence in evidence_dialogues:
        if not isinstance(evidence, dict):
            continue
        try:
            position = int(evidence.get("sequence_position"))
        except (TypeError, ValueError):
            continue
        option = str(evidence.get("option", "")).strip().upper()
        if option in set("ABCDEF"):
            position_options.setdefault(position, set()).add(option)
    return all(
        option in position_options.get(position, set())
        for position, option in enumerate(selected_sequence, start=1)
    )


def propagate_candidate_evidence_provenance(candidate: Any) -> List[Dict[str, Any]]:
    """Copy chunk-candidate option/order provenance onto its evidence items."""
    if not isinstance(candidate, dict):
        return []
    raw_evidence = candidate.get("evidence_dialogues", [])
    if not isinstance(raw_evidence, list):
        return []
    option = str(candidate.get("option", "")).strip().upper()
    try:
        sequence_position = int(candidate.get("sequence_position"))
    except (TypeError, ValueError):
        sequence_position = 0

    propagated: List[Dict[str, Any]] = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        evidence = dict(item)
        if option in set("ABCDEF"):
            evidence["option"] = option
        if sequence_position > 0:
            evidence["sequence_position"] = sequence_position
        propagated.append(evidence)
    return propagated


def normalize_answer_candidates(raw_candidates: Any, fallback: Any = None) -> List[str]:
    """Return accepted answer strings while preserving ``answer_fixed`` semantics.

    ``raw_candidates`` is the historical ``answer_fixed`` field, where a list
    means several independently accepted answers.  A list in the fallback
    ``answer`` field instead represents one multi-option answer sequence.
    """
    candidates: List[str] = []
    if isinstance(raw_candidates, list):
        candidates.extend(str(candidate) for candidate in raw_candidates if candidate not in (None, ""))
    elif raw_candidates not in (None, ""):
        candidates.append(str(raw_candidates))

    if not candidates and isinstance(fallback, list):
        answer_sequence = [
            str(option).strip()
            for option in fallback
            if option not in (None, "") and str(option).strip()
        ]
        if len(answer_sequence) == 1:
            candidates.append(answer_sequence[0])
        elif answer_sequence:
            candidates.append(f"({','.join(answer_sequence)})")
    elif not candidates and fallback not in (None, ""):
        candidates.append(str(fallback))

    if not candidates:
        candidates.append("")
    return candidates


def strip_prediction_text(text: Any) -> str:
    cleaned = str(text or "").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    if "Final Answer:" in cleaned:
        cleaned = cleaned.split("Final Answer:", 1)[1].strip()

    try:
        parsed_json = json.loads(cleaned)
    except json.JSONDecodeError:
        return cleaned

    if isinstance(parsed_json, dict):
        for key in ("answer", "final_answer", "response"):
            value = parsed_json.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return cleaned


def _parse_answer_sequence(text: Any) -> Tuple[List[str], bool]:
    raw = strip_prediction_text(text)
    if not raw:
        return [], False

    compact = raw.strip()
    parse_text = (
        compact.replace("（", "(")
        .replace("）", ")")
        .replace("【", "[")
        .replace("】", "]")
        .replace("．", ".")
        .replace("：", ":")
        .replace("，", ",")
        .replace("、", ",")
    )

    direct_match = re.fullmatch(
        r"[\[(]?\s*([A-Fa-f](?:\s*(?:,|\s)\s*[A-Fa-f])*)\s*[\])]?[.,]?",
        parse_text,
    )
    if direct_match:
        return re.findall(r"[A-Fa-f]", direct_match.group(1).upper()), False

    if re.fullmatch(r"[A-Fa-f]{1,6}", parse_text):
        return [char.upper() for char in parse_text], False

    labelled_sequence = re.search(
        r"(?:answer|final answer|correct answer|choose|pick|options?)\s*(?:is|are|:)?\s*"
        r"[\[(]?\s*([A-Fa-f](?:\s*,\s*[A-Fa-f])*)\s*[\])]?(?:\s|$|[.])",
        parse_text,
        flags=re.IGNORECASE,
    )
    if labelled_sequence:
        return re.findall(r"[A-Fa-f]", labelled_sequence.group(1).upper()), False

    # 兼容: "(B. xxx)", "[C: xxx]"
    bracketed_with_text = re.match(r"^[\[(]\s*([A-Fa-f])\s*[\.:,\-]", parse_text)
    if bracketed_with_text:
        return [bracketed_with_text.group(1).upper()], False

    if re.search(r"\([A-Fa-f]\)\([A-Fa-f]\)", parse_text) or re.search(r"\[[A-Fa-f]\]\[[A-Fa-f]\]", parse_text):
        return [], True

    leading_letter = re.match(
        r"^(?:option\s*)?[\[(]?\s*([A-Fa-f])(?:[\s\)\]\.:,\-]|$)",
        parse_text,
        flags=re.IGNORECASE,
    )
    if leading_letter:
        return [leading_letter.group(1).upper()], False

    phrase_letter = re.search(
        r"(?:answer|final answer|correct answer|choose|pick|option)\s*(?:is|:)?\s*[\[(]?\s*([A-Fa-f])(?:[\s\)\]\.:,\-]|$)",
        parse_text,
        flags=re.IGNORECASE,
    )
    if phrase_letter:
        return [phrase_letter.group(1).upper()], False

    token_re = re.compile(
        r"\(([A-Fa-f])\)|\[([A-Fa-f])\]|[\[(]\s*([A-Fa-f])\s*[\.:,\-]|(?:option\s+)([A-Fa-f])\b",
        flags=re.IGNORECASE,
    )
    options: List[str] = []
    for match in token_re.finditer(parse_text):
        letter = next(group for group in match.groups() if group)
        options.append(letter.upper())

    return options, False


def _validate_answer_sequence(
    sequence: Sequence[str],
    question_type: Any,
    parse_malformed: bool = False,
) -> Tuple[List[str], bool]:
    normalized_type = normalize_question_type(question_type)
    normalized = [str(option).strip().upper() for option in sequence if str(option).strip()]
    malformed = parse_malformed or any(option not in set("ABCDEF") for option in normalized)
    if len(normalized) != len(set(normalized)):
        malformed = True
    if normalized_type == SINGLE_CHOICE and len(normalized) > 1:
        malformed = True
    return normalized, malformed


def parse_question_answer(text: Any, question_type: Any = None) -> Tuple[List[str], bool]:
    """Parse an answer while preserving order for ordering questions."""
    sequence, malformed = _parse_answer_sequence(text)
    return _validate_answer_sequence(sequence, question_type, malformed)


def parse_mcq_pred_answers(text: Any) -> Tuple[Set[str], bool]:
    """Legacy single-choice parser retained for existing callers."""
    sequence, malformed = parse_question_answer(text, SINGLE_CHOICE)
    return set(sequence), malformed


def parse_mcq_gt_answers(text: Any) -> Set[str]:
    raw = str(text or "").strip()
    if not raw:
        return set()

    leading_letter = re.match(r"^([A-Fa-f])(?:[\s\)\]\.:,，、\-]|$)", raw, flags=re.IGNORECASE)
    if leading_letter:
        return {leading_letter.group(1).upper()}

    extracted, _ = parse_mcq_pred_answers(raw)
    return extracted


def score_mcq_prediction(
    prediction_text: Any,
    answer_candidates: Sequence[Any],
    question_type: Any = None,
) -> Dict[str, Any]:
    normalized_type = normalize_question_type(question_type)
    pred_sequence, pred_malformed = parse_question_answer(prediction_text, normalized_type)
    pred_options = set(pred_sequence)

    matched_answer = ""
    matched_gt_sequence: List[str] = []
    candidate_gt_options: List[List[str]] = []

    for candidate in answer_candidates:
        gt_sequence, gt_malformed = parse_question_answer(candidate, normalized_type)
        if gt_sequence and not gt_malformed:
            displayed_gt = gt_sequence if normalized_type == ORDERING else sorted(gt_sequence)
            candidate_gt_options.append(displayed_gt)
        if normalized_type == ORDERING:
            is_match = pred_sequence == gt_sequence
        else:
            is_match = pred_options == set(gt_sequence)
        if not pred_malformed and not gt_malformed and is_match and gt_sequence:
            matched_answer = str(candidate)
            matched_gt_sequence = gt_sequence
            break

    displayed_prediction = pred_sequence if normalized_type == ORDERING else sorted(pred_options)
    result: Dict[str, Any] = {
        "is_correct": bool(matched_answer),
        "score": 1.0 if matched_answer else 0.0,
        "prediction_malformed": pred_malformed,
        "question_type": normalized_type,
        "predicted_options": displayed_prediction,
        "ground_truth_options": candidate_gt_options,
    }
    if matched_answer:
        result["matched_answer"] = matched_answer
        result["matched_ground_truth"] = (
            matched_gt_sequence if normalized_type == ORDERING else sorted(matched_gt_sequence)
        )

    return result
