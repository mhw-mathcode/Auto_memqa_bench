import json
import re
from typing import Any, Dict, List, Sequence, Set, Tuple


def normalize_answer_candidates(raw_candidates: Any, fallback: Any = None) -> List[str]:
    candidates: List[str] = []
    if isinstance(raw_candidates, list):
        candidates.extend(str(candidate) for candidate in raw_candidates if candidate not in (None, ""))
    elif raw_candidates not in (None, ""):
        candidates.append(str(raw_candidates))

    if not candidates and fallback not in (None, ""):
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


def parse_mcq_pred_answers(text: Any) -> Tuple[Set[str], bool]:
    raw = strip_prediction_text(text)
    if not raw:
        return set(), False

    compact = raw.strip()
    parse_text = (
        compact.replace("（", "(")
        .replace("）", ")")
        .replace("【", "[")
        .replace("】", "]")
        .replace("．", ".")
        .replace("：", ":")
    )

    if re.fullmatch(r"[A-Fa-f]{1,5}", parse_text):
        return {char.upper() for char in compact}, False

    if re.fullmatch(r"[\[(]?\s*[A-Fa-f]\s*[\])]?,?", parse_text):
        letter = re.search(r"([A-Fa-f])", parse_text)
        return ({letter.group(1).upper()} if letter else set()), False

    # 兼容: "(B. xxx)", "[C: xxx]"
    bracketed_with_text = re.match(r"^[\[(]\s*([A-Fa-f])\s*[\.:,\-]", parse_text)
    if bracketed_with_text:
        return {bracketed_with_text.group(1).upper()}, False

    if re.search(r"\([A-Fa-f]\)\([A-Fa-f]\)", parse_text) or re.search(r"\[[A-Fa-f]\]\[[A-Fa-f]\]", parse_text):
        return set(), True

    leading_letter = re.match(
        r"^(?:option\s*)?[\[(]?\s*([A-Fa-f])(?:[\s\)\]\.:,\-]|$)",
        parse_text,
        flags=re.IGNORECASE,
    )
    if leading_letter:
        return {leading_letter.group(1).upper()}, False

    phrase_letter = re.search(
        r"(?:answer|final answer|correct answer|choose|pick|option)\s*(?:is|:)?\s*[\[(]?\s*([A-Fa-f])(?:[\s\)\]\.:,\-]|$)",
        parse_text,
        flags=re.IGNORECASE,
    )
    if phrase_letter:
        return {phrase_letter.group(1).upper()}, False

    token_re = re.compile(
        r"\(([A-Fa-f])\)|\[([A-Fa-f])\]|[\[(]\s*([A-Fa-f])\s*[\.:,\-]|(?:option\s+)([A-Fa-f])\b",
        flags=re.IGNORECASE,
    )
    options: Set[str] = set()
    for match in token_re.finditer(parse_text):
        letter = next(group for group in match.groups() if group)
        options.add(letter.upper())

    return options, False


def parse_mcq_gt_answers(text: Any) -> Set[str]:
    raw = str(text or "").strip()
    if not raw:
        return set()

    leading_letter = re.match(r"^([A-Fa-f])(?:[\s\)\]\.:,，、\-]|$)", raw, flags=re.IGNORECASE)
    if leading_letter:
        return {leading_letter.group(1).upper()}

    extracted, _ = parse_mcq_pred_answers(raw)
    return extracted


def score_mcq_prediction(prediction_text: Any, answer_candidates: Sequence[Any]) -> Dict[str, Any]:
    pred_options, pred_malformed = parse_mcq_pred_answers(prediction_text)

    matched_answer = ""
    matched_gt_options: Set[str] = set()
    candidate_gt_options: List[List[str]] = []

    for candidate in answer_candidates:
        gt_options = parse_mcq_gt_answers(candidate)
        if gt_options:
            candidate_gt_options.append(sorted(gt_options))
        if not pred_malformed and pred_options == gt_options and gt_options:
            matched_answer = str(candidate)
            matched_gt_options = gt_options
            break

    result: Dict[str, Any] = {
        "is_correct": bool(matched_answer),
        "score": 1.0 if matched_answer else 0.0,
        "prediction_malformed": pred_malformed,
        "predicted_options": sorted(pred_options),
        "ground_truth_options": candidate_gt_options,
    }
    if matched_answer:
        result["matched_answer"] = matched_answer
        result["matched_ground_truth"] = sorted(matched_gt_options)

    return result
