"""Strict public representation for benchmark QA items."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from src.mcq_scoring import normalize_question_type
from src.question_formatting import (
    build_question_with_options,
    extract_core_question_text,
    normalize_option_lines,
)


PUBLIC_QA_FIELDS = (
    "qa_id",
    "character",
    "category",
    "question_type",
    "question",
    "option",
    "answer",
    "label",
    "evidence_dialogues",
    "reasoning_steps",
)
QUESTION_TYPES = {"single_choice", "multiple_choice", "ordering"}
_OPTION_LINE_RE = re.compile(r"^([A-F])\.\s+(.+)$")
_RAW_OPTION_LABEL_RE = re.compile(r"^([A-Za-z])(?:[.．):：])\s*")
_ANSWER_RE = re.compile(r"^\(([A-F](?:,[A-F])*)\)$")
_CANNOT_INFER_RE = re.compile(
    r"^cannot\s+(?:infer|be\s+inferred|determine|be\s+determined)\b",
    re.IGNORECASE,
)


def make_qa_id(title_key: str, source_index: int) -> str:
    """Return the stable public ID assigned before any QA filtering."""
    title = str(title_key or "").strip()
    if not title:
        raise ValueError("title_key must be non-empty")
    if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 1:
        raise ValueError("source_index must be a positive integer")
    return f"{title}-Q{source_index:04d}"


def _option_body(line: str) -> str:
    match = _OPTION_LINE_RE.fullmatch(line)
    if not match:
        raise ValueError(f"invalid option label: {line!r}")
    return match.group(2).strip()


def _validate_raw_option_labels(raw_options: Any) -> None:
    if isinstance(raw_options, dict):
        for key in raw_options:
            key_text = str(key).strip().upper()
            if key_text not in set("ABCDEF"):
                raise ValueError(f"invalid option label: {key!r}")
        return
    if not isinstance(raw_options, list):
        raise ValueError("option must be an array")
    for raw_option in raw_options:
        if not isinstance(raw_option, str) or not raw_option.strip():
            raise ValueError("option entries must be non-empty strings")
        label_match = _RAW_OPTION_LABEL_RE.match(raw_option.strip())
        if label_match:
            label = label_match.group(1).upper()
            if label not in set("ABCDEF") or raw_option.strip()[1] in ":：":
                raise ValueError(f"invalid option label: {label}")


def _normalized_options(raw_options: Any, question_type: str) -> list[str]:
    _validate_raw_option_labels(raw_options)
    option_lines = normalize_option_lines(raw_options, question_type)
    if not option_lines:
        raise ValueError("option must not be empty")

    labels: list[str] = []
    bodies: set[str] = set()
    for line in option_lines:
        match = _OPTION_LINE_RE.fullmatch(line)
        if not match:
            raise ValueError(f"invalid option label: {line!r}")
        label, body = match.group(1), match.group(2).strip()
        if label in labels:
            raise ValueError(f"duplicate option label: {label}")
        normalized_body = re.sub(r"\s+", " ", body).casefold()
        if normalized_body in bodies:
            raise ValueError("duplicate normalized option body")
        labels.append(label)
        bodies.add(normalized_body)

    expected_labels = list("ABCDEF"[: len(labels)])
    if labels != expected_labels:
        raise ValueError("option labels must be consecutive from A")
    return [f"{label}. {_option_body(line)}" for label, line in zip(labels, option_lines)]


def _parse_answer(value: Any, question_type: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("answer must not be empty")
    compact = value.strip().upper()
    if not compact.startswith("(") and not compact.endswith(")"):
        compact = f"({compact})"
    match = re.fullmatch(r"\(\s*([A-F](?:\s*[,，]\s*[A-F])*)\s*\)", compact)
    if not match:
        raise ValueError("answer must use parenthesized option letters")
    sequence = [letter.upper() for letter in re.findall(r"[A-F]", match.group(1))]
    if len(sequence) != len(set(sequence)):
        raise ValueError("answer contains repeated option letters")
    if question_type == "single_choice" and len(sequence) != 1:
        raise ValueError("single_choice answer must contain exactly one letter")
    return sequence


def _remap_option_references(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [_remap_option_references(item, mapping) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, child in value.items():
        if key == "option":
            result[key] = _remap_explicit_option_value(child, mapping)
            continue
        result[key] = _remap_option_references(child, mapping)
    return result


def _remap_explicit_option_value(value: Any, mapping: dict[str, str]) -> Any:
    """Remap exact option tokens nested below an explicit ``option`` field."""
    if isinstance(value, str):
        stripped = value.strip().upper()
        return mapping.get(stripped, value)
    if isinstance(value, list):
        return [_remap_explicit_option_value(item, mapping) for item in value]
    if isinstance(value, dict):
        return {
            key: _remap_explicit_option_value(child, mapping)
            for key, child in value.items()
        }
    return value


def _abstain_option_indices(option_lines: list[str]) -> list[int]:
    matches = []
    for index, line in enumerate(option_lines):
        if _CANNOT_INFER_RE.match(_option_body(line).casefold()):
            matches.append(index)
    return matches


def _normalize_abstain(
    normalized: dict[str, Any],
    option_lines: list[str],
    answer: list[str],
) -> tuple[list[str], list[str], bool, dict[str, str]]:
    if normalized.get("label") != "Abstain":
        return option_lines, answer, False, {}
    if normalized["question_type"] != "single_choice":
        raise ValueError("Abstain items must be single_choice")
    matches = _abstain_option_indices(option_lines)
    if len(matches) != 1:
        raise ValueError("Abstain requires exactly one cannot-infer option")
    abstain_index = matches[0]
    if len(option_lines) < 6:
        raise ValueError("Abstain requires option F")

    mapping: dict[str, str] = {}
    if abstain_index != 5:
        abstain_label = chr(ord("A") + abstain_index)
        mapping = {abstain_label: "F", "F": abstain_label}
        swapped = list(option_lines)
        swapped[abstain_index], swapped[5] = swapped[5], swapped[abstain_index]
        option_lines = [
            f"{chr(ord('A') + index)}. {_option_body(line)}"
            for index, line in enumerate(swapped)
        ]
        answer = [mapping.get(letter, letter) for letter in answer]
        normalized["evidence_dialogues"] = _remap_option_references(
            normalized.get("evidence_dialogues"), mapping
        )
        normalized["reasoning_steps"] = _remap_option_references(
            normalized.get("reasoning_steps"), mapping
        )
    if answer != ["F"]:
        raise ValueError("Abstain answer must be (F)")
    return option_lines, answer, bool(mapping), mapping


def normalize_public_qa(item: dict, title_key: str, source_index: int) -> tuple[dict, dict]:
    """Normalize one legacy QA item into the exact publication schema."""
    if not isinstance(item, dict):
        raise ValueError("QA item must be an object")
    question_type = normalize_question_type(item.get("question_type"))
    option_lines = _normalized_options(item.get("option"), question_type)
    answer = _parse_answer(item.get("answer"), question_type)
    option_labels = {line[0] for line in option_lines}
    if not set(answer).issubset(option_labels):
        raise ValueError("answer references an out-of-range option")
    if question_type == "multiple_choice":
        answer = sorted(answer)

    normalized: dict[str, Any] = {
        "qa_id": make_qa_id(title_key, source_index),
        "character": deepcopy(item.get("character")),
        "category": deepcopy(item.get("category")),
        "question_type": question_type,
        "question": "",
        "option": option_lines,
        "answer": "",
        "label": deepcopy(item.get("label")),
        "evidence_dialogues": deepcopy(item.get("evidence_dialogues", [])),
        "reasoning_steps": deepcopy(item.get("reasoning_steps", [])),
    }
    option_lines, answer, abstain_permuted, permutation = _normalize_abstain(
        normalized, option_lines, answer
    )
    normalized["option"] = option_lines
    normalized["answer"] = f"({','.join(answer)})"
    core_stem = extract_core_question_text(
        item.get("question", ""), unknown_placeholder=""
    )
    if not core_stem:
        raise ValueError("question stem must not be empty")
    normalized["question"] = build_question_with_options(
        core_stem, option_lines, question_type
    )
    errors = validate_public_qa(normalized)
    if errors:
        raise ValueError("invalid normalized QA: " + "; ".join(errors))
    audit = {
        "qa_id": normalized["qa_id"],
        "question_type": question_type,
        "abstain_permuted": abstain_permuted,
        "option_permutation": permutation,
    }
    return normalized, audit


def _validate_answer_text(answer: Any, question_type: str, option_labels: set[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(answer, str):
        return ["answer must be a string"]
    match = _ANSWER_RE.fullmatch(answer)
    if not match:
        return ["answer must use canonical parenthesized syntax"]
    letters = match.group(1).split(",")
    if len(letters) != len(set(letters)):
        errors.append("answer contains repeated option letters")
    if question_type == "single_choice" and len(letters) != 1:
        errors.append("single_choice answer must contain exactly one letter")
    if not set(letters).issubset(option_labels):
        errors.append("answer references an out-of-range option")
    if question_type == "multiple_choice":
        expected = sorted(letters, key=lambda letter: ord(letter))
        if letters != expected:
            errors.append("multiple_choice answer must use canonical option order")
    return errors


def validate_public_qa(item: dict) -> list[str]:
    """Return deterministic validation errors for one public QA item."""
    if not isinstance(item, dict):
        return ["QA item must be an object"]
    errors: list[str] = []
    if tuple(item.keys()) != PUBLIC_QA_FIELDS:
        errors.append("public fields must exactly match the ordered schema")

    qa_id = item.get("qa_id")
    if not isinstance(qa_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*-Q\d{4}", qa_id):
        errors.append("malformed qa_id")

    question_type = item.get("question_type")
    if question_type not in QUESTION_TYPES:
        errors.append("question_type must be canonical")

    option_lines = item.get("option")
    labels: list[str] = []
    bodies: set[str] = set()
    if not isinstance(option_lines, list) or not option_lines:
        errors.append("option must be a non-empty array")
        option_lines = []
    for line in option_lines:
        if not isinstance(line, str):
            errors.append("option entries must be strings")
            continue
        match = _OPTION_LINE_RE.fullmatch(line)
        if not match:
            errors.append(f"invalid option label: {line!r}")
            continue
        label, body = match.group(1), match.group(2).strip()
        if label in labels:
            errors.append(f"duplicate option label: {label}")
        labels.append(label)
        normalized_body = re.sub(r"\s+", " ", body).casefold()
        if normalized_body in bodies:
            errors.append("duplicate normalized option body")
        bodies.add(normalized_body)
    if labels != list("ABCDEF"[: len(labels)]):
        errors.append("option labels must be consecutive from A")

    if question_type in QUESTION_TYPES and option_lines:
        question = item.get("question")
        if not isinstance(question, str):
            errors.append("question must be a string")
        else:
            core_stem = extract_core_question_text(question, unknown_placeholder="")
            expected_question = build_question_with_options(
                core_stem, option_lines, question_type
            )
            if question.replace("\r\n", "\n").replace("\r", "\n") != expected_question:
                errors.append("question rendering is not synchronized with option")
        errors.extend(_validate_answer_text(item.get("answer"), question_type, set(labels)))

    if not isinstance(item.get("evidence_dialogues"), list):
        errors.append("evidence_dialogues must be an array")
    if not isinstance(item.get("reasoning_steps"), list):
        errors.append("reasoning_steps must be an array")

    if item.get("label") == "Abstain":
        if question_type != "single_choice":
            errors.append("Abstain items must be single_choice")
        abstain_positions = []
        for index, line in enumerate(option_lines):
            if not isinstance(line, str):
                continue
            option_match = _OPTION_LINE_RE.fullmatch(line)
            if option_match and _CANNOT_INFER_RE.match(option_match.group(2).casefold()):
                abstain_positions.append(index)
        if len(abstain_positions) != 1 or abstain_positions != [5]:
            errors.append("Abstain cannot-infer option must be unique and F")
        if item.get("answer") != "(F)":
            errors.append("Abstain answer must be (F)")
    return errors


__all__ = [
    "PUBLIC_QA_FIELDS",
    "make_qa_id",
    "normalize_public_qa",
    "validate_public_qa",
]
