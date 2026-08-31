import copy
import re
from typing import Any, Dict, List, Tuple

from src.mcq_scoring import (
    MULTIPLE_SELECT_INSTRUCTION,
    ORDERING_INSTRUCTION,
    SINGLE_CHOICE,
    SINGLE_CHOICE_INSTRUCTION,
    get_answer_instruction,
    normalize_question_type,
)
from src.utils import normalize_dataset_records


ANSWER_INSTRUCTION_MARKERS = (
    "Please provide the option corresponding to the only correct answer",
    "Please provide all correct options enclosed in parentheses",
    "Please provide the options in the correct order enclosed in parentheses",
    "You need to select the correct answer from the following options:",
)

ANSWER_INSTRUCTION_SUFFIXES = (
    SINGLE_CHOICE_INSTRUCTION,
    MULTIPLE_SELECT_INSTRUCTION,
    ORDERING_INSTRUCTION,
)


def strip_answer_instruction_suffix(question_text: str) -> str:
    """Remove a trailing user-facing answer instruction while retaining options."""
    text = (question_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    folded_text = text.casefold()
    for instruction in ANSWER_INSTRUCTION_SUFFIXES:
        if folded_text.endswith(instruction.casefold()):
            return text[: -len(instruction)].rstrip()
    return text


def extract_core_question_text(question_text: str, unknown_placeholder: str = "") -> str:
    """Extract the question stem from either raw or previously formatted text."""
    text = (question_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return unknown_placeholder

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return unknown_placeholder

    first_option_idx = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"^[A-Fa-f][\.．\)]\s+", line)
        ),
        None,
    )
    if first_option_idx is not None:
        stem = "\n".join(lines[:first_option_idx]).strip()
        if stem:
            return re.sub(
                r"^Please\s+answer\s+the\s+question:\s*",
                "",
                stem,
                flags=re.IGNORECASE,
            ) or unknown_placeholder

    for marker in ANSWER_INSTRUCTION_MARKERS:
        marker_index = text.casefold().find(marker.casefold())
        if marker_index >= 0:
            return text[:marker_index].strip() or unknown_placeholder

    return re.sub(
        r"^Please\s+answer\s+the\s+question:\s*",
        "",
        lines[0],
        flags=re.IGNORECASE,
    ) or unknown_placeholder


def _strip_option_prefix(value: Any) -> str:
    text = str(value or "").strip()
    match = re.match(r"^[A-Fa-f][\.．\)]\s*(.*)$", text)
    return match.group(1).strip() if match else text


def normalize_option_lines(option_value: Any, question_type: Any = None) -> List[str]:
    """Normalize option labels without changing their source order."""
    option_lines: List[str] = []
    if isinstance(option_value, dict):
        normalized_keys = {str(key).strip().upper(): value for key, value in option_value.items()}
        for letter in "ABCDEF":
            if letter not in normalized_keys:
                continue
            body = _strip_option_prefix(normalized_keys[letter])
            if body:
                option_lines.append(f"{letter}. {body}")
    elif isinstance(option_value, list):
        next_letter_ord = ord("A")
        for raw_option in option_value:
            text = str(raw_option or "").strip()
            if not text:
                continue
            prefix_match = re.match(r"^([A-Fa-f])[\.．\)]\s*(.*)$", text)
            if prefix_match:
                letter = prefix_match.group(1).upper()
                option_lines.append(f"{letter}. {prefix_match.group(2).strip()}")
                next_letter_ord = max(next_letter_ord, ord(letter) + 1)
            else:
                letter = chr(next_letter_ord)
                option_lines.append(f"{letter}. {text}")
                next_letter_ord += 1
    elif option_value not in (None, ""):
        option_lines.append(f"A. {_strip_option_prefix(option_value)}")

    has_f = any(re.match(r"^F[\.．\)]\s+", line, flags=re.IGNORECASE) for line in option_lines)
    if normalize_question_type(question_type) == SINGLE_CHOICE and not has_f:
        option_lines.append("F. Cannot infer the answer based on the given information.")
    return option_lines


def build_question_with_options(
    core_question: str,
    option_lines: List[str],
    question_type: Any = None,
) -> str:
    option_block = "\n".join(option_lines)
    return (
        f"{core_question.strip()}\n"
        f"{option_block}\n"
        f"{get_answer_instruction(question_type)}"
    )


def format_questions_with_options(input_data: Any) -> Tuple[List[Dict[str, Any]], int]:
    """Format all questions while preserving legacy single-choice behavior."""
    formatted_data = normalize_dataset_records(copy.deepcopy(input_data))
    formatted_count = 0
    for section in formatted_data:
        qa_list = section.get("qa", [])
        if not isinstance(qa_list, list):
            continue
        for qa_item in qa_list:
            if not isinstance(qa_item, dict):
                continue
            qa_item.setdefault("question_type", "single_choice")
            question_type = normalize_question_type(qa_item.get("question_type"))
            qa_item["question_type"] = question_type
            option_lines = normalize_option_lines(qa_item.get("option", []), question_type)
            core_question = extract_core_question_text(
                qa_item.get("question", ""),
                unknown_placeholder=str(qa_item.get("question", "")).strip(),
            )
            qa_item["option"] = option_lines
            qa_item["question"] = build_question_with_options(
                core_question,
                option_lines,
                question_type,
            )
            formatted_count += 1
    return formatted_data, formatted_count
