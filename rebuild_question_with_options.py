#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Any


ANSWER_PROMPT = (
    "Please provide the option corresponding to the only correct answer, enclosed in parentheses, e.g., (X)."
)


def strip_option_prefix(text: str) -> str:
    """Remove leading option marker like 'A. ' or 'C) ' from an option string."""
    return re.sub(r"^\s*[A-Fa-f]\s*[\.|\)]\s*", "", text).strip()


def extract_original_question(question_text: str) -> str:
    """Keep only the stem before appended options and answer prompt."""
    text = question_text.strip()

    # Remove trailing standard answer prompt if present.
    text = re.sub(
        re.escape(ANSWER_PROMPT) + r"\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).rstrip()

    # Remove everything from the first option line (A.) onward.
    text = re.split(r"\n\s*A\s*[\.|\)]\s*", text, maxsplit=1)[0].strip()
    return text


def normalize_options(raw_options: Any) -> list[str]:
    """Normalize option field to ordered A-F text list."""
    if isinstance(raw_options, list):
        if len(raw_options) < 6:
            raise ValueError(f"option 数量不足 6 个，实际为 {len(raw_options)}")
        return [str(x) for x in raw_options[:6]]

    if isinstance(raw_options, dict):
        normalized_map = {str(k).upper(): v for k, v in raw_options.items()}
        letters = ["A", "B", "C", "D", "E", "F"]
        missing = [k for k in letters if k not in normalized_map]
        if missing:
            raise ValueError(f"option 字典缺少键: {', '.join(missing)}")
        return [str(normalized_map[k]) for k in letters]

    raise ValueError(f"不支持的 option 类型: {type(raw_options).__name__}")


def build_option_list(options: list[str]) -> list[str]:
    """Build the final A-F option list in the requested format."""
    lines = []
    for i, letter in enumerate(["A", "B", "C", "D", "E", "F"]):
        option_text = strip_option_prefix(options[i])
        lines.append(f"{letter}. {option_text}")
    return lines


def rebuild_question(stem: str, option_lines: list[str]) -> str:
    lines = [stem]
    lines.extend(option_lines)
    lines.append(ANSWER_PROMPT)
    return "\n".join(lines)


def normalize_evidence_field(node: dict[str, Any]) -> int:
    """Normalize evidence-related field name to `evidence_dialogues`."""
    changed = 0

    if "evidence_dialogues" in node:
        if "evidence" in node:
            del node["evidence"]
            changed += 1
        if "evidences" in node:
            del node["evidences"]
            changed += 1
        return changed

    if "evidence" in node:
        node["evidence_dialogues"] = node["evidence"]
        del node["evidence"]
        changed += 1
    elif "evidences" in node:
        node["evidence_dialogues"] = node["evidences"]
        del node["evidences"]
        changed += 1

    return changed


def process_node(node: Any) -> int:
    """Recursively process a JSON node and update all valid QA items."""
    changed = 0

    if isinstance(node, list):
        for item in node:
            changed += process_node(item)
        return changed

    if not isinstance(node, dict):
        return 0

    # Process current qa record if fields exist.
    if "question" in node and "option" in node:
        changed += normalize_evidence_field(node)

        stem = extract_original_question(str(node["question"]))
        options = normalize_options(node["option"])
        option_lines = build_option_list(options)
        rebuilt = rebuild_question(stem, option_lines)

        if node["option"] != option_lines:
            node["option"] = option_lines
            changed += 1

        if node["question"] != rebuilt:
            node["question"] = rebuilt
            changed += 1

    # Traverse nested structures.
    for value in node.values():
        changed += process_node(value)

    return changed


def process_file(input_file_path: Path, output_file_path: Path) -> int:
    with input_file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    changed = process_node(data)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with output_file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return changed


def collect_qa_items(node: Any) -> list[dict[str, Any]]:
    """Recursively collect qa items from any nested JSON structure."""
    collected: list[dict[str, Any]] = []

    if isinstance(node, list):
        for item in node:
            collected.extend(collect_qa_items(item))
        return collected

    if not isinstance(node, dict):
        return collected

    # Some sources are already a flat list of QA dicts instead of {"qa": [...]}. 
    # Treat dicts with question + option as a QA item directly.
    if "question" in node and "option" in node:
        collected.append(node)
        return collected

    qa_value = node.get("qa")
    if isinstance(qa_value, list):
        for qa_item in qa_value:
            if isinstance(qa_item, dict):
                collected.append(qa_item)

    for key, value in node.items():
        if key == "qa":
            # Already consumed above; skipping avoids duplicated collection.
            continue
        collected.extend(collect_qa_items(value))

    return collected


def collect_conversation_items(node: Any) -> list[dict[str, Any]]:
    """Recursively collect conversation objects from any nested JSON structure."""
    collected: list[dict[str, Any]] = []

    if isinstance(node, list):
        for item in node:
            collected.extend(collect_conversation_items(item))
        return collected

    if not isinstance(node, dict):
        return collected

    conversation_value = node.get("conversation")
    if isinstance(conversation_value, dict):
        collected.append(conversation_value)

    for value in node.values():
        collected.extend(collect_conversation_items(value))

    return collected


def iter_grouped_json_files(target_path: Path) -> list[tuple[str, list[Path]]]:
    """Collect JSON files grouped by immediate subfolder name."""
    if target_path.is_file():
        return [(target_path.stem, [target_path])]

    if not target_path.is_dir():
        raise FileNotFoundError(f"路径不存在: {target_path}")

    grouped: list[tuple[str, list[Path]]] = []
    for child in sorted(target_path.iterdir()):
        if not child.is_dir():
            continue

        json_files = sorted(path for path in child.rglob("*.json") if path.is_file())
        if not json_files:
            continue

        grouped.append((child.name, json_files))

    # 不再对输入目录根层 json 做汇总，避免生成诸如 new_qa_v0.json 的总文件。

    return grouped


def folder_name_to_output_name(folder_name: str) -> str:
    """Convert folder name like xxx_final to xxx_v0.json."""
    if folder_name.endswith("_final"):
        base = folder_name[: -len("_final")]
    else:
        base = folder_name
    return f"{base}_v0.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="遍历 JSON 文件并将 question/option 重建为原始问题 + A-F 选项（不固定 F）"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "new_qa",
        help="输入 JSON 文件或目录路径；不传则默认处理 new_qa 目录",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；不传则默认写入输入目录（如 new_qa）",
    )
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir is not None else (
        args.input if args.input.is_dir() else args.input.parent
    )

    grouped_files = iter_grouped_json_files(args.input)
    if not grouped_files:
        print("未找到可处理的 JSON 文件")
        return

    total_changed = 0
    total_files = 0
    output_count = 0

    for group_name, file_paths in grouped_files:
        output_name = folder_name_to_output_name(group_name)
        output_path = output_dir / output_name

        group_changed = 0
        merged_qa: list[dict[str, Any]] = []
        merged_conversation: dict[str, Any] | None = None

        for file_path in file_paths:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            group_changed += process_node(data)
            total_files += 1

            merged_qa.extend(collect_qa_items(data))
            if merged_conversation is None:
                conversations = collect_conversation_items(data)
                if conversations:
                    merged_conversation = conversations[0]

        merged_data: list[dict[str, Any]] = [{
            "qa": merged_qa,
            "conversation": merged_conversation or {},
        }]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)

        total_changed += group_changed
        output_count += 1

        print(f"写入: {output_path}")
        print(f"  来源文件数: {len(file_paths)}")
        print(f"  汇总 qa 数量: {len(merged_qa)}")
        print(f"  本组更新计数: {group_changed}")

    print(f"处理完成：{total_files} 个输入文件，生成 {output_count} 个输出文件")
    print(f"更新 question 数量：{total_changed}")


if __name__ == "__main__":
    main()

# python rebuild_question_with_options.py 
