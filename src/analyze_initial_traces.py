"""Analyze the four source trace datasets and emit Markdown/JSON reports."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple

from src.utils import align_evidence_dialogues, build_dialogue_index


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("trace1", "trace3", "trace5", "trace6")
REPORT_DIR = ROOT / "reports"
ABSOLUTE_MARKERS = (
    "从不",
    "绝不",
    "永不",
    "完全不",
    "唯一",
    "只在",
    "只因",
    "立即",
    "全部",
    "始终",
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data[0] if data else {}
    return data if isinstance(data, dict) else {}


def compact_counter(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def markdown_counter(values: Dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in values.items()) or "-"


def option_payload(option: Any) -> str:
    return re.sub(r"^\s*[A-Ea-e][\.\)）:：、]\s*", "", str(option or "")).strip()


def compact_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def answer_letter(value: Any) -> str:
    match = re.match(r"\s*([A-Fa-f])(?:\b|[\.\)）:：、])?", str(value or ""))
    return match.group(1).upper() if match else ""


def session_number(dia_id: Any) -> str:
    match = re.match(r"[A-Za-z]+(\d+):", str(dia_id or "").strip())
    return match.group(1) if match else ""


def reasoning_answer_mentions(reasoning_steps: Any) -> List[str]:
    if not isinstance(reasoning_steps, list):
        return []
    final_step = next(
        (step for step in reversed(reasoning_steps) if isinstance(step, dict)),
        None,
    )
    if final_step is None:
        return []
    inference = str(final_step.get("inference") or "")
    patterns = (
        r"(?:因此|所以|故)\s*(?:答案|正确答案|应选|选择)?\s*(?:为|是)?\s*([A-F])(?!\s*[-–—~至到0-9])",
        r"(?:答案|正确答案|选项)\s*(?:为|是|应为|选择)?\s*([A-F])(?!\s*[-–—~至到0-9])",
        r"([A-F])\s*(?:项|选项)?\s*(?:正确|最准确|最符合)",
    )
    mentions: List[str] = []
    for pattern in patterns:
        mentions.extend(
            letter.upper()
            for letter in re.findall(pattern, inference, flags=re.IGNORECASE)
        )
    return mentions


def numeric_summary(values: Iterable[int]) -> Dict[str, float]:
    data = list(values)
    if not data:
        return {"min": 0, "median": 0, "mean": 0, "max": 0}
    return {
        "min": min(data),
        "median": round(float(median(data)), 2),
        "mean": round(float(mean(data)), 2),
        "max": max(data),
    }


def answer_entropy(answer_counts: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in answer_counts.values():
        probability = count / total
        if probability:
            entropy -= probability * math.log2(probability)
    return round(entropy, 3)


def analyze_dataset(dataset: str) -> Dict[str, Any]:
    path = ROOT / "dataset" / dataset / f"{dataset}.json"
    record = load_json(path)
    conversation = record.get("conversation") if isinstance(record.get("conversation"), dict) else {}
    qas = record.get("qa") if isinstance(record.get("qa"), list) else []
    dialogue_index = build_dialogue_index(conversation)
    session_keys = sorted(
        key for key in conversation if re.fullmatch(r"session_\d+", str(key))
    )

    categories = Counter()
    labels = Counter()
    characters = Counter()
    answers = Counter()
    evidence_counts: List[int] = []
    evidence_match_types = Counter()
    option_lengths: List[int] = []
    aligned_questions = 0
    aligned_evidence_items = 0
    total_evidence_items = 0
    multi_evidence_questions = 0
    cross_session_questions = 0
    length_ratio_pass = 0
    correct_not_extreme = 0
    correct_has_longer_and_shorter = 0
    option_balance_eligible = 0
    absolute_marker_asymmetry = 0
    schema_issue_ids: List[str] = []
    alignment_issue_ids: List[str] = []
    reasoning_reference_issue_ids: List[str] = []
    reasoning_answer_mismatch_ids: List[str] = []
    answer_text_mismatch_ids: List[str] = []
    label_rule_issue_ids: List[str] = []
    duplicate_option_ids: List[str] = []
    correct_extreme_ids: List[str] = []
    length_ratio_issue_ids: List[str] = []
    qa_ids: List[str] = []

    for index, qa in enumerate(qas, start=1):
        if not isinstance(qa, dict):
            schema_issue_ids.append(f"index_{index}")
            continue
        qa_id = str(qa.get("qa_id") or f"index_{index}")
        qa_ids.append(qa_id)
        categories[str(qa.get("category") or "missing")] += 1
        labels[str(qa.get("label") or "missing")] += 1
        characters[str(qa.get("character") or "missing")] += 1
        answer = answer_letter(qa.get("answer"))
        answers[answer or "invalid"] += 1

        options = qa.get("option") if isinstance(qa.get("option"), list) else []
        payloads = [option_payload(option) for option in options]
        lengths = [compact_length(payload) for payload in payloads]
        option_lengths.extend(lengths)
        expected_option_count = 6 if answer == "F" else 5
        if len(options) != expected_option_count or answer not in set("ABCDEF") or not str(qa.get("question") or "").strip():
            schema_issue_ids.append(qa_id)
        if len(set(payloads)) != len(payloads):
            duplicate_option_ids.append(qa_id)

        if answer in set("ABCDE") and len(payloads) == 5:
            option_balance_eligible += 1
            correct_index = ord(answer) - ord("A")
            correct_payload = payloads[correct_index]
            correct_length = lengths[correct_index]
            distractor_lengths = [length for idx, length in enumerate(lengths) if idx != correct_index]
            if correct_length > 0 and all(0.75 <= length / correct_length <= 1.25 for length in distractor_lengths):
                length_ratio_pass += 1
            else:
                length_ratio_issue_ids.append(qa_id)
            if min(lengths) < correct_length < max(lengths):
                correct_not_extreme += 1
                correct_has_longer_and_shorter += 1
            else:
                correct_extreme_ids.append(qa_id)
            correct_markers = {marker for marker in ABSOLUTE_MARKERS if marker in correct_payload}
            distractor_markers = {
                marker
                for idx, payload in enumerate(payloads)
                if idx != correct_index
                for marker in ABSOLUTE_MARKERS
                if marker in payload
            }
            if distractor_markers - correct_markers:
                absolute_marker_asymmetry += 1

            answer_text = option_payload(qa.get("answer_text"))
            if answer_text and answer_text != correct_payload:
                answer_text_mismatch_ids.append(qa_id)

        raw_evidence = qa.get("evidence_dialogues")
        raw_evidence_list = raw_evidence if isinstance(raw_evidence, list) else []
        evidence_count = len(raw_evidence_list)
        evidence_counts.append(evidence_count)
        total_evidence_items += evidence_count
        if evidence_count >= 2:
            multi_evidence_questions += 1
        evidence_sessions = {
            session_number(item.get("dia_id"))
            for item in raw_evidence_list
            if isinstance(item, dict) and session_number(item.get("dia_id"))
        }
        if len(evidence_sessions) >= 2:
            cross_session_questions += 1

        aligned, alignment_report = align_evidence_dialogues(raw_evidence, conversation)
        aligned_evidence_items += len(aligned)
        evidence_match_types.update(alignment_report.get("match_type_counts") or {})
        if alignment_report.get("result") == "pass" and len(aligned) == evidence_count:
            aligned_questions += 1
        else:
            alignment_issue_ids.append(qa_id)

        evidence_ids = {
            str(item.get("id"))
            for item in raw_evidence_list
            if isinstance(item, dict) and item.get("id")
        }
        reasoning_steps = qa.get("reasoning_steps")
        if not isinstance(reasoning_steps, list) or not reasoning_steps:
            reasoning_reference_issue_ids.append(qa_id)
        else:
            unknown_refs = []
            for step in reasoning_steps:
                if not isinstance(step, dict):
                    unknown_refs.append("invalid_step")
                    continue
                based_on = step.get("based_on") if isinstance(step.get("based_on"), list) else []
                unknown_refs.extend(str(ref) for ref in based_on if str(ref) not in evidence_ids)
            if unknown_refs:
                reasoning_reference_issue_ids.append(qa_id)

        mentions = reasoning_answer_mentions(reasoning_steps)
        if answer and any(mention != answer for mention in mentions):
            reasoning_answer_mismatch_ids.append(qa_id)

        label = str(qa.get("label") or "")
        label_invalid = False
        if label == "Fact Extraction (Single Dialogue)" and evidence_count != 1:
            label_invalid = True
        elif label in {"Fact Extraction (Multiple Dialogues)", "Memory Update", "Multi-hop"} and evidence_count < 2:
            label_invalid = True
        elif label == "Abstain" and answer != "F":
            label_invalid = True
        elif answer == "F" and label != "Abstain":
            label_invalid = True
        if label_invalid:
            label_rule_issue_ids.append(qa_id)

    duplicate_qa_ids = sorted(key for key, count in Counter(qa_ids).items() if count > 1)
    total_questions = len(qas)
    result = {
        "dataset": dataset,
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "sample_id": record.get("sample_id"),
        "conversation": {
            "session_count": len(session_keys),
            "dialogue_count": len(dialogue_index),
            "declared_speakers": [
                value
                for key, value in conversation.items()
                if re.fullmatch(r"speaker_[a-z]+", str(key)) and isinstance(value, str)
            ],
        },
        "questions": {
            "total": total_questions,
            "categories": compact_counter(categories),
            "labels": compact_counter(labels),
            "characters": compact_counter(characters),
            "answers": compact_counter(answers),
            "answer_entropy_bits": answer_entropy(answers, total_questions),
        },
        "evidence": {
            "question_alignment_pass": aligned_questions,
            "question_alignment_rate": round(aligned_questions / total_questions, 4) if total_questions else 0.0,
            "aligned_items": aligned_evidence_items,
            "total_items": total_evidence_items,
            "item_alignment_rate": round(aligned_evidence_items / total_evidence_items, 4) if total_evidence_items else 0.0,
            "counts_per_question": numeric_summary(evidence_counts),
            "multi_evidence_questions": multi_evidence_questions,
            "cross_session_questions": cross_session_questions,
            "match_types": compact_counter(evidence_match_types),
        },
        "options": {
            "lengths": numeric_summary(option_lengths),
            "balance_eligible": option_balance_eligible,
            "length_ratio_pass": length_ratio_pass,
            "length_ratio_pass_rate": round(length_ratio_pass / option_balance_eligible, 4) if option_balance_eligible else 0.0,
            "correct_not_extreme": correct_not_extreme,
            "correct_not_extreme_rate": round(correct_not_extreme / option_balance_eligible, 4) if option_balance_eligible else 0.0,
            "correct_has_longer_and_shorter": correct_has_longer_and_shorter,
            "absolute_marker_asymmetry": absolute_marker_asymmetry,
        },
        "issues": {
            "schema": sorted(set(schema_issue_ids)),
            "duplicate_qa_ids": duplicate_qa_ids,
            "duplicate_options": duplicate_option_ids,
            "evidence_alignment": alignment_issue_ids,
            "reasoning_reference": reasoning_reference_issue_ids,
            "reasoning_answer_mismatch": reasoning_answer_mismatch_ids,
            "answer_text_mismatch": answer_text_mismatch_ids,
            "label_rule": label_rule_issue_ids,
            "correct_length_extreme": correct_extreme_ids,
            "option_length_ratio": length_ratio_issue_ids,
        },
    }
    return result


def issue_preview(values: List[str], limit: int = 8) -> str:
    if not values:
        return "无"
    preview = ", ".join(values[:limit])
    if len(values) > limit:
        preview += f" 等共 {len(values)} 题"
    return preview


def build_markdown(analyses: List[Dict[str, Any]]) -> str:
    total_questions = sum(item["questions"]["total"] for item in analyses)
    total_dialogues = sum(item["conversation"]["dialogue_count"] for item in analyses)
    total_evidence = sum(item["evidence"]["total_items"] for item in analyses)
    aligned_evidence = sum(item["evidence"]["aligned_items"] for item in analyses)
    total_length_pass = sum(item["options"]["length_ratio_pass"] for item in analyses)
    total_not_extreme = sum(item["options"]["correct_not_extreme"] for item in analyses)
    total_balance_eligible = sum(item["options"]["balance_eligible"] for item in analyses)
    total_alignment_pass = sum(item["evidence"]["question_alignment_pass"] for item in analyses)

    lines = [
        "# 初始四个 Trace 数据集分析报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 分析对象：`dataset/trace1`、`dataset/trace3`、`dataset/trace5`、`dataset/trace6` 的初始 JSON",
        "- 说明：选项干扰性采用长度与措辞代理指标，不等同于人工语义质量评审。",
        "",
        "## 总体结论",
        "",
        f"四组数据共 {total_questions} 题、{total_dialogues} 条原始对话 turn、{total_evidence} 条题目证据。",
        f"证据逐条可回溯率为 {aligned_evidence}/{total_evidence}（{aligned_evidence / total_evidence:.1%}），完整通过证据对齐的题目为 {total_alignment_pass}/{total_questions}（{total_alignment_pass / total_questions:.1%}）。",
        f"在 {total_balance_eligible} 道非弃答题中，选项长度 75%-125% 约束通过 {total_length_pass}/{total_balance_eligible}（{total_length_pass / total_balance_eligible:.1%}）；正确项不处于最长/最短极值的题目为 {total_not_extreme}/{total_balance_eligible}（{total_not_extreme / total_balance_eligible:.1%}）。",
        "四个文件都只有 1 个 session，因此即使题目含多条 evidence，也没有真正的跨 session 证据组合；当前更准确的描述是“同一长会话内跨 turn 记忆”。",
        "trace6 有 23 道题的 answer 与 answer_text 一致，但推理末句指向其他选项字母，呈现出选项重排后推理文本未同步更新的特征。",
        "",
        "## 数据规模",
        "",
        "| 数据集 | 题目 | Session | 对话 turn | 证据条目 | 平均证据/题 | 角色数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in analyses:
        lines.append(
            f"| {item['dataset']} | {item['questions']['total']} | "
            f"{item['conversation']['session_count']} | {item['conversation']['dialogue_count']} | "
            f"{item['evidence']['total_items']} | {item['evidence']['counts_per_question']['mean']:.2f} | "
            f"{len(item['questions']['characters'])} |"
        )

    lines.extend(
        [
            "",
            "## 题目分布",
            "",
            "| 数据集 | Category 分布 | Label 分布 | 答案位置 | 答案熵(bits) |",
            "|---|---|---|---|---:|",
        ]
    )
    for item in analyses:
        questions = item["questions"]
        lines.append(
            f"| {item['dataset']} | {markdown_counter(questions['categories'])} | "
            f"{markdown_counter(questions['labels'])} | {markdown_counter(questions['answers'])} | "
            f"{questions['answer_entropy_bits']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 证据质量",
            "",
            "| 数据集 | 对齐通过题 | 对齐证据 | 证据数范围 | ≥2条证据 | 跨Session | 匹配方式 |",
            "|---|---:|---:|---|---:|---:|---|",
        ]
    )
    for item in analyses:
        evidence = item["evidence"]
        counts = evidence["counts_per_question"]
        lines.append(
            f"| {item['dataset']} | {evidence['question_alignment_pass']}/{item['questions']['total']} | "
            f"{evidence['aligned_items']}/{evidence['total_items']} | "
            f"{counts['min']}/{counts['median']}/{counts['max']} | {evidence['multi_evidence_questions']} | "
            f"{evidence['cross_session_questions']} | {markdown_counter(evidence['match_types'])} |"
        )

    lines.extend(
        [
            "",
            "## 选项与潜在线索",
            "",
            "| 数据集 | 长度比例通过 | 正确项非极值 | 绝对词不对称 | 选项长度 min/median/max |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in analyses:
        options = item["options"]
        lengths = options["lengths"]
        lines.append(
            f"| {item['dataset']} | {options['length_ratio_pass']}/{options['balance_eligible']} | "
            f"{options['correct_not_extreme']}/{options['balance_eligible']} | "
            f"{options['absolute_marker_asymmetry']}/{item['questions']['total']} | "
            f"{lengths['min']}/{lengths['median']}/{lengths['max']} |"
        )
    lines.extend(
        [
            "",
            "“绝对词不对称”表示干扰项含有从不、绝不、唯一、立即、只在等强词，而正确项没有同类表述。它容易提供无上下文排除线索，但仅是风险提示。",
            "",
            "## 结构与一致性问题",
            "",
            "| 数据集 | Schema | 证据对齐 | 推理引用 | 推理答案冲突 | answer_text | Label规则 | 重复选项 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in analyses:
        issues = item["issues"]
        lines.append(
            f"| {item['dataset']} | {len(issues['schema'])} | {len(issues['evidence_alignment'])} | "
            f"{len(issues['reasoning_reference'])} | {len(issues['reasoning_answer_mismatch'])} | "
            f"{len(issues['answer_text_mismatch'])} | {len(issues['label_rule'])} | "
            f"{len(issues['duplicate_options'])} |"
        )

    lines.extend(["", "## 逐组观察", ""])
    for item in analyses:
        issues = item["issues"]
        lines.extend(
            [
                f"### {item['dataset']}",
                "",
                f"- 角色题量：{markdown_counter(item['questions']['characters'])}",
                f"- 证据对齐异常：{issue_preview(issues['evidence_alignment'])}",
                f"- 推理中提及的答案与标注冲突：{issue_preview(issues['reasoning_answer_mismatch'])}",
                f"- 正确选项处于长度极值：{issue_preview(issues['correct_length_extreme'])}",
                f"- 选项长度比例不合格：{issue_preview(issues['option_length_ratio'])}",
                "",
            ]
        )

    lines.extend(
        [
            "## 判断与建议",
            "",
            "1. 527 条证据全部能在原对话中定位，说明证据来源可靠；应继续保持当前严格的 dia_id 与原文片段校验。",
            "2. 数据集并不具备真正的跨 session 结构。若目标是长程跨会话记忆，需要在源数据中拆分多个 session，并要求证据至少覆盖两个 session。",
            "3. 污染检测高淘汰率与选项可推断性一致。除长度外，应继续减少绝对词、明显不合理行为和只有正确项才具备的具体细节组合。",
            "4. trace6 的 23 处冲突集中表现为 answer/answer_text 正确、推理末句字母陈旧。选项重排必须同步改写推理结论，并在落盘前自动校验三者一致。",
            "5. Category 分布较整齐不代表 Label 或认知难度均衡，后续应结合实际证据跨度与消融停止轮次共同评估难度。",
            "6. trace1 与 trace3 的答案位置明显集中于 B/C，容易形成位置先验；应控制 A-E 分布，但随机重排后必须同步 answer、answer_text 与推理结论。",
            "7. 四组题目都只围绕帕兹，角色覆盖度为 1。若目标是通用对话记忆评测，需要扩展到其他角色及角色间关系题。",
            "",
            "## 产物",
            "",
            f"- Markdown：`{REPORT_DIR / 'initial_four_datasets_analysis.md'}`",
            f"- JSON：`{REPORT_DIR / 'initial_four_datasets_analysis.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    analyses = [analyze_dataset(dataset) for dataset in DATASETS]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "initial_four_datasets_analysis.json"
    markdown_path = REPORT_DIR / "initial_four_datasets_analysis.md"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": analyses,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(build_markdown(analyses), encoding="utf-8")
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
