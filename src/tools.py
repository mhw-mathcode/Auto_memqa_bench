#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration of analysis tools for Personal Memory Dataset
整合分析工具：stats, option_perturbation, overlap_curve, sample_questions
"""

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# ============================================================================
# MODULE 1: STATS - 统计分析工具
# ============================================================================

def write_csv_stats(stats: List[Dict[str, Any]], output_csv: Path) -> None:
    """
    Write category statistics into a CSV file with format:
    版本, 类别, Script1, Script2, ..., sum
    """
    rows = []

    for file_stat in stats:
        # 从文件名解析版本，如 xxx_v0.json -> v0
        file_name = file_stat["file"]
        version = file_name.split("_v")[-1].split(".json")[0]
        version = f"v{version}"

        scripts = file_stat["scripts"]
        script_names = [s["script"] for s in scripts]

        # 收集该版本下的所有类别
        categories = set()
        for s in scripts:
            categories.update(s["category_counts"].keys())

        for category in sorted(categories):
            row = {
                "版本": version,
                "类别": category,
            }

            row_sum = 0
            for s in scripts:
                cnt = s["category_counts"].get(category, 0)
                row[s["script"]] = cnt
                row_sum += cnt

            row["sum"] = row_sum
            rows.append(row)

    # 写 CSV
    if not rows:
        return

    # 不能只看第一行：不同版本可能出现额外脚本列（例如 <unknown>）
    # 需要对所有行做并集，避免 DictWriter 因字段不在 fieldnames 中而报错。
    seen_fields = set()
    for row in rows:
        for key in row.keys():
            if key in ("版本", "类别"):
                continue
            seen_fields.add(key)

    script_fields = sorted([k for k in seen_fields if k != "sum"], key=lambda x: str(x).lower())
    fieldnames = ["版本", "类别"] + script_fields + (["sum"] if "sum" in seen_fields else [])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mixed_sort_key(value: Any) -> Tuple[int, Any]:
    """Sort numeric-like keys numerically first, then text keys alphabetically."""
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text.lower())


def should_include_qa(qa: Dict[str, Any], file_name: str) -> bool:
    """判断QA是否应该被包含在统计中"""
    lower_name = file_name.lower()
    
    # v1 and later: discard if only_evidence_check.result != "right"
    if any(v in lower_name for v in ["v1", "v2", "v3", "v4"]):
        only_check = qa.get("only_evidence_check", {})
        if isinstance(only_check, dict) and only_check.get("result") != "right":
            return False

    # v2 and later: additionally discard if round==5 and result=="right" in ablation
    if any(v in lower_name for v in ["v2", "v3", "v4"]):
        ablation = qa.get("iterative_evidence_ablation", [])
        if isinstance(ablation, list):
            for item in ablation:
                if not isinstance(item, dict):
                    continue
                if item.get("round") == 5 and item.get("result") == "right":
                    return False

    return True

def count_categories(script_obj: Dict[str, Any], file_name: str) -> Dict[str, int]:
    """统计各类别的QA数量"""
    counts: Dict[str, int] = {}
    for qa in script_obj.get("qa", []):
        if not should_include_qa(qa, file_name):
            continue
        category = qa.get("category")
        if category is None:
            continue
        key = str(category)
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_labels(script_obj: Dict[str, Any], file_name: str) -> Dict[str, int]:
    """Count the occurrences of each label across QA items."""
    counts: Dict[str, int] = {}
    for qa in script_obj.get("qa", []):
        if not should_include_qa(qa, file_name):
            continue
        label = qa.get("label")
        if label is None or label == "":
            continue
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    return counts

def collect_excluded_questions(script_obj: Dict[str, Any], file_name: str) -> List[Dict[str, Any]]:
    """Collect questions that were excluded during filtering."""
    excluded = []
    lower_name = file_name.lower()
    
    for qa in script_obj.get("qa", []):
        if should_include_qa(qa, file_name):
            continue  # Skip included questions
        
        reason = ""
        # Determine exclusion reason
        if any(v in lower_name for v in ["v1", "v2", "v3", "v4"]):
            only_check = qa.get("only_evidence_check", {})
            if isinstance(only_check, dict) and only_check.get("result") != "right":
                reason = f"only_evidence_check.result={only_check.get('result', 'N/A')}"
        
        if not reason and any(v in lower_name for v in ["v2", "v3", "v4"]):
            ablation = qa.get("iterative_evidence_ablation", [])
            if isinstance(ablation, list):
                for item in ablation:
                    if not isinstance(item, dict):
                        continue
                    if item.get("round") == 5 and item.get("result") == "right":
                        reason = "iterative_evidence_ablation round=5 result=right"
                        break
        
        # Keep most fields from original QA
        excluded_qa = {
            "qid": qa.get("qid"),
            "character": qa.get("character"),
            "category": qa.get("category"),
            "label": qa.get("label"),
            "question": qa.get("question"),
            "option": qa.get("option"),
            "answer": qa.get("answer"),
            "exclusion_reason": reason
        }
        
        # Optionally include evidence and reasoning if present
        if "evidence_dialogues" in qa:
            excluded_qa["evidence_dialogues"] = qa.get("evidence_dialogues")
        if "reasoning_steps" in qa:
            excluded_qa["reasoning_steps"] = qa.get("reasoning_steps")
        if "only_evidence_check" in qa:
            excluded_qa["only_evidence_check"] = qa.get("only_evidence_check")
        if "iterative_evidence_ablation" in qa:
            excluded_qa["iterative_evidence_ablation"] = qa.get("iterative_evidence_ablation")
        
        excluded.append(excluded_qa)
    
    return excluded


def collect_stats(input_path: Path) -> Dict[str, Any]:
    """从JSON文件收集统计信息"""
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    scripts = []
    scripts_label = []
    all_excluded = []
    
    script_data = data if isinstance(data, list) else [data]
    script_data = sorted(
        script_data,
        key=lambda obj: str(obj.get("filename", "<unknown>")).lower()
    )

    for script_obj in script_data:
        script_name = script_obj.get("filename", "<unknown>")
        category_counts = count_categories(script_obj, input_path.name)
        label_counts = count_labels(script_obj, input_path.name)
        excluded_questions = collect_excluded_questions(script_obj, input_path.name)
        
        scripts.append({
            "script": script_name,
            "category_counts": category_counts,
            "total": sum(category_counts.values()),
        })
        
        scripts_label.append({
            "script": script_name,
            "label_counts": label_counts,
        })
        
        # Add script name to excluded questions and collect
        for excl in excluded_questions:
            excl["script"] = script_name
        all_excluded.extend(excluded_questions)

    totals: Dict[str, int] = {}
    for script in scripts:
        for cat, cnt in script["category_counts"].items():
            totals[cat] = totals.get(cat, 0) + cnt

    return {
        "file": input_path.name,
        "scripts": scripts,
        "scripts_label": scripts_label,
        "totals": totals,
        "total": sum(totals.values()),
        "excluded_questions": all_excluded,
        "excluded_count": len(all_excluded)
    }


def build_question_key(excluded_qa: Dict[str, Any]) -> tuple:
    """
    Build a unique key for a question based on script, question text, and character.
    This is used to match the same question across different versions.
    """
    return (
        excluded_qa.get("script"),
        excluded_qa.get("question"),
        excluded_qa.get("character")
    )


def filter_incremental_excluded(stats: List[Dict[str, Any]]) -> None:
    """
    Filter excluded_questions to only include incremental exclusions.
    For each version after the first, remove questions that were already
    excluded in the previous version.
    
    Modifies stats in-place.
    """
    # Build a set of previously excluded question keys
    prev_excluded_keys = set()
    
    for file_stat in stats:
        current_excluded = file_stat.get("excluded_questions", [])
        
        # Filter current excluded to only those not in previous
        filtered_excluded = [
            qa for qa in current_excluded
            if build_question_key(qa) not in prev_excluded_keys
        ]
        
        # Update counts
        file_stat["excluded_questions"] = filtered_excluded
        file_stat["excluded_count"] = len(filtered_excluded)
        
        # Add current excluded to the previous set for next iteration
        for qa in current_excluded:
            prev_excluded_keys.add(build_question_key(qa))


def analyze_stats(input_dir: str = "temp", output: str = "result/category_stats.json", 
                  output_csv: str = "result/category_stats.csv") -> int:
    """主入口：统计分析"""
    input_dir_path = Path(input_dir)
    if not input_dir_path.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir_path}")

    files = sorted(input_dir_path.glob("*_v*.json"))
    stats = [collect_stats(path) for path in files]
    
    # Filter excluded_questions to show only incremental exclusions
    filter_incremental_excluded(stats)

    # 写 JSON
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 写 CSV
    output_csv_path = Path(output_csv)
    write_csv_stats(stats, output_csv_path)

    print(f"Wrote stats for {len(files)} files")
    print(f"JSON: {output_path}")
    print(f"CSV : {output_csv_path}")
    return 0


# ============================================================================
# MODULE 2: SAMPLE_QUESTIONS - 问题采样工具
# ============================================================================

def load_questions(path: str) -> List[Dict[str, Any]]:
    """从JSON加载问题"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions: List[Dict[str, Any]] = []
    for item in data:
        qa_list = item.get("qa", [])
        if isinstance(qa_list, list):
            questions.extend(qa_list)
    return questions


def group_by_category(questions: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """按类别分组"""
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for q in questions:
        category = q.get("category")
        if isinstance(category, int):
            grouped.setdefault(category, []).append(q)
    return grouped


def sample_questions(
    grouped: Dict[int, List[Dict[str, Any]]],
    per_category: int,
    rng: random.Random,
) -> Dict[int, List[Dict[str, Any]]]:
    """每个类别采样固定数量"""
    sampled: Dict[int, List[Dict[str, Any]]] = {}
    for category in range(1, 7):
        items = grouped.get(category, [])
        if len(items) < per_category:
            raise ValueError(
                f"category {category} only has {len(items)} questions, need {per_category}"
            )
        sampled[category] = rng.sample(items, per_category)
    return sampled


def dump_json_cell(value: Any) -> str:
    """JSON序列化"""
    return json.dumps(value, ensure_ascii=True)


def write_csv_samples(path: str, sampled: Dict[int, List[Dict[str, Any]]]) -> None:
    """写入采样CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Category", "question", "option", "answer", "evidence_dialogues", "reasoning_steps"]
        )
        for category in range(1, 7):
            for q in sampled[category]:
                writer.writerow(
                    [
                        f"类别{category}",
                        q.get("question", ""),
                        dump_json_cell(q.get("option", [])),
                        q.get("answer", ""),
                        dump_json_cell(q.get("evidence_dialogues", [])),
                        dump_json_cell(q.get("reasoning_steps", [])),
                    ]
                )


def sample_and_export(input_path: str = "temp/the-man-from-earth-script_v0.json",
                      output_path: str = "result/random_questions.csv",
                      seed: int = None) -> None:
    """主入口：采样问题"""
    rng = random.Random(seed)
    questions = load_questions(input_path)
    grouped = group_by_category(questions)
    sampled = sample_questions(grouped, per_category=2, rng=rng)
    write_csv_samples(output_path, sampled)
    print(f"✓ 采样完成：{output_path}")


# ============================================================================
# MODULE 3: OVERLAP_CURVE - 重叠分析工具
# ============================================================================

def load_json_records(path: str) -> List[Dict[str, Any]]:
    """加载JSON记录"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list): return obj
        if isinstance(obj, dict): return [obj]
    except json.JSONDecodeError:
        pass
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            o = json.loads(line)
            if isinstance(o, list): records.extend(o)
            elif isinstance(o, dict): records.append(o)
    return records


def make_qid(item: Dict[str, Any], fallback_i: int) -> str:
    """构建问题ID"""
    outer = item.get("outer_id", "")
    inner = item.get("inner_id", "")
    g_idx = item.get("global_index", "")
    return f"{outer}:{inner}" if outer != "" else f"idx:{g_idx or fallback_i}"


def get_blind_correct_count(item: Dict[str, Any], threshold: float = 0.5) -> int:
    """获取盲测正确计数"""
    pc = item.get("pollution_check") or item
    responses = pc.get("all_responses_and_scores", [])
    count = 0
    for r in responses:
        try:
            if float(r.get("score", 0.0)) >= threshold:
                count += 1
        except: continue
    return count


def compute_overlap_curves(
    items: List[Dict[str, Any]],
    x_min: int = 0,
    x_max: int = 10,
    correct_threshold: float = 0.5,
    metric_key: str = "mink_loss"
) -> List[Dict[str, Any]]:
    """计算重叠曲线"""
    processed_data = []
    for i, it in enumerate(items):
        loss = it.get(metric_key)
        if loss is None or math.isnan(float(loss)): continue
        processed_data.append({
            "qid": make_qid(it, i),
            "k": get_blind_correct_count(it, correct_threshold),
            "loss": float(loss)
        })

    if not processed_data:
        raise ValueError(f"Error: No valid data found for metric {metric_key}")

    # 按 Loss 升序排序（可疑度从高到低）
    processed_data.sort(key=lambda x: x["loss"])
    ranked_qids_mi = [d["qid"] for d in processed_data]
    k_map = {d["qid"]: d["k"] for d in processed_data}
    universe_qids = set(ranked_qids_mi)
    
    results = []
    prev_jaccard = None
    
    for x in range(x_min, x_max + 1):
        S1 = {qid for qid in universe_qids if k_map[qid] >= x}
        y = len(S1)
        if y == 0:
            results.append({"x": x, "y": 0, "jaccard": 0.0, "delta_j": 0.0, "overlap": 0.0})
            continue

        S2 = set(ranked_qids_mi[:y])
        intersection = len(S1 & S2)
        union = len(S1 | S2)
        current_jaccard = intersection / union if union > 0 else 0
        
        # 计算变化率 (一阶导数)
        delta = (current_jaccard - prev_jaccard) if prev_jaccard is not None else 0
        
        results.append({
            "x": x,
            "y": y,
            "jaccard": current_jaccard,
            "delta_j": delta,
            "overlap": intersection / y if y > 0 else 0
        })
        prev_jaccard = current_jaccard
        
    return results


def find_best_threshold(rows: List[Dict[str, Any]]) -> int:
    """
    寻找最佳 n：排除 x=0 后，寻找下降最慢（delta_j 最大/最接近0）的点。
    """
    # 过滤掉平凡解 x=0 和 x=1 (通常包含随机噪声)
    candidates = [r for r in rows if r["x"] >= 2]
    if not candidates:
        return 0
    
    # 寻找 delta_j 最大的点（即曲线下降的阻力位/平台期）
    best_row = max(candidates, key=lambda r: r["delta_j"])
    return best_row["x"]


def plot_curves(rows: List[Dict[str, Any]], out_prefix: str, metric_name: str, best_n: int):
    """绘制重叠曲线"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib未安装，跳过绘图")
        return
    
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    jaccards = [r["jaccard"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_y = 'tab:blue'
    ax1.set_xlabel('Blind Test Correct Count (x)')
    ax1.set_ylabel('Sample Count (Y)', color=color_y)
    ax1.plot(xs, ys, marker='s', color=color_y, alpha=0.6, label='Suspected Count')
    ax1.tick_params(axis='y', labelcolor=color_y)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Jaccard Similarity', color='black')
    ax2.plot(xs, jaccards, marker='o', color='tab:green', linewidth=3, label='Jaccard Index')
    ax2.set_ylim(0, 1.1)

    # 标注最佳阈值
    ax2.axvline(x=best_n, color='red', linestyle='--', alpha=0.8)
    ax2.annotate(f'Recommended n={best_n}', xy=(best_n, 0.5), color='red', fontweight='bold', ha='right')

    plt.title(f'Calibration Analysis ({metric_name})\nFinding Consensus Plateau')
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=300)
    print(f"✓ 图表保存：{out_prefix}.png")


def analyze_overlap(input_path: str, metric: str = "avg_loss", threshold: float = 0.5,
                   out_prefix: str = "calibration_result") -> Dict[str, Any]:
    """主入口：分析重叠"""
    items = load_json_records(input_path)
    results = compute_overlap_curves(items, metric_key=metric, correct_threshold=threshold)

    best_n = find_best_threshold(results)

    # 输出调试表格
    print(f"\n{'x':<5} | {'Count (y)':<10} | {'Jaccard':<10} | {'Delta_J':<10}")
    print("-" * 45)
    for r in results:
        mark = " <--" if r["x"] == best_n else ""
        print(f"{r['x']:<5} | {r['y']:<10} | {r['jaccard']:<10.4f} | {r['delta_j']:<10.4f}{mark}")

    print(f"\n结论：推荐阈值 n = {best_n}")
    print(f"依据：在 x={best_n} 处，模型表现与记忆探测的一致性下降最显著变缓（拐点）。")

    plot_curves(results, out_prefix, metric, best_n)
    
    return {"best_n": best_n, "results": results}


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="Integrated analysis tools for Personal Memory Dataset")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 统计分析子命令
    stats_parser = subparsers.add_parser("stats", help="Analyze category statistics")
    stats_parser.add_argument("--input-dir", default="temp", help="Directory containing *_v*.json files")
    stats_parser.add_argument("--output", default="result/category_stats.json", help="Output JSON path")
    stats_parser.add_argument("--output-csv", default="result/category_stats.csv", help="Output CSV path")

    # 采样子命令
    sample_parser = subparsers.add_parser("sample", help="Sample questions from dataset")
    sample_parser.add_argument("--input", default="temp/the-man-from-earth-script_v0.json", help="Input JSON path")
    sample_parser.add_argument("--output", default="result/random_questions.csv", help="Output CSV path")
    sample_parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # 重叠分析子命令
    overlap_parser = subparsers.add_parser("overlap", help="Analyze overlap curves")
    overlap_parser.add_argument("--input", required=True, help="Input JSON path")
    overlap_parser.add_argument("--metric", default="avg_loss", help="Metric key")
    overlap_parser.add_argument("--threshold", type=float, default=0.5, help="Correct threshold")
    overlap_parser.add_argument("--out-prefix", default="calibration_result", help="Output prefix")

    args = parser.parse_args()

    if args.command == "stats":
        analyze_stats(args.input_dir, args.output, args.output_csv)
    elif args.command == "sample":
        sample_and_export(args.input, args.output, args.seed)
    elif args.command == "overlap":
        analyze_overlap(args.input, args.metric, args.threshold, args.out_prefix)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
# 1. 运行完整流水线
python main.py --run the-man-from-earth-script

# 2. 统计各版本的数据分布
python -m src.tools stats

# 3. 从最终版本采样用于人工评审
python -m src.tools sample --input temp/the-man-from-earth-script_v3.json --seed 2024

# 4. 分析污染检测结果
python -m src.tools overlap --input temp/contamination.json --metric mink_loss
"""
