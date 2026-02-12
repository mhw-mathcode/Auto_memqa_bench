#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import numpy as np
from typing import Any, Dict, List, Tuple

# =====================
# 1. 数据加载逻辑
# =====================

def load_json_records(path: str) -> List[Dict[str, Any]]:
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
    outer = item.get("outer_id", "")
    inner = item.get("inner_id", "")
    g_idx = item.get("global_index", "")
    return f"{outer}:{inner}" if outer != "" else f"idx:{g_idx or fallback_i}"

def get_blind_correct_count(item: Dict[str, Any], threshold: float = 0.5) -> int:
    pc = item.get("pollution_check") or item
    responses = pc.get("all_responses_and_scores", [])
    count = 0
    for r in responses:
        try:
            if float(r.get("score", 0.0)) >= threshold:
                count += 1
        except: continue
    return count

# =====================
# 2. 核心分析逻辑 (新增变化率计算)
# =====================

def compute_overlap_curves(
    items: List[Dict[str, Any]],
    x_min: int = 0,
    x_max: int = 10,
    correct_threshold: float = 0.5,
    metric_key: str = "mink_loss"
) -> List[Dict[str, Any]]:
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

# =====================
# 3. 自动判定逻辑 (Elbow Point 检测)
# =====================

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

# =====================
# 4. 绘图与主流程
# =====================

def plot_curves(rows: List[Dict[str, Any]], out_prefix: str, metric_name: str, best_n: int):
    import matplotlib.pyplot as plt
    
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--metric", default="avg_loss")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_prefix", default="calibration_result")
    args = parser.parse_args()

    items = load_json_records(args.input)
    results = compute_overlap_curves(items, metric_key=args.metric, correct_threshold=args.threshold)

    best_n = find_best_threshold(results)

    # 输出调试表格
    print(f"\n{'x':<5} | {'Count (y)':<10} | {'Jaccard':<10} | {'Delta_J':<10}")
    print("-" * 45)
    for r in results:
        mark = " <--" if r["x"] == best_n else ""
        print(f"{r['x']:<5} | {r['y']:<10} | {r['jaccard']:<10.4f} | {r['delta_j']:<10.4f}{mark}")

    print(f"\n结论：推荐阈值 n = {best_n}")
    print(f"依据：在 x={best_n} 处，模型表现与记忆探测的一致性下降最显著变缓（拐点）。")

    plot_curves(results, args.out_prefix, args.metric, best_n)

if __name__ == "__main__":
    main()

# python overlap_curve.py --input temp/membership_results_api.json 

