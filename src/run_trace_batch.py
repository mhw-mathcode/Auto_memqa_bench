"""Run the four trace datasets sequentially and build a consolidated report."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
DATASETS = ("trace1", "trace3", "trace5", "trace6")


def classify_pollution_reason(reason: str) -> str:
    """Coarsely separate claimed source memory from option-only inference."""
    text = reason.lower()
    specific_memory_markers = (
        "plot",
        "storyline",
        "lore",
        "episode",
        "the game",
        "the film",
        "the movie",
        "the novel",
        "the series",
        "narrative context of",
        "known fact",
        "pre-trained",
        "recognizable scenario",
    )
    inference_markers = (
        "common sense",
        "logical",
        "logic",
        "eliminat",
        "plausib",
        "linguistic",
        "behavior",
        "the alternatives",
        "the other options",
        "trope",
    )
    if any(marker in text for marker in specific_memory_markers):
        return "claimed_specific_memory"
    if any(marker in text for marker in inference_markers):
        return "plausibility_or_elimination"
    return "other"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_qas(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = load_json(path)
    records = data if isinstance(data, list) else [data]
    result: List[Dict[str, Any]] = []
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("qa"), list):
            result.extend(item for item in record["qa"] if isinstance(item, dict))
    return result


def latest_dataset_runs(dataset: str) -> List[Path]:
    return sorted(
        (path for path in RUNS_DIR.glob(f"{dataset}_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )


def find_created_run(dataset: str, previous: set[Path]) -> Optional[Path]:
    created = [path for path in latest_dataset_runs(dataset) if path not in previous]
    return created[-1] if created else None


def counter_to_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def parse_stage_times(log_text: str) -> Dict[str, str]:
    times: Dict[str, str] = {}
    for match in re.finditer(r"- 步骤\s+(\d):[^\n]*:\s+([^\n]+)", log_text):
        times[f"stage_{match.group(1)}"] = match.group(2).strip()
    total_matches = re.findall(r"- total_elapsed:\s+([^\n]+)", log_text)
    if total_matches:
        times["total"] = total_matches[-1].strip()
    return times


def count_log_diagnostics(log_text: str) -> Dict[str, int]:
    return {
        "evidence_alignment_retries": log_text.count(
            "reason=correct_non_f_answer_without_verifiable_new_evidence"
        ),
        "evidence_json_repaired": len(
            re.findall(r"EVENT \| evidence_json_parse \| status=repaired", log_text)
        ),
        "evidence_json_retries": len(
            re.findall(r"EVENT \| evidence_json_parse \| status=retry", log_text)
        ),
        "qa_only_retries": len(
            re.findall(r"EVENT \| qa_only_answer \| status=retry", log_text)
        ),
        "qa_only_failures": len(
            re.findall(r"EVENT \| qa_only_answer \| status=failed", log_text)
        ),
        "timeouts": len(re.findall(r"timed out|timeout", log_text, flags=re.IGNORECASE)),
        "connection_errors": len(re.findall(r"Connection error", log_text, flags=re.IGNORECASE)),
        "http_500_errors": len(re.findall(r"Error code: 500|InternalServerError", log_text)),
        "evidence_retry_exhausted": log_text.count(
            "correct_non_f_answer_without_verifiable_new_evidence_after_retries"
        ),
    }


def analyze_run(dataset: str, run_dir: Optional[Path], return_code: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "dataset": dataset,
        "return_code": return_code,
        "run_dir": str(run_dir) if run_dir else "",
        "status": "failed",
    }
    if run_dir is None:
        result["error"] = "run directory was not created"
        return result

    log_path = run_dir / "run.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    temp_dir = run_dir / "temp"
    result_dir = run_dir / "result"
    paths = {
        "v0": temp_dir / f"{dataset}_v0.json",
        "v1_refined": temp_dir / f"{dataset}_v1_refined.json",
        "v2a": temp_dir / f"{dataset}_v2a.json",
        "v2b": temp_dir / f"{dataset}_v2b.json",
        "v3": temp_dir / f"{dataset}_v3.json",
        "final": result_dir / f"{dataset}_final.json",
    }
    qas = {name: load_qas(path) for name, path in paths.items()}
    counts = {name: len(items) for name, items in qas.items()}
    result["artifacts"] = {name: str(path) for name, path in paths.items()}
    result["counts"] = counts
    result["status"] = (
        "success"
        if return_code == 0 and paths["final"].exists() and "- status: success" in log_text
        else "failed"
    )
    result["stage_times"] = parse_stage_times(log_text)
    result["diagnostics"] = count_log_diagnostics(log_text)

    v2b_qas = qas["v2b"]
    only_results = Counter()
    ablation_results = Counter()
    ablation_reasons = Counter()
    ablation_stop_rounds = Counter()
    needs_rerun = 0
    evidence_filter_ids: List[str] = []
    ablation_filter_ids: List[str] = []
    for index, qa in enumerate(v2b_qas, start=1):
        qa_id = str(qa.get("qa_id") or f"index_{index}")
        only = qa.get("only_evidence_check") or {}
        only_result = str(only.get("result") or "missing")
        only_results[only_result] += 1
        if only.get("should_filter") is True or only.get("passed") is False:
            if only_result not in {"skipped_abstain", "missing"}:
                evidence_filter_ids.append(qa_id)

        summary = qa.get("iterative_evidence_ablation_summary") or {}
        if summary:
            ablation_results[str(summary.get("result") or "missing")] += 1
            ablation_reasons[str(summary.get("reason") or "missing")] += 1
            if summary.get("stop_round") is not None:
                ablation_stop_rounds[str(summary.get("stop_round"))] += 1
            if summary.get("should_filter") is True:
                ablation_filter_ids.append(qa_id)
            if summary.get("needs_rerun") is True:
                needs_rerun += 1
        else:
            ablation_results["not_run"] += 1

    result["only_evidence"] = {
        "results": counter_to_dict(only_results),
        "filtered": len(evidence_filter_ids),
        "filtered_qa_ids": evidence_filter_ids,
    }
    result["ablation"] = {
        "results": counter_to_dict(ablation_results),
        "reasons": counter_to_dict(ablation_reasons),
        "stop_rounds": counter_to_dict(ablation_stop_rounds),
        "filtered": len(ablation_filter_ids),
        "filtered_qa_ids": ablation_filter_ids,
        "needs_rerun": needs_rerun,
    }

    pollution_results = Counter()
    pollution_scores = Counter()
    pollution_reason_basis = Counter()
    pollution_reason_by_result: Dict[str, Counter] = {}
    pollution_response_count = 0
    pollution_reason_count = 0
    suspected_ids: List[str] = []
    for index, qa in enumerate(qas["v3"], start=1):
        check = qa.get("pollution_check") or {}
        check_result = str(check.get("result") or "skipped")
        pollution_results[check_result] += 1
        if check_result != "skipped":
            score_key = f"{check.get('correct_count', 0)}/{check.get('total_count', 0)}"
            pollution_scores[score_key] += 1
        for response in check.get("all_responses_and_scores") or []:
            if not isinstance(response, dict):
                continue
            pollution_response_count += 1
            reason = str(response.get("reason") or "").strip()
            if not reason:
                continue
            pollution_reason_count += 1
            basis = classify_pollution_reason(reason)
            pollution_reason_basis[basis] += 1
            pollution_reason_by_result.setdefault(check_result, Counter())[basis] += 1
        if check_result == "suspected":
            suspected_ids.append(str(qa.get("qa_id") or f"index_{index}"))
    result["pollution"] = {
        "results": counter_to_dict(pollution_results),
        "score_distribution": counter_to_dict(pollution_scores),
        "filtered": len(suspected_ids),
        "filtered_qa_ids": suspected_ids,
        "response_count": pollution_response_count,
        "reason_count": pollution_reason_count,
        "reason_coverage": (
            round(pollution_reason_count / pollution_response_count, 4)
            if pollution_response_count
            else 0.0
        ),
        "reason_basis": counter_to_dict(pollution_reason_basis),
        "reason_basis_by_result": {
            key: counter_to_dict(value) for key, value in sorted(pollution_reason_by_result.items())
        },
    }
    result["retention_rate"] = (
        round(counts["final"] / counts["v0"], 4) if counts["v0"] else 0.0
    )
    return result


def markdown_counter(values: Dict[str, int]) -> str:
    if not values:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in values.items())


def build_report(batch_dir: Path, analyses: List[Dict[str, Any]]) -> str:
    successful = [item for item in analyses if item["status"] == "success"]
    original_total = sum(item.get("counts", {}).get("v0", 0) for item in successful)
    final_total = sum(item.get("counts", {}).get("final", 0) for item in successful)
    total_only = sum(item.get("only_evidence", {}).get("filtered", 0) for item in successful)
    total_ablation = sum(item.get("ablation", {}).get("filtered", 0) for item in successful)
    total_pollution = sum(item.get("pollution", {}).get("filtered", 0) for item in successful)
    total_needs_rerun = sum(item.get("ablation", {}).get("needs_rerun", 0) for item in successful)

    lines = [
        "# 四个 Trace 数据集筛选分析报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 批次目录：`{batch_dir}`",
        "- 运行范围：步骤 0–4（生成/复用、精炼、仅证据检查、五轮迭代消融、三轮污染检查、最终汇总）",
        "- 污染判定：无上下文独立采样 3 轮，3/3 全部答对才标记为 suspected。",
        "",
        "## 总览",
        "",
        "| 数据集 | 状态 | v0 | v1 | v2a | v2b | 进入污染(v3) | final | 保留率 | 总耗时 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in analyses:
        counts = item.get("counts", {})
        lines.append(
            "| {dataset} | {status} | {v0} | {v1} | {v2a} | {v2b} | {v3} | {final} | {rate:.1%} | {elapsed} |".format(
                dataset=item["dataset"],
                status=item["status"],
                v0=counts.get("v0", 0),
                v1=counts.get("v1_refined", 0),
                v2a=counts.get("v2a", 0),
                v2b=counts.get("v2b", 0),
                v3=counts.get("v3", 0),
                final=counts.get("final", 0),
                rate=item.get("retention_rate", 0.0),
                elapsed=item.get("stage_times", {}).get("total", "-"),
            )
        )
    lines.extend(
        [
            "",
            "## 删除来源",
            "",
            "| 数据集 | 仅证据筛除 | 五轮消融筛除 | 污染筛除 | 消融需复跑但保留 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in analyses:
        lines.append(
            f"| {item['dataset']} | {item.get('only_evidence', {}).get('filtered', 0)} | "
            f"{item.get('ablation', {}).get('filtered', 0)} | "
            f"{item.get('pollution', {}).get('filtered', 0)} | "
            f"{item.get('ablation', {}).get('needs_rerun', 0)} |"
        )

    lines.extend(["", "## 消融结果", ""])
    for item in analyses:
        lines.extend(
            [
                f"### {item['dataset']}",
                "",
                f"- 结果分布：{markdown_counter(item.get('ablation', {}).get('results', {}))}",
                f"- 停止原因：{markdown_counter(item.get('ablation', {}).get('reasons', {}))}",
                f"- 停止轮次：{markdown_counter(item.get('ablation', {}).get('stop_rounds', {}))}",
                "",
            ]
        )

    lines.extend(
        [
            "## 污染检查",
            "",
            "| 数据集 | 结果分布 | 三轮得分分布 |",
            "|---|---|---|",
        ]
    )
    for item in analyses:
        lines.append(
            f"| {item['dataset']} | {markdown_counter(item.get('pollution', {}).get('results', {}))} | "
            f"{markdown_counter(item.get('pollution', {}).get('score_distribution', {}))} |"
        )

    lines.extend(
        [
            "",
            "## 污染理由审计",
            "",
            "理由分类为关键词辅助审计，不参与污染判分。claimed_specific_memory 表示模型声称依据具体作品、情节或已知事实；plausibility_or_elimination 表示主要依靠常识、选项排除或叙事合理性。",
            "",
            "| 数据集 | 有理由/回答 | 理由覆盖率 | 理由依据分布 | suspected 理由依据 | good 理由依据 |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for item in analyses:
        pollution = item.get("pollution", {})
        by_result = pollution.get("reason_basis_by_result", {})
        lines.append(
            f"| {item['dataset']} | {pollution.get('reason_count', 0)}/{pollution.get('response_count', 0)} | "
            f"{pollution.get('reason_coverage', 0.0):.1%} | "
            f"{markdown_counter(pollution.get('reason_basis', {}))} | "
            f"{markdown_counter(by_result.get('suspected', {}))} | "
            f"{markdown_counter(by_result.get('good', {}))} |"
        )

    lines.extend(
        [
            "",
            "## 运行稳定性",
            "",
            "| 数据集 | 证据补取重试 | 证据重试耗尽终态 | JSON修复 | JSON重试 | 无上下文重试 | 无上下文失败 | 超时 | 连接错误 | HTTP 500 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in analyses:
        diag = item.get("diagnostics", {})
        lines.append(
            f"| {item['dataset']} | {diag.get('evidence_alignment_retries', 0)} | "
            f"{diag.get('evidence_retry_exhausted', 0)} | "
            f"{diag.get('evidence_json_repaired', 0)} | {diag.get('evidence_json_retries', 0)} | "
            f"{diag.get('qa_only_retries', 0)} | {diag.get('qa_only_failures', 0)} | "
            f"{diag.get('timeouts', 0)} | {diag.get('connection_errors', 0)} | "
            f"{diag.get('http_500_errors', 0)} |"
        )

    lines.extend(["", "## 结论", ""])
    if successful:
        retention = final_total / original_total if original_total else 0.0
        lines.append(
            f"本批次成功完成 {len(successful)}/{len(analyses)} 个数据集，共从 {original_total} 道 v0 题中保留 "
            f"{final_total} 道，整体保留率为 {retention:.1%}。"
        )
        lines.append(
            f"删除来源合计：仅证据检查 {total_only} 道、五轮消融 {total_ablation} 道、污染检查 {total_pollution} 道。"
        )
        if total_needs_rerun:
            lines.append(
                f"另有 {total_needs_rerun} 道题因证据提取重试耗尽被标记 needs_rerun，但按当前规则保留，未被误删。"
            )
        ranked = sorted(successful, key=lambda item: item.get("retention_rate", 0.0))
        lines.append(
            f"保留率最低的是 {ranked[0]['dataset']}（{ranked[0]['retention_rate']:.1%}），最高的是 "
            f"{ranked[-1]['dataset']}（{ranked[-1]['retention_rate']:.1%}）。"
        )
        dominant = max(
            (("仅证据", total_only), ("迭代消融", total_ablation), ("污染检查", total_pollution)),
            key=lambda pair: pair[1],
        )
        lines.append(f"本批次主要删除来源为{dominant[0]}（{dominant[1]} 道）。")
    else:
        lines.append("本批次没有完整成功的数据集，请依据运行稳定性与各 run.log 定位失败原因。")

    failed = [item for item in analyses if item["status"] != "success"]
    if failed:
        lines.append("未完成数据集：" + "、".join(item["dataset"] for item in failed) + "。")

    lines.extend(["", "## 产物位置", ""])
    for item in analyses:
        lines.append(f"- {item['dataset']}：`{item.get('run_dir') or '未创建'}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = RUNS_DIR / f"trace_batch_{batch_stamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    analyses: List[Dict[str, Any]] = []

    for index, dataset in enumerate(DATASETS, start=1):
        previous = set(latest_dataset_runs(dataset))
        print(f"BATCH START {index}/{len(DATASETS)} dataset={dataset}", flush=True)
        started = time.time()
        completed = subprocess.run(
            [
                sys.executable,
                "-u",
                "main.py",
                "--run",
                dataset,
                "--start",
                "0",
                "--end",
                "4",
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=False,
        )
        run_dir = find_created_run(dataset, previous)
        analysis = analyze_run(dataset, run_dir, completed.returncode)
        analysis["wall_time_seconds"] = round(time.time() - started, 2)
        analyses.append(analysis)
        print(
            f"BATCH END {index}/{len(DATASETS)} dataset={dataset} "
            f"status={analysis['status']} final={analysis.get('counts', {}).get('final', 0)} "
            f"run_dir={analysis.get('run_dir', '')}",
            flush=True,
        )

    report_json = batch_dir / "analysis_report.json"
    report_md = batch_dir / "analysis_report.md"
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "batch_dir": str(batch_dir),
                "datasets": analyses,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    report_md.write_text(build_report(batch_dir, analyses), encoding="utf-8")
    print(f"BATCH REPORT {report_md}", flush=True)
    return 0 if all(item["status"] == "success" for item in analyses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
