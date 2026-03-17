import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from full_context import FullContextRunner, resolve_output_path
from src import mcq_scoring
from src.utils import normalize_dataset_records


def _coerce_numeric_key(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _sort_key(value: Any) -> Tuple[int, Any]:
    coerced = _coerce_numeric_key(value)
    if isinstance(coerced, int):
        return (0, coerced)
    return (1, str(coerced))


def _normalize_answer_candidates(raw_candidates: Any, fallback: Any) -> List[str]:
    return mcq_scoring.normalize_answer_candidates(raw_candidates, fallback)


def _strip_prediction_text(text: Any) -> str:
    return mcq_scoring.strip_prediction_text(text)


def _parse_mcq_pred_answers(text: Any) -> Tuple[Set[str], bool]:
    return mcq_scoring.parse_mcq_pred_answers(text)


def _parse_mcq_gt_answers(text: Any) -> Set[str]:
    return mcq_scoring.parse_mcq_gt_answers(text)


def _score_mcq_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    answer_candidates = _normalize_answer_candidates(raw_result.get("answer_fixed"), raw_result.get("answer"))
    score_result = mcq_scoring.score_mcq_prediction(raw_result.get("response", ""), answer_candidates)

    scored_result = dict(raw_result)
    scored_result["mcq_score"] = score_result.get("score", 0.0)
    scored_result["prediction_malformed"] = score_result.get("prediction_malformed", False)
    scored_result["predicted_options"] = score_result.get("predicted_options", [])
    scored_result["ground_truth_options"] = score_result.get("ground_truth_options", [])
    if score_result.get("matched_answer"):
        scored_result["matched_answer"] = score_result.get("matched_answer")
    if score_result.get("matched_ground_truth"):
        scored_result["matched_ground_truth"] = score_result.get("matched_ground_truth")

    return scored_result


def _iter_result_entries(data: Any) -> Iterable[Tuple[Any, int, Dict[str, Any]]]:
    if isinstance(data, dict):
        if isinstance(data.get("qa"), list):
            normalized = normalize_dataset_records(data)
            for conv_idx, item in enumerate(normalized):
                for question_idx, question_item in enumerate(item.get("qa", [])):
                    yield conv_idx, question_idx, question_item
            return

        for conv_key in sorted(data.keys(), key=_sort_key):
            item = data[conv_key]
            if isinstance(item, list):
                for question_idx, question_item in enumerate(item):
                    if isinstance(question_item, dict):
                        yield conv_key, question_idx, question_item
            elif isinstance(item, dict) and isinstance(item.get("qa"), list):
                for question_idx, question_item in enumerate(item.get("qa", [])):
                    if isinstance(question_item, dict):
                        yield conv_key, question_idx, question_item
        return

    if isinstance(data, list):
        normalized = normalize_dataset_records(data)
        for conv_idx, item in enumerate(normalized):
            for question_idx, question_item in enumerate(item.get("qa", [])):
                yield conv_idx, question_idx, question_item


def evaluate_results_data(data: Any) -> Dict[str, List[Dict[str, Any]]]:
    scored_map: Dict[Any, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    found_any = False

    for conv_idx, question_idx, item in _iter_result_entries(data):
        found_any = True
        scored_map[conv_idx][question_idx] = _score_mcq_result(item)

    if not found_any:
        raise ValueError("No evaluable result entries found in the input file.")

    serialized: Dict[str, List[Dict[str, Any]]] = {}
    for conv_idx in sorted(scored_map.keys(), key=_sort_key):
        question_map = scored_map[conv_idx]
        serialized[str(conv_idx)] = [
            question_map[q_idx] for q_idx in sorted(question_map.keys(), key=_sort_key)
        ]
    return serialized


def write_metrics_summary(results_dict: Dict[str, List[Dict[str, Any]]], output_path: Path) -> None:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    all_items: List[Dict[str, Any]] = []

    for items in results_dict.values():
        for item in items:
            all_items.append(item)
            grouped[_coerce_numeric_key(item.get("category", "Unknown"))].append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n" + "=" * 40 + "\n")
        file.write("   FULL CONTEXT MCQ SUMMARY\n")
        file.write("=" * 40 + "\n")

        if not all_items:
            file.write("No results to summarize.\n")
            return

        header = f"{'Category':<10} | {'Count':<6} | {'MCQ':<8}"
        file.write(header + "\n")
        file.write("-" * len(header) + "\n")

        total_score = 0.0
        total_count = 0
        for category in sorted(grouped.keys(), key=_sort_key):
            items = grouped[category]
            count = len(items)
            score_sum = sum(float(item.get("mcq_score", 0.0) or 0.0) for item in items)
            mean_score = score_sum / count if count else 0.0
            total_score += score_sum
            total_count += count
            file.write(f"{str(category):<10} | {count:<6} | {mean_score:<8.4f}\n")

        file.write("-" * len(header) + "\n")
        overall = total_score / total_count if total_count else 0.0
        file.write(f"{'ALL':<10} | {total_count:<6} | {overall:<8.4f}\n")
        file.write("=" * 40 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MCQ outputs, or optionally run full-context inference first and then evaluate."
    )
    parser.add_argument("--input_file", required=True, help="Path to a dataset file or an existing result file.")
    parser.add_argument(
        "--run_full_context",
        action="store_true",
        help="If set, run full-context inference first, then evaluate the generated outputs.",
    )
    parser.add_argument("--raw_output_file", default=None, help="Optional path for raw full-context outputs.")
    parser.add_argument("--eval_output_file", default=None, help="Optional path for MCQ-scored outputs.")
    parser.add_argument("--summary_file", default=None, help="Optional path for the summary log file.")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of worker threads.")
    parser.add_argument("--model", default=None, help="Model name for full-context inference.")
    parser.add_argument("--base_url", default=None, help="Base URL for full-context inference.")
    parser.add_argument("--api_key", default=None, help="API key for full-context inference.")
    parser.add_argument(
        "--figure_view",
        action="store_true",
        help="Include image captions in the serialized conversation when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file).resolve()
    raw_output_path = resolve_output_path(input_path, args.raw_output_file, "_full_context_raw.json")
    eval_output_path = resolve_output_path(input_path, args.eval_output_file, "_full_context_mcq_eval.json")
    summary_output_path = resolve_output_path(input_path, args.summary_file, "_full_context_mcq_summary.log")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.run_full_context:
        runner = FullContextRunner(
            output_path=raw_output_path,
            figure_view=args.figure_view,
            llm_config={
                "model": args.model,
                "base_url": args.base_url,
                "api_key": args.api_key,
            },
        )
        raw_results_dict = runner.process_data_file(str(input_path), max_workers=args.max_workers)
    else:
        with input_path.open("r", encoding="utf-8") as file:
            raw_results_dict = json.load(file)

    scored_results_dict = evaluate_results_data(raw_results_dict)

    eval_output_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_output_path.open("w", encoding="utf-8") as file:
        json.dump(scored_results_dict, file, indent=4, ensure_ascii=False)

    write_metrics_summary(scored_results_dict, summary_output_path)

    if args.run_full_context:
        print(f"Raw responses saved to: {raw_output_path}")
    print(f"Scored results saved to: {eval_output_path}")
    print(f"Summary saved to: {summary_output_path}")


if __name__ == "__main__":
    main()

# python mcq_eval.py --run_full_context --input_file ../result/12_Angry_Men_final.json --max_workers 2 --model qwen3-14b --base_url https://api.vveai.com/v1 --api_key 你的APIKey
