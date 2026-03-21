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


def _load_json_if_exists(path: Path) -> Optional[Any]:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        logging.warning("Failed to load existing file %s: %s", path, exc)
        return None


def _is_llm_failure_response(response: Any) -> bool:
    return isinstance(response, str) and response.startswith("Error: Failed to get response from LLM.")


def _normalize_question_key(question: Any) -> str:
    return " ".join(str(question or "").split())


def _entry_quality(item: Dict[str, Any]) -> Tuple[int, int, float]:
    scored = _score_mcq_result(item)
    has_predicted = 1 if bool(scored.get("predicted_options")) else 0
    is_non_error = 0 if _is_llm_failure_response(item.get("response")) else 1
    score_value = float(scored.get("mcq_score", 0.0) or 0.0)
    return is_non_error, has_predicted, score_value


def _build_entry_map(data: Any) -> Dict[Any, Dict[int, Dict[str, Any]]]:
    entry_map: Dict[Any, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for conv_idx, question_idx, item in _iter_result_entries(data):
        if isinstance(item, dict):
            conv_key = _coerce_numeric_key(conv_idx)
            try:
                q_idx = int(question_idx)
            except (TypeError, ValueError):
                continue
            entry_map[conv_key][q_idx] = dict(item)
    return entry_map


def _serialize_entry_map(entry_map: Dict[Any, Dict[int, Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    serialized: Dict[str, List[Dict[str, Any]]] = {}
    for conv_idx in sorted(entry_map.keys(), key=_sort_key):
        question_map = entry_map[conv_idx]
        serialized[str(conv_idx)] = [
            question_map[q_idx] for q_idx in sorted(question_map.keys(), key=_sort_key)
        ]
    return serialized


def _merge_results_data(base_data: Any, patch_data: Any) -> Dict[str, List[Dict[str, Any]]]:
    return _merge_results_map(base_data, _build_entry_map(patch_data))


def _merge_results_map(
    base_data: Any,
    patch_map: Dict[Any, Dict[int, Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    base_map = _build_entry_map(base_data)
    question_index: Dict[str, List[Tuple[Any, int]]] = defaultdict(list)
    for conv_idx, question_map in base_map.items():
        for question_idx, item in question_map.items():
            question_key = _normalize_question_key(item.get("question") if isinstance(item, dict) else "")
            if question_key:
                question_index[question_key].append((conv_idx, question_idx))

    replaced = 0
    kept_old = 0
    matched_by_question = 0
    fallback_to_index = 0
    ambiguous_question = 0
    skipped_missing_target = 0
    for conv_idx, question_map in patch_map.items():
        for question_idx, new_item in question_map.items():
            target_conv_idx = conv_idx
            target_question_idx = question_idx

            question_key = _normalize_question_key(new_item.get("question") if isinstance(new_item, dict) else "")
            matched_positions = question_index.get(question_key, []) if question_key else []
            if len(matched_positions) == 1:
                target_conv_idx, target_question_idx = matched_positions[0]
                matched_by_question += 1
            elif len(matched_positions) > 1:
                ambiguous_question += 1
                if (conv_idx, question_idx) in matched_positions:
                    target_conv_idx, target_question_idx = conv_idx, question_idx
                    fallback_to_index += 1
                else:
                    skipped_missing_target += 1
                    continue
            else:
                fallback_to_index += 1

            old_item = base_map.get(target_conv_idx, {}).get(target_question_idx)
            if old_item is None:
                skipped_missing_target += 1
                continue

            if _entry_quality(new_item) > _entry_quality(old_item):
                base_map[target_conv_idx][target_question_idx] = new_item
                replaced += 1
            else:
                kept_old += 1

    logging.info(
        "Incremental merge completed. replaced=%d, kept_old=%d, matched_by_question=%d, fallback_to_index=%d, ambiguous_question=%d, skipped_missing_target=%d",
        replaced,
        kept_old,
        matched_by_question,
        fallback_to_index,
        ambiguous_question,
        skipped_missing_target,
    )
    return _serialize_entry_map(base_map)


def _collect_dataset_indices(data: Any) -> Set[Tuple[int, int]]:
    normalized = normalize_dataset_records(data)
    indices: Set[Tuple[int, int]] = set()
    for conv_idx, item in enumerate(normalized):
        for question_idx, _ in enumerate(item.get("qa", [])):
            indices.add((conv_idx, question_idx))
    return indices


def _collect_result_indices(data: Any) -> Set[Tuple[int, int]]:
    indices: Set[Tuple[int, int]] = set()
    for conv_idx, question_idx, _ in _iter_result_entries(data):
        conv_key = _coerce_numeric_key(conv_idx)
        if isinstance(conv_key, int):
            indices.add((conv_key, question_idx))
    return indices


def _collect_empty_prediction_targets(scored_data: Any) -> Set[Tuple[int, int]]:
    targets: Set[Tuple[int, int]] = set()
    for conv_idx, question_idx, item in _iter_result_entries(scored_data):
        conv_key = _coerce_numeric_key(conv_idx)
        if not isinstance(conv_key, int):
            continue
        predicted_options = item.get("predicted_options") if isinstance(item, dict) else None
        has_empty_prediction = not isinstance(predicted_options, list) or len(predicted_options) == 0
        is_llm_error = _is_llm_failure_response(item.get("response") if isinstance(item, dict) else None)
        if has_empty_prediction or is_llm_error:
            targets.add((conv_key, question_idx))
    return targets


def _collect_retry_question_keys(scored_data: Any) -> Set[str]:
    question_keys: Set[str] = set()
    for _conv_idx, _question_idx, item in _iter_result_entries(scored_data):
        if not isinstance(item, dict):
            continue
        predicted_options = item.get("predicted_options")
        has_empty_prediction = not isinstance(predicted_options, list) or len(predicted_options) == 0
        is_llm_error = _is_llm_failure_response(item.get("response"))
        if not (has_empty_prediction or is_llm_error):
            continue
        question_key = _normalize_question_key(item.get("question"))
        if question_key:
            question_keys.add(question_key)
    return question_keys


def _collect_retry_targets_by_question(
    dataset_data: Any,
    retry_question_keys: Set[str],
) -> Tuple[Set[Tuple[int, int]], int, int]:
    targets: Set[Tuple[int, int]] = set()
    question_to_indices: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    normalized = normalize_dataset_records(dataset_data)
    for conv_idx, item in enumerate(normalized):
        for question_idx, question_item in enumerate(item.get("qa", [])):
            question_key = _normalize_question_key(question_item.get("question") if isinstance(question_item, dict) else "")
            if question_key:
                question_to_indices[question_key].append((conv_idx, question_idx))

    unmatched_questions = 0
    ambiguous_questions = 0
    for question_key in retry_question_keys:
        matched_indices = question_to_indices.get(question_key, [])
        if not matched_indices:
            unmatched_questions += 1
            continue
        if len(matched_indices) > 1:
            ambiguous_questions += 1
        for index_pair in matched_indices:
            targets.add(index_pair)

    return targets, unmatched_questions, ambiguous_questions


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
        with input_path.open("r", encoding="utf-8") as file:
            dataset_data = json.load(file)

        existing_eval_data = _load_json_if_exists(eval_output_path)
        existing_raw_data = _load_json_if_exists(raw_output_path)

        runner = FullContextRunner(
            output_path=raw_output_path,
            figure_view=args.figure_view,
            llm_config={
                "model": args.model,
                "base_url": args.base_url,
                "api_key": args.api_key,
            },
        )

        can_incremental_retry = False
        retry_targets: Set[Tuple[int, int]] = set()

        if existing_eval_data is not None:
            retry_question_keys = _collect_retry_question_keys(existing_eval_data)
            retry_targets, unmatched_questions, ambiguous_questions = _collect_retry_targets_by_question(
                dataset_data,
                retry_question_keys,
            )
            if retry_question_keys and not retry_targets:
                logging.warning(
                    "Existing eval file %s has retry candidates, but none matched current dataset by question. Falling back to full rerun.",
                    eval_output_path,
                )
            else:
                can_incremental_retry = True
                logging.info(
                    "Question-key retry planning: retry_questions=%d, mapped_targets=%d, unmatched_questions=%d, ambiguous_questions=%d",
                    len(retry_question_keys),
                    len(retry_targets),
                    unmatched_questions,
                    ambiguous_questions,
                )

        if can_incremental_retry:
            if retry_targets:
                logging.info(
                    "Found existing eval file %s. Rerunning only %d entries mapped by question (empty predicted_options or LLM error responses).",
                    eval_output_path,
                    len(retry_targets),
                )
                runner.process_data_file(
                    str(input_path),
                    max_workers=args.max_workers,
                    target_questions=retry_targets,
                    persist_output=False,
                )
                base_results = existing_raw_data if existing_raw_data is not None else existing_eval_data
                raw_results_dict = _merge_results_map(base_results, runner.results)
            else:
                logging.info(
                    "Found existing eval file %s. No empty predicted_options or LLM error responses found, skip full-context rerun.",
                    eval_output_path,
                )
                raw_results_dict = existing_raw_data if existing_raw_data is not None else existing_eval_data

            raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_output_path.open("w", encoding="utf-8") as file:
                json.dump(raw_results_dict, file, indent=4, ensure_ascii=False)
        else:
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

# python mcq_eval.py --run_full_context --input_file ../result/the-man-from-earth-script_final.json --max_workers 2 --model Qwen/Qwen3-14B --base_url https://api.siliconflow.cn/v1 --api_key 你的APIKey
