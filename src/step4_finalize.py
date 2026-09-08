"""Final schema and semantic QA gate used by pipeline Step 4."""

from __future__ import annotations

import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from src.benchmark_qa_schema import normalize_public_qa
from src.pipeline_utils import log_event, print_kv
from src.step0_qa_generate import call_openai_json
from src.utils import align_evidence_dialogues, write_json_file


ALLOWED_LABELS = {
    "Fact Extraction (Single Dialogue)",
    "Fact Extraction (Multiple Dialogues)",
    "Memory Update",
    "Multi-hop",
    "Abstain",
}

FINAL_SEMANTIC_REVIEW_PROMPT = """
You are the final quality gate for a long-range narrative-memory benchmark.
Review exactly one QA item against its cited, source-aligned evidence.

This is a deletion-only gate. Do not rewrite the QA. Return `drop` whenever a
substantive semantic defect exists; otherwise return `keep`.

Check all of the following:

1. The question is grammatical, self-contained, unambiguous, and asks one
   determinate thing.
2. The keyed answer is uniquely supported by the cited evidence and contains
   no unsupported detail.
3. No distractor is also correct; options are mutually exclusive, comparable,
   plausible, and not answer-revealing.
4. The evidence is sufficient, relevant, and non-redundant, and the reasoning
   does not introduce claims absent from that evidence.
5. `category` matches the memory capability actually tested.
6. `label` matches the evidence/reasoning organization. Multiple-dialogue
   questions may use distant evidence in the same session; cross-session
   evidence is preferred but is not required.
7. Memory Update tracks the same state before and after; Multi-hop contains a
   necessary intermediate inference; Lessons Learned includes an explicit
   experience-to-lesson link; Plans & Commitments distinguishes intention from
   completion; Abstain has no A-E option supported by the provided evidence.
8. The item requires no external knowledge, guessed identity/addressee,
   speculative emotion, or unsupported causality.

Do not drop an item merely because its evidence comes from one session when the
dialogue positions are sufficiently separated and jointly necessary.

Return ONLY this JSON shape:

{
  "decision": "keep" or "drop",
  "issues": [
    {"code": "short_machine_code", "detail": "specific explanation"}
  ],
  "summary": "one-sentence assessment"
}

QA item:

{qa_item}
"""


def validate_final_qa_schema(
    qa_item: Dict[str, Any],
    conversation: Dict[str, Any],
) -> List[str]:
    """Return deterministic structural defects without mutating the QA."""
    errors: List[str] = []
    if not isinstance(qa_item, dict):
        return ["qa_item_not_object"]

    character = str(qa_item.get("character") or "").strip()
    if not character:
        errors.append("missing_character")

    try:
        category = int(qa_item.get("category"))
    except (TypeError, ValueError):
        category = 0
    if category not in range(1, 8):
        errors.append("invalid_category")

    if qa_item.get("label") not in ALLOWED_LABELS:
        errors.append("invalid_label")

    try:
        # This validates the canonical public fields while tolerating the
        # pipeline's auxiliary audit fields and legacy answer syntax such as A.
        normalize_public_qa(qa_item, "final-review", 1)
    except (TypeError, ValueError) as exc:
        errors.append(f"invalid_public_schema: {exc}")

    raw_evidence = qa_item.get("evidence_dialogues")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        errors.append("empty_or_invalid_evidence_dialogues")
        evidence_ids = set()
    else:
        evidence_id_list = [
            str(item.get("id") or "").strip()
            for item in raw_evidence
            if isinstance(item, dict)
        ]
        evidence_ids = set(evidence_id_list)
        expected_ids = [f"E{index}" for index in range(1, len(raw_evidence) + 1)]
        if evidence_id_list != expected_ids:
            errors.append("non_contiguous_or_duplicate_evidence_ids")
        aligned, report = align_evidence_dialogues(raw_evidence, conversation)
        if report.get("result") != "pass" or len(aligned) != len(raw_evidence):
            errors.append(
                "invalid_evidence_alignment: "
                + json.dumps(report.get("invalid_items", []), ensure_ascii=False)
            )

    reasoning = qa_item.get("reasoning_steps")
    if not isinstance(reasoning, list) or not reasoning:
        errors.append("empty_or_invalid_reasoning_steps")
    else:
        for index, step in enumerate(reasoning, start=1):
            if not isinstance(step, dict):
                errors.append(f"reasoning_{index}_not_object")
                continue
            if step.get("step") != index:
                errors.append(f"reasoning_{index}_non_contiguous_step")
            if not str(step.get("inference") or "").strip():
                errors.append(f"reasoning_{index}_empty_inference")
            based_on = step.get("based_on")
            if not isinstance(based_on, list) or not based_on:
                errors.append(f"reasoning_{index}_invalid_based_on")
            elif any(str(ref) not in evidence_ids for ref in based_on):
                errors.append(f"reasoning_{index}_unknown_evidence_reference")

    return errors


def review_qa_semantics(
    qa_item: Dict[str, Any],
    llm_config,
) -> Dict[str, Any]:
    """Ask the configured model for a strict keep/drop semantic verdict."""
    review_fields = (
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
    review_item = {key: qa_item.get(key) for key in review_fields if key in qa_item}
    prompt = FINAL_SEMANTIC_REVIEW_PROMPT.replace(
        "{qa_item}",
        json.dumps(review_item, ensure_ascii=False, indent=2),
    )
    result = call_openai_json(
        answer_prompt=prompt,
        model=llm_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
    )
    decision = str(result.get("decision") or "").strip().lower()
    issues = result.get("issues", [])
    if decision not in {"keep", "drop"}:
        raise ValueError(f"invalid semantic decision: {decision!r}")
    if not isinstance(issues, list):
        raise ValueError("semantic review issues must be a list")
    if decision == "drop" and not issues:
        raise ValueError("semantic drop decision must include at least one issue")
    return {
        "decision": decision,
        "issues": issues,
        "summary": str(result.get("summary") or "").strip(),
    }


def finalize_qa_records(
    input_data: Any,
    llm_config,
    max_workers: int = 1,
    enable_schema_check: bool = True,
    enable_semantic_check: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Delete QA failing deterministic schema or model semantic review."""
    records = copy.deepcopy(input_data)
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        raise ValueError("finalize input must be a record or list of records")

    audit: Dict[str, Any] = {
        "schema_version": 1,
        "before": 0,
        "kept": 0,
        "removed": 0,
        "removed_by_schema": 0,
        "removed_by_semantic": 0,
        "items": [],
    }
    semantic_jobs = []
    schema_drops = set()

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        conversation = record.get("conversation", {})
        qa_items = record.get("qa", [])
        if not isinstance(conversation, dict) or not isinstance(qa_items, list):
            raise ValueError(f"record {record_index} has invalid conversation or qa")
        filename = str(record.get("filename") or f"record_{record_index}")
        audit["before"] += len(qa_items)
        for qa_index, qa_item in enumerate(qa_items):
            schema_errors = (
                validate_final_qa_schema(qa_item, conversation)
                if enable_schema_check
                else []
            )
            if schema_errors:
                schema_drops.add((record_index, qa_index))
                audit["removed_by_schema"] += 1
                audit["items"].append(
                    {
                        "filename": filename,
                        "qa_index": qa_index,
                        "qa_id": qa_item.get("qa_id") if isinstance(qa_item, dict) else None,
                        "question": qa_item.get("question") if isinstance(qa_item, dict) else None,
                        "decision": "drop",
                        "stage": "schema",
                        "issues": schema_errors,
                    }
                )
                continue
            semantic_jobs.append((record_index, qa_index, filename, qa_item))

    semantic_results: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if enable_semantic_check and semantic_jobs:
        workers = min(max(1, int(max_workers or 1)), len(semantic_jobs))
        failures = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="final-semantic") as executor:
            futures = {
                executor.submit(review_qa_semantics, qa_item, llm_config): (
                    record_index,
                    qa_index,
                    filename,
                )
                for record_index, qa_index, filename, qa_item in semantic_jobs
            }
            for future in as_completed(futures):
                record_index, qa_index, filename = futures[future]
                try:
                    semantic_results[(record_index, qa_index)] = future.result()
                except Exception as exc:
                    failures.append((filename, qa_index, str(exc)))
        if failures:
            raise RuntimeError(
                "Step 4 semantic review incomplete; final output was not written: "
                + json.dumps(failures[:20], ensure_ascii=False)
            )

    for record_index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("qa"), list):
            continue
        filename = str(record.get("filename") or f"record_{record_index}")
        survivors = []
        for qa_index, qa_item in enumerate(record["qa"]):
            if (record_index, qa_index) in schema_drops:
                continue
            result = semantic_results.get(
                (record_index, qa_index),
                {"decision": "keep", "issues": [], "summary": "semantic check disabled"},
            )
            if result["decision"] == "drop":
                audit["removed_by_semantic"] += 1
                audit["items"].append(
                    {
                        "filename": filename,
                        "qa_index": qa_index,
                        "qa_id": qa_item.get("qa_id"),
                        "question": qa_item.get("question"),
                        "decision": "drop",
                        "stage": "semantic",
                        "issues": result["issues"],
                        "summary": result["summary"],
                    }
                )
                continue
            survivors.append(qa_item)
        record["qa"] = survivors
        audit["kept"] += len(survivors)

    audit["removed"] = audit["before"] - audit["kept"]
    return records, audit


def finalize_qa_file(
    input_path: str,
    output_path: str,
    llm_config,
    max_workers: int = 1,
    enable_schema_check: bool = True,
    enable_semantic_check: bool = True,
) -> str:
    """Run the final gate, write survivors, and persist a deletion audit."""
    with open(input_path, "r", encoding="utf-8") as handle:
        input_data = json.load(handle)

    final_data, audit = finalize_qa_records(
        input_data,
        llm_config=llm_config,
        max_workers=max_workers,
        enable_schema_check=enable_schema_check,
        enable_semantic_check=enable_semantic_check,
    )
    write_json_file(final_data, output_path, indent=4)
    audit_path = os.path.splitext(output_path)[0] + "_review.json"
    write_json_file(audit, audit_path, indent=2)

    log_event(
        "final_qa_review",
        status="success",
        before=audit["before"],
        kept=audit["kept"],
        removed=audit["removed"],
        removed_by_schema=audit["removed_by_schema"],
        removed_by_semantic=audit["removed_by_semantic"],
        output=output_path,
        audit=audit_path,
    )
    print_kv("removed_by_final_schema", audit["removed_by_schema"], indent=4)
    print_kv("removed_by_final_semantic", audit["removed_by_semantic"], indent=4)
    print_kv("final_review_audit", audit_path, indent=4)
    return output_path


__all__ = [
    "finalize_qa_file",
    "finalize_qa_records",
    "review_qa_semantics",
    "validate_final_qa_schema",
]
