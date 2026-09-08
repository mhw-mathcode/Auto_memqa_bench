#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Personal Memory Dataset 处理流水线主入口
支持通过配置文件管理所有参数

流程说明:
  步骤 0: v0 生成原始问答对
  步骤 1: v0 → v1_refined 问题精炼重构
  步骤 2: v1_refined → v2a → v2b 题目合理性检测
  步骤 3: v2b → v3 污染检查
  步骤 4: v3 → final 生成最终版本
"""

import os
import sys
import argparse
import time
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import Namespace
from config import get_config, VersionManager
from src.pipeline_utils import (
    PipelinePaths,
    RunWorkspaceLock,
    create_run_workspace,
    infer_resume_start_step,
    log_event,
    log_subsection,
    print_pipeline_overview,
    print_log_section,
    print_kv,
    print_run_footer,
    print_run_header,
    print_stage_footer,
    print_stage_header,
    resolve_step_input,
    open_run_workspace,
    run_with_temp_filtered_input,
    setup_run_logging,
    summarize_step_config,
)


STEP_0_GENERATE_QA = "step_0_generate_qa"
STEP_1_REFINE_QA = "step_1_refine_qa"
STEP_2_EVIDENCE_CHECK = "step_2_evidence_check"
STEP_3_POLLUTION_CHECK = "step_3_pollution_check"
STEP_4_FINALIZE = "step_4_finalize"

STAGE_META = {
    0: {
        "name": "生成原始问答对",
        "purpose": "从输入剧本/对话中生成或复用 v0 初始问答。",
    },
    1: {
        "name": "问题精炼与重构",
        "purpose": "合并相似问题、修复冲突表达，形成进入合理性检测前的 v1_refined。",
    },
    2: {
        "name": "题目合理性检测",
        "purpose": "通过仅证据回答与删除证据后的消融测试，验证题目是否被证据支持且证据必要。",
    },
    3: {
        "name": "污染检查",
        "purpose": "在不改变选项顺序的前提下检查无上下文回答、选项泄漏与潜在污染，并输出 v3。",
    },
    4: {
        "name": "生成最终版本",
        "purpose": "累积应用前序规则，再执行最终 schema 与语义质量门禁，删除不合格 QA。",
    },
}

def show_config():
    """显示当前配置"""
    config_loader = get_config()
    config_loader.print_config_summary()


def show_versions():
    """显示版本信息"""
    VersionManager.print_version_info()


def resolve_stage_workers(config_loader, pipeline_cfg, step_key: str, field: str = "max_workers") -> int:
    """Resolve a positive per-stage worker count with the pipeline value as fallback."""
    pipeline_fallback = pipeline_cfg.get("max_workers", 4)
    fallback = pipeline_fallback
    if field != "max_workers":
        fallback = config_loader.get_step_flag(step_key, "max_workers", pipeline_fallback)
    raw_value = config_loader.get_step_flag(step_key, field, fallback)
    try:
        workers = max(1, int(raw_value))
    except (TypeError, ValueError):
        workers = max(1, int(fallback or 1))
        log_event(
            "stage_workers",
            status="fallback",
            step=step_key,
            field=field,
            configured=raw_value,
            resolved=workers,
        )
    return workers


def build_answer_args(step_llm, max_workers: int, **worker_overrides) -> Namespace:
    """构造检测/污染阶段复用的简化参数对象。"""
    return Namespace(
        answer_llm_model=step_llm.model,
        answer_llm_base_url=step_llm.base_url,
        answer_llm_api_key=step_llm.api_key,
        max_workers=max_workers,
        **worker_overrides,
    )


def format_questions_file(path: str, stage_label: str) -> bool:
    """将题目统一格式化为 question + A-E/F options 展示文本，不改变选项顺序。"""
    try:
        from src.step3_pollution_check import format_questions_with_options
        from src.utils import load_json_file, write_json_file

        formatted_data, formatted_count = format_questions_with_options(load_json_file(path))
        write_json_file(formatted_data, path, indent=4)
        log_event(
            "format_questions_with_options",
            status="success",
            stage=stage_label,
            formatted_questions=formatted_count,
            output=path,
        )
        return True
    except Exception as exc:
        log_event(
            "format_questions_with_options",
            status="failed",
            stage=stage_label,
            error=exc,
            output=path,
        )
        return False


def ensure_v0_input(paths: PipelinePaths, input_dir: str, dataset_name: str, input_target: str = None) -> bool:
    """当跳过步骤 0 时，确保 v0 文件存在；不存在则从源文件复制。"""
    if os.path.exists(paths.v0):
        return format_questions_file(paths.v0, "prepare_v0_existing")

    if input_target and os.path.isdir(input_target):
        try:
            from src.utils import load_json_file, write_json_file

            records = []
            for filename in sorted(name for name in os.listdir(input_target) if name.endswith(".json")):
                file_path = os.path.join(input_target, filename)
                data = load_json_file(file_path)
                if isinstance(data, list):
                    record = data[0] if data and isinstance(data[0], dict) else {}
                elif isinstance(data, dict):
                    record = data
                else:
                    record = {}
                if not record:
                    continue
                records.append(
                    {
                        "filename": filename,
                        "conversation": record.get("conversation", {}),
                        "qa": record.get("qa", []),
                    }
                )
            if not records:
                log_event("prepare_v0_input", status="failed", reason="json_records_missing", source=input_target)
                return False
            write_json_file(records, paths.v0, indent=2)
            log_event("prepare_v0_input", status="success", source=input_target, output=paths.v0, records=len(records))
            return format_questions_file(paths.v0, "prepare_v0_folder")
        except Exception as exc:
            log_event("prepare_v0_input", status="failed", reason="folder_prepare_failed", source=input_target, error=exc)
            return False

    source_path = input_target if input_target and os.path.isfile(input_target) else paths.source_dataset_path(input_dir)
    if not os.path.exists(source_path):
        log_event("prepare_v0_input", status="failed", reason="source_missing", source=source_path)
        return False

    shutil.copy(source_path, paths.v0)
    log_event("prepare_v0_input", status="success", source=source_path, output=paths.v0)
    return format_questions_file(paths.v0, "prepare_v0_copied")


def normalize_evidence_mode(raw_mode: str) -> str:
    """Normalize evidence-check mode to the current pipeline modes."""
    mode = str(raw_mode).strip().lower()
    if mode not in {"full", "v2a", "v2b"}:
        log_event("normalize_evidence_mode", status="warning", invalid_mode=mode, fallback="full")
        return "full"
    return mode


def run_generate_stage(config_loader, dataset_name: str, input_dir: str, paths: PipelinePaths, input_target: str = None) -> bool:
    """步骤 0：生成原始问答对。"""
    log_event("stage_0_generate_qa", status="start")
    from src.step0_qa_generate import generate_v0

    step_llm = config_loader.get_step_llm(STEP_0_GENERATE_QA)
    force_generate_new_qa = config_loader.get_step_flag(
        STEP_0_GENERATE_QA,
        "force_generate_new_qa",
        False,
    )
    batch_size = config_loader.get_step_flag(
        STEP_0_GENERATE_QA,
        "speaker_batch_size",
        8,
    )
    enable_self_reflection = config_loader.get_step_flag(
        STEP_0_GENERATE_QA,
        "enable_self_reflection",
        True,
    )
    max_workers = resolve_stage_workers(
        config_loader,
        config_loader.get_pipeline_config(),
        STEP_0_GENERATE_QA,
    )

    generated_v0_path = generate_v0(
        dataset_name,
        input_dir,
        paths.v0,
        step_llm,
        force_generate_new_qa=bool(force_generate_new_qa),
        initial_batch_size=int(batch_size),
        max_workers=max_workers,
        input_target=input_target,
        enable_self_reflection=bool(enable_self_reflection),
    )
    if not generated_v0_path:
        return False
    return format_questions_file(paths.v0, "stage_0_v0")


def run_refine_stage(config_loader, paths: PipelinePaths) -> bool:
    """步骤 1：问题精炼与重构。"""
    log_event("stage_1_refine_qa", status="start")
    from src.step1_new_qa import new_qa_main

    input_path = resolve_step_input("步骤 1", paths.v0, [])
    if not input_path:
        return False

    step_llm = config_loader.get_step_llm(STEP_1_REFINE_QA)
    max_workers = resolve_stage_workers(
        config_loader,
        config_loader.get_pipeline_config(),
        STEP_1_REFINE_QA,
    )
    new_qa_main(
        input_path,
        paths.v1_refined,
        api_key=step_llm.api_key,
        base_url=step_llm.base_url,
        model=step_llm.model,
        max_workers=max_workers,
    )
    return format_questions_file(paths.v1_refined, "stage_1_v1_refined")


def run_evidence_check_stage(config_loader, pipeline_cfg, paths: PipelinePaths) -> bool:
    """步骤 2：题目合理性检测。"""
    log_event("stage_2_evidence_check", status="start")
    from src.step2_evidence_check import evidence_check_main

    input_path = resolve_step_input("步骤 2", paths.v1_refined, [paths.v0])
    if not input_path:
        return False

    step_llm = config_loader.get_step_llm(STEP_2_EVIDENCE_CHECK)
    stage_workers = resolve_stage_workers(config_loader, pipeline_cfg, STEP_2_EVIDENCE_CHECK)
    only_evidence_workers = resolve_stage_workers(
        config_loader,
        pipeline_cfg,
        STEP_2_EVIDENCE_CHECK,
        "only_evidence_max_workers",
    )
    iterative_ablation_workers = resolve_stage_workers(
        config_loader,
        pipeline_cfg,
        STEP_2_EVIDENCE_CHECK,
        "iterative_ablation_max_workers",
    )
    try:
        checkpoint_every_questions = max(
            1,
            int(
                config_loader.get_step_flag(
                    STEP_2_EVIDENCE_CHECK,
                    "checkpoint_every_questions",
                    1,
                )
                or 1
            ),
        )
    except (TypeError, ValueError):
        checkpoint_every_questions = 1
        log_event(
            "evidence_checkpoint_config",
            status="fallback",
            resolved=checkpoint_every_questions,
        )
    ablation_settings = {
        key: config_loader.get_step_flag(
            STEP_2_EVIDENCE_CHECK,
            key,
            default,
        )
        for key, default in {
            "ablation_context_limit": 32768,
            "ablation_prompt_safety_tokens": 4096,
            "ablation_chunk_tokens": 8192,
            "ablation_retrieval_chunks": 6,
            "ablation_chunk_max_workers": 4,
        }.items()
    }
    args = build_answer_args(
        step_llm,
        stage_workers,
        only_evidence_max_workers=only_evidence_workers,
        iterative_ablation_max_workers=iterative_ablation_workers,
        checkpoint_every_questions=checkpoint_every_questions,
        **ablation_settings,
    )
    mode = normalize_evidence_mode(
        config_loader.get_step_flag(STEP_2_EVIDENCE_CHECK, "mode", "full")
    )

    if mode == "v2a":
        log_event("evidence_check_mode", mode="v2a_only_evidence")
        output_path, _ = evidence_check_main(
            args,
            input_path,
            paths.v2a,
            only_evidence=1,
            except_evidence=0,
        )
    elif mode == "v2b":
        v2b_input_path = resolve_step_input(
            "步骤 2(v2b)",
            paths.v2a,
            [paths.v1_refined, paths.v0],
        )
        if not v2b_input_path:
            return False

        log_event("evidence_check_mode", mode="v2b_iterative_ablation", input=v2b_input_path)
        output_path, _ = evidence_check_main(
            args,
            v2b_input_path,
            paths.v2b,
            only_evidence=0,
            except_evidence=1,
        )
    else:
        log_event("evidence_check_mode", mode="full_v2a_to_v2b")
        output_path, _ = evidence_check_main(args, input_path, paths.v2b)

    log_event("stage_2_evidence_check", status="success", output=output_path)
    return True


def run_pollution_stage(config_loader, pipeline_cfg, paths: PipelinePaths) -> bool:
    """步骤 3：污染检查。"""
    log_event("stage_3_pollution_check", status="start")
    from src.step3_pollution_check import pollution_check_main

    input_path = resolve_step_input(
        "步骤 3",
        paths.v2b,
        [paths.v2a, paths.v1_refined, paths.v0],
    )
    if not input_path:
        return False

    step_llm = config_loader.get_step_llm(STEP_3_POLLUTION_CHECK)
    max_workers = resolve_stage_workers(config_loader, pipeline_cfg, STEP_3_POLLUTION_CHECK)
    args = build_answer_args(step_llm, max_workers)
    enable_contamination_check = config_loader.get_step_flag(
        STEP_3_POLLUTION_CHECK,
        "enable_contamination_check",
        False,
    )
    cleanup_temp_files = config_loader.get_step_flag(
        STEP_3_POLLUTION_CHECK,
        "cleanup_temp_files",
        True,
    )

    run_with_temp_filtered_input(
        input_path,
        ["v2a", "v2b"],
        "step3",
        lambda filtered_path: pollution_check_main(
            args,
            filtered_path,
            paths.v3,
            enable_contamination_check=enable_contamination_check,
            cleanup_temp_files=cleanup_temp_files,
        ),
        temp_dir=paths.temp_dir,
    )
    return True


def run_finalize_stage(config_loader, pipeline_cfg, paths: PipelinePaths) -> bool:
    """步骤 4：累积过滤后执行最终 schema 与语义质量门禁。"""
    log_event("stage_4_finalize", status="start")
    from src.step4_finalize import finalize_qa_file

    input_path = resolve_step_input(
        "步骤 4",
        paths.v3,
        [paths.v2b, paths.v2a, paths.v1_refined, paths.v0],
    )
    if not input_path:
        return False

    max_workers = resolve_stage_workers(config_loader, pipeline_cfg, STEP_4_FINALIZE)
    enable_schema_check = bool(
        config_loader.get_step_flag(STEP_4_FINALIZE, "enable_schema_check", True)
    )
    enable_semantic_check = bool(
        config_loader.get_step_flag(STEP_4_FINALIZE, "enable_semantic_check", True)
    )
    step_llm = config_loader.get_step_llm(STEP_4_FINALIZE)
    if enable_semantic_check and not step_llm.model:
        step_llm = config_loader.get_step_llm(STEP_3_POLLUTION_CHECK)
        log_event(
            "stage_4_llm",
            status="fallback",
            reason="step_4_llm_missing",
            source=STEP_3_POLLUTION_CHECK,
            model=step_llm.model,
        )
    if enable_semantic_check and not step_llm.model:
        log_event("stage_4_finalize", status="failed", reason="semantic_llm_missing")
        return False

    def run_final_review(filtered_path: str):
        if not format_questions_file(filtered_path, "stage_4_pre_review"):
            return ""
        return finalize_qa_file(
            filtered_path,
            paths.final,
            llm_config=step_llm,
            max_workers=max_workers,
            enable_schema_check=enable_schema_check,
            enable_semantic_check=enable_semantic_check,
        )

    finalized_path = run_with_temp_filtered_input(
        input_path,
        ["v2a", "v2b", "pollution"],
        "step4",
        run_final_review,
        temp_dir=paths.temp_dir,
        max_workers=max_workers,
    )
    return bool(finalized_path and os.path.exists(paths.final))


def print_timing_summary(step_times: dict, total_time: float):
    """输出流水线耗时统计。"""
    print_log_section("PIPELINE COMPLETED")
    log_subsection("Stage elapsed", indent=2)
    for step_name, step_time in step_times.items():
        minutes = int(step_time // 60)
        seconds = step_time % 60
        print_kv(step_name, f"{minutes}m {seconds:.2f}s ({step_time:.2f}s)", indent=4)
    total_minutes = int(total_time // 60)
    total_seconds = total_time % 60
    print_kv("total", f"{total_minutes}m {total_seconds:.2f}s ({total_time:.2f}s)", indent=4)


def _safe_dataset_name(value: str) -> str:
    """Convert a dataset name or path into a safe run/output stem."""
    raw = str(value or "").strip().strip("\"'")
    if not raw:
        return "dataset"
    stem = os.path.splitext(os.path.basename(raw))[0] if raw.lower().endswith(".json") else os.path.basename(raw)
    stem = stem or raw
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stem)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "dataset"


def resolve_run_target(target: str, pipeline_cfg: dict) -> tuple[str, str]:
    """
    Resolve a CLI/config target.

    Returns (dataset_name, input_target). input_target is an absolute file/dir
    path when the target points to a concrete JSON file or folder; otherwise it
    is None and the legacy dataset/<name> lookup is used.
    """
    input_dir = pipeline_cfg.get("input_dir", "dataset")
    raw_target = str(target or "").strip().strip("\"'")
    if not raw_target:
        return "", None

    candidates = [raw_target]
    if not os.path.isabs(raw_target):
        candidates.append(os.path.join(input_dir, raw_target))

    for candidate in candidates:
        abs_candidate = os.path.abspath(candidate)
        if os.path.isfile(abs_candidate) or os.path.isdir(abs_candidate):
            return _safe_dataset_name(abs_candidate), abs_candidate

    return _safe_dataset_name(raw_target), None


def get_configured_run_targets(pipeline_cfg: dict) -> list[str]:
    """Read batch targets from config.json pipeline settings."""
    targets = []
    for key in ("run_targets", "input_files", "input_dirs"):
        value = pipeline_cfg.get(key, [])
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            targets.extend(str(item).strip() for item in value if str(item).strip())
    return targets


def run_one_target(target: str, start_step: int, end_step: int, pipeline_cfg: dict) -> bool:
    """Run one target in the current process."""
    dataset_name, input_target = resolve_run_target(target, pipeline_cfg)
    if not dataset_name:
        log_event("run_target", status="failed", reason="empty_target", target=target)
        return False

    workspace = create_run_workspace(dataset_name, pipeline_cfg)
    with RunWorkspaceLock(workspace.root_dir):
        cleanup_logging = setup_run_logging(workspace.log_path)
        try:
            log_event(
                "run_workspace",
                status="created",
                target=target,
                dataset_name=dataset_name,
                input_target=input_target,
                root=workspace.root_dir,
                log=workspace.log_path,
            )
            return run_pipeline(
                dataset_name,
                start_step,
                end_step,
                run_temp_dir=workspace.temp_dir,
                run_output_dir=workspace.output_dir,
                run_workspace=workspace,
                input_target=input_target,
            )
        finally:
            cleanup_logging()


def run_resume_target(
    run_dir: str,
    start_step: int | None,
    end_step: int | None,
    pipeline_cfg: dict,
) -> bool:
    """Resume one existing run in place without creating a new timestamp directory."""
    workspace, dataset_name = open_run_workspace(run_dir)
    paths = PipelinePaths(dataset_name, workspace.temp_dir, workspace.output_dir)
    config_loader = get_config()
    evidence_mode = normalize_evidence_mode(
        config_loader.get_step_flag(STEP_2_EVIDENCE_CHECK, "mode", "full")
    )
    inferred_start = infer_resume_start_step(paths, evidence_mode)
    resolved_start = inferred_start if start_step is None else start_step
    resolved_end = 4 if end_step is None else end_step
    if resolved_start > resolved_end:
        raise ValueError(
            f"恢复阶段范围无效: start={resolved_start}, end={resolved_end}"
        )

    with RunWorkspaceLock(workspace.root_dir):
        cleanup_logging = setup_run_logging(workspace.log_path, append=True)
        try:
            print_log_section("RUN RESUME")
            print_kv("run_dir", workspace.root_dir)
            print_kv("dataset", dataset_name)
            print_kv("inferred_start", inferred_start)
            print_kv("step_range", f"{resolved_start} -> {resolved_end}")
            return run_pipeline(
                dataset_name,
                resolved_start,
                resolved_end,
                run_temp_dir=workspace.temp_dir,
                run_output_dir=workspace.output_dir,
                run_workspace=workspace,
                input_target=None,
            )
        finally:
            cleanup_logging()


def run_batch_targets(targets: list[str], args, pipeline_cfg: dict) -> bool:
    """Run multiple configured targets in parallel subprocesses."""
    unique_targets = []
    seen = set()
    for target in targets:
        normalized = str(target).strip()
        if normalized and normalized not in seen:
            unique_targets.append(normalized)
            seen.add(normalized)

    if not unique_targets:
        log_event("batch_run", status="failed", reason="empty_targets")
        return False

    try:
        max_workers = max(1, int(pipeline_cfg.get("batch_max_workers", 1)))
    except (TypeError, ValueError):
        max_workers = 1
    max_workers = min(max_workers, len(unique_targets))

    log_event("batch_run", status="start", targets=len(unique_targets), max_workers=max_workers)

    def run_child(target: str) -> tuple[str, int]:
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--run",
            target,
            "--start",
            str(args.start),
            "--end",
            str(args.end),
            "--config",
            args.config,
        ]
        completed = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return target, completed.returncode

    results = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pipeline-batch") as executor:
        future_map = {executor.submit(run_child, target): target for target in unique_targets}
        for future in as_completed(future_map):
            target, return_code = future.result()
            results.append((target, return_code))
            log_event("batch_run_target", status="success" if return_code == 0 else "failed", target=target, return_code=return_code)

    failed = [(target, code) for target, code in results if code != 0]
    log_event("batch_run", status="success" if not failed else "failed", failed=len(failed), total=len(results))
    return not failed


def run_pipeline(
    dataset_name: str,
    start_step: int = 0,
    end_step: int = 4,
    run_temp_dir: str = None,
    run_output_dir: str = None,
    run_workspace=None,
    input_target: str = None,
):
    """
    运行完整流水线

    Args:
        dataset_name: 数据集名称（不含扩展名）
        start_step: 起始步骤（0-4）
        end_step: 结束步骤（0-4）
    """
    pipeline_start_time = time.time()
    step_times = {}
    config_loader = get_config()
    pipeline_cfg = config_loader.get_pipeline_config()

    def should_run_step(step_idx: int, step_key: str) -> bool:
        """步骤执行条件：在 start/end 范围内且未配置 skip。"""
        if not (start_step <= step_idx <= end_step):
            return False
        try:
            is_skip = bool(config_loader.get_step_flag(step_key, "skip", False))
        except ValueError:
            is_skip = False
        if is_skip:
            log_event("skip_stage", status="skipped", stage=step_idx, reason="skip=true")
            return False
        return True

    input_dir = pipeline_cfg.get('input_dir', 'dataset')
    temp_dir = run_temp_dir or pipeline_cfg.get('temp_dir', 'temp')
    output_dir = run_output_dir or pipeline_cfg.get('output_dir', 'result')

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    paths = PipelinePaths(dataset_name, temp_dir, output_dir)

    print_run_header(dataset_name, start_step, end_step, run_workspace, pipeline_cfg)
    print_pipeline_overview(dataset_name, start_step, end_step)

    def get_stage_config_summary(step_key: str) -> dict:
        try:
            return summarize_step_config(config_loader.get_step_config(step_key))
        except ValueError:
            return {"config": "missing"}

    def execute_logged_stage(
        stage_idx: int,
        step_key: str,
        inputs: dict,
        outputs: dict,
        action,
        timing_label: str,
    ) -> bool:
        meta = STAGE_META[stage_idx]
        print_stage_header(
            stage_idx,
            meta["name"],
            meta["purpose"],
            inputs=inputs,
            outputs=outputs,
            config_summary=get_stage_config_summary(step_key),
        )
        step_start = time.time()
        try:
            ok = bool(action())
        except Exception as exc:
            elapsed = time.time() - step_start
            print_stage_footer(
                stage_idx,
                meta["name"],
                "failed(exception)",
                elapsed,
                outputs=outputs,
                note=str(exc),
            )
            raise

        elapsed = time.time() - step_start
        if ok:
            step_times[timing_label] = elapsed
            print_stage_footer(
                stage_idx,
                meta["name"],
                "success",
                elapsed,
                outputs=outputs,
            )
            return True

        print_stage_footer(
            stage_idx,
            meta["name"],
            "failed",
            elapsed,
            outputs=outputs,
            note="阶段返回失败，详见上方 Execution detail。",
        )
        return False

    def finish(status: str) -> bool:
        total_time = time.time() - pipeline_start_time
        if status == "success":
            print_timing_summary(step_times, total_time)
        print_run_footer(status, total_time, paths, step_times)
        return status == "success"

    try:
        if should_run_step(0, STEP_0_GENERATE_QA):
            def _stage0_action():
                if not run_generate_stage(config_loader, dataset_name, input_dir, paths, input_target=input_target):
                    return False
                log_event("stage_0_generate_qa", status="success", output=paths.v0)
                return True

            if not execute_logged_stage(
                0,
                STEP_0_GENERATE_QA,
                {"dataset_input": input_target or os.path.join(input_dir, dataset_name)},
                {"v0": paths.v0},
                _stage0_action,
                "步骤 0: 生成原始问答对",
            ):
                return finish("failed")
        elif not ensure_v0_input(paths, input_dir, dataset_name, input_target=input_target):
            return finish("failed")

        if should_run_step(1, STEP_1_REFINE_QA):
            if not execute_logged_stage(
                1,
                STEP_1_REFINE_QA,
                {"v0": paths.v0},
                {"v1_refined": paths.v1_refined},
                lambda: run_refine_stage(config_loader, paths),
                "步骤 1: 问题精炼与重构",
            ):
                return finish("failed")

        if should_run_step(2, STEP_2_EVIDENCE_CHECK):
            if not execute_logged_stage(
                2,
                STEP_2_EVIDENCE_CHECK,
                {"v1_refined": paths.v1_refined, "fallback_v0": paths.v0},
                {"v2a": paths.v2a, "v2b": paths.v2b},
                lambda: run_evidence_check_stage(config_loader, pipeline_cfg, paths),
                "步骤 2: 题目合理性检测",
            ):
                return finish("failed")

        if should_run_step(3, STEP_3_POLLUTION_CHECK):
            if not execute_logged_stage(
                3,
                STEP_3_POLLUTION_CHECK,
                {"v2b": paths.v2b, "fallback_v2a": paths.v2a},
                {"v3": paths.v3},
                lambda: run_pollution_stage(config_loader, pipeline_cfg, paths),
                "步骤 3: 污染检查",
            ):
                return finish("failed")

        if should_run_step(4, STEP_4_FINALIZE):
            if not execute_logged_stage(
                4,
                STEP_4_FINALIZE,
                {"v3": paths.v3, "fallback_v2b": paths.v2b},
                {"final": paths.final},
                lambda: run_finalize_stage(config_loader, pipeline_cfg, paths),
                "步骤 4: 生成最终版本",
            ):
                return finish("failed")

        return finish("success")
    except Exception:
        print_run_footer("failed(exception)", time.time() - pipeline_start_time, paths, step_times)
        raise

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Personal Memory Dataset 处理流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 显示配置信息
  python main.py --show-config
  
  # 显示版本信息
  python main.py --show-versions
  
  # 运行完整流水线
  python main.py --run An-Enemy-of-the-People

  # 直接运行单个 JSON 文件
  python main.py --run dataset/standard_ebooks_trace/dracula.json

  # 一次运行多个文件或文件夹
  python main.py --run dataset/standard_ebooks_trace/dracula.json dataset/standard_ebooks_trace/jane_eyre.json
  
  # 从指定步骤开始运行
  python main.py --run An-Enemy-of-the-People --start 2 --end 4
  
  # 只运行特定步骤
  python main.py --run An-Enemy-of-the-People --start 3 --end 3

  # 从步骤 0 开始完整生成
  python main.py --run An-Enemy-of-the-People --start 0 --end 4

  # 从已有运行目录继续
  python main.py --resume-run runs/An-Enemy-of-the-People_20260828_120000
        """
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='显示当前配置信息'
    )
    
    parser.add_argument(
        '--show-versions',
        action='store_true',
        help='显示版本信息'
    )
    
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument(
        '--run',
        type=str,
        nargs='+',
        metavar='DATASET',
        help='运行流水线，指定数据集名称、JSON 文件路径或文件夹路径；可一次传多个目标'
    )

    run_group.add_argument(
        '--resume-run',
        type=str,
        metavar='RUN_DIR',
        help='复用已有运行目录，从第一个未完成阶段继续运行'
    )
    
    parser.add_argument(
        '--start',
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4],
        help='起始步骤 (0-4)；普通运行默认 0，恢复运行默认自动推断'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4],
        help='结束步骤 (0-4)，默认为 4'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='配置文件路径，默认为 config.json'
    )
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # 加载配置
    if args.config != 'config.json':
        config_loader = get_config()
        config_loader.load_config(args.config)
    
    # 处理命令
    if args.show_config:
        show_config()
    elif args.show_versions:
        show_versions()
    elif args.resume_run:
        try:
            ok = run_resume_target(
                args.resume_run,
                args.start,
                args.end,
                get_config().get_pipeline_config(),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            log_event("resume_run", status="failed", error=str(exc))
            ok = False
        sys.exit(0 if ok else 1)
    elif args.run:
        start_step = 0 if args.start is None else args.start
        end_step = 4 if args.end is None else args.end
        if start_step > end_step:
            log_event("cli_args", status="failed", reason="start_step_greater_than_end_step", start=start_step, end=end_step)
            sys.exit(1)
        pipeline_cfg = get_config().get_pipeline_config()
        if len(args.run) == 1:
            ok = run_one_target(args.run[0], start_step, end_step, pipeline_cfg)
            sys.exit(0 if ok else 1)
        args.start = start_step
        args.end = end_step
        ok = run_batch_targets(args.run, args, pipeline_cfg)
        sys.exit(0 if ok else 1)
    else:
        pipeline_cfg = get_config().get_pipeline_config()
        configured_targets = get_configured_run_targets(pipeline_cfg)
        if configured_targets:
            start_step = 0 if args.start is None else args.start
            end_step = 4 if args.end is None else args.end
            if start_step > end_step:
                log_event("cli_args", status="failed", reason="start_step_greater_than_end_step", start=start_step, end=end_step)
                sys.exit(1)
            args.start = start_step
            args.end = end_step
            ok = run_batch_targets(configured_targets, args, pipeline_cfg)
            sys.exit(0 if ok else 1)
        parser.print_help()

if __name__ == "__main__":
    main()

# python -u main.py --resume-run runs_remain/a-kiss-for-the-petals-remembering-how-we-met_revised_fixed_20260906_183426 --start 3
# python -u main.py --resume-run runs/the-house-in-fata-morgana-a-requiem-for-innocence_revised_20260828_132537 --start 3
