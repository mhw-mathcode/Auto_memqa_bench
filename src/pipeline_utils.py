"""Pipeline 编排辅助工具。

这个模块只放和流水线编排相关的通用能力：
- 版本文件路径管理
- 自动运行日志
- 步骤输入回退
- 累积过滤规则
- 临时过滤文件

具体的题目生成、重构、合理性检测和污染检查逻辑仍保留在各自业务模块中。
"""

import json
import os
import re
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Optional


PIPELINE_OVERVIEW = [
    "步骤 0: v0 生成原始问答对",
    "步骤 1: v0 → v1_refined 问题精炼重构",
    "步骤 2: v1_refined → v2a → v2b 题目合理性检测",
    "步骤 3: v2b → v3 污染检查",
    "步骤 4: v3 → final 生成最终版本",
]


@dataclass(frozen=True)
class PipelinePaths:
    """集中管理一次数据集运行中会用到的版本文件路径。"""

    dataset_name: str
    temp_dir: str
    output_dir: str

    @property
    def v0(self) -> str:
        return os.path.join(self.temp_dir, f"{self.dataset_name}_v0.json")

    @property
    def v1_refined(self) -> str:
        return os.path.join(self.temp_dir, f"{self.dataset_name}_v1_refined.json")

    @property
    def v2a(self) -> str:
        return os.path.join(self.temp_dir, f"{self.dataset_name}_v2a.json")

    @property
    def v2b(self) -> str:
        return os.path.join(self.temp_dir, f"{self.dataset_name}_v2b.json")

    @property
    def v3(self) -> str:
        return os.path.join(self.temp_dir, f"{self.dataset_name}_v3.json")

    @property
    def final(self) -> str:
        return os.path.join(self.output_dir, f"{self.dataset_name}_final.json")

    def source_dataset_path(self, input_dir: str) -> str:
        return os.path.join(input_dir, self.dataset_name, f"{self.dataset_name}_1.json")


@dataclass(frozen=True)
class RunWorkspace:
    """一次 pipeline 运行对应的独立工作目录。"""

    root_dir: str
    temp_dir: str
    output_dir: str
    log_path: str


class RunWorkspaceLock:
    """Exclusive process lock for mutating one existing run directory."""

    def __init__(self, run_dir: os.PathLike | str):
        self.lock_path = os.path.join(os.path.abspath(os.fspath(run_dir)), ".pipeline.lock")
        self.token = uuid.uuid4().hex
        self.acquired = False

    @staticmethod
    def _pid_is_running(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def acquire(self) -> None:
        payload = json.dumps({"pid": os.getpid(), "token": self.token})
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                try:
                    with open(self.lock_path, "r", encoding="utf-8") as handle:
                        existing = json.load(handle)
                    existing_pid = int(existing.get("pid", 0))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    existing_pid = 0
                if self._pid_is_running(existing_pid):
                    raise RuntimeError(
                        f"运行目录正在被另一个进程使用 (pid={existing_pid})"
                    )
                try:
                    os.remove(self.lock_path)
                except FileNotFoundError:
                    pass
                continue
            else:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                self.acquired = True
                return
        raise RuntimeError(f"无法获取运行目录锁: {self.lock_path}")

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            with open(self.lock_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("token") == self.token:
                os.remove(self.lock_path)
        except FileNotFoundError:
            pass
        finally:
            self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release()
        return False


def create_run_workspace(dataset_name: str, pipeline_cfg: dict) -> RunWorkspace:
    """创建 runs/{dataset}_{timestamp}/ 目录结构。"""
    runs_dir = pipeline_cfg.get("runs_dir", "runs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join(runs_dir, f"{dataset_name}_{timestamp}")
    temp_dir = os.path.join(root_dir, "temp")
    output_dir = os.path.join(root_dir, "result")

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    return RunWorkspace(
        root_dir=root_dir,
        temp_dir=temp_dir,
        output_dir=output_dir,
        log_path=os.path.join(root_dir, "run.log"),
    )


_STAGE_FILE_RE = re.compile(r"^(?P<dataset>.+)_(?:v0|v1_refined|v2a|v2b|v3)\.json$")


def open_run_workspace(run_dir: os.PathLike | str) -> tuple[RunWorkspace, str]:
    """Open an existing run directory and infer its single dataset name."""
    root_dir = os.path.abspath(os.fspath(run_dir))
    temp_dir = os.path.join(root_dir, "temp")
    output_dir = os.path.join(root_dir, "result")
    if not os.path.isdir(root_dir):
        raise ValueError(f"运行目录不存在: {root_dir}")
    if not os.path.isdir(temp_dir):
        raise ValueError(f"运行目录缺少 temp: {temp_dir}")

    dataset_names = {
        match.group("dataset")
        for name in os.listdir(temp_dir)
        if (match := _STAGE_FILE_RE.match(name))
    }
    if not dataset_names:
        raise ValueError(f"temp 中没有可识别的阶段文件: {temp_dir}")
    if len(dataset_names) > 1:
        raise ValueError(f"temp 中包含多个数据集: {sorted(dataset_names)}")

    os.makedirs(output_dir, exist_ok=True)
    workspace = RunWorkspace(
        root_dir=root_dir,
        temp_dir=temp_dir,
        output_dir=output_dir,
        log_path=os.path.join(root_dir, "run.log"),
    )
    return workspace, next(iter(dataset_names))


def _all_questions_terminal(path: str, phase: str) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as handle:
        data = _normalize_records_for_stats(json.load(handle))
    for record in data:
        qa_items = record.get("qa", [])
        if not isinstance(qa_items, list):
            continue
        for question in qa_items:
            if not isinstance(question, dict):
                return False
            if phase == "v2a":
                result = question.get("only_evidence_check")
                if not isinstance(result, dict) or "result" not in result:
                    return False
            else:
                only_check = question.get("only_evidence_check")
                if (
                    isinstance(only_check, dict)
                    and "result" in only_check
                    and only_check.get("result") != "right"
                ):
                    continue
                summary = question.get("iterative_evidence_ablation_summary")
                if (
                    not isinstance(summary, dict)
                    or "result" not in summary
                    or summary.get("result") == "needs_rerun"
                    or summary.get("needs_rerun") is True
                ):
                    return False
    return True


def infer_resume_start_step(paths: PipelinePaths, evidence_mode: str = "full") -> int:
    """Infer the earliest unfinished pipeline stage in an existing workspace."""
    upstream_exists = any(
        os.path.isfile(path)
        for path in (paths.v1_refined, paths.v2a, paths.v2b, paths.v3, paths.final)
    )
    if not os.path.isfile(paths.v0) and not upstream_exists:
        return 0
    if not os.path.isfile(paths.v1_refined) and not any(
        os.path.isfile(path) for path in (paths.v2a, paths.v2b, paths.v3, paths.final)
    ):
        return 1

    normalized_mode = str(evidence_mode or "full").strip().lower()
    if normalized_mode == "v2a":
        step2_complete = _all_questions_terminal(paths.v2a, "v2a")
    elif normalized_mode == "v2b":
        step2_complete = _all_questions_terminal(paths.v2b, "v2b")
    else:
        step2_complete = (
            _all_questions_terminal(paths.v2a, "v2a")
            and _all_questions_terminal(paths.v2b, "v2b")
        )
    if not step2_complete:
        return 2
    if not os.path.isfile(paths.v3):
        return 3
    return 4


class TeeLogger:
    """同时写入终端和运行日志文件。"""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, message):
        self.stream.write(message)
        self.log_file.write(message)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return bool(getattr(self.stream, "isatty", lambda: False)())


def setup_run_logging(log_path: str, append: bool = False):
    """为一次 pipeline 运行创建日志文件，返回清理函数。"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a" if append else "w", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeLogger(original_stdout, log_file)
    sys.stderr = TeeLogger(original_stderr, log_file)

    def cleanup():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.flush()
        log_file.close()

    return cleanup


def _log_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """把秒数格式化为便于人工阅读的耗时。"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rest = seconds % 60
    return f"{minutes}m {rest:.2f}s"


def format_file_size(size_bytes: int) -> str:
    """把文件大小格式化为 KB/MB。"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def print_log_section(title: str, fill: str = "=") -> None:
    """打印统一样式的日志分节标题。"""
    width = 88
    print("\n" + fill * width)
    print(title)
    print(fill * width)


def print_kv(label: str, value: Any, indent: int = 2) -> None:
    """打印统一键值行。"""
    prefix = " " * indent
    print(f"{prefix}- {label}: {value}")


def log_event(event: str, status: str = "info", indent: int = 2, **details: Any) -> None:
    """打印统一事件日志，适合阶段内部简短记录。"""
    prefix = " " * indent
    print(f"{prefix}EVENT | {event} | status={status}")
    for key, value in details.items():
        if value is None:
            continue
        print_kv(key, value, indent=indent + 2)


def log_subsection(title: str, indent: int = 2) -> None:
    """打印轻量子标题，避免阶段内部到处使用大分隔线。"""
    prefix = " " * indent
    print(f"\n{prefix}{title}")


def summarize_step_config(step_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """提取不会泄露密钥的步骤配置摘要。"""
    if not isinstance(step_cfg, dict):
        return {}

    summary: Dict[str, Any] = {}
    for key in (
        "function",
        "description",
        "api_required",
        "skip",
        "mode",
        "force_generate_new_qa",
        "speaker_batch_size",
        "max_workers",
        "only_evidence_max_workers",
        "iterative_ablation_max_workers",
        "checkpoint_every_questions",
        "enable_contamination_check",
        "cleanup_temp_files",
    ):
        if key in step_cfg:
            summary[key] = step_cfg[key]

    llm_cfg = step_cfg.get("llm", {})
    if isinstance(llm_cfg, dict):
        summary["llm.model"] = llm_cfg.get("model", "")
        summary["llm.base_url"] = llm_cfg.get("base_url", "")
        if llm_cfg.get("api_key"):
            summary["llm.api_key"] = "***"

    return summary


def _normalize_records_for_stats(data: Any):
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if any(key in data for key in ("conversation", "qa", "speaker_a", "speaker_b")):
            return [data]
        return [item for item in data.values() if isinstance(item, dict)]
    return []


def summarize_json_file(path: str) -> Dict[str, Any]:
    """汇总 JSON 产物的存在性、大小和核心数据量。"""
    snapshot: Dict[str, Any] = {
        "path": path,
        "exists": os.path.exists(path),
    }

    if not snapshot["exists"]:
        return snapshot

    if os.path.isdir(path):
        json_files = [name for name in os.listdir(path) if name.endswith(".json")]
        snapshot["type"] = "directory"
        snapshot["json_files"] = len(json_files)
        return snapshot

    stat = os.stat(path)
    snapshot["type"] = "file"
    snapshot["size"] = format_file_size(stat.st_size)
    snapshot["modified_at"] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        snapshot["json_error"] = str(exc)
        return snapshot

    records = _normalize_records_for_stats(data)
    snapshot["records"] = len(records)
    snapshot["qa_items"] = 0
    snapshot["sessions"] = 0
    snapshot["dialogues"] = 0

    for record in records:
        qa_section = record.get("qa", [])
        if isinstance(qa_section, dict):
            snapshot["qa_items"] += len(qa_section)
        elif isinstance(qa_section, list):
            snapshot["qa_items"] += len(qa_section)

        conversation = record.get("conversation", {})
        if isinstance(conversation, dict):
            session_keys = [
                key for key in conversation.keys()
                if key.startswith("session_") and not key.endswith("_date_time")
            ]
            snapshot["sessions"] += len(session_keys)
            for key in session_keys:
                turns = conversation.get(key, [])
                if isinstance(turns, list):
                    snapshot["dialogues"] += len(turns)

    return snapshot


def print_file_snapshot(label: str, path: str, indent: int = 4) -> None:
    """打印单个文件产物快照。"""
    snapshot = summarize_json_file(path)
    prefix = " " * indent
    if not snapshot.get("exists"):
        print(f"{prefix}- {label}: missing | {path}")
        return

    if snapshot.get("type") == "directory":
        print(f"{prefix}- {label}: directory; json_files={snapshot.get('json_files', 0)} | {path}")
        return

    parts = [
        f"exists",
        f"size={snapshot.get('size')}",
        f"records={snapshot.get('records', '-')}",
        f"qa={snapshot.get('qa_items', '-')}",
    ]
    if snapshot.get("sessions", 0):
        parts.append(f"sessions={snapshot.get('sessions')}")
    if snapshot.get("dialogues", 0):
        parts.append(f"dialogues={snapshot.get('dialogues')}")
    if snapshot.get("json_error"):
        parts.append(f"json_error={snapshot.get('json_error')}")
    print(f"{prefix}- {label}: {'; '.join(parts)} | {path}")


def print_run_header(
    dataset_name: str,
    start_step: int,
    end_step: int,
    workspace: Optional[RunWorkspace],
    pipeline_cfg: Dict[str, Any],
) -> None:
    """写入 run.log 开头的运行摘要。"""
    print_log_section("RUN START")
    print_kv("start_time", _log_now())
    print_kv("dataset", dataset_name)
    print_kv("step_range", f"{start_step} -> {end_step}")
    if workspace:
        print_kv("run_dir", workspace.root_dir)
        print_kv("temp_dir", workspace.temp_dir)
        print_kv("result_dir", workspace.output_dir)
        print_kv("log_file", workspace.log_path)

    log_subsection("Pipeline config", indent=2)
    for key, value in pipeline_cfg.items():
        print_kv(key, value, indent=4)


def print_stage_header(
    stage_idx: int,
    stage_name: str,
    purpose: str,
    *,
    inputs: Optional[Dict[str, str]] = None,
    outputs: Optional[Dict[str, str]] = None,
    config_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """写入阶段开始日志，包括目标、输入、预计输出和配置摘要。"""
    print_log_section(f"STAGE {stage_idx} START | {stage_name}", fill="-")
    print_kv("time", _log_now())
    print_kv("purpose", purpose)

    if config_summary:
        log_subsection("Config", indent=2)
        for key, value in config_summary.items():
            print_kv(key, value, indent=4)

    if inputs:
        log_subsection("Inputs", indent=2)
        for label, path in inputs.items():
            print_file_snapshot(label, path, indent=4)

    if outputs:
        log_subsection("Expected outputs", indent=2)
        for label, path in outputs.items():
            print_kv(label, path, indent=4)

    log_subsection("Execution detail", indent=2)


def print_stage_footer(
    stage_idx: int,
    stage_name: str,
    status: str,
    elapsed: float,
    *,
    outputs: Optional[Dict[str, str]] = None,
    note: Optional[str] = None,
) -> None:
    """写入阶段结束日志，包括状态、耗时和实际产物快照。"""
    print_log_section(f"STAGE {stage_idx} END | {stage_name}", fill="-")
    print_kv("time", _log_now())
    print_kv("status", status)
    print_kv("elapsed", format_duration(elapsed))
    if note:
        print_kv("note", note)

    if outputs:
        log_subsection("Output snapshots", indent=2)
        for label, path in outputs.items():
            print_file_snapshot(label, path, indent=4)


def print_run_footer(
    status: str,
    total_time: float,
    paths: Optional[PipelinePaths],
    step_times: Dict[str, float],
) -> None:
    """写入 run.log 结尾的总览。"""
    print_log_section("RUN SUMMARY")
    print_kv("end_time", _log_now())
    print_kv("status", status)
    print_kv("total_elapsed", format_duration(total_time))

    if step_times:
        log_subsection("Stage elapsed", indent=2)
        for step_name, elapsed in step_times.items():
            print_kv(step_name, format_duration(elapsed), indent=4)

    if paths:
        log_subsection("Final artifacts", indent=2)
        artifact_paths = {
            "v0": paths.v0,
            "v1_refined": paths.v1_refined,
            "v2a": paths.v2a,
            "v2b": paths.v2b,
            "v3": paths.v3,
            "final": paths.final,
        }
        for label, path in artifact_paths.items():
            print_file_snapshot(label, path, indent=4)


def print_pipeline_overview(dataset_name: str, start_step: int, end_step: int):
    print_log_section("PIPELINE OVERVIEW")
    print_kv("dataset", dataset_name)
    print_kv("step_range", f"{start_step} -> {end_step}")
    log_subsection("Step plan", indent=2)
    for line in PIPELINE_OVERVIEW:
        print_kv("step", line, indent=4)


def resolve_step_input(
    step_label: str,
    preferred_path: str,
    fallback_paths: Iterable[str],
) -> Optional[str]:
    """按顺序选择存在的输入文件，支持步骤间自动回退。"""
    checked = []
    ordered_candidates = [preferred_path] + list(fallback_paths)

    for candidate in ordered_candidates:
        if not candidate or candidate in checked:
            continue
        checked.append(candidate)
        if os.path.exists(candidate):
            if candidate != preferred_path:
                log_event(
                    "resolve_step_input",
                    status="fallback",
                    step=step_label,
                    missing=os.path.basename(preferred_path),
                    selected=os.path.basename(candidate),
                )
            return candidate

    checked_text = ", ".join(os.path.basename(path) for path in checked)
    log_event("resolve_step_input", status="failed", step=step_label, checked=checked_text)
    return None


def _passes_only_evidence_rule(qa_item: dict) -> bool:
    only_check = qa_item.get("only_evidence_check")
    if isinstance(only_check, dict):
        if only_check.get("skipped") is True and only_check.get("passed") is True:
            return True
        if only_check.get("result") == "skipped_abstain":
            return True
        return only_check.get("result") == "right"
    return True


def _passes_ablation_rule(qa_item: dict) -> bool:
    """消融规则：只有连续五轮删除证据后仍答对才过滤。"""
    summary = qa_item.get("iterative_evidence_ablation_summary")
    if isinstance(summary, dict):
        if "should_filter" in summary:
            return not bool(summary.get("should_filter"))
        if summary.get("reason") == "max_rounds_reached_while_still_right":
            return False
        if summary.get("passed") is True:
            return True

    ablation = qa_item.get("iterative_evidence_ablation")
    if not isinstance(ablation, list):
        return True

    if any(isinstance(record, dict) and record.get("result") == "wrong" for record in ablation):
        return True

    last_record = ablation[-1] if ablation and isinstance(ablation[-1], dict) else {}
    if last_record.get("round") == 5 and last_record.get("result") == "right":
        return False

    return True


def _passes_pollution_rule(qa_item: dict) -> bool:
    pollution_check = qa_item.get("pollution_check")
    if not isinstance(pollution_check, dict):
        return True
    if "result" not in pollution_check:
        return True
    return pollution_check.get("result") == "good"


RULE_CHECKERS = {
    "v2a": _passes_only_evidence_rule,
    "v2b": _passes_ablation_rule,
    "pollution": _passes_pollution_rule,
}


def apply_cumulative_rules(
    input_path: str,
    rule_names: Iterable[str],
    stage_label: str,
    output_path: Optional[str] = None,
    max_workers: int = 1,
):
    """对输入文件应用累积删题规则；仅在指定 output_path 时落盘。"""
    rule_names = list(rule_names)
    unknown_rules = [rule for rule in rule_names if rule not in RULE_CHECKERS]
    if unknown_rules:
        raise ValueError(f"未知过滤规则: {unknown_rules}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    total_before = 0
    total_after = 0
    removed_by_rule = {rule: 0 for rule in rule_names}
    normalized_workers = max(1, int(max_workers or 1))

    def evaluate_item(qa_item):
        for rule in rule_names:
            checker = RULE_CHECKERS.get(rule)
            if checker and not checker(qa_item):
                return qa_item, rule
        return qa_item, None

    for section in data:
        qa_list = section.get("qa", [])
        if not isinstance(qa_list, list):
            continue

        effective_workers = min(normalized_workers, max(1, len(qa_list)))
        if effective_workers == 1:
            evaluated_items = map(evaluate_item, qa_list)
        else:
            with ThreadPoolExecutor(
                max_workers=effective_workers,
                thread_name_prefix="finalize-rules",
            ) as executor:
                evaluated_items = list(executor.map(evaluate_item, qa_list))

        filtered = []
        for qa_item, removed_rule in evaluated_items:
            total_before += 1

            if removed_rule:
                removed_by_rule[removed_rule] += 1
                continue

            filtered.append(qa_item)
            total_after += 1

        section["qa"] = filtered

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    removed_total = total_before - total_after
    log_event(
        "apply_cumulative_rules",
        status="success",
        stage=stage_label,
        before=total_before,
        after=total_after,
        removed=removed_total,
        max_workers=normalized_workers,
    )
    for rule in rule_names:
        print_kv(f"removed_by_{rule}", removed_by_rule.get(rule, 0), indent=4)

    return data


def run_with_temp_filtered_input(
    input_path: str,
    rule_names: Iterable[str],
    stage_label: str,
    runner: Callable[[str], object],
    temp_dir: Optional[str] = None,
):
    """在临时文件中传递过滤结果，执行后立即删除，避免持久化中间文件。"""
    filtered_data = apply_cumulative_rules(input_path, rule_names, stage_label)

    temp_file_path = None
    try:
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"{stage_label}_",
            encoding="utf-8",
            delete=False,
            dir=temp_dir,
        ) as temp_file:
            json.dump(filtered_data, temp_file, indent=4, ensure_ascii=False)
            temp_file_path = temp_file.name

        return runner(temp_file_path)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
