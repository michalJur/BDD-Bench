from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any

from bdd_bench.common.diff_parser import parse_diff_git_header_paths
from bdd_bench.common.reporting import (
    DEFAULT_REPORT_CONFIG_DESCRIPTION,
    is_default_report_config,
)
from bdd_bench.common.selection import matches_repo
from bdd_bench.common.test_file_rules import is_non_bdd_test_path, is_test_path


DEFAULT_OUTPUT_DIR = Path("output_evaluation")
DEFAULT_MARKDOWN_NAME = "evaluation_runs_summary.md"
CHAIN_BLOCKED_PREFIX = "Blocked by release-chain stop:"
TRAJECTORY_TIMESTAMP_RE = re.compile(r"(?:^|_)(\d{8}_\d{6})(?:[._]|$)")


@dataclass
class Counts:
    available: int = 0
    reported: int = 0
    resolved: int = 0
    attempted: int = 0
    attempted_resolved: int = 0
    blocked: int = 0
    failed_tests: int = 0
    failed_apply: int = 0
    errors: int = 0
    infra: int = 0
    skipped: int = 0
    not_parsed: int = 0
    missing_after_attempt: int = 0
    missing_not_started: int = 0

    def add(self, other: "Counts") -> None:
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))


@dataclass
class LocCounts:
    available: int | None = 0
    resolved: int | None = 0

    def add(self, other: "LocCounts") -> None:
        if self.available is None or other.available is None:
            self.available = None
            self.resolved = None
            return
        self.available += other.available
        self.resolved = (self.resolved or 0) + (other.resolved or 0)


@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    agent_seconds: float = 0.0
    cost: float = 0.0
    has_tokens: bool = False
    has_agent_time: bool = False
    has_cost: bool = False

    def add(self, other: "UsageStats") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.agent_seconds += other.agent_seconds
        self.cost += other.cost
        self.has_tokens = self.has_tokens or other.has_tokens
        self.has_agent_time = self.has_agent_time or other.has_agent_time
        self.has_cost = self.has_cost or other.has_cost


@dataclass
class RunStats:
    run_id: str
    run_dir: Path
    agent: str
    model: str
    effort: str
    mode: str
    progression: str
    result_path: Path | None
    counts: Counts
    loc_counts: LocCounts
    usage: UsageStats
    repo_counts: dict[str, Counts] = field(default_factory=dict)
    repo_failure_reasons: dict[str, str] = field(default_factory=dict)
    config_error: str | None = None
    result_error: str | None = None

    @property
    def group_key(self) -> tuple[str, str, str]:
        return self.agent, self.model, self.effort


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize evaluation runs and write a Markdown report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Evaluation output directory containing run folders.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help=("Markdown destination. Defaults to " "OUTPUT/evaluation_runs_summary.md."),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Find run directories recursively instead of only direct children.",
    )
    parser.add_argument(
        "--include-old",
        action="store_true",
        help="Include paths under OUTPUT/OLD when --recursive is used.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_reported_result_file(run_dir: Path) -> Path | None:
    result_dir = run_dir / "results"
    if not result_dir.is_dir():
        return None
    report_files = sorted(result_dir.glob("*.md"), key=lambda path: path.stat().st_mtime)
    for report_path in reversed(report_files):
        result_path = report_path.with_suffix(".json")
        if result_path.is_file():
            return result_path
    return None


def _message_usage(message: dict[str, Any]) -> dict[str, Any] | None:
    usage = message.get("usage")
    if isinstance(usage, dict):
        return usage
    extra = message.get("extra")
    if not isinstance(extra, dict):
        return None
    response = extra.get("response")
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    return usage if isinstance(usage, dict) else None


def _usage_token_count(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
    return 0


def _trajectory_start_timestamp(path: Path) -> float | None:
    match = TRAJECTORY_TIMESTAMP_RE.search(path.name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return None


def _trajectory_usage(path: Path) -> UsageStats:
    stats = UsageStats()
    try:
        trajectory = _load_json(path)
    except Exception:
        return stats

    timestamps: list[float] = []
    fallback_cost = 0.0
    messages = trajectory.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            usage = _message_usage(message)
            if usage is not None:
                input_tokens = _usage_token_count(usage, "input_tokens", "prompt_tokens")
                output_tokens = _usage_token_count(
                    usage,
                    "output_tokens",
                    "completion_tokens",
                )
                total_tokens = _usage_token_count(usage, "total_tokens")
                stats.input_tokens += input_tokens
                stats.output_tokens += output_tokens
                stats.total_tokens += total_tokens or input_tokens + output_tokens
                stats.has_tokens = True

            extra = message.get("extra")
            if not isinstance(extra, dict):
                continue
            timestamp = extra.get("timestamp")
            if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
                timestamps.append(float(timestamp))
            message_cost = extra.get("cost")
            if isinstance(message_cost, (int, float)) and not isinstance(message_cost, bool):
                fallback_cost += float(message_cost)

    info = trajectory.get("info")
    model_stats = info.get("model_stats") if isinstance(info, dict) else None
    instance_cost = model_stats.get("instance_cost") if isinstance(model_stats, dict) else None
    if isinstance(instance_cost, (int, float)) and not isinstance(instance_cost, bool):
        stats.cost = float(instance_cost)
        stats.has_cost = True
    elif fallback_cost:
        stats.cost = fallback_cost
        stats.has_cost = True

    if timestamps:
        start_timestamp = _trajectory_start_timestamp(path)
        if start_timestamp is None:
            start_timestamp = min(timestamps)
        end_timestamp = max(timestamps)
        if end_timestamp >= start_timestamp:
            stats.agent_seconds = end_timestamp - start_timestamp
            stats.has_agent_time = True
    return stats


def _usage_from_trajectories(run_dir: Path) -> UsageStats:
    usage = UsageStats()
    for path in sorted(run_dir.glob("trajectories/*/*.traj.json")):
        usage.add(_trajectory_usage(path))
    return usage


def _candidate_run_dirs(output_dir: Path, *, recursive: bool, include_old: bool) -> list[Path]:
    if recursive:
        candidates = [path.parent for path in output_dir.glob("**/config.json")]
    else:
        candidates = [
            path
            for path in output_dir.iterdir()
            if path.is_dir() and (path / "config.json").is_file()
        ]
    if not include_old:
        old_root = output_dir / "OLD"
        candidates = [
            path for path in candidates if old_root not in path.parents and path != old_root
        ]
    return sorted(set(candidates), key=lambda path: str(path))


def _dict_value(source: dict[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key)
    return value if isinstance(value, dict) else {}


def _config_agent(config: dict[str, Any], result: dict[str, Any] | None) -> str:
    agent_config = _dict_value(config, "agent_config")
    result_payload = result or {}
    return str(
        config.get("agent_name")
        or agent_config.get("agent_name")
        or result_payload.get("agent_name")
        or "-"
    )


def _config_model(config: dict[str, Any]) -> str:
    agent = _dict_value(config, "agent")
    agent_config = _dict_value(config, "agent_config")
    mini_swe = _dict_value(config, "mini_swe")
    return str(
        agent.get("model_name")
        or agent_config.get("model_name")
        or mini_swe.get("model_name")
        or "-"
    )


def _config_effort(config: dict[str, Any]) -> str:
    agent = _dict_value(config, "agent")
    agent_config = _dict_value(config, "agent_config")
    mini_swe = _dict_value(config, "mini_swe")
    return str(
        agent.get("reasoning_effort")
        or agent_config.get("reasoning_effort")
        or mini_swe.get("reasoning_effort")
        or "-"
    )


def _run_progression(config: dict[str, Any]) -> str:
    execution = _dict_value(config, "execution")
    return str(execution.get("progression_mode") or "-")


def _result_repo(instance_id: str) -> str:
    return instance_id.split("#", maxsplit=1)[0] if "#" in instance_id else "-"


def _instance_repo(instance: dict[str, Any]) -> str:
    owner = instance.get("owner")
    repo = instance.get("repo")
    if isinstance(owner, str) and isinstance(repo, str):
        return f"{owner}/{repo}"
    return str(repo or "-")


def _resolve_instances_file(path_text: Any, *, run_dir: Path) -> Path | None:
    if not isinstance(path_text, str) or not path_text.strip():
        return None
    raw = Path(path_text)
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, run_dir / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


@cache
def _load_instances(path: Path) -> tuple[dict[str, Any], ...]:
    payload = _load_json(path)
    instances = payload.get("instances")
    if not isinstance(instances, list):
        return ()
    return tuple(instance for instance in instances if isinstance(instance, dict))


def _selected_instances(config: dict[str, Any], *, run_dir: Path) -> list[dict[str, Any]]:
    instances_file = _resolve_instances_file(config.get("instances_file"), run_dir=run_dir)
    if instances_file is None:
        return []

    try:
        instances = _load_instances(instances_file)
    except Exception:
        return []

    selection = _dict_value(config, "selection")
    selected_repo = selection.get("repo")
    selected_instance_id = selection.get("instance_id")

    selected: list[dict[str, Any]] = []
    for instance in instances:
        if isinstance(selected_instance_id, str) and (
            instance.get("instance_id") != selected_instance_id
        ):
            continue
        owner = instance.get("owner")
        repo_name = instance.get("repo")
        if not matches_repo(
            owner if isinstance(owner, str) else None,
            repo_name if isinstance(repo_name, str) else None,
            selected_repo if isinstance(selected_repo, str) else None,
        ):
            continue
        selected.append(instance)
    return selected


def _available_by_repo(instances: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for instance in instances:
        counts[_instance_repo(instance)] += 1
    return dict(counts)


def _instance_id(instance: dict[str, Any]) -> str:
    return str(instance.get("instance_id") or "")


def _clean_diff_path(raw: str) -> str:
    path = raw.strip().split("\t", maxsplit=1)[0]
    if path in {"", "/dev/null"}:
        return ""
    return path[2:] if path.startswith(("a/", "b/")) else path


@cache
def _production_python_added_lines(path: Path, repo: str) -> int | None:
    if not path.is_file():
        return None

    current_path = ""
    added_lines = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("diff --git "):
            parsed = parse_diff_git_header_paths(line)
            if parsed is not None:
                current_path = _clean_diff_path(parsed[1]) or _clean_diff_path(parsed[0])
            continue
        if line.startswith("+++ "):
            current_path = _clean_diff_path(line[4:]) or current_path
            continue
        if line.startswith(("--- ", "@@")):
            continue
        if (
            current_path.endswith(".py")
            and not is_test_path(current_path, repo=repo)
            and not is_non_bdd_test_path(current_path, repo=repo)
            and line.startswith("+")
            and line[1:].strip()
        ):
            added_lines += 1
    return added_lines


def _resolve_instance_path(path_text: Any, *, run_dir: Path) -> Path | None:
    if not isinstance(path_text, str) or not path_text.strip():
        return None
    raw = Path(path_text)
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, run_dir / raw]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _loc_counts(
    instances: list[dict[str, Any]],
    result_payload: dict[str, Any] | None,
    *,
    run_dir: Path,
) -> LocCounts:
    weights: dict[str, int] = {}
    for instance in instances:
        instance_id = _instance_id(instance)
        patches = _dict_value(instance, "patches")
        patch_path = _resolve_instance_path(patches.get("golden_patch"), run_dir=run_dir)
        if not instance_id or patch_path is None:
            return LocCounts(available=None, resolved=None)
        weight = _production_python_added_lines(patch_path, _instance_repo(instance))
        if weight is None:
            return LocCounts(available=None, resolved=None)
        weights[instance_id] = weight

    if result_payload is None:
        return LocCounts(available=sum(weights.values()), resolved=0)
    results = result_payload.get("results")
    if not isinstance(results, list):
        return LocCounts(available=None, resolved=None)

    resolved_weight = 0
    for item in results:
        if not isinstance(item, dict) or not item.get("resolved"):
            continue
        instance_id = str(item.get("instance_id") or "")
        if instance_id not in weights:
            return LocCounts(available=None, resolved=None)
        resolved_weight += weights[instance_id]
    return LocCounts(available=sum(weights.values()), resolved=resolved_weight)


def _is_chain_blocked(result: dict[str, Any]) -> bool:
    error = result.get("error")
    return isinstance(error, str) and error.startswith(CHAIN_BLOCKED_PREFIX)


def _is_infra_error(result: dict[str, Any]) -> bool:
    error = result.get("error")
    return isinstance(error, str) and error.startswith("Infrastructure error:")


def _add_result(counts: Counts, result: dict[str, Any]) -> None:
    counts.reported += 1
    resolved = bool(result.get("resolved"))
    chain_blocked = _is_chain_blocked(result)
    if resolved:
        counts.resolved += 1
    if not chain_blocked:
        counts.attempted += 1
        if resolved:
            counts.attempted_resolved += 1
    if chain_blocked:
        counts.blocked += 1
    elif _is_infra_error(result):
        counts.infra += 1
    elif result.get("error"):
        counts.errors += 1
    elif not result.get("patch_applied_successfully", False):
        counts.failed_apply += 1
    elif not result.get("all_tests_passed", False):
        counts.failed_tests += 1


def _counts_from_result(result_payload: dict[str, Any], available_total: int) -> Counts:
    counts = Counts(available=available_total)
    results = result_payload.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                _add_result(counts, item)
    else:
        counts.reported = int(result_payload.get("total_instances") or 0)
        counts.resolved = int(result_payload.get("resolved_instances") or 0)
        counts.attempted = int(result_payload.get("attempted_instances") or 0)
        counts.attempted_resolved = int(result_payload.get("attempted_resolved_instances") or 0)
        counts.blocked = int(result_payload.get("chain_blocked_instances") or 0)

    counts.failed_tests = int(result_payload.get("failed_tests") or counts.failed_tests)
    counts.failed_apply = int(result_payload.get("failed_patch_apply") or counts.failed_apply)
    counts.errors = int(result_payload.get("errors") or counts.errors)
    counts.infra = int(result_payload.get("infra_error_instances") or counts.infra)
    counts.skipped = int(result_payload.get("skipped_instances") or 0)
    counts.not_parsed = int(result_payload.get("not_parsed_instances") or 0)
    counts.missing_after_attempt = int(
        result_payload.get("missing_patch_after_attempt_instances") or 0
    )
    counts.missing_not_started = int(result_payload.get("missing_patch_not_started_instances") or 0)
    return counts


def _repo_counts_from_result(
    result_payload: dict[str, Any],
    available_by_repo: dict[str, int],
) -> dict[str, Counts]:
    repo_counts: dict[str, Counts] = {
        repo: Counts(available=available) for repo, available in sorted(available_by_repo.items())
    }
    results = result_payload.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            repo = _result_repo(str(item.get("instance_id") or ""))
            repo_counts.setdefault(repo, Counts())
            _add_result(repo_counts[repo], item)
    return dict(sorted(repo_counts.items()))


def _failure_reason_from_result(result: dict[str, Any]) -> str:
    instance_id = str(result.get("instance_id") or "-")
    error = result.get("error")
    if isinstance(error, str) and error:
        return f"{instance_id}: error - {error}"
    if not result.get("patch_applied_successfully", False):
        return f"{instance_id}: patch failed to apply"
    if not result.get("all_tests_passed", False):
        failed_count = result.get("failed_test_count")
        if isinstance(failed_count, int) and failed_count > 0:
            noun = "test" if failed_count == 1 else "tests"
            return f"{instance_id}: tests failed - {failed_count} {noun} failed"
        return f"{instance_id}: tests failed"
    return f"{instance_id}: not resolved"


def _repo_failure_reasons_from_result(result_payload: dict[str, Any]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    first_failures = result_payload.get("lifecycle_first_failures")
    if isinstance(first_failures, list):
        for item in first_failures:
            if not isinstance(item, dict):
                continue
            segment_id = item.get("segment_id")
            repo = str(segment_id).split("::", maxsplit=1)[0] if segment_id else ""
            if not repo:
                instance_id = item.get("instance_id")
                repo = _result_repo(str(instance_id or ""))
            if not repo or repo == "-":
                continue
            instance_id = str(item.get("instance_id") or repo)
            category = str(item.get("category") or "failed")
            reason = str(item.get("reason") or "not resolved")
            reasons[repo] = f"{instance_id}: {category} - {reason}"

    results = result_payload.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict) or item.get("resolved"):
                continue
            if _is_chain_blocked(item):
                continue
            repo = _result_repo(str(item.get("instance_id") or ""))
            reasons.setdefault(repo, _failure_reason_from_result(item))
    return dict(sorted(reasons.items()))


def collect_runs(
    output_dir: Path,
    *,
    recursive: bool,
    include_old: bool,
) -> list[RunStats]:
    runs: list[RunStats] = []
    for run_dir in _candidate_run_dirs(output_dir, recursive=recursive, include_old=include_old):
        config_path = run_dir / "config.json"
        result_path = _latest_reported_result_file(run_dir)
        if result_path is None:
            continue
        config: dict[str, Any] = {}
        result_payload: dict[str, Any] | None = None
        config_error = None
        result_error = None
        try:
            config = _load_json(config_path)
        except Exception as error:
            config_error = str(error)
        if config_error is not None or not is_default_report_config(config):
            continue
        if result_path is not None:
            try:
                result_payload = _load_json(result_path)
            except Exception as error:
                result_error = str(error)

        selected_instances = _selected_instances(config, run_dir=run_dir)
        available_by_repo = _available_by_repo(selected_instances)
        available_total = sum(available_by_repo.values())
        if result_payload is None:
            counts = Counts(available=available_total)
            repo_counts = {
                repo: Counts(available=count) for repo, count in available_by_repo.items()
            }
            repo_failure_reasons = {}
        else:
            counts = _counts_from_result(result_payload, available_total)
            repo_counts = _repo_counts_from_result(result_payload, available_by_repo)
            repo_failure_reasons = _repo_failure_reasons_from_result(result_payload)

        runs.append(
            RunStats(
                run_id=run_dir.name,
                run_dir=run_dir,
                agent=_config_agent(config, result_payload),
                model=_config_model(config),
                effort=_config_effort(config),
                mode=str(config.get("mode") or "-"),
                progression=_run_progression(config),
                result_path=result_path,
                counts=counts,
                loc_counts=_loc_counts(
                    selected_instances,
                    result_payload,
                    run_dir=run_dir,
                ),
                usage=_usage_from_trajectories(run_dir),
                repo_counts=repo_counts,
                repo_failure_reasons=repo_failure_reasons,
                config_error=config_error,
                result_error=result_error,
            )
        )
    return runs


def _pct(numerator: int | None, denominator: int | None) -> str:
    if numerator is None or denominator is None or denominator <= 0:
        return "-"
    return f"{100 * numerator / denominator:.1f}%"


def _fraction(numerator: int | None, denominator: int | None) -> str:
    if numerator is None or denominator is None:
        return "-"
    if denominator <= 0:
        return f"{numerator}/0"
    return f"{numerator}/{denominator}"


def _format_tokens(usage: UsageStats) -> str:
    if not usage.has_tokens:
        return "-"
    return f"{usage.total_tokens:,}"


def _format_duration(usage: UsageStats) -> str:
    if not usage.has_agent_time:
        return "-"
    total_seconds = max(0, round(usage.agent_seconds))
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02}h {minutes:02}m"
    if hours:
        return f"{hours}h {minutes:02}m"
    if minutes:
        return f"{minutes}m {seconds:02}s"
    return f"{seconds}s"


def _format_cost(usage: UsageStats) -> str:
    if not usage.has_cost:
        return "-"
    return f"${usage.cost:,.3f}"


def _escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = [
        "| " + " | ".join(_escape_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_cell(value) for value in row) + " |")
    return lines


def _result_label(run: RunStats) -> str:
    return run.result_path.name if run.result_path is not None else "-"


def _runs_by_group(runs: list[RunStats]) -> dict[tuple[str, str, str], list[RunStats]]:
    grouped: dict[tuple[str, str, str], list[RunStats]] = defaultdict(list)
    for run in runs:
        grouped[run.group_key].append(run)
    return dict(sorted(grouped.items()))


def _group_counts(runs: list[RunStats]) -> dict[tuple[str, str, str], Counts]:
    grouped: dict[tuple[str, str, str], Counts] = {}
    for run in runs:
        grouped.setdefault(run.group_key, Counts()).add(run.counts)
    return dict(sorted(grouped.items()))


def _group_loc_counts(runs: list[RunStats]) -> dict[tuple[str, str, str], LocCounts]:
    grouped: dict[tuple[str, str, str], LocCounts] = {}
    for run in runs:
        grouped.setdefault(run.group_key, LocCounts()).add(run.loc_counts)
    return dict(sorted(grouped.items()))


def _group_usage(runs: list[RunStats]) -> dict[tuple[str, str, str], UsageStats]:
    grouped: dict[tuple[str, str, str], UsageStats] = {}
    for run in runs:
        grouped.setdefault(run.group_key, UsageStats()).add(run.usage)
    return dict(sorted(grouped.items()))


def _comparison_repos(runs: list[RunStats]) -> list[str]:
    return sorted({repo for run in runs for repo in run.repo_counts if repo and repo != "-"})


def _repo_resolution_rows(repos: list[str], runs: list[RunStats]) -> list[list[object]]:
    return [
        [
            run.run_id,
            *[run.repo_counts.get(repo, Counts()).resolved for repo in repos],
        ]
        for run in runs
    ]


def _append_repo_comparison(
    lines: list[str],
    grouped_runs: dict[tuple[str, str, str], list[RunStats]],
    *,
    runs: list[RunStats],
) -> None:
    selected_repos = _comparison_repos(runs)
    lines.append("## Repo Side-by-Side Comparison")
    lines.append("")
    lines.append("Values are resolved instance counts.")
    if not selected_repos:
        lines.append("")
        lines.append("_No matching repos._")
        return

    for (agent, model, effort), group_runs in grouped_runs.items():
        lines.append("")
        lines.append(f"### `{agent}` / `{model}` / `{effort}`")
        lines.append("")
        headers = ["Run", *selected_repos]
        lines.extend(_markdown_table(headers, _repo_resolution_rows(selected_repos, group_runs)))


def render_markdown(
    runs: list[RunStats],
    *,
    output_dir: Path,
    recursive: bool,
    include_old: bool,
) -> str:
    lines: list[str] = []
    lines.append("# Evaluation Runs Summary")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append(f"- Search mode: `{'recursive' if recursive else 'top-level'}`")
    lines.append(f"- Run config filter: `{DEFAULT_REPORT_CONFIG_DESCRIPTION}`")
    if recursive:
        lines.append(f"- Include `OLD`: `{'yes' if include_old else 'no'}`")
    lines.append(f"- Completed runs with reports: {len(runs)}")
    lines.append(
        "- Usage metrics: current trajectory files; tokens include repeated/cached API "
        "input and output, and agent time is cumulative trajectory wall time."
    )
    lines.append(
        "- Completion metrics: instances are unweighted counts; weighted LOC is the "
        "number of non-empty lines added to non-test Python files by each instance's "
        "golden patch."
    )
    lines.append("")

    lines.append("## Agent Aggregates")
    aggregate_rows: list[list[object]] = []
    grouped_counts = _group_counts(runs)
    grouped_loc_counts = _group_loc_counts(runs)
    grouped_usage = _group_usage(runs)
    grouped_runs = _runs_by_group(runs)
    for (agent, model, effort), counts in grouped_counts.items():
        loc_counts = grouped_loc_counts[(agent, model, effort)]
        usage = grouped_usage[(agent, model, effort)]
        aggregate_rows.append(
            [
                agent,
                model,
                effort,
                len(grouped_runs[(agent, model, effort)]),
                _fraction(counts.resolved, counts.available),
                _pct(counts.resolved, counts.available),
                _fraction(loc_counts.resolved, loc_counts.available),
                _pct(loc_counts.resolved, loc_counts.available),
                _fraction(counts.attempted_resolved, counts.attempted),
                _pct(counts.attempted_resolved, counts.attempted),
                counts.blocked,
                counts.failed_tests,
                counts.errors,
                _format_tokens(usage),
                _format_duration(usage),
                _format_cost(usage),
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "Agent",
                "Model",
                "Effort",
                "Runs",
                "Completed Instances/Available",
                "Completed Instances %",
                "Completed Weighted LOC/Available LOC",
                "Completed Weighted LOC %",
                "Attempted Completed/Attempted",
                "Attempted %",
                "Blocked",
                "Failed Tests",
                "Errors",
                "Tokens Used",
                "Agent Time",
                "Cost",
            ],
            aggregate_rows,
        )
    )
    lines.append("")

    lines.append("## Runs By Agent")
    for (agent, model, effort), group_runs in grouped_runs.items():
        lines.append("")
        lines.append(f"### `{agent}` / `{model}` / `{effort}`")
        lines.append("")
        run_rows: list[list[object]] = []
        for run in group_runs:
            c = run.counts
            run_rows.append(
                [
                    run.run_id,
                    run.progression,
                    _fraction(c.resolved, c.available),
                    _pct(c.resolved, c.available),
                    _fraction(run.loc_counts.resolved, run.loc_counts.available),
                    _pct(run.loc_counts.resolved, run.loc_counts.available),
                    _fraction(c.attempted_resolved, c.attempted),
                    _pct(c.attempted_resolved, c.attempted),
                    c.blocked,
                    c.failed_tests,
                    c.errors,
                    c.missing_after_attempt,
                    c.missing_not_started,
                    _format_tokens(run.usage),
                    _format_duration(run.usage),
                    _format_cost(run.usage),
                    _result_label(run),
                ]
            )
        lines.extend(
            _markdown_table(
                [
                    "Run",
                    "Progression",
                    "Completed Instances/Available",
                    "Completed Instances %",
                    "Completed Weighted LOC/Available LOC",
                    "Completed Weighted LOC %",
                    "Attempted Completed/Attempted",
                    "Attempted %",
                    "Blocked",
                    "Failed Tests",
                    "Errors",
                    "Missing After Attempt",
                    "Missing Not Started",
                    "Tokens Used",
                    "Agent Time",
                    "Cost",
                    "Result",
                ],
                run_rows,
            )
        )
    lines.append("")

    _append_repo_comparison(
        lines,
        grouped_runs,
        runs=runs,
    )

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output.resolve()
    markdown_output = args.markdown_output
    if markdown_output is None:
        markdown_output = output_dir / DEFAULT_MARKDOWN_NAME
    elif not markdown_output.is_absolute():
        markdown_output = Path.cwd() / markdown_output

    runs = collect_runs(
        output_dir,
        recursive=bool(args.recursive),
        include_old=bool(args.include_old),
    )
    report = render_markdown(
        runs,
        output_dir=output_dir,
        recursive=bool(args.recursive),
        include_old=bool(args.include_old),
    )
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(report, encoding="utf-8")
    print(f"Wrote {markdown_output}")


if __name__ == "__main__":
    main()
