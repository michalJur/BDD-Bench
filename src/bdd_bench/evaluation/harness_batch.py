# mypy: ignore-errors
from __future__ import annotations

import concurrent.futures
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from bdd_bench.common.docker_client import ensure_docker_available
from bdd_bench.common.test_file_rules import is_test_path
from bdd_bench.evaluation.harness_agents import Agent
from bdd_bench.evaluation.harness_metadata import (
    immutable_image_ref,
    load_metadata,
    sanitize_label,
)
from bdd_bench.evaluation.harness_models import EvaluationResult, EvaluationSummary
from bdd_bench.evaluation.patch_utils import extract_patch_paths
from bdd_bench.evaluation.result_classification import (
    chain_blocked_reason as _chain_blocked_reason,
    classify_evaluation_result,
    is_chain_blocked_error as _is_chain_blocked_error,
    is_disallowed_patch_error as _is_disallowed_test_patch_error,
    is_infra_error as _is_infra_error,
    is_parse_error as _is_parse_error,
)
from bdd_bench.evaluation.run_config import agent_config_from_run_config


def _text_indicates_infra_denial(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    normalized = text.lower()
    return (
        "permission denied while trying to connect to the docker daemon socket" in normalized
        or ("docker.sock" in normalized and "operation not permitted" in normalized)
        or "blocked by environment restrictions before any repo work could start" in normalized
        or ("blocked by host permissions" in normalized and "docker exec" in normalized)
        or "bubblewrap is unavailable" in normalized
        or "no system bwrap" in normalized
        or "missing `bwrap`" in normalized
        or "missing bwrap" in normalized
    )


def _text_indicates_retryable_generation_failure(text: str | None) -> bool:
    """Return True for transient agent-runtime failures worth retrying."""
    if not isinstance(text, str):
        return False
    normalized = text.lower()
    retryable_markers = [
        "rate_limit",
        "you've hit your limit",
        "you have hit your limit",
        "too many requests",
        'api_error_status":429',
        "api_error_status: 429",
        "status code: 429",
        "http 429",
        "429",
        "overloaded",
        "temporarily unavailable",
        "service unavailable",
        "internal server error",
        "gateway timeout",
        "connection reset",
        "connection refused",
        "connection aborted",
        "network error",
        "timed out",
        "timeout",
    ]
    return any(marker in normalized for marker in retryable_markers)


def _is_fatal_generation_error(error: str | None) -> bool:
    """Return True for agent failures where continuing the batch will repeat the same failure."""
    if not isinstance(error, str):
        return False
    normalized = error.lower()
    fatal_markers = [
        "could not load credentials",
        "invalid api key",
        "incorrect api key",
        "missing api key",
        "authentication failed",
        "invalid authentication",
        "invalid credentials",
        "not logged in",
        "login required",
        "insufficient quota",
        "quota exceeded",
        "billing hard limit",
        "context_length_exceeded",
        "context length exceeded",
        "maximum context length",
        "token limit",
        "too many tokens",
    ]
    return any(marker in normalized for marker in fatal_markers)


class _HarnessBatch:
    """Batch (multi-instance) operations and results reporting for EvaluationHarness."""

    # Set by EvaluationHarness.__init__.
    workers: int

    def _run_chain_segment_jobs(
        self,
        grouped_items: dict[str, list[Any]],
        process_segment: Callable[[str, list[Any]], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run independent chain segments concurrently and return deterministic results."""
        segment_ids = sorted(grouped_items)
        if not segment_ids:
            return []

        max_workers = min(max(1, self.workers), len(segment_ids))
        if max_workers == 1:
            outcomes: list[dict[str, Any]] = []
            for segment_id in segment_ids:
                outcome = process_segment(segment_id, grouped_items[segment_id])
                outcomes.append(outcome)
                if outcome.get("fatal_generation_error"):
                    break
            return outcomes

        logging.info(
            "Running %d independent release-chain segment(s) with %d worker(s)",
            len(segment_ids),
            max_workers,
        )
        outcomes_by_segment: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_segment = {
                executor.submit(
                    process_segment,
                    segment_id,
                    grouped_items[segment_id],
                ): segment_id
                for segment_id in segment_ids
            }
            for future in concurrent.futures.as_completed(future_to_segment):
                segment_id = future_to_segment[future]
                outcomes_by_segment[segment_id] = future.result()

        return [outcomes_by_segment[segment_id] for segment_id in segment_ids]

    def _result_status_label(self, result: EvaluationResult) -> str:
        if _is_parse_error(result.error):
            return "NOT_PARSED"
        if self._is_infra_denied_empty_patch_result(result):
            return "INFRA_DENIED_EMPTY_PATCH"
        if _is_infra_error(result.error):
            return "INFRA_ERROR"
        if _is_chain_blocked_error(result.error):
            return "CHAIN_BLOCKED"
        if result.resolved:
            return "RESOLVED"
        if result.error:
            return "ERROR"
        if not result.patch_applied_successfully:
            return "PATCH_FAILED"
        if not result.all_tests_passed:
            return "TESTS_FAILED"
        return "FAILED"

    @staticmethod
    def _running_evaluation_counts(results: list[EvaluationResult]) -> dict[str, int]:
        not_parsed = sum(1 for result in results if _is_parse_error(result.error))
        evaluable = [result for result in results if not _is_parse_error(result.error)]
        resolved = sum(1 for result in evaluable if result.resolved)
        infra_denied_empty_patch = sum(
            1
            for result in evaluable
            if result.detailed_results.get("infra_denied_empty_patch", False)
        )
        failed_apply = sum(
            1
            for result in evaluable
            if not result.patch_applied_successfully
            and not result.error
            and not result.detailed_results.get("infra_denied_empty_patch", False)
        )
        failed_tests = sum(
            1
            for result in evaluable
            if (
                result.patch_applied_successfully
                and not result.all_tests_passed
                and not result.error
                and not result.detailed_results.get("infra_denied_empty_patch", False)
            )
        )
        errors = sum(
            1
            for result in evaluable
            if (
                result.error
                and not result.detailed_results.get("infra_denied_empty_patch", False)
                and not _is_chain_blocked_error(result.error)
            )
        )
        chain_blocked = sum(1 for result in evaluable if _is_chain_blocked_error(result.error))
        return {
            "resolved": resolved,
            "infra_denied_empty_patch": infra_denied_empty_patch,
            "failed_tests": failed_tests,
            "failed_apply": failed_apply,
            "errors": errors,
            "chain_blocked": chain_blocked,
            "not_parsed": not_parsed,
        }

    @staticmethod
    def _running_test_run_counts(run_payloads: list[dict[str, Any]]) -> dict[str, int]:
        errors = sum(1 for payload in run_payloads if payload.get("error"))
        completed = len(run_payloads) - errors
        return {
            "completed": completed,
            "errors": errors,
        }

    @staticmethod
    def _payload_status_label(payload: dict[str, Any]) -> str:
        return "ERROR" if payload.get("error") else "DONE"

    def _emit_evaluation_progress(
        self,
        *,
        progress_bar: tqdm,
        results: list[EvaluationResult],
        completed: int,
        total: int,
        last_label: str,
        status: str,
    ) -> None:
        self._annotate_special_result_categories(results)
        counts = self._running_evaluation_counts(results)
        progress_bar.set_postfix(
            {
                "resolved": counts["resolved"],
                "infra_denied": counts["infra_denied_empty_patch"],
                "failed_tests": counts["failed_tests"],
                "failed_apply": counts["failed_apply"],
                "chain_blocked": counts["chain_blocked"],
                "errors": counts["errors"],
                "last": last_label[:24],
            },
            refresh=False,
        )
        tqdm.write(
            f"[{completed}/{total}] {last_label}: {status} "
            f"(resolved={counts['resolved']} "
            f"infra_denied={counts['infra_denied_empty_patch']} "
            f"failed_tests={counts['failed_tests']} "
            f"failed_apply={counts['failed_apply']} "
            f"chain_blocked={counts['chain_blocked']} "
            f"errors={counts['errors']} "
            f"not_parsed={counts['not_parsed']})"
        )

    def _emit_test_run_progress(
        self,
        *,
        progress_bar: tqdm,
        run_payloads: list[dict[str, Any]],
        completed: int,
        total: int,
        last_label: str,
        status: str,
        skipped: int = 0,
    ) -> None:
        counts = self._running_test_run_counts(run_payloads)
        progress_bar.set_postfix(
            {
                "completed": counts["completed"],
                "errors": counts["errors"],
                "skipped": skipped,
                "last": last_label[:24],
            },
            refresh=False,
        )
        tqdm.write(
            f"[{completed}/{total}] {last_label}: {status} "
            f"(completed={counts['completed']} errors={counts['errors']} skipped={skipped})"
        )

    @staticmethod
    def _trajectory_json_indicates_terminal_attempt(trajectory_json_path: Path) -> bool:
        """Return True when a trajectory JSON records a terminal generation outcome."""
        try:
            payload = json.loads(trajectory_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False

        messages = payload.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if not isinstance(message, dict):
                    continue
                if message.get("role") != "exit":
                    continue
                extra = message.get("extra")
                if isinstance(extra, dict):
                    exit_status = extra.get("exit_status")
                    if isinstance(exit_status, str) and exit_status.strip():
                        return True
                # Some trajectories may include role=exit without extra fields.
                return True

        return False

    @staticmethod
    def _trajectory_log_indicates_terminal_attempt(trajectory_log_path: Path) -> bool:
        """Fallback parser for terminal attempt markers in textual trajectory logs."""
        try:
            text = trajectory_log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

        if "role=exit" in text:
            return True
        return False

    @staticmethod
    def _trajectory_attempt_stem(path: Path) -> str:
        name = path.name
        for suffix in (".traj.json", ".live.log", ".log"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return path.stem

    @staticmethod
    def _trajectory_payload_indicates_infra_denial(payload: Any) -> bool:
        if isinstance(payload, str):
            return _text_indicates_infra_denial(payload)
        if isinstance(payload, dict):
            return any(
                _HarnessBatch._trajectory_payload_indicates_infra_denial(value)
                for value in payload.values()
            )
        if isinstance(payload, list):
            return any(
                _HarnessBatch._trajectory_payload_indicates_infra_denial(value) for value in payload
            )
        return False

    @staticmethod
    def _trajectory_payload_indicates_retryable_generation_failure(payload: Any) -> bool:
        if isinstance(payload, str):
            return _text_indicates_retryable_generation_failure(payload)
        if isinstance(payload, dict):
            return any(
                _HarnessBatch._trajectory_payload_indicates_retryable_generation_failure(value)
                for value in payload.values()
            )
        if isinstance(payload, list):
            return any(
                _HarnessBatch._trajectory_payload_indicates_retryable_generation_failure(value)
                for value in payload
            )
        return False

    @staticmethod
    def _trajectory_json_indicates_retryable_generation_failure(
        trajectory_json_path: Path,
    ) -> bool:
        try:
            payload = json.loads(trajectory_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return _HarnessBatch._trajectory_payload_indicates_retryable_generation_failure(payload)

    @staticmethod
    def _trajectory_log_indicates_retryable_generation_failure(
        trajectory_log_path: Path,
    ) -> bool:
        try:
            text = trajectory_log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return _text_indicates_retryable_generation_failure(text)

    @staticmethod
    def _trajectory_json_indicates_infra_denial(trajectory_json_path: Path) -> bool:
        try:
            payload = json.loads(trajectory_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return _HarnessBatch._trajectory_payload_indicates_infra_denial(payload)

    @staticmethod
    def _trajectory_log_indicates_infra_denial(trajectory_log_path: Path) -> bool:
        try:
            text = trajectory_log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return _text_indicates_infra_denial(text)

    def _latest_generation_attempt_paths(self, instance_id: str) -> tuple[Path | None, Path | None]:
        trajectory_dir = self.output_dir / "trajectories" / sanitize_label(instance_id)  # type: ignore[attr-defined]
        if not trajectory_dir.is_dir():
            return None, None

        latest_artifact: Path | None = None
        latest_mtime = 0.0
        for path in trajectory_dir.iterdir():
            if not path.is_file():
                continue
            if not (
                path.name.endswith(".traj.json")
                or path.name.endswith(".log")
                or path.name.endswith(".live.log")
            ):
                continue
            mtime = path.stat().st_mtime
            if (
                latest_artifact is None
                or mtime > latest_mtime
                or (mtime == latest_mtime and path.name > latest_artifact.name)
            ):
                latest_artifact = path
                latest_mtime = mtime

        if latest_artifact is None:
            return None, None

        attempt_stem = self._trajectory_attempt_stem(latest_artifact)
        trajectory_json_path = trajectory_dir / f"{attempt_stem}.traj.json"
        trajectory_log_path = trajectory_dir / f"{attempt_stem}.log"
        return (
            trajectory_json_path if trajectory_json_path.is_file() else None,
            trajectory_log_path if trajectory_log_path.is_file() else None,
        )

    def _has_generation_attempt_artifacts(self, instance_id: str) -> bool:
        """Return True when trajectory artifacts indicate a terminal generation attempt."""
        trajectory_json_path, trajectory_log_path = self._latest_generation_attempt_paths(
            instance_id
        )

        if trajectory_json_path is not None:
            if self._trajectory_json_indicates_terminal_attempt(trajectory_json_path):
                return True
            return (
                trajectory_log_path is not None
                and self._trajectory_log_indicates_terminal_attempt(trajectory_log_path)
            )

        # Backward-compatible fallback in case only textual trajectory logs are present.
        if trajectory_log_path is not None:
            return self._trajectory_log_indicates_terminal_attempt(trajectory_log_path)

        return False

    def _has_infra_denied_generation_artifacts(self, instance_id: str) -> bool:
        cache = getattr(self, "_infra_denied_generation_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_infra_denied_generation_cache", cache)
        if instance_id in cache:
            return bool(cache[instance_id])

        trajectory_json_path, trajectory_log_path = self._latest_generation_attempt_paths(
            instance_id
        )
        has_denial = False
        if trajectory_json_path is not None:
            has_denial = self._trajectory_json_indicates_infra_denial(trajectory_json_path)
        if not has_denial and trajectory_log_path is not None:
            has_denial = self._trajectory_log_indicates_infra_denial(trajectory_log_path)
        cache[instance_id] = has_denial
        return has_denial

    def _has_retryable_generation_failure_artifacts(self, instance_id: str) -> bool:
        trajectory_json_path, trajectory_log_path = self._latest_generation_attempt_paths(
            instance_id
        )
        has_retryable_failure = False
        if trajectory_json_path is not None:
            has_retryable_failure = self._trajectory_json_indicates_retryable_generation_failure(
                trajectory_json_path,
            )
        if not has_retryable_failure and trajectory_log_path is not None:
            has_retryable_failure = self._trajectory_log_indicates_retryable_generation_failure(
                trajectory_log_path,
            )
        return has_retryable_failure

    def _prior_terminal_generation_failure_result(
        self,
        *,
        agent_name: str,
        instance_id: str,
    ) -> EvaluationResult | None:
        from bdd_bench.evaluation.harness_single import _prior_terminal_agent_failure_result

        return _prior_terminal_agent_failure_result(
            output_dir=self.output_dir,  # type: ignore[attr-defined]
            instance_id=instance_id,
            agent_name=agent_name,
        )

    def cleanup_retryable_generation_artifacts(
        self,
        *,
        agent_name: str,
        metadata_paths: list[Path],
    ) -> dict[str, object]:
        """Remove stale transient generation artifacts so continuation can retry."""
        cleaned_labels: list[str] = []
        for metadata_path in metadata_paths:
            try:
                data = load_metadata(metadata_path)
                instance_id = data["instance_id"]
            except Exception as error:
                logging.warning("Could not load metadata from %s: %s", metadata_path, error)
                continue

            patch_file = self.resolve_patch_file(  # type: ignore[attr-defined]
                patches_dir=self.patches_dir_for_agent(agent_name),  # type: ignore[attr-defined]
                agent_name=agent_name,
                instance_id=instance_id,
            )
            if patch_file is not None:
                continue
            if (
                self._prior_terminal_generation_failure_result(
                    agent_name=agent_name,
                    instance_id=instance_id,
                )
                is not None
            ):
                continue
            if not self._has_retryable_generation_failure_artifacts(instance_id):
                continue

            safe_instance_id = sanitize_label(instance_id)
            for artifact_dir in [
                self.output_dir / "trajectories" / safe_instance_id,  # type: ignore[attr-defined]
                self.output_dir / "patches" / safe_instance_id,  # type: ignore[attr-defined]
                self.output_dir / "eval_logs" / safe_instance_id,  # type: ignore[attr-defined]
            ]:
                if artifact_dir.exists():
                    shutil.rmtree(artifact_dir)
            infra_cache = getattr(self, "_infra_denied_generation_cache", None)
            if isinstance(infra_cache, dict):
                infra_cache.pop(instance_id, None)
            cleaned_labels.append(instance_id)

        return {"count": len(cleaned_labels), "instances": cleaned_labels}

    def _is_infra_denied_empty_patch_result(self, result: EvaluationResult) -> bool:
        if result.detailed_results.get("infra_denied_empty_patch", False):
            return True
        if result.resolved or _is_infra_error(result.error):
            return False
        if not isinstance(result.patch_generated, str) or result.patch_generated.strip():
            return False
        instance_id = result.instance_id
        if instance_id not in self.instances:  # type: ignore[attr-defined]
            return False
        return self._has_infra_denied_generation_artifacts(instance_id)

    def _annotate_special_result_categories(self, results: list[EvaluationResult]) -> None:
        for result in results:
            if self._is_infra_denied_empty_patch_result(result):
                result.detailed_results["infra_denied_empty_patch"] = True
            else:
                result.detailed_results.pop("infra_denied_empty_patch", None)

    def _result_failure_category(self, result: EvaluationResult) -> str:
        return classify_evaluation_result(
            resolved=result.resolved,
            error=result.error,
            patch_applied_successfully=result.patch_applied_successfully,
            all_tests_passed=result.all_tests_passed,
            detailed_results=result.detailed_results,
        )

    def _result_failure_reason(self, result: EvaluationResult) -> str:
        if _is_chain_blocked_error(result.error):
            return _chain_blocked_reason(result.error)
        if result.error:
            return result.error
        if not result.patch_applied_successfully:
            return "patch did not apply"
        if not result.all_tests_passed:
            count = result.failed_test_count
            return f"{count} test failed" if count == 1 else f"{count} tests failed"
        return ""

    def _lifecycle_first_failures(
        self,
        results: list[EvaluationResult],
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[tuple[int, str, EvaluationResult]]] = {}
        for result in results:
            position = self._instance_chain_position(result.instance_id)
            if position is None:
                continue
            segment_id, stage_index = position
            grouped.setdefault(segment_id, []).append((stage_index, result.instance_id, result))

        first_failures: list[dict[str, Any]] = []
        for segment_id in sorted(grouped):
            ordered = sorted(grouped[segment_id], key=lambda item: item[0])
            survived = 0
            for stage_index, instance_id, result in ordered:
                if result.resolved:
                    survived += 1
                    continue
                first_failures.append(
                    {
                        "segment_id": segment_id,
                        "instance_id": instance_id,
                        "stage_index": stage_index,
                        "survived_stages": survived,
                        "total_reported_stages": len(ordered),
                        "blocked_after": sum(
                            1
                            for _idx, later_label, later_result in ordered
                            if (
                                later_label != instance_id
                                and _is_chain_blocked_error(later_result.error)
                            )
                        ),
                        "category": self._result_failure_category(result),
                        "reason": self._result_failure_reason(result),
                    }
                )
                break
            else:
                first_failures.append(
                    {
                        "segment_id": segment_id,
                        "instance_id": None,
                        "stage_index": None,
                        "survived_stages": survived,
                        "total_reported_stages": len(ordered),
                        "blocked_after": 0,
                        "category": "all_resolved",
                        "reason": "",
                    }
                )

        return first_failures

    def _lifecycle_stop_reason_for_report_result(
        self,
        result: EvaluationResult,
    ) -> str | None:
        if result.resolved:
            return None
        if _is_chain_blocked_error(result.error):
            return _chain_blocked_reason(result.error)
        if _is_parse_error(result.error):
            return f"stage_not_parsed:{result.instance_id}"
        if result.error:
            return f"stage_generation_failed:{result.instance_id}"
        return f"stage_not_resolved:{result.instance_id}"

    def _with_lifecycle_blocked_report_results(
        self,
        *,
        agent_name: str,
        metadata_paths: list[Path],
        results: list[EvaluationResult],
    ) -> tuple[list[EvaluationResult], set[str]]:
        selected_chain_labels: list[str] = []
        for metadata_path in metadata_paths:
            try:
                instance_id = load_metadata(metadata_path)["instance_id"]
            except Exception:
                continue
            if instance_id not in self.instances:  # type: ignore[attr-defined]
                continue
            if self._instance_chain_position(instance_id) is None:
                continue
            selected_chain_labels.append(instance_id)

        if not selected_chain_labels or not self._supports_release_chain_progression(
            selected_chain_labels,
        ):
            return results, set()

        result_by_label = {result.instance_id: result for result in results}
        grouped_labels: dict[str, list[tuple[int, str]]] = {}
        for instance_id in selected_chain_labels:
            position = self._instance_chain_position(instance_id)
            if position is None:
                continue
            segment_id, stage_index = position
            grouped_labels.setdefault(segment_id, []).append((stage_index, instance_id))

        ordered_results: list[EvaluationResult] = []
        emitted_labels: set[str] = set()
        synthesized_blocked_labels: set[str] = set()
        for segment_id in sorted(grouped_labels):
            segment_stop_reason: str | None = None
            for _stage_index, instance_id in sorted(grouped_labels[segment_id]):
                if segment_stop_reason is not None:
                    result = self._blocked_chain_result(
                        agent_name=agent_name,
                        instance_id=instance_id,
                        reason=segment_stop_reason,
                    )
                    synthesized_blocked_labels.add(instance_id)
                else:
                    result = result_by_label.get(instance_id)
                    if result is None:
                        continue
                    segment_stop_reason = self._lifecycle_stop_reason_for_report_result(result)
                ordered_results.append(result)
                emitted_labels.add(instance_id)

        ordered_results.extend(
            result for result in results if result.instance_id not in emitted_labels
        )
        return ordered_results, synthesized_blocked_labels

    def _build_evaluation_summary(
        self,
        *,
        agent_name: str,
        results: list[EvaluationResult],
        skipped_instances: int = 0,
        missing_patch_after_attempt_instances: int = 0,
        missing_patch_not_started_instances: int = 0,
    ) -> EvaluationSummary:
        self._annotate_special_result_categories(results)
        summary_results = list(results)
        not_parsed_count = sum(1 for result in results if _is_parse_error(result.error))
        parseable_results = [result for result in results if not _is_parse_error(result.error)]
        chain_blocked_count = sum(
            1 for result in parseable_results if _is_chain_blocked_error(result.error)
        )
        attempted_results = [
            result for result in summary_results if not _is_chain_blocked_error(result.error)
        ]
        attempted_resolved_count = sum(1 for result in attempted_results if result.resolved)
        resolved_count = sum(1 for result in summary_results if result.resolved)
        infra_denied_empty_patch_count = sum(
            1 for result in parseable_results if self._is_infra_denied_empty_patch_result(result)
        )
        failed_apply_count = sum(
            1
            for result in parseable_results
            if (
                not result.patch_applied_successfully
                and not result.error
                and not self._is_infra_denied_empty_patch_result(result)
            )
        )
        failed_tests_count = sum(
            1
            for result in parseable_results
            if (
                result.patch_applied_successfully
                and not result.all_tests_passed
                and not result.error
                and not self._is_infra_denied_empty_patch_result(result)
            )
        )
        error_count = sum(
            1
            for result in parseable_results
            if (
                result.error
                and not self._is_infra_denied_empty_patch_result(result)
                and not _is_chain_blocked_error(result.error)
            )
        )
        disallowed_test_patch_count = sum(
            1 for result in parseable_results if _is_disallowed_test_patch_error(result.error)
        )
        infra_error_count = sum(1 for result in parseable_results if _is_infra_error(result.error))
        include_lifecycle_diagnostics = (
            getattr(self, "progression_mode", "basic") == "lifecycle" or chain_blocked_count > 0
        )
        return EvaluationSummary(
            agent_name=agent_name,
            total_instances=len(summary_results),
            resolved_instances=resolved_count,
            failed_patch_apply=failed_apply_count,
            failed_tests=failed_tests_count,
            errors=error_count,
            resolution_rate=resolved_count / len(summary_results) if summary_results else 0.0,
            results=results,
            skipped_instances=skipped_instances,
            not_parsed_instances=not_parsed_count,
            infra_error_instances=infra_error_count,
            infra_denied_empty_patch_instances=infra_denied_empty_patch_count,
            missing_patch_after_attempt_instances=missing_patch_after_attempt_instances,
            missing_patch_not_started_instances=missing_patch_not_started_instances,
            disallowed_test_patch_instances=disallowed_test_patch_count,
            attempted_instances=len(attempted_results),
            attempted_resolved_instances=attempted_resolved_count,
            attempted_resolution_rate=(
                attempted_resolved_count / len(attempted_results) if attempted_results else 0.0
            ),
            chain_blocked_instances=chain_blocked_count,
            lifecycle_first_failures=(
                self._lifecycle_first_failures(summary_results)
                if include_lifecycle_diagnostics
                else []
            ),
        )

    def _read_run_mini_swe_config(self) -> dict[str, Any] | None:
        """Load mini_swe settings from the run config when available."""
        config_path = getattr(self, "config_path", None)
        if not isinstance(config_path, Path) or not config_path.is_file():
            return None

        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logging.warning(f"Failed to read run config from {config_path}")
            return None

        if not isinstance(payload, dict):
            return None
        config = agent_config_from_run_config(payload)
        config.pop("agent_name", None)
        if not isinstance(config.get("model_name"), str):
            return None
        agent_config = payload.get("agent_config")
        if isinstance(agent_config, dict):
            reasoning_effort = agent_config.get("reasoning_effort")
            if (
                "reasoning_effort" not in config
                and isinstance(reasoning_effort, str)
                and reasoning_effort.strip()
            ):
                config["reasoning_effort"] = reasoning_effort.strip()
        config.setdefault("reasoning_effort", None)
        return config

    def _instance_chain_position(self, instance_id: str) -> tuple[str, int] | None:
        instance = self.instances.get(instance_id)  # type: ignore[attr-defined]
        if not isinstance(instance, dict):
            return None
        segment_id = instance.get("chain_id")
        stage_index = instance.get("chain_stage_index")
        if not isinstance(segment_id, str) or not segment_id.strip():
            return None
        if not isinstance(stage_index, int) or stage_index <= 0:
            return None
        return segment_id.strip(), stage_index

    def _supports_release_chain_progression(self, instance_ids: list[str]) -> bool:
        if not instance_ids:
            return False
        for instance_id in instance_ids:
            if self._instance_chain_position(instance_id) is None:
                return False
        return True

    def _instance_patch_path(self, instance_id: str, patch_key: str) -> str | None:
        instance = self.instances.get(instance_id)  # type: ignore[attr-defined]
        if not isinstance(instance, dict):
            return None
        patches = instance.get("patches")
        if not isinstance(patches, dict):
            return None
        patch_path = patches.get(patch_key)
        if not isinstance(patch_path, str) or not patch_path.strip():
            return None
        return patch_path.strip()

    def _instance_evaluation_test_patch_key(self, instance_id: str) -> str:
        instance = self.instances.get(instance_id)  # type: ignore[attr-defined]
        if isinstance(instance, dict):
            patch_key = instance.get("evaluation_test_patch")
            if isinstance(patch_key, str) and patch_key.strip():
                return patch_key.strip()
        return "golden_test_patch"

    def _instance_image_tag(self, instance_id: str) -> str | None:
        instance = self.instances.get(instance_id)  # type: ignore[attr-defined]
        if not isinstance(instance, dict):
            return None
        return immutable_image_ref(instance)

    def _instance_command_instance_id(self, instance_id: str) -> str | None:
        return instance_id if instance_id in self.instances else None  # type: ignore[attr-defined]

    def _segment_base_git_ref(self, segment_id: str) -> str | None:
        stage_map = self._segment_stage_instance_ids(segment_id)
        first_instance_id = stage_map.get(1)
        if not isinstance(first_instance_id, str) or not first_instance_id.strip():
            return None
        return "HEAD"

    def _instance_repo_name(self, instance_id: str) -> str | None:
        instance = self.instances.get(instance_id)  # type: ignore[attr-defined]
        if not isinstance(instance, dict):
            return None
        repo_name = instance.get("repo")
        if isinstance(repo_name, str) and repo_name.strip():
            return repo_name.strip()
        return None

    def _segment_stage_instance_ids(self, segment_id: str) -> dict[int, str]:
        stage_map: dict[int, str] = {}
        for instance_id, instance in self.instances.items():  # type: ignore[attr-defined]
            if not isinstance(instance, dict):
                continue
            instance_segment_id = instance.get("chain_id")
            stage_index = instance.get("chain_stage_index")
            if (
                not isinstance(instance_segment_id, str)
                or instance_segment_id.strip() != segment_id
            ):
                continue
            if not isinstance(stage_index, int) or stage_index <= 0:
                continue
            if not isinstance(instance_id, str) or not instance_id.strip():
                continue
            stage_map[stage_index] = instance_id
        return stage_map

    def _required_bridge_instance_ids(
        self,
        *,
        segment_id: str,
        stage_index: int,
        previous_stage_index: int | None,
    ) -> tuple[list[str] | None, str | None]:
        if stage_index <= 0:
            return None, "missing_or_invalid_chain_stage_index"

        stage_map = self._segment_stage_instance_ids(segment_id)
        if stage_index not in stage_map:
            return None, f"missing_chain_stage_instance:{segment_id}:{stage_index}"

        if previous_stage_index is None:
            bridge_indexes = list(range(1, stage_index))
        else:
            if stage_index <= previous_stage_index:
                return None, (
                    "invalid_chain_stage_number_order:"
                    f"{segment_id}:{previous_stage_index}:{stage_index}"
                )
            bridge_indexes = list(range(previous_stage_index + 1, stage_index))

        bridge_instance_ids: list[str] = []
        for bridge_stage_index in bridge_indexes:
            bridge_instance_id = stage_map.get(bridge_stage_index)
            if not isinstance(bridge_instance_id, str) or not bridge_instance_id.strip():
                return None, f"missing_chain_stage_instance:{segment_id}:{bridge_stage_index}"
            bridge_instance_ids.append(bridge_instance_id)
        return bridge_instance_ids, None

    def _append_required_bridge_patches(
        self,
        *,
        segment_id: str,
        instance_id: str,
        stage_index: int,
        previous_stage_index: int | None,
        destination_setup_patches: list[str],
        destination_code_patches: list[str],
    ) -> str | None:
        # Bridge code patches must be applied AFTER all carried test patches (a
        # stage's code change can depend on its own test-side file moves, e.g.
        # a dir -> symlink conversion). Env goes to setup and code goes to a
        # separate list so callers can order them as: env -> tests -> code.
        bridge_instance_ids, bridge_error = self._required_bridge_instance_ids(
            segment_id=segment_id,
            stage_index=stage_index,
            previous_stage_index=previous_stage_index,
        )
        if bridge_error is not None:
            return bridge_error
        if bridge_instance_ids is None:
            return "missing_bridge_stage_labels"

        for bridge_instance_id in bridge_instance_ids:
            bridge_env_patch = self._instance_patch_path(bridge_instance_id, "golden_env_patch")
            bridge_env_patch_path = (
                Path(bridge_env_patch)
                if isinstance(bridge_env_patch, str) and bridge_env_patch
                else None
            )
            if bridge_env_patch_path is not None and bridge_env_patch_path.is_file():
                destination_setup_patches.append(str(bridge_env_patch_path))

            bridge_patch = self._instance_patch_path(bridge_instance_id, "golden_patch")
            bridge_patch_path = (
                Path(bridge_patch) if isinstance(bridge_patch, str) and bridge_patch else None
            )
            if bridge_patch_path is None or not bridge_patch_path.is_file():
                missing = (
                    bridge_patch_path
                    if bridge_patch_path is not None
                    else f"{bridge_instance_id}:golden_patch.diff"
                )
                return f"missing_bridge_code_patch:{missing}"
            destination_code_patches.append(str(bridge_patch_path))
        return None

    def _append_current_env_patch(
        self,
        *,
        instance_id: str,
        destination_setup_patches: list[str],
    ) -> None:
        env_patch = self._instance_patch_path(instance_id, "golden_env_patch")
        if not isinstance(env_patch, str) or not env_patch.strip():
            return
        normalized = env_patch.strip()
        if normalized not in destination_setup_patches:
            destination_setup_patches.append(normalized)

    def assemble_chain_stage_stack(
        self, instance_id: str, *, agent_name: str
    ) -> dict[str, Any] | None:
        """Assemble the exact setup/test stack an eval used for a lifecycle stage.

        Replays all prior (resolved) stages of the segment, accumulating their
        cumulative visible+hidden test patches and — in lifecycle mode — their
        agent code patches, mirroring the eval-with-hidden-tests construction.
        Returns None for a non-chained instance (caller falls back to the plain
        single-instance stack).
        """
        position = self._instance_chain_position(instance_id)
        if position is None:
            return None
        segment_id, stage_index = position
        stage_map = self._segment_stage_instance_ids(segment_id)
        include_hidden = bool(getattr(self, "include_hidden_tests", True))

        prior_setup: list[str] = []
        prior_code: list[str] = []
        cumulative_visible: list[str] = []
        cumulative_hidden: list[str] = []
        for idx in sorted(k for k in stage_map if k < stage_index):
            prior_label = stage_map[idx]
            stack = self._resolve_current_test_patch_stack_for_instance(
                prior_label, include_hidden=include_hidden
            )
            if stack is None:
                raise RuntimeError(f"missing test patch for prior chain stage {prior_label}")
            setup_patches, current_patch = stack
            self._append_test_patch_stack_to_cumulative(
                instance_id=prior_label,
                setup_patches=setup_patches,
                current_patch=current_patch,
                destination_visible_test_patches=cumulative_visible,
                destination_hidden_test_patches=cumulative_hidden if include_hidden else None,
            )
            # Lifecycle: carry the prior stage's env + agent code patch forward.
            self._append_current_env_patch(
                instance_id=prior_label, destination_setup_patches=prior_setup
            )
            agent_patch = self._agent_patch_path(agent_name=agent_name, instance_id=prior_label)
            if not agent_patch.is_file():
                raise RuntimeError(f"missing prior-stage agent patch: {agent_patch}")
            prior_code.append(str(agent_patch))

        current_stack = self._resolve_current_test_patch_stack_for_instance(
            instance_id, include_hidden=include_hidden
        )
        if current_stack is None:
            raise RuntimeError(f"missing test patch for {instance_id}")
        current_test_setup, current_test_patch = current_stack

        stage_setup_patches = (
            list(prior_setup)
            + cumulative_visible
            + cumulative_hidden
            + prior_code
            + current_test_setup
        )
        return {
            "setup_patch_paths": stage_setup_patches,
            "test_patch": current_test_patch,
            "image_tag": self._instance_image_tag(instance_id),
            "base_git_ref": self._segment_base_git_ref(segment_id),
            "command_instance_id": self._instance_command_instance_id(instance_id),
        }

    def assemble_classic_stage_stack(self, instance_id: str) -> dict[str, Any] | None:
        """Assemble the exact independent baseline used for a basic/classic stage.

        Classic evaluation accumulates prior golden code/environment patches and
        cumulative tests, never patches produced by the evaluated agent.
        """
        position = self._instance_chain_position(instance_id)
        if position is None:
            return None
        segment_id, stage_index = position
        stage_map = self._segment_stage_instance_ids(segment_id)
        include_hidden = bool(getattr(self, "include_hidden_tests", True))

        prior_setup: list[str] = []
        prior_golden_code: list[str] = []
        cumulative_visible: list[str] = []
        cumulative_hidden: list[str] = []
        for idx in sorted(k for k in stage_map if k < stage_index):
            prior_label = stage_map[idx]
            stack = self._resolve_current_test_patch_stack_for_instance(
                prior_label, include_hidden=include_hidden
            )
            if stack is None:
                raise RuntimeError(f"missing test patch for prior classic stage {prior_label}")
            setup_patches, current_patch = stack
            self._append_test_patch_stack_to_cumulative(
                instance_id=prior_label,
                setup_patches=setup_patches,
                current_patch=current_patch,
                destination_visible_test_patches=cumulative_visible,
                destination_hidden_test_patches=cumulative_hidden if include_hidden else None,
            )
            self._append_current_env_patch(
                instance_id=prior_label, destination_setup_patches=prior_setup
            )
            golden_patch = self._instance_patch_path(prior_label, "golden_patch")
            if not isinstance(golden_patch, str) or not golden_patch.strip():
                raise RuntimeError(f"missing golden patch for prior classic stage {prior_label}")
            prior_golden_code.append(golden_patch.strip())

        current_stack = self._resolve_current_test_patch_stack_for_instance(
            instance_id, include_hidden=include_hidden
        )
        if current_stack is None:
            raise RuntimeError(f"missing test patch for {instance_id}")
        current_test_setup, current_test_patch = current_stack

        return {
            "setup_patch_paths": (
                prior_setup
                + cumulative_visible
                + cumulative_hidden
                + prior_golden_code
                + current_test_setup
            ),
            "test_patch": current_test_patch,
            "image_tag": self._instance_image_tag(instance_id),
            "base_git_ref": self._segment_base_git_ref(segment_id),
            "command_instance_id": self._instance_command_instance_id(instance_id),
        }

    def _append_required_bridge_test_patches(
        self,
        *,
        segment_id: str,
        instance_id: str,
        stage_index: int,
        previous_stage_index: int | None,
        destination_visible_test_patches: list[str],
        destination_hidden_test_patches: list[str] | None = None,
    ) -> str | None:
        bridge_instance_ids, bridge_error = self._required_bridge_instance_ids(
            segment_id=segment_id,
            stage_index=stage_index,
            previous_stage_index=previous_stage_index,
        )
        if bridge_error is not None:
            return bridge_error
        if bridge_instance_ids is None:
            return "missing_bridge_stage_labels"

        include_hidden = destination_hidden_test_patches is not None and bool(
            getattr(self, "include_hidden_tests", True)
        )
        for bridge_instance_id in bridge_instance_ids:
            stack = self._resolve_current_test_patch_stack_for_instance(
                bridge_instance_id,
                include_hidden=include_hidden,
            )
            if stack is None:
                silver = self._instance_patch_path(bridge_instance_id, "silver_test_patch")
                silver_patch = Path(silver) if isinstance(silver, str) and silver else None
                golden = self._instance_patch_path(bridge_instance_id, "golden_test_patch")
                golden_patch = Path(golden) if isinstance(golden, str) and golden else None
                missing = (
                    silver_patch
                    if silver_patch is not None
                    else golden_patch
                    if golden_patch is not None
                    else f"{bridge_instance_id}:silver_test_patch.diff|golden_test_patch.diff"
                )
                return f"missing_bridge_test_patch:{missing}"

            setup_patches, current_patch = stack
            self._append_test_patch_stack_to_cumulative(
                instance_id=bridge_instance_id,
                setup_patches=setup_patches,
                current_patch=current_patch,
                destination_visible_test_patches=destination_visible_test_patches,
                destination_hidden_test_patches=destination_hidden_test_patches,
            )

        return None

    def _resolve_visible_test_patch_for_instance(self, instance_id: str) -> str | None:
        silver = self._instance_patch_path(instance_id, "silver_test_patch")
        if silver is not None and Path(silver).is_file():
            return silver
        golden = self._instance_patch_path(instance_id, "golden_test_patch")
        if golden is not None and Path(golden).is_file():
            return golden
        return None

    def _resolve_hidden_test_patch_for_instance(self, instance_id: str) -> str | None:
        if self._instance_evaluation_test_patch_key(instance_id) != "bronze_test_patch":
            return None
        bronze = self._instance_patch_path(instance_id, "bronze_test_patch")
        if bronze is not None and Path(bronze).is_file():
            return bronze
        return None

    @staticmethod
    def _append_unique_path(paths: list[str], candidate: str | None) -> None:
        if candidate is None:
            return
        if candidate not in paths:
            paths.append(candidate)

    def _append_test_patch_stack_to_cumulative(
        self,
        *,
        instance_id: str,
        setup_patches: list[str],
        current_patch: str,
        destination_visible_test_patches: list[str],
        destination_hidden_test_patches: list[str] | None = None,
    ) -> None:
        """Keep visible test setup separate from hidden-only grading setup."""
        hidden_patch = self._resolve_hidden_test_patch_for_instance(instance_id)

        for patch_path in setup_patches:
            if (
                destination_hidden_test_patches is not None
                and hidden_patch is not None
                and patch_path == hidden_patch
            ):
                self._append_unique_path(destination_hidden_test_patches, patch_path)
            else:
                self._append_unique_path(destination_visible_test_patches, patch_path)

        if (
            destination_hidden_test_patches is not None
            and hidden_patch is not None
            and current_patch == hidden_patch
        ):
            self._append_unique_path(destination_hidden_test_patches, current_patch)
        else:
            self._append_unique_path(destination_visible_test_patches, current_patch)

    def _resolve_current_test_patch_stack_for_instance(
        self,
        instance_id: str,
        *,
        include_hidden: bool,
    ) -> tuple[list[str], str] | None:
        visible = self._resolve_visible_test_patch_for_instance(instance_id)
        if visible is None:
            return None

        hidden = (
            self._resolve_hidden_test_patch_for_instance(instance_id) if include_hidden else None
        )
        current = hidden or visible
        setup_patches: list[str] = []

        golden = self._instance_patch_path(instance_id, "golden_test_patch")
        if golden is not None and Path(golden).is_file() and golden != current:
            self._append_unique_path(setup_patches, golden)

        silver = self._instance_patch_path(instance_id, "silver_test_patch")
        if silver is not None and Path(silver).is_file() and silver != current:
            self._append_unique_path(setup_patches, silver)

        return setup_patches, current

    def _resolve_test_patch_for_instance(self, instance_id: str) -> str | None:
        """Return the configured evaluation test patch if it exists on disk."""
        patch_key = self._instance_evaluation_test_patch_key(instance_id)
        selected = self._instance_patch_path(instance_id, patch_key)
        if selected is not None and Path(selected).is_file():
            return selected
        silver = self._instance_patch_path(instance_id, "silver_test_patch")
        if silver is not None and Path(silver).is_file():
            return silver
        golden = self._instance_patch_path(instance_id, "golden_test_patch")
        if golden is not None and Path(golden).is_file():
            return golden
        return None

    def _write_cumulative_chain_test_patch(
        self,
        *,
        agent_name: str,
        segment_id: str,
        instance_id: str,
        patch_paths: list[str],
    ) -> Path:
        destination = (
            self.output_dir
            / "chain_eval"
            / agent_name
            / sanitize_label(segment_id)
            / f"{sanitize_label(instance_id)}.test.diff"
        )  # type: ignore[attr-defined]
        destination.parent.mkdir(parents=True, exist_ok=True)
        merged = bytearray()
        for patch_path in patch_paths:
            patch_file = Path(patch_path)
            if not patch_file.is_file():
                continue
            patch_bytes = patch_file.read_bytes()
            if not patch_bytes.strip():
                continue
            if merged and not merged.endswith(b"\n"):
                merged.extend(b"\n")
            merged.extend(patch_bytes)
            if not merged.endswith(b"\n"):
                merged.extend(b"\n")
        destination.write_bytes(bytes(merged))
        return destination

    def _blocked_chain_result(
        self,
        *,
        agent_name: str,
        instance_id: str,
        reason: str,
    ) -> EvaluationResult:
        return self._error_evaluation_result(  # type: ignore[attr-defined]
            instance_id=instance_id,
            agent_name=agent_name,
            error=f"Blocked by release-chain stop: {reason}",
        )

    def _disallowed_test_patch_result_if_any(
        self,
        *,
        agent_name: str,
        instance_id: str,
        patch_file: Path,
        repo: str | None = None,
    ) -> EvaluationResult | None:
        from bdd_bench.common.test_file_rules import is_env_path, is_non_bdd_test_path
        from bdd_bench.evaluation.harness_single import _disallowed_patch_result

        if not patch_file.is_file():
            return None
        patch = patch_file.read_bytes().decode("utf-8", errors="replace")
        disallowed_files = sorted(
            {
                path
                for path in extract_patch_paths(patch)
                if (
                    is_test_path(path, repo=repo)
                    or is_non_bdd_test_path(path, repo=repo)
                    or is_env_path(path, repo=repo)
                )
            },
        )
        if not disallowed_files:
            return None
        return _disallowed_patch_result(
            output_dir=self.output_dir,  # type: ignore[attr-defined]
            instance_id=instance_id,
            agent_name=agent_name,
            patch_file=patch_file,
            patch_text=patch,
            disallowed_files=disallowed_files,
        )

    def generate_patches_batch(
        self,
        agent: Agent,
        metadata_paths: list[Path],
        *,
        skip_existing_patch: bool = False,
    ) -> dict[str, Any]:
        """Generate patches for multiple instances without running evaluation."""
        filtered_paths: list[Path] = []
        for metadata_path in metadata_paths:
            try:
                data = load_metadata(metadata_path)
                instance_id = data["instance_id"]
                if instance_id in self.instances:  # type: ignore[attr-defined]
                    filtered_paths.append(metadata_path)
                else:
                    logging.info(f"Skipping {metadata_path.name}: not in evaluation manifest")
            except Exception as error:
                logging.warning(f"Could not load metadata from {metadata_path}: {error}")

        if not filtered_paths:
            logging.warning("No instances found in evaluation manifest for patch generation")
            return {
                "agent_name": agent.name,
                "total_instances": 0,
                "generated": 0,
                "skipped_existing": 0,
                "errors": 0,
                "results": [],
            }

        logging.info(
            f"Generating patches with {agent.name} on {len(filtered_paths)} instance(s) "
            f"(filtered from {len(metadata_paths)} total) with {self.workers} worker(s)"  # type: ignore[attr-defined]
        )
        if agent.requires_runtime_context:
            ensure_docker_available()

        progression_mode = getattr(self, "progression_mode", "basic")
        chain_instance_ids: list[str] = []
        if progression_mode in {"basic", "lifecycle"}:
            for path in filtered_paths:
                try:
                    chain_instance_ids.append(load_metadata(path)["instance_id"])
                except Exception:
                    continue
        use_chain_progression = progression_mode in {"basic", "lifecycle"} and (
            self._supports_release_chain_progression(chain_instance_ids)
        )
        stop_on_stage_failure = progression_mode == "lifecycle"

        results: list[EvaluationResult] = []
        if use_chain_progression:
            grouped_paths: dict[str, list[tuple[int, Path, str]]] = {}
            for metadata_path in filtered_paths:
                try:
                    data = load_metadata(metadata_path)
                except Exception as error:
                    logging.warning("Could not load metadata from %s: %s", metadata_path, error)
                    results.append(
                        EvaluationResult(
                            instance_id=str(metadata_path),
                            agent_name=agent.name,
                            error=f"Failed to load metadata: {error}",
                        )
                    )
                    continue
                instance_id = data["instance_id"]
                position = self._instance_chain_position(instance_id)
                if position is None:
                    results.append(
                        EvaluationResult(
                            instance_id=instance_id,
                            agent_name=agent.name,
                            error=f"Missing chain position for {instance_id}",
                        )
                    )
                    continue
                segment_id, stage_index = position
                grouped_paths.setdefault(segment_id, []).append(
                    (stage_index, metadata_path, instance_id)
                )

            completed = 0
            fatal_generation_error = False
            with tqdm(
                total=len(filtered_paths),
                desc=f"Generating {agent.name} patches",
                unit="instance",
            ) as progress_bar:
                for segment_id in sorted(grouped_paths):
                    ordered = sorted(grouped_paths[segment_id], key=lambda item: item[0])
                    segment_stop_reason: str | None = None
                    previous_stage_index: int | None = None
                    segment_base_git_ref = self._segment_base_git_ref(segment_id)
                    if not isinstance(segment_base_git_ref, str) or not segment_base_git_ref:
                        segment_stop_reason = f"missing_segment_base_git_ref:{segment_id}"
                    prior_setup_patches: list[str] = []
                    prior_code_patches: list[str] = []
                    cumulative_visible_test_patch_paths: list[str] = []

                    for stage_index, metadata_path, instance_id in ordered:
                        if segment_stop_reason is not None:
                            result = self._blocked_chain_result(
                                agent_name=agent.name,
                                instance_id=instance_id,
                                reason=segment_stop_reason,
                            )
                            status = "SKIPPED"
                        elif (
                            previous_stage_index is not None and stage_index <= previous_stage_index
                        ):
                            segment_stop_reason = f"invalid_chain_stage_index_order:{previous_stage_index}:{stage_index}"
                            result = self._error_evaluation_result(  # type: ignore[attr-defined]
                                instance_id=instance_id,
                                agent_name=agent.name,
                                error=segment_stop_reason,
                            )
                            status = "ERROR"
                        else:
                            bridge_error = self._append_required_bridge_patches(
                                segment_id=segment_id,
                                instance_id=instance_id,
                                stage_index=stage_index,
                                previous_stage_index=previous_stage_index,
                                destination_setup_patches=prior_setup_patches,
                                destination_code_patches=prior_code_patches,
                            )
                            if bridge_error is not None:
                                segment_stop_reason = bridge_error
                                result = self._error_evaluation_result(  # type: ignore[attr-defined]
                                    instance_id=instance_id,
                                    agent_name=agent.name,
                                    error=segment_stop_reason,
                                )
                                status = "ERROR"
                            else:
                                bridge_test_error = self._append_required_bridge_test_patches(
                                    segment_id=segment_id,
                                    instance_id=instance_id,
                                    stage_index=stage_index,
                                    previous_stage_index=previous_stage_index,
                                    destination_visible_test_patches=cumulative_visible_test_patch_paths,
                                )
                                if bridge_test_error is not None:
                                    segment_stop_reason = bridge_test_error
                                    result = self._error_evaluation_result(  # type: ignore[attr-defined]
                                        instance_id=instance_id,
                                        agent_name=agent.name,
                                        error=segment_stop_reason,
                                    )
                                    status = "ERROR"
                                else:
                                    current_test_patch_stack = (
                                        self._resolve_current_test_patch_stack_for_instance(
                                            instance_id,
                                            include_hidden=False,
                                        )
                                    )
                                    if current_test_patch_stack is None:
                                        segment_stop_reason = (
                                            f"missing_chain_test_patch:{instance_id}"
                                        )
                                        result = self._error_evaluation_result(  # type: ignore[attr-defined]
                                            instance_id=instance_id,
                                            agent_name=agent.name,
                                            error=segment_stop_reason,
                                        )
                                        status = "ERROR"
                                    else:
                                        current_test_setup_patches, current_test_patch = (
                                            current_test_patch_stack
                                        )
                                        stage_image_override = self._instance_image_tag(instance_id)
                                        stage_command_instance_id = (
                                            self._instance_command_instance_id(instance_id)
                                        )
                                        stage_setup_patches = list(prior_setup_patches)
                                        stage_setup_patches.extend(
                                            cumulative_visible_test_patch_paths
                                        )
                                        stage_setup_patches.extend(prior_code_patches)
                                        stage_setup_patches.extend(current_test_setup_patches)
                                        result = self.generate_patch_single(  # type: ignore[attr-defined]
                                            agent,
                                            metadata_path,
                                            skip_existing_patch=skip_existing_patch,
                                            setup_patch_paths=stage_setup_patches,
                                            test_patch_path_override=current_test_patch,
                                            image_tag_override=stage_image_override,
                                            base_git_ref_override=segment_base_git_ref,
                                            command_instance_id_override=stage_command_instance_id,
                                        )
                                        if result.error:
                                            status = "ERROR"
                                            if _is_fatal_generation_error(result.error):
                                                logging.error(
                                                    "Stopping patch generation after fatal agent "
                                                    "failure on %s: %s",
                                                    instance_id,
                                                    result.error,
                                                )
                                                results.append(result)
                                                completed += 1
                                                progress_bar.set_postfix(
                                                    {"last": instance_id[:24], "status": status},
                                                    refresh=False,
                                                )
                                                tqdm.write(
                                                    f"[{completed}/{len(filtered_paths)}] "
                                                    f"{instance_id}: {status}"
                                                )
                                                progress_bar.update(1)
                                                generated_count = sum(
                                                    1
                                                    for item in results
                                                    if not item.error
                                                    and not item.detailed_results.get(
                                                        "skipped_existing_patch", False
                                                    )
                                                )
                                                skipped_existing_count = sum(
                                                    1
                                                    for item in results
                                                    if item.detailed_results.get(
                                                        "skipped_existing_patch", False
                                                    )
                                                )
                                                error_count = sum(
                                                    1 for item in results if item.error
                                                )
                                                return {
                                                    "agent_name": agent.name,
                                                    "total_instances": len(results),
                                                    "generated": generated_count,
                                                    "skipped_existing": skipped_existing_count,
                                                    "errors": error_count,
                                                    "results": results,
                                                }
                                            if stop_on_stage_failure:
                                                segment_stop_reason = (
                                                    f"stage_generation_failed:{instance_id}"
                                                )
                                        elif result.detailed_results.get(
                                            "skipped_existing_patch", False
                                        ):
                                            status = "SKIPPED"
                                        else:
                                            status = "GENERATED"

                                        if not result.error:
                                            self._append_test_patch_stack_to_cumulative(
                                                instance_id=instance_id,
                                                setup_patches=current_test_setup_patches,
                                                current_patch=current_test_patch,
                                                destination_visible_test_patches=(
                                                    cumulative_visible_test_patch_paths
                                                ),
                                            )
                                            if progression_mode == "lifecycle":
                                                patch_file = self._agent_patch_path(  # type: ignore[attr-defined]
                                                    agent_name=agent.name,
                                                    instance_id=instance_id,
                                                )
                                                if not Path(patch_file).is_file():
                                                    segment_stop_reason = (
                                                        f"missing_generated_patch:{instance_id}"
                                                    )
                                                else:
                                                    self._append_current_env_patch(
                                                        instance_id=instance_id,
                                                        destination_setup_patches=prior_setup_patches,
                                                    )
                                                    prior_code_patches.append(str(patch_file))
                                                    previous_stage_index = stage_index
                                            else:
                                                golden_patch = self._instance_patch_path(
                                                    instance_id, "golden_patch"
                                                )
                                                if (
                                                    golden_patch is None
                                                    or not Path(golden_patch).is_file()
                                                ):
                                                    segment_stop_reason = (
                                                        f"missing_chain_golden_patch:{instance_id}"
                                                    )
                                                else:
                                                    self._append_current_env_patch(
                                                        instance_id=instance_id,
                                                        destination_setup_patches=prior_setup_patches,
                                                    )
                                                    prior_code_patches.append(golden_patch)
                                                    previous_stage_index = stage_index

                        results.append(result)
                        completed += 1
                        progress_bar.set_postfix(
                            {"last": instance_id[:24], "status": status},
                            refresh=False,
                        )
                        tqdm.write(f"[{completed}/{len(filtered_paths)}] {instance_id}: {status}")
                        progress_bar.update(1)

            generated_count = sum(
                1
                for result in results
                if not result.error
                and not result.detailed_results.get("skipped_existing_patch", False)
            )
            skipped_existing_count = sum(
                1
                for result in results
                if result.detailed_results.get("skipped_existing_patch", False)
            )
            error_count = sum(1 for result in results if result.error)
            return {
                "agent_name": agent.name,
                "total_instances": len(results),
                "generated": generated_count,
                "skipped_existing": skipped_existing_count,
                "errors": error_count,
                "results": results,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:  # type: ignore[attr-defined]
            future_to_path = {
                executor.submit(
                    self.generate_patch_single,  # type: ignore[attr-defined]
                    agent,
                    metadata_path,
                    skip_existing_patch=skip_existing_patch,
                ): metadata_path
                for metadata_path in filtered_paths
            }

            completed = 0
            fatal_generation_error = False
            with tqdm(
                total=len(filtered_paths),
                desc=f"Generating {agent.name} patches",
                unit="instance",
            ) as progress_bar:
                for future in concurrent.futures.as_completed(future_to_path):
                    metadata_path = future_to_path[future]
                    completed += 1
                    fatal_generation_error = False
                    try:
                        result = future.result()
                        results.append(result)
                        skipped_existing = bool(
                            result.detailed_results.get("skipped_existing_patch", False),
                        )
                        if result.error:
                            status = "ERROR"
                        elif skipped_existing:
                            status = "SKIPPED"
                        else:
                            status = "GENERATED"
                        logging.info(
                            f"[{completed}/{len(filtered_paths)}] {metadata_path.name}: {status}"
                        )
                        fatal_generation_error = _is_fatal_generation_error(result.error)
                        if fatal_generation_error:
                            logging.error(
                                "Stopping patch generation after fatal agent failure on %s: %s",
                                metadata_path.stem,
                                result.error,
                            )
                            for pending_future in future_to_path:
                                if pending_future is not future:
                                    pending_future.cancel()
                        progress_bar.set_postfix(
                            {"last": metadata_path.stem[:24], "status": status}, refresh=False
                        )
                        tqdm.write(
                            f"[{completed}/{len(filtered_paths)}] {metadata_path.stem}: {status}"
                        )
                    except Exception as error:
                        logging.exception(
                            f"Unexpected error generating patch for {metadata_path}: {error}"
                        )
                        results.append(
                            EvaluationResult(
                                instance_id=str(metadata_path),
                                agent_name=agent.name,
                                error=f"Unexpected error: {error}",
                            )
                        )
                        progress_bar.set_postfix(
                            {"last": metadata_path.stem[:24], "status": "ERROR"},
                            refresh=False,
                        )
                        tqdm.write(
                            f"[{completed}/{len(filtered_paths)}] {metadata_path.stem}: ERROR"
                        )
                    finally:
                        progress_bar.update(1)
                    if fatal_generation_error:
                        break

        generated_count = sum(
            1
            for result in results
            if not result.error and not result.detailed_results.get("skipped_existing_patch", False)
        )
        skipped_existing_count = sum(
            1 for result in results if result.detailed_results.get("skipped_existing_patch", False)
        )
        error_count = sum(1 for result in results if result.error)
        return {
            "agent_name": agent.name,
            "total_instances": len(results),
            "generated": generated_count,
            "skipped_existing": skipped_existing_count,
            "errors": error_count,
            "results": results,
        }

    def _resolve_metadata_patch_pairs(
        self,
        *,
        agent_name: str,
        metadata_paths: list[Path],
        patches_dir: Path,
        skip_missing_patches: bool,
    ) -> tuple[
        list[tuple[Path, Path, str]],
        list[str],
        list[EvaluationResult],
        int,
        list[str],
        list[str],
    ]:
        pairs: list[tuple[Path, Path, str]] = []
        missing_patch_instances: list[str] = []
        missing_patch_after_attempt_instances: list[str] = []
        missing_patch_not_started_instances: list[str] = []
        preexisting_results: list[EvaluationResult] = []
        skipped_instances = 0

        for metadata_path in metadata_paths:
            try:
                data = load_metadata(metadata_path)
                instance_id = data["instance_id"]
                if instance_id not in self.instances:  # type: ignore[attr-defined]
                    logging.info(f"Skipping {metadata_path.name}: not in evaluation manifest")
                    skipped_instances += 1
                    continue

                patch_file = self.resolve_patch_file(  # type: ignore[attr-defined]
                    patches_dir=patches_dir,
                    agent_name=agent_name,
                    instance_id=instance_id,
                )
                if patch_file is None:
                    missing_patch_instances.append(instance_id)
                    prior_failure_result = self._prior_terminal_generation_failure_result(
                        agent_name=agent_name,
                        instance_id=instance_id,
                    )
                    attempted_generation = (
                        prior_failure_result is not None
                        or self._has_generation_attempt_artifacts(instance_id)
                    )
                    if attempted_generation:
                        missing_patch_after_attempt_instances.append(instance_id)
                        if prior_failure_result is not None:
                            preexisting_results.append(prior_failure_result)
                        else:
                            preexisting_results.append(
                                self._error_evaluation_result(  # type: ignore[attr-defined]
                                    instance_id=instance_id,
                                    agent_name=agent_name,
                                    error=(
                                        "Patch not emitted after attempted agent generation "
                                        f"in {patches_dir}"
                                    ),
                                )
                            )
                    else:
                        missing_patch_not_started_instances.append(instance_id)
                    logging.warning(f"Patch not found for {instance_id} in {patches_dir}")
                    if skip_missing_patches and not attempted_generation:
                        skipped_instances += 1
                    elif not skip_missing_patches and not attempted_generation:
                        preexisting_results.append(
                            self._error_evaluation_result(  # type: ignore[attr-defined]
                                instance_id=instance_id,
                                agent_name=agent_name,
                                error=f"Patch not found in {patches_dir}",
                            )
                        )
                    continue

                pairs.append((metadata_path, patch_file, instance_id))
            except Exception as error:
                logging.warning(f"Could not load metadata from {metadata_path}: {error}")
                skipped_instances += 1

        return (
            pairs,
            missing_patch_instances,
            preexisting_results,
            skipped_instances,
            missing_patch_after_attempt_instances,
            missing_patch_not_started_instances,
        )

    def run_tests_batch_from_patches(
        self,
        *,
        agent_name: str,
        metadata_paths: list[Path],
        patches_dir: Path,
        skip_missing_patches: bool = True,
        skip_instances_with_existing_eval_logs: bool = False,
    ) -> tuple[dict[str, Any], list[str]]:
        (
            pairs,
            missing_patch_instances,
            preexisting_results,
            skipped_instances,
            missing_patch_after_attempt_instances,
            missing_patch_not_started_instances,
        ) = self._resolve_metadata_patch_pairs(
            agent_name=agent_name,
            metadata_paths=metadata_paths,
            patches_dir=patches_dir,
            skip_missing_patches=skip_missing_patches,
        )
        skipped_existing_eval_logs = 0

        # Precompute existing-log dirs once. In chain-progression mode we cannot
        # drop instances with existing logs from the iteration because the chain
        # needs every prior stage's agent code patch to be accumulated into
        # prior_setup_patches (in lifecycle mode) — dropping them would make the
        # bridge fall back to golden patches and the current stage's diff would
        # then fail to apply against a different base state.
        existing_logs_for_instance_id: dict[str, Path] = {}
        if skip_instances_with_existing_eval_logs and pairs:
            for _metadata_path, _patch_file, instance_id in pairs:
                existing_log_dir = self._latest_eval_log_dir(  # type: ignore[attr-defined]
                    agent_name=agent_name,
                    instance_id=instance_id,
                )
                if existing_log_dir is None:
                    continue
                run_log_paths = self._run_log_paths_from_dir(existing_log_dir)  # type: ignore[attr-defined]
                if len(run_log_paths) >= self.runs:  # type: ignore[attr-defined]
                    existing_logs_for_instance_id[instance_id] = existing_log_dir

        progression_mode = getattr(self, "progression_mode", "basic")
        stop_on_stage_failure = progression_mode == "lifecycle"
        chain_instance_ids = [instance_id for _m, _p, instance_id in pairs]
        use_chain_progression = (
            progression_mode in {"basic", "lifecycle"}
            and bool(pairs)
            and self._supports_release_chain_progression(chain_instance_ids)
        )

        if use_chain_progression:
            # Keep every chain stage in iteration order; per-stage we decide whether
            # to actually run tests or replay state from the existing eval logs.
            pairs_to_run: list[tuple[Path, Path, str]] = list(pairs)
        elif existing_logs_for_instance_id:
            filtered_pairs: list[tuple[Path, Path, str]] = []
            for metadata_path, patch_file, instance_id in pairs:
                existing_log_dir = existing_logs_for_instance_id.get(instance_id)
                if existing_log_dir is not None:
                    skipped_existing_eval_logs += 1
                    skipped_instances += 1
                    logging.info(
                        f"Skipping test run for {instance_id}: existing evaluation logs in "
                        f"{existing_log_dir}"
                    )
                    continue
                filtered_pairs.append((metadata_path, patch_file, instance_id))
            pairs_to_run = filtered_pairs
        else:
            pairs_to_run = list(pairs)

        if not pairs_to_run and not preexisting_results:
            logging.warning("No instances found with both metadata and patch files to run")
            return {
                "agent_name": agent_name,
                "total_instances": 0,
                "completed": 0,
                "errors": 0,
                "skipped_instances": skipped_instances,
                "skipped_existing_eval_logs": skipped_existing_eval_logs,
                "missing_patch_after_attempt_instances": len(missing_patch_after_attempt_instances),
                "missing_patch_not_started_instances": len(missing_patch_not_started_instances),
            }, missing_patch_instances

        run_payloads: list[dict[str, Any]] = []
        if pairs_to_run:
            if use_chain_progression:
                ensure_docker_available()
                grouped_pairs: dict[str, list[tuple[int, Path, Path, str]]] = {}
                for metadata_path, patch_file, instance_id in pairs_to_run:
                    position = self._instance_chain_position(instance_id)
                    if position is None:
                        continue
                    segment_id, stage_index = position
                    grouped_pairs.setdefault(segment_id, []).append(
                        (stage_index, metadata_path, patch_file, instance_id)
                    )
                completed = 0
                with tqdm(
                    total=len(pairs_to_run),
                    desc=f"Run tests for existing {agent_name} patches",
                    unit="instance",
                ) as progress_bar:
                    for segment_id in sorted(grouped_pairs):
                        ordered = sorted(grouped_pairs[segment_id], key=lambda item: item[0])
                        segment_stop_reason: str | None = None
                        previous_stage_index: int | None = None
                        segment_base_git_ref = self._segment_base_git_ref(segment_id)
                        if not isinstance(segment_base_git_ref, str) or not segment_base_git_ref:
                            segment_stop_reason = f"missing_segment_base_git_ref:{segment_id}"
                        # Accumulated code setup patches applied before each instance.
                        # In basic mode: golden code patches from prior stages.
                        # In lifecycle mode: agent code patches from prior stages.
                        prior_setup_patches: list[str] = []
                        prior_code_patches: list[str] = []
                        cumulative_visible_test_patch_paths: list[str] = []
                        cumulative_hidden_test_patch_paths: list[str] = []
                        include_hidden_tests = bool(getattr(self, "include_hidden_tests", True))

                        for stage_index, metadata_path, patch_file, instance_id in ordered:
                            status = "DONE"
                            if segment_stop_reason is not None:
                                skipped_instances += 1
                                logging.info(
                                    "Skipping %s in %s: chain already stopped (%s).",
                                    instance_id,
                                    segment_id,
                                    segment_stop_reason,
                                )
                                status = "SKIPPED"
                            elif (
                                previous_stage_index is not None
                                and stage_index <= previous_stage_index
                            ):
                                segment_stop_reason = f"invalid_chain_stage_index_order:{previous_stage_index}:{stage_index}"
                                logging.warning(
                                    "Stopping segment %s at %s: %s",
                                    segment_id,
                                    instance_id,
                                    segment_stop_reason,
                                )
                                run_payloads.append(
                                    {
                                        "instance_id": instance_id,
                                        "error": segment_stop_reason,
                                    }
                                )
                                status = "ERROR"
                            else:
                                bridge_error = self._append_required_bridge_patches(
                                    segment_id=segment_id,
                                    instance_id=instance_id,
                                    stage_index=stage_index,
                                    previous_stage_index=previous_stage_index,
                                    destination_setup_patches=prior_setup_patches,
                                    destination_code_patches=prior_code_patches,
                                )
                                if bridge_error is not None:
                                    segment_stop_reason = bridge_error
                                    logging.warning(
                                        "Stopping segment %s at %s during bridge replay: %s",
                                        segment_id,
                                        instance_id,
                                        segment_stop_reason,
                                    )
                                    run_payloads.append(
                                        {
                                            "instance_id": instance_id,
                                            "error": segment_stop_reason,
                                        }
                                    )
                                    status = "ERROR"
                                else:
                                    bridge_test_error = self._append_required_bridge_test_patches(
                                        segment_id=segment_id,
                                        instance_id=instance_id,
                                        stage_index=stage_index,
                                        previous_stage_index=previous_stage_index,
                                        destination_visible_test_patches=cumulative_visible_test_patch_paths,
                                        destination_hidden_test_patches=(
                                            cumulative_hidden_test_patch_paths
                                            if include_hidden_tests
                                            else None
                                        ),
                                    )
                                    if bridge_test_error is not None:
                                        segment_stop_reason = bridge_test_error
                                        logging.warning(
                                            "Stopping segment %s at %s during bridge test replay: %s",
                                            segment_id,
                                            instance_id,
                                            segment_stop_reason,
                                        )
                                        run_payloads.append(
                                            {
                                                "instance_id": instance_id,
                                                "error": segment_stop_reason,
                                            }
                                        )
                                        status = "ERROR"
                                    else:
                                        current_test_patch_stack = (
                                            self._resolve_current_test_patch_stack_for_instance(
                                                instance_id,
                                                include_hidden=include_hidden_tests,
                                            )
                                        )
                                        if current_test_patch_stack is None:
                                            segment_stop_reason = (
                                                f"missing_chain_test_patch:{instance_id}"
                                            )
                                            logging.warning(
                                                "Stopping segment %s at %s: %s",
                                                segment_id,
                                                instance_id,
                                                segment_stop_reason,
                                            )
                                            run_payloads.append(
                                                {
                                                    "instance_id": instance_id,
                                                    "error": segment_stop_reason,
                                                }
                                            )
                                            status = "ERROR"
                                        else:
                                            current_test_setup_patches, current_test_patch = (
                                                current_test_patch_stack
                                            )
                                            stage_image_override = self._instance_image_tag(
                                                instance_id
                                            )
                                            stage_command_instance_id = (
                                                self._instance_command_instance_id(instance_id)
                                            )
                                            stage_resolved = False
                                            existing_log_dir_for_stage = (
                                                existing_logs_for_instance_id.get(instance_id)
                                            )
                                            if existing_log_dir_for_stage is not None:
                                                # Replay path: don't run tests, just evaluate
                                                # existing logs so the chain accumulator can
                                                # carry the agent code patch forward.
                                                skipped_existing_eval_logs += 1
                                                skipped_instances += 1
                                                logging.info(
                                                    f"Replaying chain state for {instance_id} "
                                                    f"from existing evaluation logs in "
                                                    f"{existing_log_dir_for_stage}"
                                                )
                                                evaluation_result = (
                                                    self.evaluate_single_test_results_from_logs(  # type: ignore[attr-defined]
                                                        agent_name=agent_name,
                                                        metadata_path=metadata_path,
                                                        patch_file=patch_file,
                                                        log_dir=existing_log_dir_for_stage,
                                                    )
                                                )
                                                stage_resolved = evaluation_result.resolved
                                                status = (
                                                    "REPLAY-RESOLVED"
                                                    if stage_resolved
                                                    else "REPLAY-FAILED"
                                                )
                                                if not stage_resolved and stop_on_stage_failure:
                                                    segment_stop_reason = (
                                                        f"stage_not_resolved:{instance_id}"
                                                    )
                                                    logging.warning(
                                                        "Stopping segment %s at %s: %s",
                                                        segment_id,
                                                        instance_id,
                                                        segment_stop_reason,
                                                    )
                                            else:
                                                stage_setup_patches = list(prior_setup_patches)
                                                stage_setup_patches.extend(
                                                    cumulative_visible_test_patch_paths
                                                )
                                                stage_setup_patches.extend(
                                                    cumulative_hidden_test_patch_paths
                                                )
                                                stage_setup_patches.extend(prior_code_patches)
                                                stage_setup_patches.extend(
                                                    current_test_setup_patches
                                                )

                                                payload = self.run_tests_single_from_patch(  # type: ignore[attr-defined]
                                                    agent_name=agent_name,
                                                    metadata_path=metadata_path,
                                                    patch_file=patch_file,
                                                    setup_patch_paths=stage_setup_patches,
                                                    test_patch_path_override=current_test_patch,
                                                    image_tag_override=stage_image_override,
                                                    base_git_ref_override=segment_base_git_ref,
                                                    command_instance_id_override=stage_command_instance_id,
                                                )
                                                run_payloads.append(payload)
                                                status = self._payload_status_label(payload)
                                                run_error = payload.get("error")
                                                if isinstance(run_error, str) and run_error:
                                                    if stop_on_stage_failure:
                                                        segment_stop_reason = (
                                                            f"stage_test_run_failed:{instance_id}"
                                                        )
                                                        logging.warning(
                                                            "Stopping segment %s at %s: %s (%s)",
                                                            segment_id,
                                                            instance_id,
                                                            segment_stop_reason,
                                                            run_error,
                                                        )
                                                else:
                                                    log_dir_raw = payload.get("log_dir")
                                                    log_dir = (
                                                        Path(log_dir_raw)
                                                        if isinstance(log_dir_raw, str)
                                                        and log_dir_raw
                                                        else None
                                                    )
                                                    if log_dir is None:
                                                        if stop_on_stage_failure:
                                                            segment_stop_reason = (
                                                                f"missing_stage_logs:{instance_id}"
                                                            )
                                                            logging.warning(
                                                                "Stopping segment %s at %s: %s",
                                                                segment_id,
                                                                instance_id,
                                                                segment_stop_reason,
                                                            )
                                                    else:
                                                        evaluation_result = self.evaluate_single_test_results_from_logs(  # type: ignore[attr-defined]
                                                            agent_name=agent_name,
                                                            metadata_path=metadata_path,
                                                            patch_file=patch_file,
                                                            log_dir=log_dir,
                                                        )
                                                        stage_resolved = evaluation_result.resolved
                                                        if (
                                                            not stage_resolved
                                                            and stop_on_stage_failure
                                                        ):
                                                            segment_stop_reason = (
                                                                f"stage_not_resolved:{instance_id}"
                                                            )
                                                            logging.warning(
                                                                "Stopping segment %s at %s: %s",
                                                                segment_id,
                                                                instance_id,
                                                                segment_stop_reason,
                                                            )

                                            # Accumulate setup patches for subsequent stages.
                                            if status != "ERROR" and status != "SKIPPED":
                                                self._append_test_patch_stack_to_cumulative(
                                                    instance_id=instance_id,
                                                    setup_patches=current_test_setup_patches,
                                                    current_patch=current_test_patch,
                                                    destination_visible_test_patches=(
                                                        cumulative_visible_test_patch_paths
                                                    ),
                                                    destination_hidden_test_patches=(
                                                        cumulative_hidden_test_patch_paths
                                                        if include_hidden_tests
                                                        else None
                                                    ),
                                                )
                                                if progression_mode == "lifecycle":
                                                    # Lifecycle: carry agent code patch forward.
                                                    if stage_resolved:
                                                        self._append_current_env_patch(
                                                            instance_id=instance_id,
                                                            destination_setup_patches=prior_setup_patches,
                                                        )
                                                        prior_code_patches.append(str(patch_file))
                                                        previous_stage_index = stage_index
                                                else:
                                                    # Basic: carry golden code patch forward.
                                                    golden_patch = self._instance_patch_path(
                                                        instance_id, "golden_patch"
                                                    )
                                                    if (
                                                        golden_patch is None
                                                        or not Path(golden_patch).is_file()
                                                    ):
                                                        segment_stop_reason = f"missing_chain_golden_patch:{instance_id}"
                                                        logging.warning(
                                                            "Stopping segment %s at %s: %s",
                                                            segment_id,
                                                            instance_id,
                                                            segment_stop_reason,
                                                        )
                                                    else:
                                                        self._append_current_env_patch(
                                                            instance_id=instance_id,
                                                            destination_setup_patches=prior_setup_patches,
                                                        )
                                                        prior_code_patches.append(golden_patch)
                                                        previous_stage_index = stage_index

                            completed += 1
                            self._emit_test_run_progress(
                                progress_bar=progress_bar,
                                run_payloads=run_payloads,
                                completed=completed,
                                total=len(pairs_to_run),
                                last_label=instance_id,
                                status=status,
                                skipped=skipped_instances,
                            )
                            progress_bar.update(1)
            else:
                logging.info(
                    f"Running tests for existing patches ({agent_name}) on {len(pairs_to_run)} instance(s) "
                    f"with {self.workers} worker(s)"  # type: ignore[attr-defined]
                )
                ensure_docker_available()
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:  # type: ignore[attr-defined]
                    future_to_pair = {
                        executor.submit(
                            self.run_tests_single_from_patch,  # type: ignore[attr-defined]
                            agent_name=agent_name,
                            metadata_path=metadata_path,
                            patch_file=patch_file,
                        ): (metadata_path, patch_file)
                        for metadata_path, patch_file, _instance_id in pairs_to_run
                    }
                    completed = 0
                    with tqdm(
                        total=len(pairs_to_run),
                        desc=f"Run tests for existing {agent_name} patches",
                        unit="instance",
                    ) as progress_bar:
                        for future in concurrent.futures.as_completed(future_to_pair):
                            metadata_path, patch_file = future_to_pair[future]
                            completed += 1
                            try:
                                payload = future.result()
                                run_payloads.append(payload)
                                status = "DONE"
                                if payload.get("error"):
                                    status = "ERROR"
                                logging.info(
                                    f"[{completed}/{len(pairs_to_run)}] {metadata_path.name} "
                                    f"({patch_file.name}): {status}"
                                )
                                self._emit_test_run_progress(
                                    progress_bar=progress_bar,
                                    run_payloads=run_payloads,
                                    completed=completed,
                                    total=len(pairs_to_run),
                                    last_label=metadata_path.stem,
                                    status=status,
                                    skipped=skipped_instances,
                                )
                            except Exception as error:
                                logging.exception(
                                    f"Unexpected error running tests for {metadata_path}: {error}",
                                )
                                run_payloads.append(
                                    {
                                        "instance_id": str(metadata_path),
                                        "error": f"Unexpected error: {error}",
                                    }
                                )
                                self._emit_test_run_progress(
                                    progress_bar=progress_bar,
                                    run_payloads=run_payloads,
                                    completed=completed,
                                    total=len(pairs_to_run),
                                    last_label=metadata_path.stem,
                                    status="ERROR",
                                    skipped=skipped_instances,
                                )
                            finally:
                                progress_bar.update(1)

        run_error_count = sum(1 for payload in run_payloads if payload.get("error"))
        total_errors = run_error_count + sum(1 for result in preexisting_results if result.error)
        return {
            "agent_name": agent_name,
            "total_instances": len(run_payloads) + len(preexisting_results),
            "completed": len(run_payloads) - run_error_count,
            "errors": total_errors,
            "skipped_instances": skipped_instances,
            "skipped_existing_eval_logs": skipped_existing_eval_logs,
            "missing_patch_after_attempt_instances": len(missing_patch_after_attempt_instances),
            "missing_patch_not_started_instances": len(missing_patch_not_started_instances),
        }, missing_patch_instances

    def evaluate_test_results_batch_from_patches(
        self,
        *,
        agent_name: str,
        metadata_paths: list[Path],
        patches_dir: Path,
        skip_missing_patches: bool = True,
    ) -> tuple[EvaluationSummary, list[str], list[str]]:
        (
            pairs,
            missing_patch_instances,
            results,
            skipped_instances,
            missing_patch_after_attempt_instances,
            missing_patch_not_started_instances,
        ) = self._resolve_metadata_patch_pairs(
            agent_name=agent_name,
            metadata_paths=metadata_paths,
            patches_dir=patches_dir,
            skip_missing_patches=skip_missing_patches,
        )
        missing_eval_log_instances: list[str] = []
        evaluable_items: list[tuple[Path, Path, Path, str]] = []

        for metadata_path, patch_file, instance_id in pairs:
            log_dir = self._latest_eval_log_dir(  # type: ignore[attr-defined]
                agent_name=agent_name,
                instance_id=instance_id,
            )
            if log_dir is None:
                repo_hint = None
                try:
                    meta = load_metadata(metadata_path)
                    repo_hint = meta.get("repo")
                except Exception:
                    pass
                disallowed_result = self._disallowed_test_patch_result_if_any(
                    agent_name=agent_name,
                    instance_id=instance_id,
                    patch_file=patch_file,
                    repo=repo_hint,
                )
                if disallowed_result is not None:
                    results.append(disallowed_result)
                    logging.warning(
                        "Patch for %s modifies disallowed files; counting as disallowed patch.",
                        instance_id,
                    )
                    continue
                missing_eval_log_instances.append(instance_id)
                logging.warning(f"Missing evaluation logs for {instance_id}")
                skipped_instances += 1
                continue
            evaluable_items.append((metadata_path, patch_file, log_dir, instance_id))

        if not evaluable_items and not results:
            logging.warning("No evaluation logs found to evaluate")
            summary = self._build_evaluation_summary(
                agent_name=agent_name,
                results=[],
                skipped_instances=skipped_instances,
                missing_patch_after_attempt_instances=len(missing_patch_after_attempt_instances),
                missing_patch_not_started_instances=len(missing_patch_not_started_instances),
            )
            return summary, missing_patch_instances, missing_eval_log_instances

        if evaluable_items:
            progression_mode = getattr(self, "progression_mode", "basic")
            stop_on_stage_failure = progression_mode == "lifecycle"
            chain_instance_ids = [instance_id for _m, _p, _l, instance_id in evaluable_items]
            use_chain_progression = progression_mode in {"basic", "lifecycle"} and (
                self._supports_release_chain_progression(chain_instance_ids)
            )
            if use_chain_progression:
                grouped_items: dict[str, list[tuple[int, Path, Path, Path, str]]] = {}
                for metadata_path, patch_file, log_dir, instance_id in evaluable_items:
                    position = self._instance_chain_position(instance_id)
                    if position is None:
                        continue
                    segment_id, stage_index = position
                    grouped_items.setdefault(segment_id, []).append(
                        (stage_index, metadata_path, patch_file, log_dir, instance_id)
                    )
                completed = 0
                with tqdm(
                    total=len(evaluable_items),
                    desc=f"Evaluate test results for {agent_name}",
                    unit="instance",
                ) as progress_bar:
                    for segment_id in sorted(grouped_items):
                        ordered_items = sorted(grouped_items[segment_id], key=lambda item: item[0])
                        segment_stop_reason: str | None = None
                        for (
                            _stage_index,
                            metadata_path,
                            patch_file,
                            log_dir,
                            instance_id,
                        ) in ordered_items:
                            if segment_stop_reason is not None:
                                result = self._blocked_chain_result(
                                    agent_name=agent_name,
                                    instance_id=instance_id,
                                    reason=segment_stop_reason,
                                )
                            else:
                                result = self.evaluate_single_test_results_from_logs(  # type: ignore[attr-defined]
                                    agent_name=agent_name,
                                    metadata_path=metadata_path,
                                    patch_file=patch_file,
                                    log_dir=log_dir,
                                )
                                if not result.resolved and stop_on_stage_failure:
                                    segment_stop_reason = f"stage_not_resolved:{instance_id}"
                            results.append(result)
                            completed += 1
                            self._emit_evaluation_progress(
                                progress_bar=progress_bar,
                                results=results,
                                completed=completed,
                                total=len(evaluable_items),
                                last_label=instance_id,
                                status=self._result_status_label(result),
                            )
                            progress_bar.update(1)
            else:
                logging.info(
                    f"Evaluating test results for {agent_name} on {len(evaluable_items)} instance(s) "
                    f"with {self.workers} worker(s)"  # type: ignore[attr-defined]
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:  # type: ignore[attr-defined]
                    future_to_item = {
                        executor.submit(
                            self.evaluate_single_test_results_from_logs,  # type: ignore[attr-defined]
                            agent_name=agent_name,
                            metadata_path=metadata_path,
                            patch_file=patch_file,
                            log_dir=log_dir,
                        ): (metadata_path, patch_file, log_dir)
                        for metadata_path, patch_file, log_dir, _instance_id in evaluable_items
                    }
                    completed = 0
                    with tqdm(
                        total=len(evaluable_items),
                        desc=f"Evaluate test results for {agent_name}",
                        unit="instance",
                    ) as progress_bar:
                        for future in concurrent.futures.as_completed(future_to_item):
                            metadata_path, patch_file, log_dir = future_to_item[future]
                            completed += 1
                            try:
                                result = future.result()
                                results.append(result)
                                resolved = sum(1 for item in results if item.resolved)
                                status = "RESOLVED" if result.resolved else "FAILED"
                                logging.info(
                                    f"[{completed}/{len(evaluable_items)}] {metadata_path.name} "
                                    f"({patch_file.name}; {log_dir.name}): {status} "
                                    f"(Progress: {resolved}/{completed} resolved, "
                                    f"{100 * resolved / completed:.1f}%)"
                                )
                                self._emit_evaluation_progress(
                                    progress_bar=progress_bar,
                                    results=results,
                                    completed=completed,
                                    total=len(evaluable_items),
                                    last_label=metadata_path.stem,
                                    status=self._result_status_label(result),
                                )
                            except Exception as error:
                                logging.exception(
                                    f"Unexpected error evaluating test results for {metadata_path}: {error}",
                                )
                                results.append(
                                    self._error_evaluation_result(  # type: ignore[attr-defined]
                                        instance_id=str(metadata_path),
                                        agent_name=agent_name,
                                        error=f"Unexpected error: {error}",
                                    )
                                )
                                self._emit_evaluation_progress(
                                    progress_bar=progress_bar,
                                    results=results,
                                    completed=completed,
                                    total=len(evaluable_items),
                                    last_label=metadata_path.stem,
                                    status="ERROR",
                                )
                            finally:
                                progress_bar.update(1)

        synthesized_blocked_labels: set[str] = set()
        if getattr(self, "progression_mode", "basic") == "lifecycle":
            results, synthesized_blocked_labels = self._with_lifecycle_blocked_report_results(
                agent_name=agent_name,
                metadata_paths=metadata_paths,
                results=results,
            )
            skipped_instances = max(0, skipped_instances - len(synthesized_blocked_labels))

        missing_patch_after_attempt_count = len(
            set(missing_patch_after_attempt_instances) - synthesized_blocked_labels,
        )
        missing_patch_not_started_count = len(
            set(missing_patch_not_started_instances) - synthesized_blocked_labels,
        )

        summary = self._build_evaluation_summary(
            agent_name=agent_name,
            results=results,
            skipped_instances=skipped_instances,
            missing_patch_after_attempt_instances=missing_patch_after_attempt_count,
            missing_patch_not_started_instances=missing_patch_not_started_count,
        )
        self._save_results(summary)  # type: ignore[attr-defined]
        reported_labels = {result.instance_id for result in results}
        unreported_missing_patch_instances = [
            instance_id
            for instance_id in missing_patch_instances
            if instance_id not in reported_labels
        ]
        return summary, unreported_missing_patch_instances, missing_eval_log_instances

    def _evaluate_pairs_in_release_chain(
        self,
        *,
        agent_name: str,
        pairs: list[tuple[Path, Path, str]],
        progression_mode: str,
    ) -> list[EvaluationResult]:
        stop_on_stage_failure = progression_mode == "lifecycle"
        grouped_pairs: dict[str, list[tuple[int, Path, Path, str]]] = {}
        for metadata_path, patch_file, instance_id in pairs:
            position = self._instance_chain_position(instance_id)
            if position is None:
                continue
            segment_id, stage_index = position
            grouped_pairs.setdefault(segment_id, []).append(
                (stage_index, metadata_path, patch_file, instance_id)
            )

        results: list[EvaluationResult] = []
        completed = 0
        with tqdm(
            total=len(pairs),
            desc=f"Evaluating existing {agent_name} patches",
            unit="instance",
        ) as progress_bar:
            for segment_id in sorted(grouped_pairs):
                ordered_items = sorted(grouped_pairs[segment_id], key=lambda item: item[0])
                segment_stop_reason: str | None = None
                previous_stage_index: int | None = None
                segment_base_git_ref = self._segment_base_git_ref(segment_id)
                if not isinstance(segment_base_git_ref, str) or not segment_base_git_ref:
                    segment_stop_reason = f"missing_segment_base_git_ref:{segment_id}"
                prior_setup_patches: list[str] = []
                prior_code_patches: list[str] = []
                cumulative_visible_test_patch_paths: list[str] = []
                cumulative_hidden_test_patch_paths: list[str] = []
                include_hidden_tests = bool(getattr(self, "include_hidden_tests", True))

                for stage_index, metadata_path, patch_file, instance_id in ordered_items:
                    if segment_stop_reason is not None:
                        result = self._blocked_chain_result(
                            agent_name=agent_name,
                            instance_id=instance_id,
                            reason=segment_stop_reason,
                        )
                    elif previous_stage_index is not None and stage_index <= previous_stage_index:
                        segment_stop_reason = (
                            f"invalid_chain_stage_index_order:{previous_stage_index}:{stage_index}"
                        )
                        result = self._error_evaluation_result(  # type: ignore[attr-defined]
                            instance_id=instance_id,
                            agent_name=agent_name,
                            error=segment_stop_reason,
                        )
                    else:
                        bridge_error = self._append_required_bridge_patches(
                            segment_id=segment_id,
                            instance_id=instance_id,
                            stage_index=stage_index,
                            previous_stage_index=previous_stage_index,
                            destination_setup_patches=prior_setup_patches,
                            destination_code_patches=prior_code_patches,
                        )
                        if bridge_error is not None:
                            segment_stop_reason = bridge_error
                            result = self._error_evaluation_result(  # type: ignore[attr-defined]
                                instance_id=instance_id,
                                agent_name=agent_name,
                                error=segment_stop_reason,
                            )
                        else:
                            bridge_test_error = self._append_required_bridge_test_patches(
                                segment_id=segment_id,
                                instance_id=instance_id,
                                stage_index=stage_index,
                                previous_stage_index=previous_stage_index,
                                destination_visible_test_patches=cumulative_visible_test_patch_paths,
                                destination_hidden_test_patches=(
                                    cumulative_hidden_test_patch_paths
                                    if include_hidden_tests
                                    else None
                                ),
                            )
                            if bridge_test_error is not None:
                                segment_stop_reason = bridge_test_error
                                result = self._error_evaluation_result(  # type: ignore[attr-defined]
                                    instance_id=instance_id,
                                    agent_name=agent_name,
                                    error=segment_stop_reason,
                                )
                            else:
                                current_test_patch_stack = (
                                    self._resolve_current_test_patch_stack_for_instance(
                                        instance_id,
                                        include_hidden=include_hidden_tests,
                                    )
                                )
                                if current_test_patch_stack is None:
                                    segment_stop_reason = f"missing_chain_test_patch:{instance_id}"
                                    result = self._error_evaluation_result(  # type: ignore[attr-defined]
                                        instance_id=instance_id,
                                        agent_name=agent_name,
                                        error=segment_stop_reason,
                                    )
                                else:
                                    current_test_setup_patches, current_test_patch = (
                                        current_test_patch_stack
                                    )
                                    stage_image_override = self._instance_image_tag(instance_id)
                                    stage_command_instance_id = self._instance_command_instance_id(
                                        instance_id
                                    )
                                    stage_setup_patches = list(prior_setup_patches)
                                    stage_setup_patches.extend(cumulative_visible_test_patch_paths)
                                    stage_setup_patches.extend(cumulative_hidden_test_patch_paths)
                                    stage_setup_patches.extend(prior_code_patches)
                                    stage_setup_patches.extend(current_test_setup_patches)

                                    result = self.evaluate_single_from_patch(  # type: ignore[attr-defined]
                                        agent_name=agent_name,
                                        metadata_path=metadata_path,
                                        patch_file=patch_file,
                                        setup_patch_paths=stage_setup_patches,
                                        test_patch_path_override=current_test_patch,
                                        image_tag_override=stage_image_override,
                                        base_git_ref_override=segment_base_git_ref,
                                        command_instance_id_override=stage_command_instance_id,
                                    )
                                    if not result.resolved and stop_on_stage_failure:
                                        segment_stop_reason = f"stage_not_resolved:{instance_id}"

                                    self._append_test_patch_stack_to_cumulative(
                                        instance_id=instance_id,
                                        setup_patches=current_test_setup_patches,
                                        current_patch=current_test_patch,
                                        destination_visible_test_patches=(
                                            cumulative_visible_test_patch_paths
                                        ),
                                        destination_hidden_test_patches=(
                                            cumulative_hidden_test_patch_paths
                                            if include_hidden_tests
                                            else None
                                        ),
                                    )
                                    if progression_mode == "lifecycle":
                                        if result.resolved:
                                            self._append_current_env_patch(
                                                instance_id=instance_id,
                                                destination_setup_patches=prior_setup_patches,
                                            )
                                            prior_code_patches.append(str(patch_file))
                                            previous_stage_index = stage_index
                                    else:
                                        golden_patch = self._instance_patch_path(
                                            instance_id, "golden_patch"
                                        )
                                        if golden_patch is None or not Path(golden_patch).is_file():
                                            segment_stop_reason = (
                                                f"missing_chain_golden_patch:{instance_id}"
                                            )
                                        else:
                                            self._append_current_env_patch(
                                                instance_id=instance_id,
                                                destination_setup_patches=prior_setup_patches,
                                            )
                                            prior_code_patches.append(golden_patch)
                                            previous_stage_index = stage_index

                    results.append(result)
                    completed += 1
                    self._emit_evaluation_progress(
                        progress_bar=progress_bar,
                        results=results,
                        completed=completed,
                        total=len(pairs),
                        last_label=instance_id,
                        status=self._result_status_label(result),
                    )
                    progress_bar.update(1)

        return results

    def evaluate_batch_from_patches(
        self,
        *,
        agent_name: str,
        metadata_paths: list[Path],
        patches_dir: Path,
        skip_missing_patches: bool = True,
    ) -> tuple[EvaluationSummary, list[str]]:
        """Evaluate pre-generated patches matched by instance ID."""
        (
            pairs,
            missing_patch_instances,
            results,
            skipped_instances,
            missing_patch_after_attempt_instances,
            missing_patch_not_started_instances,
        ) = self._resolve_metadata_patch_pairs(
            agent_name=agent_name,
            metadata_paths=metadata_paths,
            patches_dir=patches_dir,
            skip_missing_patches=skip_missing_patches,
        )

        if not pairs and not results:
            logging.warning("No instances found with both metadata and patch files to evaluate")
            summary = self._build_evaluation_summary(
                agent_name=agent_name,
                results=[],
                skipped_instances=skipped_instances,
                missing_patch_after_attempt_instances=len(missing_patch_after_attempt_instances),
                missing_patch_not_started_instances=len(missing_patch_not_started_instances),
            )
            return summary, missing_patch_instances

        if pairs:
            progression_mode = getattr(self, "progression_mode", "basic")
            chain_instance_ids = [instance_id for _metadata_path, _patch_file, instance_id in pairs]
            use_chain_progression = progression_mode in {"basic", "lifecycle"} and (
                self._supports_release_chain_progression(chain_instance_ids)
            )
            if use_chain_progression:
                logging.info(
                    "Evaluating existing patches for %s in release-chain progression mode '%s' "
                    "on %d instance(s).",
                    agent_name,
                    progression_mode,
                    len(pairs),
                )
                ensure_docker_available()
                results.extend(
                    self._evaluate_pairs_in_release_chain(
                        agent_name=agent_name,
                        pairs=pairs,
                        progression_mode=progression_mode,
                    )
                )
            else:
                logging.info(
                    f"Evaluating existing patches for {agent_name} on {len(pairs)} instance(s) "
                    f"with {self.workers} worker(s)"  # type: ignore[attr-defined]
                )
                ensure_docker_available()

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:  # type: ignore[attr-defined]
                    future_to_pair = {
                        executor.submit(
                            self.evaluate_single_from_patch,  # type: ignore[attr-defined]
                            agent_name=agent_name,
                            metadata_path=metadata_path,
                            patch_file=patch_file,
                        ): (metadata_path, patch_file)
                        for metadata_path, patch_file, _instance_id in pairs
                    }

                    completed = 0
                    with tqdm(
                        total=len(pairs),
                        desc=f"Evaluating existing {agent_name} patches",
                        unit="instance",
                    ) as progress_bar:
                        for future in concurrent.futures.as_completed(future_to_pair):
                            metadata_path, patch_file = future_to_pair[future]
                            completed += 1
                            try:
                                result = future.result()
                                results.append(result)
                                resolved = sum(1 for item in results if item.resolved)
                                status = "RESOLVED" if result.resolved else "FAILED"
                                logging.info(
                                    f"[{completed}/{len(pairs)}] {metadata_path.name} "
                                    f"({patch_file.name}): {status} "
                                    f"(Progress: {resolved}/{completed} resolved, "
                                    f"{100 * resolved / completed:.1f}%)"
                                )
                                self._emit_evaluation_progress(
                                    progress_bar=progress_bar,
                                    results=results,
                                    completed=completed,
                                    total=len(pairs),
                                    last_label=metadata_path.stem,
                                    status=self._result_status_label(result),
                                )
                            except Exception as error:
                                logging.exception(
                                    f"Unexpected error evaluating patch for {metadata_path}: {error}",
                                )
                                results.append(
                                    self._error_evaluation_result(  # type: ignore[attr-defined]
                                        instance_id=str(metadata_path),
                                        agent_name=agent_name,
                                        error=f"Unexpected error: {error}",
                                    )
                                )
                                self._emit_evaluation_progress(
                                    progress_bar=progress_bar,
                                    results=results,
                                    completed=completed,
                                    total=len(pairs),
                                    last_label=metadata_path.stem,
                                    status="ERROR",
                                )
                            finally:
                                progress_bar.update(1)

        summary = self._build_evaluation_summary(
            agent_name=agent_name,
            results=results,
            skipped_instances=skipped_instances,
            missing_patch_after_attempt_instances=len(missing_patch_after_attempt_instances),
            missing_patch_not_started_instances=len(missing_patch_not_started_instances),
        )
        self._save_results(summary)  # type: ignore[attr-defined]
        return summary, missing_patch_instances

    def _generate_and_evaluate_chain_segment(
        self,
        *,
        agent: Agent,
        segment_id: str,
        segment_items: list[tuple[int, Path, str]],
        existing_logs_for_instance_id: dict[str, Path],
        skip_existing_patch: bool,
        progression_mode: str,
        stop_on_stage_failure: bool,
    ) -> dict[str, Any]:
        """Generate and evaluate one chain sequentially."""
        results: list[EvaluationResult] = []
        generated_count = 0
        skipped_existing_count = 0
        generation_error_count = 0
        fatal_generation_error = False
        ordered = sorted(segment_items, key=lambda item: item[0])
        segment_stop_reason: str | None = None
        previous_stage_index: int | None = None
        segment_base_git_ref = self._segment_base_git_ref(segment_id)
        if not isinstance(segment_base_git_ref, str) or not segment_base_git_ref:
            segment_stop_reason = f"missing_segment_base_git_ref:{segment_id}"
        prior_setup_patches: list[str] = []
        prior_code_patches: list[str] = []
        cumulative_visible_test_patch_paths: list[str] = []
        cumulative_hidden_test_patch_paths: list[str] = []
        include_hidden_tests = bool(getattr(self, "include_hidden_tests", True))

        for stage_index, metadata_path, instance_id in ordered:
            if segment_stop_reason is not None:
                results.append(
                    self._blocked_chain_result(
                        agent_name=agent.name,
                        instance_id=instance_id,
                        reason=segment_stop_reason,
                    )
                )
                continue
            if previous_stage_index is not None and stage_index <= previous_stage_index:
                segment_stop_reason = (
                    f"invalid_chain_stage_index_order:{previous_stage_index}:{stage_index}"
                )
                generation_error_count += 1
                results.append(
                    self._error_evaluation_result(  # type: ignore[attr-defined]
                        instance_id=instance_id,
                        agent_name=agent.name,
                        error=segment_stop_reason,
                    )
                )
                continue
            bridge_error = self._append_required_bridge_patches(
                segment_id=segment_id,
                instance_id=instance_id,
                stage_index=stage_index,
                previous_stage_index=previous_stage_index,
                destination_setup_patches=prior_setup_patches,
                destination_code_patches=prior_code_patches,
            )
            if bridge_error is not None:
                segment_stop_reason = bridge_error
                generation_error_count += 1
                results.append(
                    self._error_evaluation_result(  # type: ignore[attr-defined]
                        instance_id=instance_id,
                        agent_name=agent.name,
                        error=segment_stop_reason,
                    )
                )
                continue
            bridge_test_error = self._append_required_bridge_test_patches(
                segment_id=segment_id,
                instance_id=instance_id,
                stage_index=stage_index,
                previous_stage_index=previous_stage_index,
                destination_visible_test_patches=cumulative_visible_test_patch_paths,
                destination_hidden_test_patches=(
                    cumulative_hidden_test_patch_paths if include_hidden_tests else None
                ),
            )
            if bridge_test_error is not None:
                segment_stop_reason = bridge_test_error
                generation_error_count += 1
                results.append(
                    self._error_evaluation_result(  # type: ignore[attr-defined]
                        instance_id=instance_id,
                        agent_name=agent.name,
                        error=segment_stop_reason,
                    )
                )
                continue

            visible_test_patch_stack = self._resolve_current_test_patch_stack_for_instance(
                instance_id,
                include_hidden=False,
            )
            current_test_patch_stack = self._resolve_current_test_patch_stack_for_instance(
                instance_id,
                include_hidden=include_hidden_tests,
            )
            if visible_test_patch_stack is None or current_test_patch_stack is None:
                segment_stop_reason = f"missing_chain_test_patch:{instance_id}"
                generation_error_count += 1
                results.append(
                    self._error_evaluation_result(  # type: ignore[attr-defined]
                        instance_id=instance_id,
                        agent_name=agent.name,
                        error=segment_stop_reason,
                    )
                )
                continue
            visible_test_setup_patches, visible_test_patch = visible_test_patch_stack
            current_test_setup_patches, current_test_patch = current_test_patch_stack
            stage_image_override = self._instance_image_tag(instance_id)
            stage_command_instance_id = self._instance_command_instance_id(instance_id)
            generation_setup_patches = list(prior_setup_patches)
            generation_setup_patches.extend(cumulative_visible_test_patch_paths)
            generation_setup_patches.extend(prior_code_patches)
            generation_setup_patches.extend(visible_test_setup_patches)

            generation_result = self.generate_patch_single(  # type: ignore[attr-defined]
                agent,
                metadata_path,
                skip_existing_patch=skip_existing_patch,
                setup_patch_paths=generation_setup_patches,
                test_patch_path_override=visible_test_patch,
                image_tag_override=stage_image_override,
                base_git_ref_override=segment_base_git_ref,
                command_instance_id_override=stage_command_instance_id,
            )
            if generation_result.error:
                generation_error_count += 1
                results.append(generation_result)
                if _is_fatal_generation_error(generation_result.error):
                    logging.error(
                        "Stopping chain segment after fatal agent failure on %s: %s",
                        instance_id,
                        generation_result.error,
                    )
                    fatal_generation_error = True
                    break
                if stop_on_stage_failure:
                    segment_stop_reason = f"stage_generation_failed:{instance_id}"
                    continue
                # Basic mode continues chain progression using golden baselines.
                self._append_test_patch_stack_to_cumulative(
                    instance_id=instance_id,
                    setup_patches=current_test_setup_patches,
                    current_patch=current_test_patch,
                    destination_visible_test_patches=cumulative_visible_test_patch_paths,
                    destination_hidden_test_patches=(
                        cumulative_hidden_test_patch_paths if include_hidden_tests else None
                    ),
                )
                golden_patch = self._instance_patch_path(instance_id, "golden_patch")
                if golden_patch is None or not Path(golden_patch).is_file():
                    segment_stop_reason = f"missing_chain_golden_patch:{instance_id}"
                    continue
                self._append_current_env_patch(
                    instance_id=instance_id,
                    destination_setup_patches=prior_setup_patches,
                )
                prior_code_patches.append(golden_patch)
                previous_stage_index = stage_index
                continue
            if generation_result.detailed_results.get("skipped_existing_patch", False):
                skipped_existing_count += 1
            else:
                generated_count += 1

            patch_file = self._agent_patch_path(  # type: ignore[attr-defined]
                agent_name=agent.name,
                instance_id=instance_id,
            )
            existing_log_dir_for_stage = existing_logs_for_instance_id.get(instance_id)
            if existing_log_dir_for_stage is not None:
                evaluation_result = self.evaluate_single_test_results_from_logs(  # type: ignore[attr-defined]
                    agent_name=agent.name,
                    metadata_path=metadata_path,
                    patch_file=patch_file,
                    log_dir=existing_log_dir_for_stage,
                )
            else:
                evaluation_result = self.evaluate_single_from_patch(  # type: ignore[attr-defined]
                    agent_name=agent.name,
                    metadata_path=metadata_path,
                    patch_file=patch_file,
                    setup_patch_paths=[
                        *prior_setup_patches,
                        *cumulative_visible_test_patch_paths,
                        *cumulative_hidden_test_patch_paths,
                        *prior_code_patches,
                        *current_test_setup_patches,
                    ],
                    test_patch_path_override=current_test_patch,
                    image_tag_override=stage_image_override,
                    base_git_ref_override=segment_base_git_ref,
                    command_instance_id_override=stage_command_instance_id,
                )
            results.append(evaluation_result)
            if not evaluation_result.resolved and stop_on_stage_failure:
                segment_stop_reason = f"stage_not_resolved:{instance_id}"
                continue

            self._append_test_patch_stack_to_cumulative(
                instance_id=instance_id,
                setup_patches=current_test_setup_patches,
                current_patch=current_test_patch,
                destination_visible_test_patches=cumulative_visible_test_patch_paths,
                destination_hidden_test_patches=(
                    cumulative_hidden_test_patch_paths if include_hidden_tests else None
                ),
            )
            if progression_mode == "lifecycle":
                if evaluation_result.resolved:
                    self._append_current_env_patch(
                        instance_id=instance_id,
                        destination_setup_patches=prior_setup_patches,
                    )
                    prior_code_patches.append(str(patch_file))
                    previous_stage_index = stage_index
            else:
                golden_patch = self._instance_patch_path(instance_id, "golden_patch")
                if golden_patch is None or not Path(golden_patch).is_file():
                    segment_stop_reason = f"missing_chain_golden_patch:{instance_id}"
                    continue
                self._append_current_env_patch(
                    instance_id=instance_id,
                    destination_setup_patches=prior_setup_patches,
                )
                prior_code_patches.append(golden_patch)
                previous_stage_index = stage_index

        return {
            "segment_id": segment_id,
            "results": results,
            "generated": generated_count,
            "skipped_existing": skipped_existing_count,
            "generation_errors": generation_error_count,
            "fatal_generation_error": fatal_generation_error,
        }

    def generate_and_evaluate_batch(
        self,
        agent: Agent,
        metadata_paths: list[Path],
        *,
        skip_existing_patch: bool = False,
        skip_instances_with_existing_eval_logs: bool = False,
    ) -> tuple[EvaluationSummary, dict[str, int]]:
        """Generate patches and evaluate each instance when its patch is available."""
        filtered_paths: list[Path] = []
        skipped_instances = 0
        for metadata_path in metadata_paths:
            try:
                data = load_metadata(metadata_path)
                instance_id = data["instance_id"]
                if instance_id in self.instances:  # type: ignore[attr-defined]
                    filtered_paths.append(metadata_path)
                else:
                    logging.info(f"Skipping {metadata_path.name}: not in evaluation manifest")
                    skipped_instances += 1
            except Exception as error:
                logging.warning(f"Could not load metadata from {metadata_path}: {error}")
                skipped_instances += 1

        if not filtered_paths:
            logging.warning("No instances found in evaluation manifest to process")
            summary = self._build_evaluation_summary(
                agent_name=agent.name,
                results=[],
                skipped_instances=skipped_instances,
            )
            return summary, {"generated": 0, "skipped_existing": 0, "generation_errors": 0}

        progression_mode = getattr(self, "progression_mode", "basic")
        chain_instance_ids: list[str] = []
        if progression_mode in {"basic", "lifecycle"}:
            for path in filtered_paths:
                try:
                    chain_instance_ids.append(load_metadata(path)["instance_id"])
                except Exception:
                    continue
        supports_chain_progression = progression_mode in {"basic", "lifecycle"} and (
            self._supports_release_chain_progression(chain_instance_ids)
        )
        stop_on_stage_failure = progression_mode == "lifecycle"
        if supports_chain_progression:
            logging.info(
                "Generating and evaluating %s in release-chain progression mode '%s' "
                "on %d instance(s) with %d chain worker(s).",
                agent.name,
                progression_mode,
                len(filtered_paths),
                self.workers,  # type: ignore[attr-defined]
            )
            ensure_docker_available()
            grouped_paths: dict[str, list[tuple[int, Path, str]]] = {}
            for metadata_path in filtered_paths:
                try:
                    data = load_metadata(metadata_path)
                except Exception as error:
                    logging.warning("Could not load metadata from %s: %s", metadata_path, error)
                    skipped_instances += 1
                    continue
                instance_id = data["instance_id"]
                position = self._instance_chain_position(instance_id)
                if position is None:
                    skipped_instances += 1
                    continue
                segment_id, stage_index = position
                grouped_paths.setdefault(segment_id, []).append(
                    (stage_index, metadata_path, instance_id)
                )

            results: list[EvaluationResult] = []
            generated_count = 0
            skipped_existing_count = 0
            generation_error_count = 0
            existing_logs_for_instance_id: dict[str, Path] = {}
            if skip_instances_with_existing_eval_logs:
                for ordered in grouped_paths.values():
                    for _stage_index, _metadata_path, instance_id in ordered:
                        existing_log_dir = self._latest_eval_log_dir(  # type: ignore[attr-defined]
                            agent_name=agent.name,
                            instance_id=instance_id,
                        )
                        if existing_log_dir is None:
                            continue
                        run_log_paths = self._run_log_paths_from_dir(existing_log_dir)  # type: ignore[attr-defined]
                        if len(run_log_paths) >= self.runs:  # type: ignore[attr-defined]
                            existing_logs_for_instance_id[instance_id] = existing_log_dir

            if self.workers > 1 and len(grouped_paths) > 1:  # type: ignore[attr-defined]
                segment_outcomes = self._run_chain_segment_jobs(
                    grouped_paths,
                    lambda segment_id, segment_items: (
                        self._generate_and_evaluate_chain_segment(
                            agent=agent,
                            segment_id=segment_id,
                            segment_items=segment_items,
                            existing_logs_for_instance_id=existing_logs_for_instance_id,
                            skip_existing_patch=skip_existing_patch,
                            progression_mode=progression_mode,
                            stop_on_stage_failure=stop_on_stage_failure,
                        )
                    ),
                )
                for outcome in segment_outcomes:
                    results.extend(outcome["results"])
                    generated_count += outcome["generated"]
                    skipped_existing_count += outcome["skipped_existing"]
                    generation_error_count += outcome["generation_errors"]

                summary = self._build_evaluation_summary(
                    agent_name=agent.name,
                    results=results,
                    skipped_instances=skipped_instances,
                )
                self._save_results(summary)  # type: ignore[attr-defined]
                generation_stats = {
                    "generated": generated_count,
                    "skipped_existing": skipped_existing_count,
                    "generation_errors": generation_error_count,
                }
                return summary, generation_stats

            for segment_id in sorted(grouped_paths):
                ordered = sorted(grouped_paths[segment_id], key=lambda item: item[0])
                segment_stop_reason: str | None = None
                previous_stage_index: int | None = None
                segment_base_git_ref = self._segment_base_git_ref(segment_id)
                if not isinstance(segment_base_git_ref, str) or not segment_base_git_ref:
                    segment_stop_reason = f"missing_segment_base_git_ref:{segment_id}"
                prior_setup_patches: list[str] = []
                prior_code_patches: list[str] = []
                cumulative_visible_test_patch_paths: list[str] = []
                cumulative_hidden_test_patch_paths: list[str] = []
                include_hidden_tests = bool(getattr(self, "include_hidden_tests", True))

                for stage_index, metadata_path, instance_id in ordered:
                    if segment_stop_reason is not None:
                        results.append(
                            self._blocked_chain_result(
                                agent_name=agent.name,
                                instance_id=instance_id,
                                reason=segment_stop_reason,
                            )
                        )
                        continue
                    if previous_stage_index is not None and stage_index <= previous_stage_index:
                        segment_stop_reason = (
                            f"invalid_chain_stage_index_order:{previous_stage_index}:{stage_index}"
                        )
                        generation_error_count += 1
                        results.append(
                            self._error_evaluation_result(  # type: ignore[attr-defined]
                                instance_id=instance_id,
                                agent_name=agent.name,
                                error=segment_stop_reason,
                            )
                        )
                        continue
                    bridge_error = self._append_required_bridge_patches(
                        segment_id=segment_id,
                        instance_id=instance_id,
                        stage_index=stage_index,
                        previous_stage_index=previous_stage_index,
                        destination_setup_patches=prior_setup_patches,
                        destination_code_patches=prior_code_patches,
                    )
                    if bridge_error is not None:
                        segment_stop_reason = bridge_error
                        generation_error_count += 1
                        results.append(
                            self._error_evaluation_result(  # type: ignore[attr-defined]
                                instance_id=instance_id,
                                agent_name=agent.name,
                                error=segment_stop_reason,
                            )
                        )
                        continue
                    bridge_test_error = self._append_required_bridge_test_patches(
                        segment_id=segment_id,
                        instance_id=instance_id,
                        stage_index=stage_index,
                        previous_stage_index=previous_stage_index,
                        destination_visible_test_patches=cumulative_visible_test_patch_paths,
                        destination_hidden_test_patches=(
                            cumulative_hidden_test_patch_paths if include_hidden_tests else None
                        ),
                    )
                    if bridge_test_error is not None:
                        segment_stop_reason = bridge_test_error
                        generation_error_count += 1
                        results.append(
                            self._error_evaluation_result(  # type: ignore[attr-defined]
                                instance_id=instance_id,
                                agent_name=agent.name,
                                error=segment_stop_reason,
                            )
                        )
                        continue

                    visible_test_patch_stack = self._resolve_current_test_patch_stack_for_instance(
                        instance_id,
                        include_hidden=False,
                    )
                    current_test_patch_stack = self._resolve_current_test_patch_stack_for_instance(
                        instance_id,
                        include_hidden=include_hidden_tests,
                    )
                    if visible_test_patch_stack is None or current_test_patch_stack is None:
                        segment_stop_reason = f"missing_chain_test_patch:{instance_id}"
                        generation_error_count += 1
                        results.append(
                            self._error_evaluation_result(  # type: ignore[attr-defined]
                                instance_id=instance_id,
                                agent_name=agent.name,
                                error=segment_stop_reason,
                            )
                        )
                        continue
                    visible_test_setup_patches, visible_test_patch = visible_test_patch_stack
                    current_test_setup_patches, current_test_patch = current_test_patch_stack
                    stage_image_override = self._instance_image_tag(instance_id)
                    stage_command_instance_id = self._instance_command_instance_id(instance_id)
                    generation_setup_patches = list(prior_setup_patches)
                    generation_setup_patches.extend(cumulative_visible_test_patch_paths)
                    generation_setup_patches.extend(prior_code_patches)
                    generation_setup_patches.extend(visible_test_setup_patches)

                    generation_result = self.generate_patch_single(  # type: ignore[attr-defined]
                        agent,
                        metadata_path,
                        skip_existing_patch=skip_existing_patch,
                        setup_patch_paths=generation_setup_patches,
                        test_patch_path_override=visible_test_patch,
                        image_tag_override=stage_image_override,
                        base_git_ref_override=segment_base_git_ref,
                        command_instance_id_override=stage_command_instance_id,
                    )
                    if generation_result.error:
                        generation_error_count += 1
                        results.append(generation_result)
                        if _is_fatal_generation_error(generation_result.error):
                            logging.error(
                                "Stopping generate+evaluate after fatal agent failure on %s: %s",
                                instance_id,
                                generation_result.error,
                            )
                            summary = self._build_evaluation_summary(
                                agent_name=agent.name,
                                results=results,
                                skipped_instances=skipped_instances,
                            )
                            self._save_results(summary)  # type: ignore[attr-defined]
                            return summary, {
                                "generated": generated_count,
                                "skipped_existing": skipped_existing_count,
                                "generation_errors": generation_error_count,
                            }
                        if stop_on_stage_failure:
                            segment_stop_reason = f"stage_generation_failed:{instance_id}"
                            continue
                        # Basic mode continues chain progression using golden baselines.
                        self._append_test_patch_stack_to_cumulative(
                            instance_id=instance_id,
                            setup_patches=current_test_setup_patches,
                            current_patch=current_test_patch,
                            destination_visible_test_patches=cumulative_visible_test_patch_paths,
                            destination_hidden_test_patches=(
                                cumulative_hidden_test_patch_paths if include_hidden_tests else None
                            ),
                        )
                        golden_patch = self._instance_patch_path(instance_id, "golden_patch")
                        if golden_patch is None or not Path(golden_patch).is_file():
                            segment_stop_reason = f"missing_chain_golden_patch:{instance_id}"
                            continue
                        self._append_current_env_patch(
                            instance_id=instance_id,
                            destination_setup_patches=prior_setup_patches,
                        )
                        prior_code_patches.append(golden_patch)
                        previous_stage_index = stage_index
                        continue
                    if generation_result.detailed_results.get("skipped_existing_patch", False):
                        skipped_existing_count += 1
                    else:
                        generated_count += 1

                    patch_file = self._agent_patch_path(
                        agent_name=agent.name, instance_id=instance_id
                    )  # type: ignore[attr-defined]
                    existing_log_dir_for_stage = existing_logs_for_instance_id.get(instance_id)
                    if existing_log_dir_for_stage is not None:
                        evaluation_result = self.evaluate_single_test_results_from_logs(  # type: ignore[attr-defined]
                            agent_name=agent.name,
                            metadata_path=metadata_path,
                            patch_file=patch_file,
                            log_dir=existing_log_dir_for_stage,
                        )
                    else:
                        evaluation_result = self.evaluate_single_from_patch(  # type: ignore[attr-defined]
                            agent_name=agent.name,
                            metadata_path=metadata_path,
                            patch_file=patch_file,
                            setup_patch_paths=[
                                *prior_setup_patches,
                                *cumulative_visible_test_patch_paths,
                                *cumulative_hidden_test_patch_paths,
                                *prior_code_patches,
                                *current_test_setup_patches,
                            ],
                            test_patch_path_override=current_test_patch,
                            image_tag_override=stage_image_override,
                            base_git_ref_override=segment_base_git_ref,
                            command_instance_id_override=stage_command_instance_id,
                        )
                    results.append(evaluation_result)
                    if not evaluation_result.resolved and stop_on_stage_failure:
                        segment_stop_reason = f"stage_not_resolved:{instance_id}"
                        continue

                    self._append_test_patch_stack_to_cumulative(
                        instance_id=instance_id,
                        setup_patches=current_test_setup_patches,
                        current_patch=current_test_patch,
                        destination_visible_test_patches=cumulative_visible_test_patch_paths,
                        destination_hidden_test_patches=(
                            cumulative_hidden_test_patch_paths if include_hidden_tests else None
                        ),
                    )
                    if progression_mode == "lifecycle":
                        if evaluation_result.resolved:
                            self._append_current_env_patch(
                                instance_id=instance_id,
                                destination_setup_patches=prior_setup_patches,
                            )
                            prior_code_patches.append(str(patch_file))
                            previous_stage_index = stage_index
                    else:
                        golden_patch = self._instance_patch_path(instance_id, "golden_patch")
                        if golden_patch is None or not Path(golden_patch).is_file():
                            segment_stop_reason = f"missing_chain_golden_patch:{instance_id}"
                            continue
                        self._append_current_env_patch(
                            instance_id=instance_id,
                            destination_setup_patches=prior_setup_patches,
                        )
                        prior_code_patches.append(golden_patch)
                        previous_stage_index = stage_index

            summary = self._build_evaluation_summary(
                agent_name=agent.name,
                results=results,
                skipped_instances=skipped_instances,
            )
            self._save_results(summary)  # type: ignore[attr-defined]
            generation_stats = {
                "generated": generated_count,
                "skipped_existing": skipped_existing_count,
                "generation_errors": generation_error_count,
            }
            return summary, generation_stats

        logging.info(
            f"Generating and evaluating {agent.name} on {len(filtered_paths)} instance(s) "
            f"(filtered from {len(metadata_paths)} total) with {self.workers} worker(s)"  # type: ignore[attr-defined]
        )
        ensure_docker_available()

        results: list[EvaluationResult] = []
        generated_count = 0
        skipped_existing_count = 0
        generation_error_count = 0

        def _generate_then_evaluate(metadata_path: Path) -> dict[str, Any]:
            generation_result = self.generate_patch_single(  # type: ignore[attr-defined]
                agent,
                metadata_path,
                skip_existing_patch=skip_existing_patch,
            )
            if generation_result.error:
                return {
                    "result": generation_result,
                    "generated": False,
                    "skipped_existing": False,
                    "generation_error": True,
                }

            skipped_existing = bool(
                generation_result.detailed_results.get("skipped_existing_patch", False),
            )
            try:
                data = load_metadata(metadata_path)
                instance_id = data["instance_id"]
            except Exception as error:
                return {
                    "result": EvaluationResult(
                        instance_id=str(metadata_path),
                        agent_name=agent.name,
                        error=f"Failed to reload metadata for evaluation: {error}",
                    ),
                    "generated": not skipped_existing,
                    "skipped_existing": skipped_existing,
                    "generation_error": False,
                }

            patch_file = self._agent_patch_path(agent_name=agent.name, instance_id=instance_id)  # type: ignore[attr-defined]
            existing_log_dir = None
            if skip_instances_with_existing_eval_logs:
                candidate_log_dir = self._latest_eval_log_dir(  # type: ignore[attr-defined]
                    agent_name=agent.name,
                    instance_id=instance_id,
                )
                if candidate_log_dir is not None:
                    run_log_paths = self._run_log_paths_from_dir(candidate_log_dir)  # type: ignore[attr-defined]
                    if len(run_log_paths) >= self.runs:  # type: ignore[attr-defined]
                        existing_log_dir = candidate_log_dir
            if existing_log_dir is not None:
                evaluation_result = self.evaluate_single_test_results_from_logs(  # type: ignore[attr-defined]
                    agent_name=agent.name,
                    metadata_path=metadata_path,
                    patch_file=patch_file,
                    log_dir=existing_log_dir,
                )
            else:
                evaluation_result = self.evaluate_single_from_patch(  # type: ignore[attr-defined]
                    agent_name=agent.name,
                    metadata_path=metadata_path,
                    patch_file=patch_file,
                )
            return {
                "result": evaluation_result,
                "generated": not skipped_existing,
                "skipped_existing": skipped_existing,
                "generation_error": False,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:  # type: ignore[attr-defined]
            future_to_path = {
                executor.submit(_generate_then_evaluate, metadata_path): metadata_path
                for metadata_path in filtered_paths
            }

            completed = 0
            fatal_generation_error = False
            with tqdm(
                total=len(filtered_paths),
                desc=f"Generate+Evaluate {agent.name}",
                unit="instance",
            ) as progress_bar:
                for future in concurrent.futures.as_completed(future_to_path):
                    metadata_path = future_to_path[future]
                    completed += 1
                    fatal_generation_error = False
                    try:
                        payload = future.result()
                        result = payload["result"]
                        results.append(result)

                        if payload.get("generation_error"):
                            generation_error_count += 1
                        if payload.get("skipped_existing"):
                            skipped_existing_count += 1
                        if payload.get("generated"):
                            generated_count += 1

                        resolved = sum(1 for item in results if item.resolved)
                        status = "RESOLVED" if result.resolved else "FAILED"
                        logging.info(
                            f"[{completed}/{len(filtered_paths)}] {metadata_path.name}: {status} "
                            f"(Progress: {resolved}/{completed} resolved, "
                            f"{100 * resolved / completed:.1f}%)"
                        )
                        fatal_generation_error = _is_fatal_generation_error(result.error)
                        if fatal_generation_error:
                            logging.error(
                                "Stopping generate+evaluate after fatal agent failure on %s: %s",
                                metadata_path.stem,
                                result.error,
                            )
                            for pending_future in future_to_path:
                                if pending_future is not future:
                                    pending_future.cancel()
                        self._emit_evaluation_progress(
                            progress_bar=progress_bar,
                            results=results,
                            completed=completed,
                            total=len(filtered_paths),
                            last_label=metadata_path.stem,
                            status=self._result_status_label(result),
                        )
                    except Exception as error:
                        logging.exception(
                            f"Unexpected error in generate+evaluate for {metadata_path}: {error}",
                        )
                        generation_error_count += 1
                        results.append(
                            EvaluationResult(
                                instance_id=str(metadata_path),
                                agent_name=agent.name,
                                error=f"Unexpected error: {error}",
                            )
                        )
                        self._emit_evaluation_progress(
                            progress_bar=progress_bar,
                            results=results,
                            completed=completed,
                            total=len(filtered_paths),
                            last_label=metadata_path.stem,
                            status="ERROR",
                        )
                    finally:
                        progress_bar.update(1)
                    if fatal_generation_error:
                        break

        summary = self._build_evaluation_summary(
            agent_name=agent.name,
            results=results,
            skipped_instances=skipped_instances,
        )
        self._save_results(summary)  # type: ignore[attr-defined]
        generation_stats = {
            "generated": generated_count,
            "skipped_existing": skipped_existing_count,
            "generation_errors": generation_error_count,
        }
        return summary, generation_stats

    def evaluate_batch(
        self,
        agent: Agent,
        metadata_paths: list[Path],
    ) -> EvaluationSummary:
        """Evaluate an agent on multiple instances.

        Args:
            agent: The agent to evaluate.
            metadata_paths: List of paths to instance metadata JSON files.

        Returns:
            EvaluationSummary with aggregated results.
        """
        # Filter to only evaluate instances present in the evaluation manifest.
        filtered_paths: list[Path] = []
        skipped_instances = 0
        for metadata_path in metadata_paths:
            try:
                data = load_metadata(metadata_path)
                instance_id = data["instance_id"]

                if instance_id in self.instances:  # type: ignore[attr-defined]
                    filtered_paths.append(metadata_path)
                else:
                    logging.info(f"Skipping {metadata_path.name}: not in evaluation manifest")
                    skipped_instances += 1
            except Exception as e:
                logging.warning(f"Could not load metadata from {metadata_path}: {e}")
                skipped_instances += 1

        if not filtered_paths:
            logging.warning("No instances found in evaluation manifest to evaluate")
            return self._build_evaluation_summary(
                agent_name=agent.name,
                results=[],
                skipped_instances=skipped_instances,
            )

        logging.info(
            f"Evaluating {agent.name} on {len(filtered_paths)} instances "
            f"(filtered from {len(metadata_paths)} total) with {self.workers} worker(s)"  # type: ignore[attr-defined]
        )

        # Preflight docker once before worker threads are spawned.
        ensure_docker_available()

        results: list[EvaluationResult] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:  # type: ignore[attr-defined]
            future_to_path = {
                executor.submit(self.evaluate_single, agent, metadata_path): metadata_path  # type: ignore[attr-defined]
                for metadata_path in filtered_paths
            }

            completed = 0
            with tqdm(
                total=len(filtered_paths),
                desc=f"Evaluating {agent.name}",
                unit="instance",
            ) as progress_bar:
                for future in concurrent.futures.as_completed(future_to_path):
                    metadata_path = future_to_path[future]
                    completed += 1
                    fatal_generation_error = False

                    try:
                        result = future.result()
                        results.append(result)

                        # Log progress
                        resolved = sum(1 for r in results if r.resolved)
                        status = "RESOLVED" if result.resolved else "FAILED"
                        logging.info(
                            f"[{completed}/{len(filtered_paths)}] {metadata_path.name}: {status} "
                            f"(Progress: {resolved}/{completed} resolved, {100 * resolved / completed:.1f}%)"
                        )
                        fatal_generation_error = _is_fatal_generation_error(result.error)
                        if fatal_generation_error:
                            logging.error(
                                "Stopping evaluation after fatal agent failure on %s: %s",
                                metadata_path.stem,
                                result.error,
                            )
                            for pending_future in future_to_path:
                                if pending_future is not future:
                                    pending_future.cancel()
                        self._emit_evaluation_progress(
                            progress_bar=progress_bar,
                            results=results,
                            completed=completed,
                            total=len(filtered_paths),
                            last_label=metadata_path.stem,
                            status=self._result_status_label(result),
                        )
                    except Exception as e:
                        logging.exception(f"Unexpected error evaluating {metadata_path}: {e}")
                        results.append(
                            EvaluationResult(
                                instance_id=str(metadata_path),
                                agent_name=agent.name,
                                error=f"Unexpected error: {e}",
                            )
                        )
                        self._emit_evaluation_progress(
                            progress_bar=progress_bar,
                            results=results,
                            completed=completed,
                            total=len(filtered_paths),
                            last_label=metadata_path.stem,
                            status="ERROR",
                        )
                    finally:
                        progress_bar.update(1)
                    if fatal_generation_error:
                        break

        summary = self._build_evaluation_summary(
            agent_name=agent.name,
            results=results,
            skipped_instances=skipped_instances,
        )

        # Save results
        self._save_results(summary)  # type: ignore[attr-defined]

        return summary

    def _save_results(self, summary: EvaluationSummary) -> Path:
        """Save evaluation results to JSON file."""
        results_dir = self.output_dir / "results"  # type: ignore[attr-defined]
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{summary.agent_name}_{timestamp}.json"
        payload = summary.to_dict()
        mini_swe_config = self._read_run_mini_swe_config()
        if mini_swe_config is not None:
            payload["mini_swe"] = mini_swe_config

        results_file.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        logging.info(f"Results saved to {results_file}")

        # Also save markdown summary
        self._save_summary_markdown(summary, timestamp)

        return results_file

    def _save_summary_markdown(self, summary: EvaluationSummary, timestamp: str) -> Path:
        """Save evaluation summary to markdown file."""
        results_dir = self.output_dir / "results"  # type: ignore[attr-defined]
        results_dir.mkdir(parents=True, exist_ok=True)

        markdown_file = results_dir / f"{summary.agent_name}_{timestamp}.md"

        def _error_snippet(error: str) -> str:
            compact = " ".join(error.split()).replace("|", "\\|")
            return f"{compact[:240]}..." if len(compact) > 240 else compact

        def _failed_tests_text(count: int) -> str:
            return f"{count} test failed" if count == 1 else f"{count} tests failed"

        def _format_config_value(value: object) -> str:
            if value is None:
                return "_(not set)_"
            if isinstance(value, bool):
                return "`true`" if value else "`false`"
            if isinstance(value, (int, float)):
                return str(value)
            if isinstance(value, str):
                return f"`{value}`"
            return f"`{json.dumps(value, sort_keys=True)}`"

        lines: list[str] = []
        lines.append("# Evaluation Summary")
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        lines.append(f"- **Agent:** `{summary.agent_name}`")
        lines.append(f"- **Total instances:** {summary.total_instances}")
        lines.append(f"- **Skipped:** {summary.skipped_instances}")
        lines.append(
            "- **Missing patch (agent attempted but none emitted):** "
            f"{summary.missing_patch_after_attempt_instances}"
        )
        lines.append(
            "- **Missing patch (not started/interrupted):** "
            f"{summary.missing_patch_not_started_instances}"
        )
        lines.append(f"- **Not parsed:** {summary.not_parsed_instances}")
        lines.append(
            f"- **Resolved:** {summary.resolved_instances} ({100 * summary.resolution_rate:.1f}%)"
        )
        lines.append(
            "- **Infra-denied empty patch instances:** "
            f"{summary.infra_denied_empty_patch_instances}"
        )
        lines.append(f"- **Failed patch apply:** {summary.failed_patch_apply}")
        lines.append(f"- **Failed tests:** {summary.failed_tests}")
        lines.append(
            f"- **Disallowed test-patch instances:** {summary.disallowed_test_patch_instances}"
        )
        lines.append(f"- **Infra errors:** {summary.infra_error_instances}")
        lines.append(f"- **Errors:** {summary.errors}")
        lines.append("")
        if summary.chain_blocked_instances or summary.lifecycle_first_failures:
            lines.append("## Lifecycle Diagnostics")
            lines.append("")
            lines.append(
                "- **Attempted-stage resolved:** "
                f"{summary.attempted_resolved_instances}/{summary.attempted_instances} "
                f"({100 * summary.attempted_resolution_rate:.1f}%)"
            )
            lines.append(
                f"- **Blocked by prior lifecycle failure:** {summary.chain_blocked_instances}"
            )
            if summary.lifecycle_first_failures:
                lines.append("- **First chain stops:**")
                for failure in summary.lifecycle_first_failures:
                    segment_id = failure.get("segment_id", "unknown")
                    category = failure.get("category", "unknown")
                    survived = failure.get("survived_stages", 0)
                    total = failure.get("total_reported_stages", 0)
                    blocked_after = failure.get("blocked_after", 0)
                    instance_id = failure.get("instance_id")
                    if category == "all_resolved":
                        lines.append(
                            f"  - `{segment_id}`: all {survived}/{total} reported stages resolved"
                        )
                        continue
                    reason = _error_snippet(str(failure.get("reason") or ""))
                    lines.append(
                        f"  - `{segment_id}`: stopped at `{instance_id}` "
                        f"after {survived}/{total} stages, category `{category}`, "
                        f"blocked later stages {blocked_after}, reason: {reason}"
                    )
            lines.append("")
        mini_swe_config = self._read_run_mini_swe_config()
        if mini_swe_config is not None:
            lines.append("## Configuration")
            lines.append("")
            preferred_fields = [
                ("agent_version", "Agent version"),
                ("model_name", "Model used"),
                ("reasoning_effort", "Reasoning effort"),
                ("step_limit", "Step limit"),
                ("command_timeout_seconds", "Command timeout"),
            ]
            seen_fields: set[str] = set()
            for key, label in preferred_fields:
                if key in mini_swe_config:
                    lines.append(f"- **{label}:** {_format_config_value(mini_swe_config[key])}")
                    seen_fields.add(key)
            for key in sorted(k for k in mini_swe_config.keys() if k not in seen_fields):
                label = key.replace("_", " ").capitalize()
                lines.append(f"- **{label}:** {_format_config_value(mini_swe_config[key])}")
            lines.append("")
        lines.append("## Per-Instance Results")
        lines.append("")

        if not summary.results:
            lines.append("_No per-instance results available._")
            lines.append("")
            markdown_file.write_text("\n".join(lines), encoding="utf-8")
            logging.info(f"Summary saved to {markdown_file}")
            return markdown_file

        for result in summary.results:
            if _is_parse_error(result.error):
                lines.append(
                    f"- ⚠️ **NOT PARSED** `{result.instance_id}`: could not parse test results"
                )
                continue
            if result.detailed_results.get("infra_denied_empty_patch", False):
                lines.append(
                    f"- 🚧 **INFRA_DENIED_EMPTY_PATCH** `{result.instance_id}`:"
                    " generation was blocked by environment restrictions "
                    "and produced no usable patch"
                )
                continue
            if _is_infra_error(result.error):
                lines.append(
                    f"- 🔧 **INFRA_ERROR** `{result.instance_id}`:"
                    f" {_error_snippet(result.error)}"
                )
                continue
            if _is_chain_blocked_error(result.error):
                lines.append(
                    f"- ⛓️ **CHAIN_BLOCKED** `{result.instance_id}`:"
                    f" {_error_snippet(_chain_blocked_reason(result.error))}"
                )
                continue
            status = "RESOLVED" if result.resolved else "FAILED"
            emoji = "✅" if result.resolved else "❌"
            detail = ""
            if result.error:
                detail = f" error: {_error_snippet(result.error)}"
            elif not result.patch_applied_successfully:
                detail = " patch apply failed"
            elif not result.all_tests_passed:
                detail = f" {_failed_tests_text(result.failed_test_count)}"

            lines.append(f"- {emoji} **{status}** `{result.instance_id}`:{detail}")

        lines.append("")

        markdown_file.write_text("\n".join(lines), encoding="utf-8")
        logging.info(f"Summary saved to {markdown_file}")
        return markdown_file
