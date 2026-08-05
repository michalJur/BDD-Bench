from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from bdd_bench.common.docker_client import (
    exec_streaming,
    stop_container,
)
from bdd_bench.common.container_runtime import apply_patch, reset_repository
from bdd_bench.common.test_results import (
    TestRunResult,
    load_run_result_from_log,
)
from bdd_bench.evaluation.container_security import create_secure_evaluation_container
from bdd_bench.evaluation.harness_metadata import sanitize_label
from bdd_bench.evaluation.harness_models import EvaluationContext, EvaluationResult

NON_INTERACTIVE_EXEC_ENV = {
    "CI": "1",
    "DEBIAN_FRONTEND": "noninteractive",
    "GIT_TERMINAL_PROMPT": "0",
    "PIP_NO_INPUT": "1",
    "PYTHONUNBUFFERED": "1",
}

EVALUATION_TEST_TIMEOUT_SECONDS = 60 * 60
EVALUATION_TEST_KILL_GRACE_SECONDS = 30


class _HarnessExecution:
    """Low-level test execution and Docker orchestration for EvaluationHarness."""

    def _effective_evaluation_test_patch_key(self, patch_key: str | None) -> str:
        normalized = patch_key.strip() if isinstance(patch_key, str) else ""
        if not normalized:
            normalized = "golden_test_patch"
        if normalized == "bronze_test_patch" and not getattr(self, "include_hidden_tests", True):
            return "silver_test_patch"
        return normalized

    def _instance_evaluation_test_patch_key(
        self,
        instance_id: str,
        *,
        metadata_evaluation_test_patch: str | None = None,
    ) -> str:
        if isinstance(metadata_evaluation_test_patch, str):
            normalized = metadata_evaluation_test_patch.strip()
            if normalized:
                return self._effective_evaluation_test_patch_key(normalized)
        instances = getattr(self, "instances", None)
        if isinstance(instances, dict):
            instance = instances.get(instance_id)
            if isinstance(instance, dict):
                patch_key = instance.get("evaluation_test_patch")
                if isinstance(patch_key, str) and patch_key.strip():
                    return self._effective_evaluation_test_patch_key(patch_key.strip())
        return self._effective_evaluation_test_patch_key("golden_test_patch")

    def _selected_test_patch_path(
        self,
        *,
        instance_id: str,
        patch_info: dict[str, str],
        test_patch_path_override: str | None = None,
        metadata_evaluation_test_patch: str | None = None,
    ) -> str | None:
        if isinstance(test_patch_path_override, str) and test_patch_path_override.strip():
            return test_patch_path_override.strip()
        patch_key = self._instance_evaluation_test_patch_key(
            instance_id,
            metadata_evaluation_test_patch=metadata_evaluation_test_patch,
        )
        selected = patch_info.get(patch_key)
        if isinstance(selected, str) and selected.strip():
            return selected.strip()
        fallback = patch_info.get("golden_test_patch")
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
        return None

    def _normalize_setup_patch_paths(
        self,
        *,
        setup_patch_paths: list[str] | None,
        patch_info: dict[str, str],
        evaluation_test_patch: str | None = None,
    ) -> list[str]:
        normalized = [
            path for path in (setup_patch_paths or []) if isinstance(path, str) and path.strip()
        ]
        # Match dataset-generation ordering: chain setup → golden_env → golden_test → silver.
        # The silver/bronze test patches were authored on top of this exact stack during
        # generation, so applying them in the same order at eval time keeps generation and
        # evaluation aligned.
        env_patch = patch_info.get("golden_env_patch")
        if isinstance(env_patch, str) and env_patch.strip() and env_patch not in normalized:
            normalized.append(env_patch)
        if evaluation_test_patch in {"silver_test_patch", "bronze_test_patch"}:
            golden_test_patch = patch_info.get("golden_test_patch")
            if (
                isinstance(golden_test_patch, str)
                and golden_test_patch.strip()
                and golden_test_patch not in normalized
            ):
                normalized.append(golden_test_patch)
        if evaluation_test_patch == "bronze_test_patch":
            silver_patch = patch_info.get("silver_test_patch")
            if (
                isinstance(silver_patch, str)
                and silver_patch.strip()
                and silver_patch not in normalized
            ):
                normalized.append(silver_patch)
        return normalized

    def _run_tests_for_evaluation(
        self,
        context: EvaluationContext,
        agent_name: str,
        patch_file: Path | None,
        patch_info: dict[str, str],
        *,
        setup_patch_paths: list[str] | None = None,
        test_patch_path_override: str | None = None,
        base_git_ref_override: str | None = None,
        command_instance_id_override: str | None = None,
        metadata_evaluation_test_patch: str | None = None,
        build_only: bool = False,
    ) -> dict[str, Any]:
        instance_id_safe = sanitize_label(context.instance_id)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.output_dir / "eval_logs" / instance_id_safe / run_timestamp  # type: ignore[attr-defined]

        # Load precomputed command metadata from the evaluation manifest.
        test_command = self._build_test_command(  # type: ignore[attr-defined]
            context.instance_id,
            context.repo,
            command_instance_id_override=command_instance_id_override,
        )
        test_patch_path = self._selected_test_patch_path(
            instance_id=context.instance_id,
            patch_info=patch_info,
            test_patch_path_override=test_patch_path_override,
            metadata_evaluation_test_patch=metadata_evaluation_test_patch,
        )
        if not isinstance(test_patch_path, str) or not test_patch_path.strip():
            return {"error": "Failed to resolve selected test patch path for evaluation."}
        normalized_setup_patch_paths = self._normalize_setup_patch_paths(
            setup_patch_paths=setup_patch_paths,
            patch_info=patch_info,
            evaluation_test_patch=metadata_evaluation_test_patch,
        )

        # build_only: create the container in the failing state (same construction
        # as a real evaluation run) but do not run tests and do not tear it down.
        # The caller owns the live container and is responsible for stopping it.
        if build_only:
            try:
                create_secure_evaluation_container(
                    image_tag=context.image_tag,
                    container_name=context.container_name,
                )
            except Exception as e:
                stop_container(context.container_name, timeout=0)
                return {"error": f"Failed to create container: {e}"}
            try:
                reset_repository(
                    context.container_name,
                    context.repo,
                    git_ref=base_git_ref_override or "HEAD",
                    stream_to_console=not self.quiet,  # type: ignore[attr-defined]
                )
                for setup_patch_path in normalized_setup_patch_paths:
                    apply_patch(
                        context.container_name,
                        setup_patch_path,
                        allow_empty_patch=True,
                        repo_path=context.repo,
                    )
                if not apply_patch(context.container_name, test_patch_path, repo_path=context.repo):
                    stop_container(context.container_name, timeout=0)
                    return {"error": f"Failed to apply test patch: {test_patch_path}"}
                if patch_file is not None:
                    try:
                        agent_patch_applied = apply_patch(
                            context.container_name,
                            str(patch_file),
                            allow_empty_patch=True,
                            repo_path=context.repo,
                        )
                    except Exception as error:
                        stop_container(context.container_name, timeout=0)
                        return {"error": f"Failed to apply agent patch: {patch_file}: {error}"}
                    if not agent_patch_applied:
                        stop_container(context.container_name, timeout=0)
                        return {"error": f"Failed to apply agent patch: {patch_file}"}
            except Exception as e:
                stop_container(context.container_name, timeout=0)
                return {"error": f"Failed to build failing state: {e}"}
            return {
                "patch_applied": True,
                "build_only": True,
                "container_name": context.container_name,
                "test_command": test_command,
            }

        # Run tests multiple times
        run_log_paths: list[str] = []
        patch_applied = False

        for run_number in range(1, self.runs + 1):  # type: ignore[attr-defined]
            logging.info(
                f"Running evaluation for {context.instance_id} "
                f"[{agent_name}] (run {run_number}/{self.runs})"  # type: ignore[attr-defined]
            )

            # Create a fresh container for this run at its historyless baseline.
            try:
                create_secure_evaluation_container(
                    image_tag=context.image_tag,
                    container_name=context.container_name,
                )
            except Exception as e:
                stop_container(context.container_name, timeout=0)
                return {"error": f"Failed to create container: {e}"}

            try:
                reset_repository(
                    context.container_name,
                    context.repo,
                    git_ref=base_git_ref_override or "HEAD",
                    stream_to_console=not self.quiet,  # type: ignore[attr-defined]
                )
                for setup_patch_path in normalized_setup_patch_paths:
                    apply_patch(
                        context.container_name,
                        setup_patch_path,
                        allow_empty_patch=True,
                        repo_path=context.repo,
                    )
                # Apply golden test patch (required - must have tests)
                patch_applied = apply_patch(
                    context.container_name,
                    test_patch_path,
                    repo_path=context.repo,
                )
                if not patch_applied:
                    return {
                        "error": f"Failed to apply test patch: {test_patch_path}",
                        "patch_applied": False,
                    }

                # Apply agent's code patch (file always exists, may be empty)
                patch_applied = apply_patch(
                    context.container_name,
                    str(patch_file),
                    allow_empty_patch=True,
                    repo_path=context.repo,
                )
                if not patch_applied:
                    return {
                        "error": f"Failed to apply agent patch: {patch_file}",
                        "patch_applied": False,
                    }

                # Run tests
                log_path = log_dir / f"run_{run_number}.log"
                exec_env = dict(NON_INTERACTIVE_EXEC_ENV)
                exec_streaming(
                    container_name=context.container_name,
                    command=[
                        "timeout",
                        "--verbose",
                        "--signal=TERM",
                        f"--kill-after={EVALUATION_TEST_KILL_GRACE_SECONDS}s",
                        f"{EVALUATION_TEST_TIMEOUT_SECONDS}s",
                        "bash",
                        "-lc",
                        test_command,
                    ],
                    environment=exec_env,
                    log_path=log_path,
                    stream_to_console=not self.quiet,  # type: ignore[attr-defined]
                    tty=False,
                    privileged=False,
                )
                run_log_paths.append(str(log_path))

            finally:
                # Always cleanup container after each run
                stop_container(context.container_name, timeout=0)

        if not run_log_paths:
            return {"error": "No test runs completed"}

        return {
            "patch_applied": patch_applied,
            "run_count": len(run_log_paths),
            "run_log_paths": run_log_paths,
            "run_timestamp": run_timestamp,
            "log_dir": str(log_dir),
        }

    def _evaluate_test_runs_from_logs(
        self,
        *,
        run_log_paths: list[Path],
    ) -> dict[str, Any]:
        all_failed_tests: set[str] = set()
        run_results: list[TestRunResult] = []

        for run_number, log_path in enumerate(run_log_paths, start=1):
            result = load_run_result_from_log(log_path, run_number)
            if result is None:
                return {"error": f"Failed to parse test results from {log_path}"}
            run_results.append(result)
            all_failed_tests.update(result["failed_tests"])

        last_run = run_results[-1]
        all_runs_passed = all(
            r["test_exit_code"] == 0 and not r["failed_tests"] for r in run_results
        )
        any_run_passed = any(
            r["test_exit_code"] == 0 and not r["failed_tests"] for r in run_results
        )

        return {
            "all_runs_passed": all_runs_passed,
            "any_run_passed": any_run_passed,
            "last_passed_count": last_run["passed_test_count"],
            "last_failed_count": len(last_run["failed_tests"]),
            "all_failed_tests": sorted(all_failed_tests),
            "last_exit_code": last_run["test_exit_code"],
            "run_count": len(run_results),
            "runs": run_results,
        }

    def _run_evaluation(
        self,
        context: EvaluationContext,
        agent_name: str,
        patch_file: Path | None,
        patch_info: dict[str, str],
        *,
        setup_patch_paths: list[str] | None = None,
        test_patch_path_override: str | None = None,
        base_git_ref_override: str | None = None,
        command_instance_id_override: str | None = None,
        metadata_evaluation_test_patch: str | None = None,
    ) -> dict[str, Any]:
        run_data = self._run_tests_for_evaluation(
            context,
            agent_name=agent_name,
            patch_file=patch_file,
            patch_info=patch_info,
            setup_patch_paths=setup_patch_paths,
            test_patch_path_override=test_patch_path_override,
            base_git_ref_override=base_git_ref_override,
            command_instance_id_override=command_instance_id_override,
            metadata_evaluation_test_patch=metadata_evaluation_test_patch,
        )
        if "error" in run_data:
            return run_data

        raw_run_paths = run_data["run_log_paths"]
        run_log_paths = [Path(path_str) for path_str in raw_run_paths if isinstance(path_str, str)]

        parsed = self._evaluate_test_runs_from_logs(
            run_log_paths=run_log_paths,
        )
        if "error" in parsed:
            payload = dict(run_data)
            payload.update(parsed)
            return payload

        payload = dict(run_data)
        payload.update(parsed)
        return payload

    def _error_evaluation_result(
        self,
        *,
        instance_id: str,
        agent_name: str,
        error: str,
        patch_generated: str = "",
        run_count: int = 0,
        detailed_results: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        return EvaluationResult(
            instance_id=instance_id,
            agent_name=agent_name,
            patch_generated=patch_generated,
            run_count=run_count,
            error=error,
            detailed_results=detailed_results or {},
        )
