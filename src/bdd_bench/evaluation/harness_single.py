# mypy: ignore-errors
from __future__ import annotations

from datetime import datetime
import hashlib
import logging
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from bdd_bench.common.command import run_command
from bdd_bench.common.container_runtime import apply_patch, reset_repository
from bdd_bench.common.docker_client import (
    remove_container,
)
from bdd_bench.common.git_history import sanitize_container_git_history
from bdd_bench.common.image_artifacts import ensure_local_image_available
from bdd_bench.common.test_file_rules import is_env_path, is_non_bdd_test_path, is_test_path
from bdd_bench.evaluation.container_limits import container_oom_kill_count
from bdd_bench.evaluation.container_security import create_secure_evaluation_container
from bdd_bench.evaluation.harness_agents import Agent
from bdd_bench.evaluation.harness_metadata import (
    immutable_image_ref,
    load_metadata,
    sanitize_label,
    validate_metadata,
)
from bdd_bench.evaluation.harness_models import EvaluationContext, EvaluationResult
from bdd_bench.evaluation.patch_utils import extract_patch_paths
from bdd_bench.evaluation.result_classification import INFRA_ERROR_PREFIX


AGENT_FAILURE_DIAGNOSTICS_DIR = "agent_failure_diagnostics"
RESOURCE_EXHAUSTION_ERROR = (
    "Resource exhaustion: runtime agent container hit its RAM limit and was OOM-killed."
)
_MEMORY_STATE_COMMAND = (
    "echo '== memory.events =='; cat /sys/fs/cgroup/memory.events; "
    "echo '== memory.max =='; cat /sys/fs/cgroup/memory.max; "
    "echo '== memory.peak =='; cat /sys/fs/cgroup/memory.peak 2>/dev/null"
)
_DOCKER_CONTAINER_NAME_MAX_LENGTH = 255
_INVALID_DOCKER_NAME_CHARS_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _run_scoped_container_name(
    *,
    purpose: str,
    run_id: str,
    agent_name: str,
    instance_id: str,
    suffix: str | None = None,
) -> str:
    """Build a Docker-safe name scoped to one timestamped model run."""
    components = [purpose, run_id, agent_name, "bdd-bench", instance_id]
    if suffix:
        components.append(suffix)
    raw_name = "-".join(components)
    name = _INVALID_DOCKER_NAME_CHARS_RE.sub("-", raw_name).strip("-.")
    if len(name) <= _DOCKER_CONTAINER_NAME_MAX_LENGTH:
        return name
    digest = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:12]
    prefix_length = _DOCKER_CONTAINER_NAME_MAX_LENGTH - len(digest) - 1
    return f"{name[:prefix_length].rstrip('-.')}-{digest}"


def _patch_to_bytes(patch: str | bytes) -> bytes:
    if isinstance(patch, bytes):
        return patch
    return patch.encode("utf-8")


def _patch_to_text(patch: str | bytes) -> str:
    if isinstance(patch, bytes):
        return patch.decode("utf-8", errors="replace")
    return patch


def _patch_to_parser_text(patch: str | bytes) -> str:
    if isinstance(patch, bytes):
        return patch.decode("latin-1")
    return patch


def _preflight_docker_exec(container_name: str) -> str | None:
    """Run a trivial command inside the container to verify Docker access.

    Returns ``None`` on success or an error description string on failure.
    """
    try:
        completed = subprocess.run(
            ["docker", "exec", container_name, "echo", "ok"],
            check=False,
            capture_output=True,
            timeout=30,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or b"").decode("utf-8", errors="replace").strip()
            return f"docker exec returned exit code {completed.returncode}: {stderr}"
    except FileNotFoundError:
        return "docker binary not found on host"
    except subprocess.TimeoutExpired:
        return "docker exec preflight timed out after 30s"
    except OSError as exc:
        return f"OS error running docker exec: {exc}"
    return None


def _agent_execution_failure_type(error: BaseException) -> str:
    normalized = f"{type(error).__name__} {error}".lower()
    if "contextwindow" in normalized or "context window" in normalized:
        return "context_window_exceeded"
    if "apiconnection" in normalized or "api connection" in normalized:
        return "model_api_error"
    if "costlimit" in normalized or "cost limit" in normalized:
        return "cost_limit_exceeded"
    if "steplimit" in normalized or "step limit" in normalized or "turn limit" in normalized:
        return "step_limit_exceeded"
    if "repeatedformat" in normalized or "format error" in normalized:
        return "repeated_format_error"
    if "timed out" in normalized or "timeexceeded" in normalized:
        return "timeout"
    if "patch" in normalized and ("extract" in normalized or "diff" in normalized):
        return "patch_extraction_failed"
    if "without submitting" in normalized:
        return "agent_did_not_submit"
    return "agent_execution"


def _save_agent_failure_log(
    *,
    output_dir: Path,
    instance_id: str,
    agent_name: str,
    container_name: str,
    repo: str,
    error: BaseException,
) -> Path:
    """Save a basic log describing an agent failure; call while the container is alive."""
    repo_hint = shlex.quote(f"/{repo.strip('/')}" if repo.strip() else ".")
    try:
        repo_state = subprocess.run(
            [
                "docker",
                "exec",
                container_name,
                "bash",
                "-lc",
                f"cd {repo_hint} 2>/dev/null || true; "
                "git status --short; echo '--- DIFF ---'; git diff",
            ],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        repo_state = f"failed to capture repo state: {exc}\n"
    return _write_agent_failure_diagnostic(
        output_dir=output_dir,
        instance_id=instance_id,
        agent_name=agent_name,
        failure_type=_agent_execution_failure_type(error),
        error=str(error),
        error_type=type(error).__name__,
        metadata={"container": container_name},
        sections={"git status & diff": repo_state},
    )


def _write_agent_failure_diagnostic(
    *,
    output_dir: Path,
    instance_id: str,
    agent_name: str,
    failure_type: str,
    error: str,
    error_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    sections: dict[str, str] | None = None,
) -> Path:
    """Write one terminal agent diagnostic using the shared artifact location."""
    log_path = output_dir / AGENT_FAILURE_DIAGNOSTICS_DIR / f"{sanitize_label(instance_id)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"instance: {instance_id}",
        f"agent: {agent_name}",
        f"failure_type: {failure_type}",
    ]
    if error_type:
        header.append(f"error_type: {error_type}")
    for key, value in (metadata or {}).items():
        header.append(f"{key}: {value}")
    header.extend((f"error: {error}", f"written_at: {datetime.now().isoformat()}"))

    body = "\n".join(header) + "\n"
    for title, content in (sections or {}).items():
        body += f"\n== {title} ==\n{content}"
        if content and not content.endswith("\n"):
            body += "\n"
    log_path.write_text(body, encoding="utf-8")
    logging.warning(
        "Saved %s agent failure diagnostic for %s to %s",
        failure_type,
        instance_id,
        log_path,
    )
    return log_path


def _terminal_agent_failure_result(
    *,
    output_dir: Path,
    instance_id: str,
    agent_name: str,
    failure_type: str,
    error: str,
    patch_file: str | Path | None = None,
) -> EvaluationResult:
    """Return an error result with a discoverable terminal-agent diagnostic."""
    metadata: dict[str, Any] = {}
    details: dict[str, Any] = {}
    if patch_file is not None:
        metadata["patch_file"] = str(patch_file)
        details["patch_file"] = str(patch_file)
    log_path = _write_agent_failure_diagnostic(
        output_dir=output_dir,
        instance_id=instance_id,
        agent_name=agent_name,
        failure_type=failure_type,
        error=error,
        metadata=metadata,
    )
    details["agent_failure_log"] = str(log_path)
    return EvaluationResult(
        instance_id=instance_id,
        agent_name=agent_name,
        error=error,
        detailed_results=details,
    )


def _disallowed_patch_result(
    *,
    output_dir: Path,
    instance_id: str,
    agent_name: str,
    patch_file: Path,
    patch_text: str,
    disallowed_files: list[str],
) -> EvaluationResult:
    """Persist and return a result for a patch that touches forbidden paths."""
    normalized_files = sorted(set(disallowed_files))
    listed_files = ", ".join(normalized_files)
    error = f"Agent patch modifies disallowed files (test/env/config): {listed_files}"
    formatted_files = "\n".join(f"- {path}" for path in normalized_files)
    log_path = _write_agent_failure_diagnostic(
        output_dir=output_dir,
        instance_id=instance_id,
        agent_name=agent_name,
        failure_type="disallowed_patch",
        error=error,
        metadata={"patch_file": str(patch_file)},
        sections={
            "disallowed paths": formatted_files,
            "submitted patch": patch_text,
        },
    )
    return EvaluationResult(
        instance_id=instance_id,
        agent_name=agent_name,
        patch_generated=patch_text,
        error=error,
        detailed_results={
            "patch_file": str(patch_file),
            "disallowed_files": normalized_files,
            "agent_failure_log": str(log_path),
        },
    )


def _prior_terminal_agent_failure_result(
    *,
    output_dir: Path,
    instance_id: str,
    agent_name: str,
) -> EvaluationResult | None:
    """Return a preserved generation failure so continuation does not retry it."""
    log_path = output_dir / AGENT_FAILURE_DIAGNOSTICS_DIR / f"{sanitize_label(instance_id)}.log"
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    error_text = ""
    for line in text.splitlines():
        if line.startswith("error:"):
            error_text = line.partition(":")[2].strip()
            break

    if not error_text:
        error_text = f"previous agent generation attempt failed; see {log_path}"
    normalized_text = text.lower()
    if "fatalturnlimitederror" in normalized_text or "reached max session turns" in normalized_text:
        error_text = (
            f"{agent_name} run reached the configured session turn limit while "
            "generating patch. The prior attempt is preserved and will not be rerun."
        )
    return EvaluationResult(
        instance_id=instance_id,
        agent_name=agent_name,
        error=f"Agent failed to generate patch: {error_text}",
        detailed_results={
            "agent_failure_log": str(log_path),
            "skipped_existing_generation_failure": True,
        },
    )


def _resource_exhaustion_result(
    *,
    output_dir: Path,
    instance_id: str,
    agent_name: str,
    container_name: str,
    oom_kills: int,
) -> EvaluationResult:
    """Save a basic OOM log and fail the instance; call while the container is alive."""
    try:
        memory_state = subprocess.run(
            ["docker", "exec", container_name, "sh", "-c", _MEMORY_STATE_COMMAND],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        memory_state = f"failed to read memory state: {error}\n"
    log_path = _write_agent_failure_diagnostic(
        output_dir=output_dir,
        instance_id=instance_id,
        agent_name=agent_name,
        failure_type="resource_exhaustion",
        error=RESOURCE_EXHAUSTION_ERROR,
        metadata={
            "container": container_name,
            "memory_oom_kills": str(oom_kills),
        },
        sections={"memory state": memory_state},
    )
    logging.warning("%s: resource exhaustion (%s OOM kills)", instance_id, oom_kills)
    return EvaluationResult(
        instance_id=instance_id,
        agent_name=agent_name,
        error=RESOURCE_EXHAUSTION_ERROR,
        detailed_results={
            "resource_exhaustion": True,
            "memory_oom_kills": oom_kills,
            "agent_failure_log": str(log_path),
        },
    )


def _create_runtime_patch_base_ref(container_name: str) -> str:
    """Create a commit baseline in the runtime agent container and return its ref."""
    baseline_command = (
        "repo_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd); "
        'if [ -n "$(git -C "$repo_root" status --porcelain)" ]; then '
        'git -C "$repo_root" add -A; '
        'git -C "$repo_root" '
        "-c user.name='bdd-bench' "
        "-c user.email='bdd-bench@local' "
        "commit --no-gpg-sign -m 'bdd-bench runtime baseline (golden test patch)' >/dev/null; "
        "fi; "
        'git -C "$repo_root" rev-parse --verify HEAD'
    )
    completed = run_command(
        ["docker", "exec", container_name, "bash", "-lc", baseline_command],
        text=True,
    )
    stdout = completed.stdout
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
    base_ref = stdout.strip().splitlines()[-1].strip() if stdout else ""
    if not base_ref:
        raise RuntimeError(
            f"Could not determine runtime patch base ref in container {container_name}."
        )
    return base_ref


def _disallowed_paths_in_patch(patch: str | bytes, *, repo: str | None = None) -> list[str]:
    """Return paths in *patch* that the agent is not allowed to modify.

    This covers test files (BDD and non-BDD), env/build config files, and
    .feature files.  The check mirrors the exclusion logic used by
    ``GoldenAgent.generate_patch``.
    """
    return sorted(
        {
            path
            for path in extract_patch_paths(_patch_to_parser_text(patch))
            if (
                is_test_path(path, repo=repo)
                or is_non_bdd_test_path(path, repo=repo)
                or is_env_path(path, repo=repo)
            )
        },
    )


def _normalize_generation_setup_patch_paths(
    *,
    setup_patch_paths: list[str] | None,
    patch_info: dict[str, Any],
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


def _visible_runtime_test_patch_path(
    *,
    patch_info: dict[str, Any],
    visible_test_patch_key: str,
    fallback_test_patch_path: str,
) -> str:
    candidate = patch_info.get(visible_test_patch_key)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return fallback_test_patch_path


class _HarnessSingle:
    """Per-instance patch generation and evaluation workflows for EvaluationHarness."""

    def evaluate_single(
        self,
        agent: Agent,
        metadata_path: Path,
    ) -> EvaluationResult:
        logging.info(f"Evaluating {agent.name} on {metadata_path}")

        # Load and validate metadata
        try:
            data = load_metadata(metadata_path)
            validate_metadata(data)
        except Exception as e:
            return EvaluationResult(
                instance_id=str(metadata_path),
                agent_name=agent.name,
                error=f"Failed to load metadata: {e}",
            )

        image_tag = immutable_image_ref(data)
        repo = data["repo"]
        instance_id = data["instance_id"]
        patch_info = data["patches"]
        evaluation_test_patch = data.get("evaluation_test_patch")
        patch = ""

        eval_container_name = _run_scoped_container_name(
            purpose="eval",
            run_id=self.container_name_scope,  # type: ignore[attr-defined]
            agent_name=agent.name,
            instance_id=instance_id,
        )
        agent_container_name = _run_scoped_container_name(
            purpose="agent",
            run_id=self.container_name_scope,  # type: ignore[attr-defined]
            agent_name=agent.name,
            instance_id=instance_id,
        )

        # Check if instance is in the published evaluation manifest.
        if instance_id not in self.instances:  # type: ignore[attr-defined]
            return EvaluationResult(
                instance_id=instance_id,
                agent_name=agent.name,
                error=f"Instance {instance_id} not found in the evaluation manifest.",
            )

        try:
            ensure_local_image_available(image_tag)
        except Exception as error:
            return EvaluationResult(
                instance_id=instance_id,
                agent_name=agent.name,
                error=f"Failed to prepare Docker image {image_tag}: {error}",
            )

        runtime_context: dict[str, Any] | None = None
        if agent.requires_runtime_context:
            host_preflight_failure = agent.preflight_host_environment()
            if host_preflight_failure is not None:
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error=f"{INFRA_ERROR_PREFIX}{host_preflight_failure}",
                )
            try:
                test_command_for_agent = self._build_agent_test_command(  # type: ignore[attr-defined]
                    instance_id,
                    repo,
                    metadata_evaluation_test_patch=evaluation_test_patch,
                )
            except Exception as error:
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error=f"Failed to prepare agent test command: {error}",
                )

            selected_test_patch = self._selected_test_patch_path(
                instance_id=instance_id,
                patch_info=patch_info,
                metadata_evaluation_test_patch=evaluation_test_patch,
            )
            if not isinstance(selected_test_patch, str) or not selected_test_patch.strip():
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error="Metadata missing selected test patch required by runtime agent.",
                )
            visible_test_patch_key = self._agent_visible_test_patch_key(  # type: ignore[attr-defined]
                instance_id,
                metadata_evaluation_test_patch=evaluation_test_patch,
            )
            visible_test_patch = _visible_runtime_test_patch_path(
                patch_info=patch_info,
                visible_test_patch_key=visible_test_patch_key,
                fallback_test_patch_path=selected_test_patch,
            )

            remove_container(agent_container_name)
            try:
                create_secure_evaluation_container(
                    image_tag=image_tag,
                    container_name=agent_container_name,
                )
            except Exception as error:
                remove_container(agent_container_name)
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error=f"Failed to create runtime agent container: {error}",
                )
            try:
                sanitize_container_git_history(
                    container_name=agent_container_name,
                    repo_hint=repo,
                )
            except Exception as error:
                remove_container(agent_container_name)
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error=f"Failed to sanitize runtime agent history: {error}",
                )

            # Apply golden_env_patch first (if present) so the agent sees the
            # same environment the evaluation replay will use.
            golden_env_patch = patch_info.get("golden_env_patch")
            if isinstance(golden_env_patch, str) and golden_env_patch.strip():
                try:
                    apply_patch(
                        agent_container_name,
                        golden_env_patch,
                        allow_empty_patch=True,
                        repo_path=repo,
                    )
                except Exception as error:
                    remove_container(agent_container_name)
                    return _terminal_agent_failure_result(
                        output_dir=self.output_dir,  # type: ignore[attr-defined]
                        instance_id=instance_id,
                        agent_name=agent.name,
                        failure_type="agent_setup_patch_apply_failed",
                        error=f"Failed to apply golden env patch for runtime agent: {error}",
                        patch_file=golden_env_patch,
                    )

            # Match dataset-generation ordering: when the visible patch is silver or bronze,
            # the generated patches were authored on top of golden_test_patch, so we layer
            # it onto the agent's container before applying the visible patch.
            golden_test_patch_path = patch_info.get("golden_test_patch")
            apply_golden_test_patch = (
                visible_test_patch_key in {"silver_test_patch", "bronze_test_patch"}
                and isinstance(golden_test_patch_path, str)
                and golden_test_patch_path.strip()
                and golden_test_patch_path != visible_test_patch
            )
            if apply_golden_test_patch:
                try:
                    golden_test_applied = apply_patch(
                        agent_container_name,
                        golden_test_patch_path,
                        allow_empty_patch=True,
                        repo_path=repo,
                    )
                except Exception as error:
                    remove_container(agent_container_name)
                    return _terminal_agent_failure_result(
                        output_dir=self.output_dir,  # type: ignore[attr-defined]
                        instance_id=instance_id,
                        agent_name=agent.name,
                        failure_type="agent_setup_patch_apply_failed",
                        error=("Failed to apply golden test patch for runtime agent: " f"{error}"),
                        patch_file=golden_test_patch_path,
                    )
                if not golden_test_applied:
                    remove_container(agent_container_name)
                    return _terminal_agent_failure_result(
                        output_dir=self.output_dir,  # type: ignore[attr-defined]
                        instance_id=instance_id,
                        agent_name=agent.name,
                        failure_type="agent_setup_patch_apply_failed",
                        error=(
                            "Failed to apply golden test patch for runtime agent: "
                            f"{golden_test_patch_path}"
                        ),
                        patch_file=golden_test_patch_path,
                    )

            try:
                test_patch_applied = apply_patch(
                    agent_container_name,
                    visible_test_patch,
                    repo_path=repo,
                )
            except Exception as error:
                remove_container(agent_container_name)
                return _terminal_agent_failure_result(
                    output_dir=self.output_dir,  # type: ignore[attr-defined]
                    instance_id=instance_id,
                    agent_name=agent.name,
                    failure_type="agent_setup_patch_apply_failed",
                    error=f"Failed to apply selected test patch for runtime agent: {error}",
                    patch_file=visible_test_patch,
                )
            if not test_patch_applied:
                remove_container(agent_container_name)
                return _terminal_agent_failure_result(
                    output_dir=self.output_dir,  # type: ignore[attr-defined]
                    instance_id=instance_id,
                    agent_name=agent.name,
                    failure_type="agent_setup_patch_apply_failed",
                    error=(
                        "Failed to apply selected test patch for runtime agent: "
                        f"{visible_test_patch}"
                    ),
                    patch_file=visible_test_patch,
                )

            try:
                patch_base_ref = _create_runtime_patch_base_ref(agent_container_name)
            except Exception as error:
                remove_container(agent_container_name)
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error=f"Failed to create runtime patch baseline: {error}",
                )

            runtime_context = {
                "container_name": agent_container_name,
                "test_command": test_command_for_agent,
                "visible_test_patch": visible_test_patch,
                "visible_test_patch_key": visible_test_patch_key,
                "golden_test_patch": visible_test_patch,
                "patch_base_ref": patch_base_ref,
                "output_dir": str(self.output_dir),  # type: ignore[attr-defined]
                "instance_id": instance_id,
                "repo": repo,
                "agent_name": agent.name,
                "live_stream": self.live_agent_stream,  # type: ignore[attr-defined]
            }

            preflight_failure = _preflight_docker_exec(agent_container_name)
            if preflight_failure is not None:
                remove_container(agent_container_name)
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error=f"{INFRA_ERROR_PREFIX}{preflight_failure}",
                )

        # Generate patch using the agent
        logging.info(f"Running agent {agent.name} to generate patch...")
        try:
            if agent.requires_runtime_context:
                patch = agent.generate_patch(data, runtime_context=runtime_context)
                oom_kills = container_oom_kill_count(agent_container_name)
                if oom_kills:
                    return _resource_exhaustion_result(
                        output_dir=self.output_dir,  # type: ignore[attr-defined]
                        instance_id=instance_id,
                        agent_name=agent.name,
                        container_name=agent_container_name,
                        oom_kills=oom_kills,
                    )
            else:
                patch = agent.generate_patch(data)
        except Exception as e:
            detailed_results: dict[str, Any] = {}
            if agent.requires_runtime_context:
                log_path = _save_agent_failure_log(
                    output_dir=self.output_dir,  # type: ignore[attr-defined]
                    instance_id=instance_id,
                    agent_name=agent.name,
                    container_name=agent_container_name,
                    repo=repo,
                    error=e,
                )
                detailed_results["agent_failure_log"] = str(log_path)
            return EvaluationResult(
                instance_id=instance_id,
                agent_name=agent.name,
                error=f"Agent failed to generate patch: {e}",
                detailed_results=detailed_results,
            )
        finally:
            if agent.requires_runtime_context:
                remove_container(agent_container_name)

        patch_bytes = _patch_to_bytes(patch)
        patch_text = _patch_to_text(patch)
        logging.info(f"Agent generated patch of {len(patch_bytes)} bytes")

        # Write patch to temporary file (create file even if empty)
        patch_file = self._agent_patch_path(agent_name=agent.name, instance_id=instance_id)  # type: ignore[attr-defined]
        patch_file.write_bytes(patch_bytes)
        logging.info(f"Saved patch to {patch_file}")

        disallowed_files = _disallowed_paths_in_patch(patch_bytes, repo=repo)
        if disallowed_files:
            return _disallowed_patch_result(
                output_dir=self.output_dir,  # type: ignore[attr-defined]
                instance_id=instance_id,
                agent_name=agent.name,
                patch_file=patch_file,
                patch_text=patch_text,
                disallowed_files=disallowed_files,
            )

        # Ensure no leftover containers
        remove_container(eval_container_name)

        try:
            context = EvaluationContext(
                container_name=eval_container_name,
                repo=repo,
                image_tag=image_tag,
                instance_id=instance_id,
                log_path=self.output_dir,  # type: ignore[attr-defined]
            )

            # Run evaluation (handles container lifecycle for each run)
            results = self._run_evaluation(  # type: ignore[attr-defined]
                context,
                agent_name=agent.name,
                patch_file=patch_file,
                patch_info=patch_info,
                base_git_ref_override="HEAD",
                metadata_evaluation_test_patch=evaluation_test_patch,
            )

            # Parse results
            if "error" in results:
                details = dict(results)
                result_error = str(results["error"])
                if "Failed to apply" in result_error and "patch" in result_error:
                    agent_patch_failed = "agent patch" in result_error
                    failed_patch_file = (
                        str(patch_file)
                        if agent_patch_failed
                        else result_error.partition(":")[2].strip()
                    )
                    log_path = _write_agent_failure_diagnostic(
                        output_dir=self.output_dir,  # type: ignore[attr-defined]
                        instance_id=instance_id,
                        agent_name=agent.name,
                        failure_type=(
                            "agent_patch_apply_failed"
                            if agent_patch_failed
                            else "evaluation_setup_patch_apply_failed"
                        ),
                        error=result_error,
                        metadata={"patch_file": failed_patch_file},
                        sections={"submitted patch": patch_text} if agent_patch_failed else None,
                    )
                    details["agent_failure_log"] = str(log_path)
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    patch_generated=patch_text,
                    error=result_error,
                    detailed_results=details,
                )

            return EvaluationResult(
                instance_id=instance_id,
                agent_name=agent.name,
                patch_generated=patch_text,
                patch_applied_successfully=results["patch_applied"],
                all_tests_passed=results["all_runs_passed"],
                any_tests_passed=results["any_run_passed"],
                passed_test_count=results["last_passed_count"],
                failed_test_count=results["last_failed_count"],
                failed_tests=results["all_failed_tests"],
                test_exit_code=results["last_exit_code"],
                run_count=results["run_count"],
                detailed_results=results,
            )

        except Exception as e:
            logging.exception(f"Evaluation failed for {instance_id}")
            return EvaluationResult(
                instance_id=instance_id,
                agent_name=agent.name,
                patch_generated=patch_text,
                error=str(e),
            )
        # Note: Container cleanup is handled by _run_evaluation after each test run

    def generate_patch_single(
        self,
        agent: Agent,
        metadata_path: Path,
        *,
        skip_existing_patch: bool = False,
        setup_patch_paths: list[str] | None = None,
        test_patch_path_override: str | None = None,
        image_tag_override: str | None = None,
        base_git_ref_override: str | None = None,
        command_instance_id_override: str | None = None,
    ) -> EvaluationResult:
        """Generate and persist a patch for one instance without running tests."""
        logging.info(f"Generating patch for {agent.name} on {metadata_path}")

        try:
            data = load_metadata(metadata_path)
            validate_metadata(data)
        except Exception as error:
            return EvaluationResult(
                instance_id=str(metadata_path),
                agent_name=agent.name,
                error=f"Failed to load metadata: {error}",
            )

        image_tag = image_tag_override or immutable_image_ref(data)
        repo = data["repo"]
        instance_id = data["instance_id"]
        patch_info = data["patches"]
        patch_file = self._agent_patch_path(agent_name=agent.name, instance_id=instance_id)  # type: ignore[attr-defined]

        if instance_id not in self.instances:  # type: ignore[attr-defined]
            return EvaluationResult(
                instance_id=instance_id,
                agent_name=agent.name,
                error=(
                    f"Instance {instance_id} not found in the evaluation manifest. "
                    "Only valid instances can be evaluated."
                ),
            )

        if skip_existing_patch and patch_file.is_file():
            existing_patch = patch_file.read_bytes().decode("utf-8", errors="replace")
            logging.info(f"Skipping {instance_id}: patch already exists at {patch_file}")
            return EvaluationResult(
                instance_id=instance_id,
                agent_name=agent.name,
                patch_generated=existing_patch,
                detailed_results={
                    "patch_file": str(patch_file),
                    "skipped_existing_patch": True,
                },
            )
        if skip_existing_patch:
            prior_failure_result = _prior_terminal_agent_failure_result(
                output_dir=self.output_dir,  # type: ignore[attr-defined]
                instance_id=instance_id,
                agent_name=agent.name,
            )
            if prior_failure_result is not None:
                logging.info(
                    "Skipping %s: prior terminal generation failure recorded at %s",
                    instance_id,
                    prior_failure_result.detailed_results["agent_failure_log"],
                )
                return prior_failure_result

        patch = ""
        agent_container_name = _run_scoped_container_name(
            purpose="agent",
            run_id=self.container_name_scope,  # type: ignore[attr-defined]
            agent_name=agent.name,
            instance_id=instance_id,
        )

        runtime_context: dict[str, Any] | None = None
        if agent.requires_runtime_context:
            host_preflight_failure = agent.preflight_host_environment()
            if host_preflight_failure is not None:
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error=f"{INFRA_ERROR_PREFIX}{host_preflight_failure}",
                )
            try:
                ensure_local_image_available(image_tag)
            except Exception as error:
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error=f"Failed to prepare Docker image {image_tag}: {error}",
                )

            generation_test_patch_key = data.get("evaluation_test_patch")
            if test_patch_path_override == patch_info.get("silver_test_patch"):
                generation_test_patch_key = "silver_test_patch"
            elif test_patch_path_override == patch_info.get("bronze_test_patch"):
                generation_test_patch_key = "bronze_test_patch"
            normalized_generation_test_patch_key = (
                generation_test_patch_key if isinstance(generation_test_patch_key, str) else None
            )
            try:
                test_command_for_agent = self._build_agent_test_command(  # type: ignore[attr-defined]
                    instance_id,
                    repo,
                    metadata_evaluation_test_patch=normalized_generation_test_patch_key,
                    command_instance_id_override=command_instance_id_override,
                )
            except Exception as error:
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error=f"Failed to prepare agent test command: {error}",
                )

            selected_test_patch = self._selected_test_patch_path(
                instance_id=instance_id,
                patch_info=patch_info,
                test_patch_path_override=test_patch_path_override,
                metadata_evaluation_test_patch=data.get("evaluation_test_patch"),
            )
            if not isinstance(selected_test_patch, str) or not selected_test_patch.strip():
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error="Metadata missing selected test patch required by runtime agent.",
                )
            visible_test_patch_key = self._agent_visible_test_patch_key(  # type: ignore[attr-defined]
                instance_id,
                metadata_evaluation_test_patch=normalized_generation_test_patch_key,
                command_instance_id_override=command_instance_id_override,
            )
            visible_test_patch = _visible_runtime_test_patch_path(
                patch_info=patch_info,
                visible_test_patch_key=visible_test_patch_key,
                fallback_test_patch_path=selected_test_patch,
            )
            normalized_setup_patch_paths = _normalize_generation_setup_patch_paths(
                setup_patch_paths=setup_patch_paths,
                patch_info=patch_info,
                evaluation_test_patch=normalized_generation_test_patch_key,
            )
            normalized_setup_patch_paths = [
                path for path in normalized_setup_patch_paths if path != visible_test_patch
            ]

            remove_container(agent_container_name)
            try:
                create_secure_evaluation_container(
                    image_tag=image_tag,
                    container_name=agent_container_name,
                )
            except Exception as error:
                remove_container(agent_container_name)
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error=f"Failed to create runtime agent container: {error}",
                )
            try:
                reset_repository(
                    agent_container_name,
                    repo,
                    git_ref=base_git_ref_override or "HEAD",
                )
                sanitize_container_git_history(
                    container_name=agent_container_name,
                    repo_hint=repo,
                )
            except Exception as error:
                remove_container(agent_container_name)
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error=f"Failed to sanitize runtime agent history: {error}",
                )

            applying_patch_path = visible_test_patch
            try:
                for setup_patch_path in normalized_setup_patch_paths:
                    if not isinstance(setup_patch_path, str) or not setup_patch_path.strip():
                        continue
                    applying_patch_path = setup_patch_path
                    apply_patch(
                        agent_container_name,
                        setup_patch_path,
                        allow_empty_patch=True,
                        repo_path=repo,
                    )
                applying_patch_path = visible_test_patch
                test_patch_applied = apply_patch(
                    agent_container_name,
                    visible_test_patch,
                    repo_path=repo,
                )
            except Exception as error:
                remove_container(agent_container_name)
                return _terminal_agent_failure_result(
                    output_dir=self.output_dir,  # type: ignore[attr-defined]
                    instance_id=instance_id,
                    agent_name=agent.name,
                    failure_type="agent_setup_patch_apply_failed",
                    error=f"Failed to apply setup patch for runtime agent: {error}",
                    patch_file=applying_patch_path,
                )
            if not test_patch_applied:
                remove_container(agent_container_name)
                return _terminal_agent_failure_result(
                    output_dir=self.output_dir,  # type: ignore[attr-defined]
                    instance_id=instance_id,
                    agent_name=agent.name,
                    failure_type="agent_setup_patch_apply_failed",
                    error=(
                        "Failed to apply selected test patch for runtime agent: "
                        f"{visible_test_patch}"
                    ),
                    patch_file=visible_test_patch,
                )

            try:
                patch_base_ref = _create_runtime_patch_base_ref(agent_container_name)
            except Exception as error:
                remove_container(agent_container_name)
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error=f"Failed to create runtime patch baseline: {error}",
                )

            runtime_context = {
                "container_name": agent_container_name,
                "test_command": test_command_for_agent,
                "visible_test_patch": visible_test_patch,
                "visible_test_patch_key": visible_test_patch_key,
                "golden_test_patch": visible_test_patch,
                "patch_base_ref": patch_base_ref,
                "output_dir": str(self.output_dir),  # type: ignore[attr-defined]
                "instance_id": instance_id,
                "repo": repo,
                "agent_name": agent.name,
                "live_stream": self.live_agent_stream,  # type: ignore[attr-defined]
            }

            preflight_failure = _preflight_docker_exec(agent_container_name)
            if preflight_failure is not None:
                remove_container(agent_container_name)
                return EvaluationResult(
                    instance_id=instance_id,
                    agent_name=agent.name,
                    error=f"{INFRA_ERROR_PREFIX}{preflight_failure}",
                )

        try:
            if agent.requires_runtime_context:
                patch = agent.generate_patch(data, runtime_context=runtime_context)
                oom_kills = container_oom_kill_count(agent_container_name)
                if oom_kills:
                    return _resource_exhaustion_result(
                        output_dir=self.output_dir,  # type: ignore[attr-defined]
                        instance_id=instance_id,
                        agent_name=agent.name,
                        container_name=agent_container_name,
                        oom_kills=oom_kills,
                    )
            else:
                patch = agent.generate_patch(data)
        except Exception as error:
            detailed_results: dict[str, Any] = {}
            if agent.requires_runtime_context:
                log_path = _save_agent_failure_log(
                    output_dir=self.output_dir,  # type: ignore[attr-defined]
                    instance_id=instance_id,
                    agent_name=agent.name,
                    container_name=agent_container_name,
                    repo=repo,
                    error=error,
                )
                detailed_results["agent_failure_log"] = str(log_path)
            return EvaluationResult(
                instance_id=instance_id,
                agent_name=agent.name,
                error=f"Agent failed to generate patch: {error}",
                detailed_results=detailed_results,
            )
        finally:
            if agent.requires_runtime_context:
                remove_container(agent_container_name)

        patch_bytes = _patch_to_bytes(patch)
        patch_text = _patch_to_text(patch)
        patch_file.write_bytes(patch_bytes)
        logging.info(f"Saved patch to {patch_file}")

        disallowed_files = _disallowed_paths_in_patch(patch_bytes, repo=repo)
        if disallowed_files:
            return _disallowed_patch_result(
                output_dir=self.output_dir,  # type: ignore[attr-defined]
                instance_id=instance_id,
                agent_name=agent.name,
                patch_file=patch_file,
                patch_text=patch_text,
                disallowed_files=disallowed_files,
            )

        return EvaluationResult(
            instance_id=instance_id,
            agent_name=agent.name,
            patch_generated=patch_text,
            detailed_results={
                "patch_file": str(patch_file),
                "skipped_existing_patch": False,
            },
        )

    def _prepare_patch_evaluation(
        self,
        *,
        agent_name: str,
        metadata_path: Path,
        patch_file: Path,
        image_tag_override: str | None = None,
        command_instance_id_override: str | None = None,
        container_name_suffix: str | None = None,
    ) -> tuple[dict[str, Any] | None, EvaluationResult | None]:
        try:
            data = load_metadata(metadata_path)
            validate_metadata(data)
        except Exception as error:
            return None, self._error_evaluation_result(  # type: ignore[attr-defined]
                instance_id=str(metadata_path),
                agent_name=agent_name,
                error=f"Failed to load metadata: {error}",
            )

        image_tag = image_tag_override or immutable_image_ref(data)
        repo = data["repo"]
        instance_id = data["instance_id"]
        patch_info = data["patches"]

        if instance_id not in self.instances:  # type: ignore[attr-defined]
            return (
                None,
                self._error_evaluation_result(  # type: ignore[attr-defined]
                    instance_id=instance_id,
                    agent_name=agent_name,
                    error=(
                        f"Instance {instance_id} not found in the evaluation manifest. "
                        "Only valid instances can be evaluated."
                    ),
                ),
            )

        try:
            ensure_local_image_available(image_tag)
        except Exception as error:
            return None, self._error_evaluation_result(  # type: ignore[attr-defined]
                instance_id=instance_id,
                agent_name=agent_name,
                error=f"Failed to prepare Docker image {image_tag}: {error}",
            )

        if not patch_file.is_file():
            return None, self._error_evaluation_result(  # type: ignore[attr-defined]
                instance_id=instance_id,
                agent_name=agent_name,
                error=f"Patch file not found: {patch_file}",
            )

        patch = patch_file.read_bytes().decode("utf-8", errors="replace")
        disallowed_files = _disallowed_paths_in_patch(patch, repo=repo)
        if disallowed_files:
            return (
                None,
                _disallowed_patch_result(
                    output_dir=self.output_dir,  # type: ignore[attr-defined]
                    instance_id=instance_id,
                    agent_name=agent_name,
                    patch_file=patch_file,
                    patch_text=patch,
                    disallowed_files=disallowed_files,
                ),
            )

        eval_container_name = _run_scoped_container_name(
            purpose="eval",
            run_id=self.container_name_scope,  # type: ignore[attr-defined]
            agent_name=agent_name,
            instance_id=instance_id,
            suffix=container_name_suffix,
        )
        remove_container(eval_container_name)

        context = EvaluationContext(
            container_name=eval_container_name,
            repo=repo,
            image_tag=image_tag,
            instance_id=instance_id,
            log_path=self.output_dir,  # type: ignore[attr-defined]
        )
        return {
            "repo": repo,
            "instance_id": instance_id,
            "patch_info": patch_info,
            "patch": patch,
            "context": context,
            "command_instance_id_override": command_instance_id_override,
        }, None

    def run_tests_single_from_patch(
        self,
        *,
        agent_name: str,
        metadata_path: Path,
        patch_file: Path,
        setup_patch_paths: list[str] | None = None,
        test_patch_path_override: str | None = None,
        image_tag_override: str | None = None,
        base_git_ref_override: str | None = None,
        command_instance_id_override: str | None = None,
        build_only: bool = False,
        container_name_suffix: str | None = None,
    ) -> dict[str, Any]:
        prepared, error_result = self._prepare_patch_evaluation(
            agent_name=agent_name,
            metadata_path=metadata_path,
            patch_file=patch_file,
            image_tag_override=image_tag_override,
            command_instance_id_override=command_instance_id_override,
            container_name_suffix=container_name_suffix,
        )
        if error_result is not None:
            error_payload = {
                "instance_id": error_result.instance_id,
                "agent_name": agent_name,
                "metadata_path": str(metadata_path),
                "patch_file": str(patch_file),
                "patch_generated": error_result.patch_generated,
                "patch_applied": False,
                "run_count": 0,
                "run_log_paths": [],
                "log_dir": None,
                "error": error_result.error,
            }
            agent_failure_log = error_result.detailed_results.get("agent_failure_log")
            if isinstance(agent_failure_log, str):
                error_payload["agent_failure_log"] = agent_failure_log
            return error_payload

        assert prepared is not None
        instance_id = prepared["instance_id"]
        patch = prepared["patch"]
        run_data = self._run_tests_for_evaluation(  # type: ignore[attr-defined]
            prepared["context"],
            agent_name=agent_name,
            patch_file=patch_file,
            patch_info=prepared["patch_info"],
            setup_patch_paths=setup_patch_paths,
            test_patch_path_override=test_patch_path_override,
            base_git_ref_override=base_git_ref_override,
            command_instance_id_override=prepared["command_instance_id_override"],
            build_only=build_only,
        )
        if build_only:
            return {
                "instance_id": instance_id,
                "agent_name": agent_name,
                "patch_generated": patch,
                "container_name": run_data.get("container_name"),
                "build_only": True,
                "error": run_data.get("error"),
            }
        run_error = run_data.get("error")
        if isinstance(run_error, str) and run_error:
            error_payload = {
                "instance_id": instance_id,
                "agent_name": agent_name,
                "metadata_path": str(metadata_path),
                "patch_file": str(patch_file),
                "patch_generated": patch,
                "patch_applied": False,
                "run_count": 0,
                "run_log_paths": [],
                "run_timestamp": None,
                "log_dir": None,
                "error": run_error,
            }
            if "Failed to apply" in run_error and "patch" in run_error:
                agent_patch_failed = "agent patch" in run_error
                failed_patch_file = (
                    str(patch_file) if agent_patch_failed else run_error.partition(":")[2].strip()
                )
                log_path = _write_agent_failure_diagnostic(
                    output_dir=self.output_dir,  # type: ignore[attr-defined]
                    instance_id=instance_id,
                    agent_name=agent_name,
                    failure_type=(
                        "agent_patch_apply_failed"
                        if agent_patch_failed
                        else "evaluation_setup_patch_apply_failed"
                    ),
                    error=run_error,
                    metadata={"patch_file": failed_patch_file},
                    sections={"submitted patch": patch} if agent_patch_failed else None,
                )
                error_payload["agent_failure_log"] = str(log_path)
            return error_payload

        return {
            "instance_id": instance_id,
            "agent_name": agent_name,
            "metadata_path": str(metadata_path),
            "patch_file": str(patch_file),
            "patch_generated": patch,
            "patch_applied": bool(run_data["patch_applied"]),
            "run_count": int(run_data["run_count"]),
            "run_log_paths": [path for path in run_data["run_log_paths"] if isinstance(path, str)],
            "run_timestamp": run_data["run_timestamp"],
            "log_dir": run_data["log_dir"],
            "error": None,
        }

    def evaluate_single_test_results_from_logs(
        self,
        *,
        agent_name: str,
        metadata_path: Path,
        patch_file: Path,
        log_dir: Path | None = None,
    ) -> EvaluationResult:
        try:
            data = load_metadata(metadata_path)
            validate_metadata(data)
        except Exception as error:
            return self._error_evaluation_result(  # type: ignore[attr-defined]
                instance_id=str(metadata_path),
                agent_name=agent_name,
                error=f"Failed to load metadata: {error}",
            )

        instance_id = data["instance_id"]
        repo = data.get("repo")
        if instance_id not in self.instances:  # type: ignore[attr-defined]
            return self._error_evaluation_result(  # type: ignore[attr-defined]
                instance_id=instance_id,
                agent_name=agent_name,
                error=(
                    f"Instance {instance_id} not found in the evaluation manifest. "
                    "Only valid instances can be evaluated."
                ),
            )

        patch = ""
        if patch_file.is_file():
            patch = patch_file.read_bytes().decode("utf-8", errors="replace")
        else:
            return self._error_evaluation_result(  # type: ignore[attr-defined]
                instance_id=instance_id,
                agent_name=agent_name,
                error=f"Patch file not found: {patch_file}",
            )

        disallowed_files = _disallowed_paths_in_patch(patch, repo=repo)
        if disallowed_files:
            return _disallowed_patch_result(
                output_dir=self.output_dir,  # type: ignore[attr-defined]
                instance_id=instance_id,
                agent_name=agent_name,
                patch_file=patch_file,
                patch_text=patch,
                disallowed_files=disallowed_files,
            )

        if log_dir is None:
            log_dir = self._latest_eval_log_dir(agent_name=agent_name, instance_id=instance_id)  # type: ignore[attr-defined]
        if log_dir is None:
            return self._error_evaluation_result(  # type: ignore[attr-defined]
                instance_id=instance_id,
                agent_name=agent_name,
                patch_generated=patch,
                error=(
                    "No evaluation logs found for "
                    f"{instance_id} under {self.output_dir / 'eval_logs'}."  # type: ignore[attr-defined]
                ),
            )

        run_log_paths = self._run_log_paths_from_dir(log_dir)  # type: ignore[attr-defined]
        if not run_log_paths:
            return EvaluationResult(
                instance_id=instance_id,
                agent_name=agent_name,
                patch_generated=patch,
                error=f"No run logs found in {log_dir}",
                detailed_results={"log_dir": str(log_dir)},
            )

        if self.runs > len(run_log_paths):  # type: ignore[attr-defined]
            return self._error_evaluation_result(  # type: ignore[attr-defined]
                instance_id=instance_id,
                agent_name=agent_name,
                patch_generated=patch,
                run_count=len(run_log_paths),
                error=(
                    f"Missing run logs in {log_dir}: expected {self.runs}, "  # type: ignore[attr-defined]
                    f"found {len(run_log_paths)}."
                ),
                detailed_results={
                    "log_dir": str(log_dir),
                    "run_log_paths": [str(path) for path in run_log_paths],
                },
            )

        parsed = self._evaluate_test_runs_from_logs(run_log_paths=run_log_paths)  # type: ignore[attr-defined]
        if "error" in parsed:
            details = dict(parsed)
            details["log_dir"] = str(log_dir)
            details["run_log_paths"] = [str(path) for path in run_log_paths]
            return self._error_evaluation_result(  # type: ignore[attr-defined]
                instance_id=instance_id,
                agent_name=agent_name,
                patch_generated=patch,
                run_count=len(run_log_paths),
                error=parsed["error"],
                detailed_results=details,
            )

        details = dict(parsed)
        details["log_dir"] = str(log_dir)
        details["run_log_paths"] = [str(path) for path in run_log_paths]
        return EvaluationResult(
            instance_id=instance_id,
            agent_name=agent_name,
            patch_generated=patch,
            patch_applied_successfully=True,
            all_tests_passed=parsed["all_runs_passed"],
            any_tests_passed=parsed["any_run_passed"],
            passed_test_count=parsed["last_passed_count"],
            failed_test_count=parsed["last_failed_count"],
            failed_tests=parsed["all_failed_tests"],
            test_exit_code=parsed["last_exit_code"],
            run_count=parsed["run_count"],
            detailed_results=details,
        )

    def evaluate_single_from_patch(
        self,
        *,
        agent_name: str,
        metadata_path: Path,
        patch_file: Path,
        setup_patch_paths: list[str] | None = None,
        test_patch_path_override: str | None = None,
        image_tag_override: str | None = None,
        base_git_ref_override: str | None = None,
        command_instance_id_override: str | None = None,
    ) -> EvaluationResult:
        """Evaluate a single instance using a pre-generated patch file."""
        logging.info(f"Evaluating {agent_name} patch on {metadata_path} from {patch_file}")
        run_payload = self.run_tests_single_from_patch(
            agent_name=agent_name,
            metadata_path=metadata_path,
            patch_file=patch_file,
            setup_patch_paths=setup_patch_paths,
            test_patch_path_override=test_patch_path_override,
            image_tag_override=image_tag_override,
            base_git_ref_override=base_git_ref_override,
            command_instance_id_override=command_instance_id_override,
        )
        run_error = run_payload["error"]
        if isinstance(run_error, str) and run_error:
            return self._error_evaluation_result(  # type: ignore[attr-defined]
                instance_id=str(run_payload["instance_id"]),
                agent_name=agent_name,
                patch_generated=str(run_payload["patch_generated"]),
                run_count=int(run_payload["run_count"]),
                error=run_error,
                detailed_results=run_payload,
            )

        log_dir_raw = run_payload["log_dir"]
        log_dir = Path(log_dir_raw) if isinstance(log_dir_raw, str) and log_dir_raw else None
        return self.evaluate_single_test_results_from_logs(
            agent_name=agent_name,
            metadata_path=metadata_path,
            patch_file=patch_file,
            log_dir=log_dir,
        )
