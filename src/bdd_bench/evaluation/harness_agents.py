# mypy: ignore-errors
from __future__ import annotations

import contextlib
import logging
import os
import platform
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from string import Template
from typing import Any

from dotenv import load_dotenv

from bdd_bench.common.command import run_command
from bdd_bench.common.patch_capture import extract_patch_bytes_from_container
from bdd_bench.common.test_file_rules import is_env_path, is_non_bdd_test_path, is_test_path
from bdd_bench.evaluation.harness_metadata import sanitize_label
from bdd_bench.evaluation.patch_utils import (
    extract_patch_paths,
    filter_patch_excluding_paths,
)

load_dotenv()

MINI_SWE_DONE_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MINI_SWE_EXEC_ENV = {
    "CI": "1",
    "DEBIAN_FRONTEND": "noninteractive",
    "GIT_TERMINAL_PROMPT": "0",
    "PIP_NO_INPUT": "1",
    "PYTHONUNBUFFERED": "1",
}
_MINI_SWE_LOG_TEXT_LIMIT = 10_000


def _bounded_mini_swe_log_text(text: str) -> str:
    if len(text) <= _MINI_SWE_LOG_TEXT_LIMIT:
        return text
    edge_length = _MINI_SWE_LOG_TEXT_LIMIT // 2
    omitted = len(text) - (edge_length * 2)
    return text[:edge_length] + f"\n... [{omitted} characters omitted] ...\n" + text[-edge_length:]


def _mini_swe_log_role(message: dict[str, Any]) -> str:
    role = message.get("role")
    if isinstance(role, str) and role.strip():
        return role.strip()
    if message.get("object") == "response":
        return "assistant"
    if message.get("type") == "function_call_output":
        return "tool"
    return "unknown"


def _responses_api_message_text(message: dict[str, Any]) -> str:
    text_parts: list[str] = []
    output = message.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
    return "\n".join(text_parts)


def _write_command_timeout_diagnostic(
    *,
    output_dir: Path,
    instance_id: str,
    agent_name: str,
    container_name: str,
    timeout_seconds: int,
    command: str,
    cwd: str,
    partial_output: str,
    error: str,
    event_timestamp: float | None = None,
    source_trajectory: str | None = None,
    backfilled: bool = False,
) -> Path:
    """Persist one non-terminal agent command-timeout event."""
    event_time = (
        datetime.fromtimestamp(event_timestamp)
        if isinstance(event_timestamp, (int, float))
        else datetime.now()
    )
    event_stamp = event_time.strftime("%Y%m%d_%H%M%S_%f")
    log_path = (
        output_dir
        / "agent_failure_diagnostics"
        / f"{sanitize_label(instance_id)}.command_timeout.{event_stamp}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        f"instance: {instance_id}",
        f"agent: {agent_name}",
        "failure_type: command_timeout",
        "terminal: false",
        f"container: {container_name}",
        f"command_timeout_seconds: {timeout_seconds}",
    ]
    if source_trajectory:
        header.append(f"source_trajectory: {source_trajectory}")
    if backfilled:
        header.append("backfilled: true")
    header.extend((f"error: {error}", f"written_at: {event_time.isoformat()}"))

    rendered_command = f"cwd: {cwd or '<default>'}\n$ {command}\n"
    body = "\n".join(header) + "\n\n== command ==\n" + rendered_command
    if partial_output:
        body += "\n== partial output ==\n" + _bounded_mini_swe_log_text(partial_output)
        if not body.endswith("\n"):
            body += "\n"
    log_path.write_text(body, encoding="utf-8")
    return log_path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip() or raw.strip().lower() in {"default", "none"}:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, 'default', or 'none'.")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


MINI_SWE_DEFAULT_STEP_LIMIT = _env_int("BDD_BENCH_MINI_SWE_MAX_STEPS", 250)
MINI_SWE_DEFAULT_MODEL_NAME = os.environ.get("BDD_BENCH_MINI_SWE_MODEL", "gpt-5-mini")
MINI_SWE_DEFAULT_REASONING_EFFORT = "high"
MINI_SWE_DEFAULT_MAX_OUTPUT_TOKENS = _env_optional_int("BDD_BENCH_MINI_SWE_MAX_OUTPUT_TOKENS")
_MINI_SWE_MAX_OUTPUT_TOKENS_BY_MODEL = {
    "gpt-5.6": 128_000,
    "gpt-5.6-sol": 128_000,
    "gpt-5.6-terra": 128_000,
    "gpt-5.6-luna": 128_000,
    "gpt-5.5": 128_000,
    "gpt-5.4": 128_000,
    "gpt-5.4-mini": 128_000,
    "gpt-5.4-nano": 128_000,
    "gpt-5-mini": 128_000,
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": 64_000,
    "us.anthropic.claude-sonnet-4-6": 64_000,
    "us.anthropic.claude-sonnet-5": 128_000,
    "us.anthropic.claude-fable-5": 128_000,
    "us.anthropic.claude-opus-4-8": 128_000,
    "gemini-3.1-flash-lite": 65_536,
    "gemini-3.5-flash": 65_536,
    "deepseek-v4-flash": 384_000,
    "deepseek-v4-pro": 384_000,
}
MINI_SWE_DEFAULT_COST_LIMIT = _env_float("BDD_BENCH_MINI_SWE_COST_LIMIT", 10.0)
MINI_SWE_DEFAULT_COMMAND_TIMEOUT = _env_int(
    "BDD_BENCH_MINI_SWE_COMMAND_TIMEOUT",
    3600,
)
_MINI_SWE_VALID_REASONING_EFFORT_VALUES = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


def _normalize_mini_swe_reasoning_effort(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized not in _MINI_SWE_VALID_REASONING_EFFORT_VALUES:
        raise ValueError(
            "Invalid mini-swe reasoning_effort "
            f"{value!r}; expected one of {_MINI_SWE_VALID_REASONING_EFFORT_VALUES}"
        )
    return normalized


def resolve_default_reasoning_effort(model_name: str) -> str:
    """Return the reasoning level used consistently by new mini-swe runs."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")
    normalized = model_name.strip().lower()
    if "gemini-3.5-flash" in normalized:
        return "high"
    if "gemini-3.1-flash-lite" in normalized:
        return "high"
    return MINI_SWE_DEFAULT_REASONING_EFFORT


def resolve_max_output_tokens(model_name: str) -> int:
    """Return the provider-documented maximum output tokens for a model."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")
    normalized = model_name.strip().lower()
    provider_model_name = normalized.split("/", 1)[-1]
    configured = _MINI_SWE_MAX_OUTPUT_TOKENS_BY_MODEL.get(provider_model_name)
    if configured is not None:
        return configured

    try:
        import litellm

        model_info = litellm.get_model_info(normalized)
    except Exception as error:
        raise ValueError(
            f"No maximum output-token mapping is available for model {model_name!r}. "
            "Set BDD_BENCH_MINI_SWE_MAX_OUTPUT_TOKENS explicitly."
        ) from error
    discovered = model_info.get("max_output_tokens") or model_info.get("max_tokens")
    if not isinstance(discovered, int) or isinstance(discovered, bool) or discovered <= 0:
        raise ValueError(
            f"No valid maximum output-token value is registered for model {model_name!r}. "
            "Set BDD_BENCH_MINI_SWE_MAX_OUTPUT_TOKENS explicitly."
        )
    return discovered


def resolve_highest_reasoning_effort(model_name: str) -> str:
    """Backward-compatible alias for the fixed default reasoning policy."""
    return resolve_default_reasoning_effort(model_name)


class MiniSweStepLimitExceeded(RuntimeError):
    pass


class MiniSweCostLimitExceeded(RuntimeError):
    pass


def _require_mini_swe_submission(
    run_result: Any,
    *,
    api_calls: int,
    instance_cost: float,
    step_limit: int,
    cost_limit: float,
) -> None:
    exit_status = run_result.get("exit_status") if isinstance(run_result, dict) else None
    if exit_status == "Submitted":
        return
    rendered_status = str(exit_status or "unknown")
    if rendered_status == "LimitsExceeded":
        if cost_limit > 0 and instance_cost >= cost_limit:
            raise MiniSweCostLimitExceeded(
                "mini-swe-agent reached the per-instance cost limit "
                f"(cost=${instance_cost:.6f}, limit=${cost_limit:.6f}, "
                f"api_calls={api_calls})."
            )
        raise MiniSweStepLimitExceeded(
            "mini-swe-agent reached the step limit "
            f"(api_calls={api_calls}, limit={step_limit}, "
            f"cost=${instance_cost:.6f})."
        )
    raise RuntimeError(
        "mini-swe-agent exited without submitting a patch "
        f"(exit_status={rendered_status}, api_calls={api_calls}, "
        f"cost=${instance_cost:.6f})."
    )


AGENT_TASK_PROMPT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "config" / "agent_task_prompt.txt"
)
MINI_SWE_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent / "config" / "mini_swe_system_prompt.txt"
)
MINI_SWE_INSTANCE_PROMPT_PATH = (
    Path(__file__).resolve().parent / "config" / "mini_swe_instance_prompt.txt"
)

_AGENT_TASK_PROMPT_TEMPLATE: Template | None = None


@lru_cache(maxsize=1)
def resolve_mini_swe_agent_version() -> str | None:
    """Return installed mini-swe-agent distribution version if available."""
    for package_name in ("mini-swe-agent", "mini_swe_agent"):
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _get_agent_task_prompt_template() -> Template:
    global _AGENT_TASK_PROMPT_TEMPLATE
    if _AGENT_TASK_PROMPT_TEMPLATE is None:
        if not AGENT_TASK_PROMPT_TEMPLATE_PATH.is_file():
            raise FileNotFoundError(
                f"Missing agent task prompt template: {AGENT_TASK_PROMPT_TEMPLATE_PATH}"
            )
        template_text = AGENT_TASK_PROMPT_TEMPLATE_PATH.read_text(
            encoding="utf-8", errors="replace"
        )
        _AGENT_TASK_PROMPT_TEMPLATE = Template(template_text)
    return _AGENT_TASK_PROMPT_TEMPLATE


@lru_cache(maxsize=1)
def _get_mini_swe_prompt_templates() -> tuple[str, str]:
    paths = (MINI_SWE_SYSTEM_PROMPT_PATH, MINI_SWE_INSTANCE_PROMPT_PATH)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing custom mini-swe prompt template(s): " + ", ".join(missing))
    return tuple(path.read_text(encoding="utf-8", errors="replace") for path in paths)


def _configure_mini_swe_prompt(agent_config: dict[str, Any]) -> None:
    system_template, instance_template = _get_mini_swe_prompt_templates()
    agent_config["system_template"] = system_template
    agent_config["instance_template"] = instance_template


def _wrap_task_for_host_container_execution(
    *,
    task: str,
    container_name: str,
    instance: dict[str, Any],
) -> str:
    repo_hint = instance.get("repo")
    if not isinstance(repo_hint, str) or not repo_hint.strip():
        repo_hint = "."
    elif not repo_hint.startswith("/"):
        repo_hint = f"/{repo_hint}"
    return "\n".join(
        [
            "You are running from the host machine. The target git repo is inside a Docker container.",
            f"Container: {container_name}",
            f"Repo path inside container: {repo_hint}",
            "Execute repository commands only via Docker.",
            "Use this exact command shape for repo commands and test runs:",
            f"docker exec {container_name} bash -lc 'set -euo pipefail; cd {repo_hint}; <command>'",
            "Keep the entire container-side script inside the final single-quoted `bash -lc '...'` argument.",
            'Do NOT use `docker exec ... bash -lc "..."` or mixed nested quoting for container commands.',
            "Do not let the host shell expand container variables like `$PWD` or `${LC_ALL:-...}`.",
            "Do not edit host workspace files directly.",
            "All /tmp/... paths in task instructions are container paths.",
            "If you need to create/read those files from host context, use docker exec.",
            "Never write artifacts with host commands like: cat > /tmp/...",
            "Use the task instructions below as the source of truth:",
            "",
            task,
        ]
    )


class Agent(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def requires_runtime_context(self) -> bool:
        return False

    def preflight_host_environment(self) -> str | None:
        return None

    @abstractmethod
    def generate_patch(
        self,
        instance: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> str | bytes:
        pass


class DummyAgent(Agent):
    """Dummy agent that always returns an empty patch.

    Use this as a baseline or for testing the evaluation harness.
    An empty patch should result in test failures for valid instances.
    """

    @property
    def name(self) -> str:
        return "dummy"

    def generate_patch(
        self,
        instance: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        del instance, runtime_context
        return ""


class GoldenAgent(Agent):
    """Agent that returns the golden (correct) patch.

    Use this to verify the harness works correctly - should always pass.
    """

    @property
    def name(self) -> str:
        return "golden"

    def generate_patch(
        self,
        instance: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        del runtime_context
        patch_path = Path(instance["patches"]["golden_patch"])
        if not patch_path.is_file():
            return ""
        # Read as bytes to preserve exact line endings (including mixed CRLF/LF)
        patch_text = patch_path.read_bytes().decode("utf-8")
        repo = instance.get("repo")
        excluded_paths = {
            path
            for path in extract_patch_paths(patch_text)
            if (
                is_test_path(path, repo=repo)
                or is_non_bdd_test_path(path, repo=repo)
                or is_env_path(path, repo=repo)
            )
        }
        if not excluded_paths:
            return patch_text
        return filter_patch_excluding_paths(patch_text, excluded_paths)


class _ExistingContainerEnvironment:
    def __init__(
        self,
        container_name: str,
        *,
        environment: dict[str, str] | None = None,
        timeout: int = MINI_SWE_DEFAULT_COMMAND_TIMEOUT,
        stream_to_console: bool = False,
        live_log_path: Path | None = None,
        diagnostic_output_dir: Path | None = None,
        instance_id: str = "",
        agent_name: str = "mini-swe",
    ):
        self.container_name = container_name
        self.environment = dict(environment or {})
        self.timeout = timeout
        self.stream_to_console = stream_to_console
        self.live_log_path = live_log_path
        self.diagnostic_output_dir = diagnostic_output_dir
        self.instance_id = instance_id
        self.agent_name = agent_name
        if self.live_log_path is not None:
            self.live_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _restart_after_timeout(self, effective_timeout: int) -> str:
        """Reset the container process tree while preserving its writable layer."""
        try:
            completed = subprocess.run(
                ["docker", "restart", "--time", "0", self.container_name],
                check=False,
                timeout=60,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            self._remove_after_failed_restart()
            raise RuntimeError(
                f"Command timed out after {effective_timeout}s and the container "
                f"could not be restarted: {error}"
            ) from error

        if completed.returncode != 0:
            details = completed.stdout.strip() or f"exit code {completed.returncode}"
            self._remove_after_failed_restart()
            raise RuntimeError(
                f"Command timed out after {effective_timeout}s and the container "
                f"could not be restarted: {details}"
            )

        return (
            f"Command timed out after {effective_timeout}s. The container was "
            "restarted to terminate remaining processes; filesystem changes were preserved."
        )

    def _remove_after_failed_restart(self) -> None:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["docker", "rm", "--force", self.container_name],
                check=False,
                timeout=60,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def _restart_and_record_timeout(
        self,
        *,
        effective_timeout: int,
        command: str,
        cwd: str,
        partial_output: str,
    ) -> str:
        event_timestamp = time.time()
        try:
            exception_info = self._restart_after_timeout(effective_timeout)
        except Exception as error:
            self._record_timeout(
                effective_timeout=effective_timeout,
                command=command,
                cwd=cwd,
                partial_output=partial_output,
                error=str(error),
                event_timestamp=event_timestamp,
            )
            raise
        self._record_timeout(
            effective_timeout=effective_timeout,
            command=command,
            cwd=cwd,
            partial_output=partial_output,
            error=exception_info,
            event_timestamp=event_timestamp,
        )
        return exception_info

    def _record_timeout(
        self,
        *,
        effective_timeout: int,
        command: str,
        cwd: str,
        partial_output: str,
        error: str,
        event_timestamp: float,
    ) -> None:
        if self.diagnostic_output_dir is None or not self.instance_id:
            return
        try:
            _write_command_timeout_diagnostic(
                output_dir=self.diagnostic_output_dir,
                instance_id=self.instance_id,
                agent_name=self.agent_name,
                container_name=self.container_name,
                timeout_seconds=effective_timeout,
                command=command,
                cwd=cwd,
                partial_output=partial_output,
                error=error,
                event_timestamp=event_timestamp,
            )
        except OSError as diagnostic_error:
            logging.warning(
                "Could not write command-timeout diagnostic for %s: %s",
                self.instance_id,
                diagnostic_error,
            )

    @staticmethod
    def _execution_error_output(error: Exception) -> dict[str, Any]:
        return {
            "output": "",
            "returncode": -1,
            "exception_info": f"An error occurred while executing the command: {error}",
            "extra": {"exception_type": type(error).__name__, "exception": str(error)},
        }

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = str(action["command"])
        if "\x00" in command:
            return self._execution_error_output(
                ValueError(
                    "Command contains an embedded NUL byte and cannot be executed. "
                    "Reissue it without NUL characters; check for malformed Unicode "
                    "escapes such as \\u00003c."
                )
            )

        effective_timeout = timeout or self.timeout
        docker_command = ["docker", "exec"]
        for key, value in sorted(self.environment.items()):
            docker_command.extend(["-e", f"{key}={value}"])
        if cwd:
            docker_command.extend(["-w", cwd])
        docker_command.extend([self.container_name, "bash", "-lc", command])

        output: dict[str, Any]
        if not self.stream_to_console and self.live_log_path is None:
            try:
                completed = subprocess.run(
                    docker_command,
                    check=False,
                    timeout=effective_timeout,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                output = {
                    "output": completed.stdout,
                    "returncode": completed.returncode,
                    "exception_info": "",
                }
            except subprocess.TimeoutExpired as error:
                timed_out_output = error.stdout if isinstance(error.stdout, str) else ""
                exception_info = self._restart_and_record_timeout(
                    effective_timeout=effective_timeout,
                    command=command,
                    cwd=cwd,
                    partial_output=timed_out_output,
                )
                output = {
                    "output": timed_out_output,
                    "returncode": -1,
                    "exception_info": exception_info,
                    "extra": {
                        "exception_type": "TimeoutExpired",
                        "exception": exception_info,
                    },
                }
            except Exception as error:
                output = self._execution_error_output(error)
        else:
            header = f"\n[mini-swe][command]\n$ {command}\n"
            self._emit(header)
            try:
                process = subprocess.Popen(
                    docker_command,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                chunks: list[str] = []
                timed_out = False

                def _drain_stdout() -> None:
                    if process.stdout is None:
                        return
                    try:
                        for line in process.stdout:
                            chunks.append(line)
                            self._emit(line)
                    finally:
                        process.stdout.close()

                reader = threading.Thread(target=_drain_stdout, daemon=True)
                reader.start()
                try:
                    returncode = process.wait(timeout=effective_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process.kill()
                    returncode = -1
                finally:
                    try:
                        process.wait(timeout=5)
                    except Exception:
                        pass
                    reader.join(timeout=5)

                exception_info = ""
                extra: dict[str, Any] = {}
                if timed_out:
                    exception_info = self._restart_and_record_timeout(
                        effective_timeout=effective_timeout,
                        command=command,
                        cwd=cwd,
                        partial_output="".join(chunks),
                    )
                    extra = {
                        "exception_type": "TimeoutExpired",
                        "exception": exception_info,
                    }

                output = {
                    "output": "".join(chunks),
                    "returncode": returncode,
                    "exception_info": exception_info,
                }
                if extra:
                    output["extra"] = extra
            except RuntimeError:
                raise
            except Exception as error:
                output = self._execution_error_output(error)

        lines = output["output"].lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == MINI_SWE_DONE_MARKER and output["returncode"] == 0:
            submission = "".join(lines[1:])
            from minisweagent.exceptions import Submitted

            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

        return output

    def _emit(self, text: str) -> None:
        if self.stream_to_console and text:
            print(text, end="", flush=True)
        if self.live_log_path is not None and text:
            with self.live_log_path.open("a", encoding="utf-8") as handle:
                handle.write(text)

    def get_template_vars(self) -> dict[str, Any]:
        vars_map = platform.uname()._asdict()
        vars_map["working_dir"] = ""
        vars_map["container_name"] = self.container_name
        return vars_map

    def serialize(self) -> dict[str, str]:
        return {"type": "existing_container", "container_name": self.container_name}

    def cleanup(self) -> None:
        return None


class MiniSweAgent(Agent):
    def __init__(
        self,
        *,
        step_limit: int = MINI_SWE_DEFAULT_STEP_LIMIT,
        model_name: str | None = MINI_SWE_DEFAULT_MODEL_NAME,
        command_timeout: int = MINI_SWE_DEFAULT_COMMAND_TIMEOUT,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = MINI_SWE_DEFAULT_MAX_OUTPUT_TOKENS,
        cost_limit: float = MINI_SWE_DEFAULT_COST_LIMIT,
    ):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "MiniSweAgent requires a non-empty model name. "
                "Pass model_name explicitly or set BDD_BENCH_MINI_SWE_MODEL."
            )
        self.step_limit = max(1, step_limit)
        self.model_name = model_name
        self.command_timeout = max(1, command_timeout)
        self.reasoning_effort = _normalize_mini_swe_reasoning_effort(
            reasoning_effort or resolve_default_reasoning_effort(model_name)
        )
        self.max_output_tokens = max(
            1,
            max_output_tokens
            if max_output_tokens is not None
            else resolve_max_output_tokens(model_name),
        )
        self.cost_limit = max(0.0, float(cost_limit))

    @property
    def name(self) -> str:
        return "mini-swe"

    @property
    def requires_runtime_context(self) -> bool:
        return True

    def generate_patch(
        self,
        instance: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        runtime_context = runtime_context or {}
        container_name = runtime_context["container_name"].strip()
        patch_base_ref = runtime_context.get("patch_base_ref")

        try:
            from minisweagent.agents.default import DefaultAgent
            from minisweagent.config import get_config_from_spec
            from minisweagent.models import get_model
        except ImportError as error:
            raise RuntimeError(
                "mini-swe-agent is not installed. Add it to your environment and retry."
            ) from error

        task = self._build_task(instance, runtime_context)
        config = get_config_from_spec("mini.yaml")
        model_config = config["model"]
        self._configure_model(model_config)

        agent_config = config["agent"]
        _configure_mini_swe_prompt(agent_config)
        agent_config["step_limit"] = self.step_limit
        agent_config["cost_limit"] = self.cost_limit
        trajectory_json_path, trajectory_log_path, live_stream_log_path = self._trajectory_paths(
            runtime_context
        )
        agent_config["output_path"] = trajectory_json_path

        live_stream = bool(runtime_context.get("live_stream", False))
        if live_stream:
            print(f"[mini-swe] Live stream log: {live_stream_log_path}", flush=True)

        model = get_model(config=model_config)
        environment = _ExistingContainerEnvironment(
            container_name=container_name,
            environment=MINI_SWE_EXEC_ENV,
            timeout=self.command_timeout,
            stream_to_console=live_stream,
            live_log_path=live_stream_log_path if live_stream else None,
            diagnostic_output_dir=Path(str(runtime_context["output_dir"])),
            instance_id=str(runtime_context["instance_id"]),
            agent_name=self.name,
        )
        agent = DefaultAgent(model=model, env=environment, **agent_config)
        run_result: dict[str, Any] | None = None
        try:
            run_result = agent.run(task=task)
        finally:
            trajectory = agent.save(trajectory_json_path)
            self._write_and_print_trajectory_log(
                trajectory,
                trajectory_json_path=trajectory_json_path,
                log_path=trajectory_log_path,
            )
        _require_mini_swe_submission(
            run_result,
            api_calls=agent.n_calls,
            instance_cost=agent.cost,
            step_limit=self.step_limit,
            cost_limit=self.cost_limit,
        )
        return self._extract_patch(
            container_name,
            patch_base_ref=patch_base_ref,
            label=str(instance["instance_id"]),
        )

    def _configure_model(self, model_config: dict[str, Any]) -> None:
        model_config["model_name"] = self.model_name
        uses_responses_api = self.model_name.strip().lower().startswith("openai/gpt-5")
        if uses_responses_api:
            model_config["model_class"] = "litellm_response"
        else:
            model_config.pop("model_class", None)
        model_kwargs = model_config.setdefault("model_kwargs", {})
        if not isinstance(model_kwargs, dict):
            raise ValueError("mini-swe model.model_kwargs must be a mapping.")
        if self.reasoning_effort is None:
            model_kwargs.pop("reasoning_effort", None)
        else:
            model_kwargs["reasoning_effort"] = self.reasoning_effort
        if uses_responses_api:
            model_kwargs.pop("max_tokens", None)
            model_kwargs["max_output_tokens"] = self.max_output_tokens
        else:
            model_kwargs.pop("max_output_tokens", None)
            model_kwargs["max_tokens"] = self.max_output_tokens

    def _build_task(self, instance: dict[str, Any], runtime_context: dict[str, Any]) -> str:
        test_command = str(runtime_context["test_command"]).strip()
        del instance
        template = _get_agent_task_prompt_template()
        substitution_vars: dict[str, str] = {
            "test_command": test_command,
            "done_marker": MINI_SWE_DONE_MARKER,
        }
        return template.substitute(substitution_vars)

    def _trajectory_paths(self, runtime_context: dict[str, Any]) -> tuple[Path, Path, Path]:
        output_dir = runtime_context["output_dir"]
        base_dir = Path(output_dir)

        instance_id = runtime_context["instance_id"]
        safe_instance_id = instance_id.replace("/", "_").replace("#", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        trajectory_dir = base_dir / "trajectories" / safe_instance_id
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        trajectory_json_path = trajectory_dir / f"{timestamp}.traj.json"
        trajectory_log_path = trajectory_dir / f"{timestamp}.log"
        live_stream_log_path = trajectory_dir / f"{timestamp}.live.log"
        return trajectory_json_path, trajectory_log_path, live_stream_log_path

    def _write_and_print_trajectory_log(
        self,
        trajectory: dict[str, Any],
        *,
        trajectory_json_path: Path,
        log_path: Path,
    ) -> None:
        messages = trajectory.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        lines: list[str] = []
        lines.append("=" * 70)
        lines.append(f"MINI-SWE TRAJECTORY ({len(messages)} messages)")
        lines.append("=" * 70)

        for index, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                continue
            role = _mini_swe_log_role(message)
            content = message.get("content")
            lines.append(f"[{index}] role={role}")
            if isinstance(content, str) and content.strip():
                lines.append(_bounded_mini_swe_log_text(content))
            elif message.get("object") == "response":
                response_text = _responses_api_message_text(message)
                if response_text:
                    lines.append(_bounded_mini_swe_log_text(response_text))

            extra = message.get("extra")
            if isinstance(extra, dict):
                actions = extra.get("actions")
                if isinstance(actions, list) and actions:
                    lines.append("actions:")
                    for action in actions:
                        if not isinstance(action, dict):
                            continue
                        command = action.get("command")
                        if isinstance(command, str) and command.strip():
                            lines.append(f"  $ {command}")
                if message.get("type") == "function_call_output":
                    returncode = extra.get("returncode")
                    if returncode is not None:
                        lines.append(f"returncode: {returncode}")
                    raw_output = extra.get("raw_output")
                    if isinstance(raw_output, str) and raw_output:
                        lines.append("output:")
                        lines.append(_bounded_mini_swe_log_text(raw_output))
                    exception_info = extra.get("exception_info")
                    if isinstance(exception_info, str) and exception_info.strip():
                        lines.append(f"exception: {exception_info}")
            lines.append("-" * 70)

        rendered = "\n".join(lines).rstrip() + "\n"
        log_path.write_text(rendered, encoding="utf-8")
        print(f"[mini-swe] Trajectory JSON: {trajectory_json_path}", flush=True)
        print(f"[mini-swe] Trajectory log: {log_path}", flush=True)

    def _extract_patch(
        self,
        container_name: str,
        patch_base_ref: str | None = None,
        *,
        label: str | None = None,
    ) -> bytes:
        base_ref = patch_base_ref.strip() if isinstance(patch_base_ref, str) else ""
        if not base_ref:
            base_ref = "HEAD"
        if any(char.isspace() for char in base_ref):
            raise ValueError(f"Invalid patch base ref: {base_ref!r}")
        return extract_patch_bytes_from_container(
            command_runner=run_command,
            container_name=container_name,
            base_ref=base_ref,
            label=label or container_name,
            patch_label="agent patch",
            allow_empty_patch=True,
        )


def _normalize_agent_name(agent_name: str) -> str:
    if agent_name == "mini_swe":
        return "mini-swe"
    return agent_name


def _parse_mini_swe_settings(
    mini_swe_settings: dict[str, Any] | None,
) -> tuple[str, int, int] | None:
    if mini_swe_settings is None:
        return None

    model_name = mini_swe_settings.get("model_name")
    step_limit = mini_swe_settings.get("step_limit")
    command_timeout = mini_swe_settings.get("command_timeout_seconds")
    max_output_tokens = mini_swe_settings.get(
        "max_output_tokens",
        MINI_SWE_DEFAULT_MAX_OUTPUT_TOKENS,
    )
    cost_limit = mini_swe_settings.get("cost_limit", MINI_SWE_DEFAULT_COST_LIMIT)
    if (
        not isinstance(model_name, str)
        or not model_name.strip()
        or not isinstance(step_limit, int)
        or step_limit <= 0
        or not isinstance(command_timeout, int)
        or command_timeout <= 0
        or (
            max_output_tokens is not None
            and (not isinstance(max_output_tokens, int) or max_output_tokens <= 0)
        )
        or not isinstance(cost_limit, (int, float))
        or cost_limit < 0
    ):
        raise ValueError("Invalid mini_swe_settings for runtime agent.")
    return model_name, step_limit, command_timeout


def get_agent(
    agent_name: str,
    *,
    mini_swe_settings: dict[str, Any] | None = None,
) -> Agent:
    """Get an agent instance by name.

    For mini-swe, optional ``mini_swe_settings`` can be provided to pin
    model/step/timeout values loaded from a saved run config.
    """
    normalized_agent_name = _normalize_agent_name(agent_name)
    if normalized_agent_name == "dummy":
        return DummyAgent()
    if normalized_agent_name == "golden":
        return GoldenAgent()
    if normalized_agent_name == "mini-swe":
        parsed_settings = _parse_mini_swe_settings(mini_swe_settings)
        if parsed_settings is None:
            return MiniSweAgent()
        model_name, step_limit, command_timeout = parsed_settings
        return MiniSweAgent(
            model_name=model_name,
            step_limit=step_limit,
            command_timeout=command_timeout,
            reasoning_effort=mini_swe_settings.get("reasoning_effort"),
            max_output_tokens=mini_swe_settings.get(
                "max_output_tokens",
                MINI_SWE_DEFAULT_MAX_OUTPUT_TOKENS,
            ),
            cost_limit=float(mini_swe_settings.get("cost_limit", MINI_SWE_DEFAULT_COST_LIMIT)),
        )
    available = "dummy, golden, mini-swe, mini_swe"
    raise ValueError(f"Unknown agent: {agent_name}. Available: {available}")
