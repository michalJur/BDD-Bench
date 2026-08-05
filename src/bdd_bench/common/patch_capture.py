from __future__ import annotations

import shlex
import subprocess
from typing import Callable


RunCommand = Callable[..., subprocess.CompletedProcess]


def _decode_stdout_bytes(stdout: bytes | str | None) -> str:
    if stdout is None:
        return ""
    if isinstance(stdout, bytes):
        return stdout.decode("utf-8", errors="replace")
    return str(stdout)


def extract_patch_bytes_from_container(
    *,
    command_runner: RunCommand,
    container_name: str,
    base_ref: str,
    label: str,
    patch_label: str,
    repo_path: str | None = None,
    allow_empty_patch: bool = False,
) -> bytes:
    repo_expression = (
        shlex.quote(repo_path)
        if isinstance(repo_path, str) and repo_path.strip()
        else '"$repo_root"'
    )
    repo_root_prefix = ""
    if repo_expression == '"$repo_root"':
        repo_root_prefix = "repo_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd); "

    try:
        command_runner(
            [
                "docker",
                "exec",
                container_name,
                "bash",
                "-lc",
                f"{repo_root_prefix}git -C {repo_expression} add -N --all",
            ],
        )
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            f"Failed preparing {patch_label} diff index for {label}: {error}"
        ) from error

    try:
        untracked_completed = command_runner(
            [
                "docker",
                "exec",
                container_name,
                "bash",
                "-lc",
                f"{repo_root_prefix}git -C {repo_expression} ls-files --others --exclude-standard",
            ],
        )
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            f"Failed checking untracked files before extracting {patch_label} for {label}: {error}"
        ) from error

    untracked_paths = [
        line.strip()
        for line in _decode_stdout_bytes(untracked_completed.stdout).splitlines()
        if line.strip()
    ]
    if untracked_paths:
        listed_paths = ", ".join(untracked_paths)
        raise RuntimeError(
            f"Untracked files remain before extracting {patch_label} for {label}: {listed_paths}"
        )

    try:
        completed = command_runner(
            [
                "docker",
                "exec",
                container_name,
                "bash",
                "-lc",
                (
                    f"{repo_root_prefix}git -C {repo_expression} diff --binary -M -C "
                    f"{shlex.quote(base_ref)}"
                ),
            ],
        )
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            f"Failed extracting {patch_label} diff for {label} from base {base_ref}: {error}"
        ) from error

    patch_bytes = completed.stdout
    if isinstance(patch_bytes, str):
        patch_bytes = patch_bytes.encode("utf-8")
    if not patch_bytes.strip() and not allow_empty_patch:
        raise RuntimeError(
            f"Generated {patch_label} for {label} is empty when diffing from {base_ref}"
        )
    return patch_bytes
