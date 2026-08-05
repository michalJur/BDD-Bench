from __future__ import annotations

import logging
import os
import subprocess

CONTAINER_MEMORY_ENV_VAR = "BDD_BENCH_EVAL_CONTAINER_MEMORY"
CONTAINER_MEMORY_SWAP_ENV_VAR = "BDD_BENCH_EVAL_CONTAINER_MEMORY_SWAP"
CONTAINER_PIDS_ENV_VAR = "BDD_BENCH_EVAL_CONTAINER_PIDS"
DEFAULT_CONTAINER_MEMORY = "8g"
DEFAULT_CONTAINER_PIDS = "1024"


def evaluation_container_run_limits() -> dict[str, object]:
    """Return resource limits suitable for Docker ``containers.run``."""
    memory = os.environ.get(CONTAINER_MEMORY_ENV_VAR, DEFAULT_CONTAINER_MEMORY).strip()
    memory_swap = os.environ.get(CONTAINER_MEMORY_SWAP_ENV_VAR, memory).strip()
    pids = os.environ.get(CONTAINER_PIDS_ENV_VAR, DEFAULT_CONTAINER_PIDS).strip()

    options: dict[str, object] = {}
    if memory:
        options["mem_limit"] = memory
    if memory_swap:
        options["memswap_limit"] = memory_swap
    if pids:
        try:
            options["pids_limit"] = int(pids)
        except ValueError:
            logging.warning(
                "Ignoring invalid %s value %r; expected an integer.",
                CONTAINER_PIDS_ENV_VAR,
                pids,
            )
    return options


def apply_evaluation_container_limits(container_name: str) -> None:
    """Apply evaluation-only Docker resource limits to a running container."""
    command = _docker_update_command(container_name)
    if command is None:
        logging.info("Evaluation container limits disabled for %s", container_name)
        return

    completed = subprocess.run(
        command,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to apply evaluation container limits to "
            f"{container_name}: {completed.stdout.strip()}"
        )
    logging.info("Applied evaluation container limits to %s", container_name)


def container_oom_kill_count(container_name: str) -> int:
    """Return the cgroup v2 memory OOM-kill count for a container (0 if unreadable)."""
    try:
        completed = subprocess.run(
            ["docker", "exec", container_name, "cat", "/sys/fs/cgroup/memory.events"],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if completed.returncode != 0:
        return 0
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "oom_kill":
            try:
                return int(parts[1])
            except ValueError:
                return 0
    return 0


def _docker_update_command(container_name: str) -> list[str] | None:
    options = evaluation_container_run_limits()

    command = ["docker", "update"]
    memory = options.get("mem_limit")
    if isinstance(memory, str):
        command.extend(["--memory", memory])
    memory_swap = options.get("memswap_limit")
    if isinstance(memory_swap, str):
        command.extend(["--memory-swap", memory_swap])
    pids = options.get("pids_limit")
    if isinstance(pids, int):
        command.extend(["--pids-limit", str(pids)])

    if len(command) == 2:
        return None
    command.append(container_name)
    return command
