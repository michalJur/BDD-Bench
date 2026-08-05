"""Shared Docker container and Git worktree operations."""

from __future__ import annotations

import logging
import shlex
import time
from pathlib import Path
from typing import Any

import docker

from bdd_bench.common.command import run_command
from bdd_bench.common.docker_client import exec_streaming, remove_container


def create_container_from_image(
    image_tag: str,
    *,
    container_name: str,
    container_options: dict[str, Any] | None = None,
) -> str:
    """Start a long-lived container from an image, retrying a stale name once."""
    client = docker.from_env()
    net_name = "offline_testing_net"
    matching_networks = [network for network in client.networks.list(names=[net_name])]
    exact_name_matches = [network for network in matching_networks if network.name == net_name]

    if not exact_name_matches:
        network = client.networks.create(net_name, driver="bridge", internal=True)
    else:
        exact_name_matches.sort(key=lambda item: (item.attrs.get("Created", ""), item.id))
        network = exact_name_matches[0]
        if len(exact_name_matches) > 1:
            logging.warning(
                "Detected %s Docker networks named '%s'. Using network ID %s.",
                len(exact_name_matches),
                net_name,
                network.id,
            )
            for duplicate in exact_name_matches[1:]:
                try:
                    duplicate.reload()
                    if duplicate.attrs.get("Containers"):
                        logging.warning(
                            "Keeping duplicate network %s (has attached containers).", duplicate.id
                        )
                        continue
                    duplicate.remove()
                    logging.warning("Removed unused duplicate network %s.", duplicate.id)
                except docker.errors.APIError as error:
                    logging.warning(
                        "Failed to remove duplicate network %s: %s", duplicate.id, error
                    )

    create_kwargs = {
        "image": image_tag,
        "command": ["sleep", "infinity"],
        "name": container_name,
        "detach": True,
        "shm_size": "2g",
        "privileged": True,
        "auto_remove": True,
        "tty": True,
        "stdin_open": True,
        "network": network.id,
    }
    requested_options = dict(container_options or {})
    create_kwargs.update(requested_options)
    if "network_mode" in requested_options:
        create_kwargs.pop("network", None)

    try:
        container = client.containers.run(**create_kwargs)
    except docker.errors.APIError as error:
        response = getattr(error, "response", None)
        response_text = str(getattr(response, "text", "") or "")
        explanation = getattr(error, "explanation", None)
        message = (
            explanation.lower()
            if isinstance(explanation, str) and explanation.strip()
            else " ".join(str(part) for part in getattr(error, "args", ()) if part).lower()
        )
        conflict_text = f"{response_text} {message}".lower()
        if (
            getattr(error, "status_code", None) == 409
            and "name" in conflict_text
            and "in use" in conflict_text
        ):
            logging.warning(
                "Container name %s was still in use during create; removing and retrying once.",
                container_name,
            )
            remove_container(container_name)
            time.sleep(0.2)
            try:
                container = client.containers.run(**create_kwargs)
            except docker.errors.APIError as retry_error:
                raise SystemExit(
                    f"Failed to create container {container_name} from image {image_tag}: "
                    f"{retry_error!r}"
                ) from retry_error
        else:
            raise SystemExit(
                f"Failed to create container {container_name} from image {image_tag}: {error!r}"
            ) from error

    logging.info("Started container %s from image %s", container.name, image_tag)
    return container.name


def _git_repo_command_script(repo_path: str | None, command: str) -> str:
    """Run ``command`` at the resolved Git worktree root inside a container."""
    repo_hint = (repo_path or ".").strip() or "."
    quoted_hint = shlex.quote(repo_hint)
    return f"""set -euo pipefail
repo_hint={quoted_hint}
for candidate in "$repo_hint" "/$repo_hint" "."; do
  [ -n "$candidate" ] || continue
  if resolved_repo_path="$(git -C "$candidate" rev-parse --show-toplevel </dev/null 2>/dev/null)"; then
    {command}
    exit 0
  fi
done
git -C "$repo_hint" rev-parse --show-toplevel >/dev/null
"""


def reset_repository(
    container_name: str,
    repo_path: str,
    log_path: Path | None = None,
    git_ref: str = "initial-pr-commit",
    stream_to_console: bool | None = None,
) -> None:
    """Reset a container worktree to a named Git ref."""
    script = _git_repo_command_script(
        repo_path,
        f'git -C "$resolved_repo_path" checkout --force {shlex.quote(git_ref)}',
    )
    logging.info("Resetting repository in container to reference: %s...", git_ref)
    exec_streaming(
        container_name=container_name,
        command=["bash", "-lc", script],
        log_path=log_path,
        stream_to_console=stream_to_console,
    )


def apply_patch(
    container_name: str,
    patch_path: str,
    *,
    allow_empty_patch: bool = False,
    repo_path: str | None = None,
) -> bool:
    """Apply a patch inside a container worktree when it exists and is non-empty."""
    if not patch_path:
        return False
    patch_file = Path(patch_path)
    if not patch_file.is_file():
        logging.error("Patch file %s does not exist; skipping.", patch_path)
        return False
    patch_content = patch_file.read_bytes()
    if not patch_content.strip():
        logging.info("Patch file %s is empty; skipping.", patch_path)
        return allow_empty_patch

    script = _git_repo_command_script(
        repo_path,
        'git -C "$resolved_repo_path" apply --whitespace=nowarn "$patch_file"',
    )
    script = f"""set -euo pipefail
patch_file="$(mktemp)"
trap 'rm -f "$patch_file"' EXIT
cat > "$patch_file"
{script}"""
    run_command(
        ["docker", "exec", "-i", container_name, "bash", "-c", script], input_bytes=patch_content
    )
    return True
