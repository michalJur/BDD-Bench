from __future__ import annotations

from typing import Any

import docker

from bdd_bench.common.container_runtime import create_container_from_image
from bdd_bench.evaluation.container_limits import evaluation_container_run_limits


def secure_evaluation_container_options() -> dict[str, Any]:
    """Docker create options for containers that execute untrusted agent patches."""
    return {
        "privileged": False,
        "network_mode": "none",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        **evaluation_container_run_limits(),
    }


def create_secure_evaluation_container(*, image_tag: str, container_name: str) -> str:
    created_name = create_container_from_image(
        image_tag=image_tag,
        container_name=container_name,
        container_options=secure_evaluation_container_options(),
    )
    verify_evaluation_container_security(created_name)
    return created_name


def verify_evaluation_container_security(container_name: str) -> None:
    """Fail closed when Docker did not apply the requested isolation profile."""
    container = docker.from_env().containers.get(container_name)
    container.reload()
    host_config = container.attrs.get("HostConfig") or {}

    failures: list[str] = []
    if host_config.get("Privileged") is not False:
        failures.append("Privileged must be false")
    if host_config.get("NetworkMode") != "none":
        failures.append("NetworkMode must be none")
    cap_drop = {str(value).upper() for value in host_config.get("CapDrop") or []}
    if "ALL" not in cap_drop:
        failures.append("all Linux capabilities must be dropped")
    security_opt = {str(value).lower() for value in host_config.get("SecurityOpt") or []}
    if not any(value.startswith("no-new-privileges") for value in security_opt):
        failures.append("no-new-privileges must be enabled")
    binds = host_config.get("Binds") or []
    if binds:
        failures.append("host bind mounts are forbidden")
    mounts = container.attrs.get("Mounts") or host_config.get("Mounts") or []
    if mounts:
        failures.append("Docker mounts are forbidden")
    volumes_from = host_config.get("VolumesFrom") or []
    if volumes_from:
        failures.append("inherited Docker volumes are forbidden")

    if failures:
        raise RuntimeError(
            f"Evaluation container {container_name} failed isolation verification: "
            + "; ".join(failures)
        )
