from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv

from bdd_bench.common.docker_client import (
    image_exists,
    pull_image,
)

load_dotenv()

# Bump this whenever image build/runtime semantics change in a way that should
# invalidate previously published dataset images.
DATASET_VERSION = "3"
IMAGE_REGISTRY_ENV_VAR = "BDD_BENCH_IMAGE_REGISTRY"

_IMAGE_SEGMENT_PATTERN = re.compile(r"[^a-z0-9._-]+")


def _sanitize_image_segment(value: str) -> str:
    normalized = value.strip().lower()
    sanitized = _IMAGE_SEGMENT_PATTERN.sub("-", normalized).strip("-.")
    return sanitized or "unknown"


def resolve_image_registry_repository() -> str | None:
    value = os.environ.get(IMAGE_REGISTRY_ENV_VAR)
    if value is None:
        return None
    normalized = value.strip().rstrip("/")
    return normalized or None


def _render_image_ref(repository: str, tag: str) -> str:
    return f"{repository}:{tag}"


def image_schema_tag(schema_version: str = DATASET_VERSION) -> str:
    version_segment = _sanitize_image_segment(schema_version)
    return f"v{version_segment}"


def _normalize_repository(repository: str) -> str:
    normalized_repository = repository.strip().rstrip("/")
    if not normalized_repository:
        raise ValueError("Image repository cannot be empty.")
    return normalized_repository


def _identity_segment(
    *,
    owner: str,
    repo: str,
    identity_kind: str,
    identity_number: int,
) -> str:
    owner_segment = _sanitize_image_segment(owner)
    repo_segment = _sanitize_image_segment(repo)
    identity_segment = _sanitize_image_segment(f"{identity_kind}-{identity_number}")
    return "-".join([owner_segment, repo_segment, identity_segment])


def _pr_identity_segment(
    *,
    owner: str,
    repo: str,
    pr_number: int,
) -> str:
    return _identity_segment(
        owner=owner,
        repo=repo,
        identity_kind="pr",
        identity_number=pr_number,
    )


def _registry_repository_with_identity(
    registry_repository: str,
    *,
    owner: str,
    repo: str,
    identity_kind: str,
    identity_number: int,
) -> str:
    normalized_repository = _normalize_repository(registry_repository)
    identity_segment = _identity_segment(
        owner=owner,
        repo=repo,
        identity_kind=identity_kind,
        identity_number=identity_number,
    )
    return f"{normalized_repository}-{identity_segment}"


def build_registry_image_ref(
    registry_repository: str,
    *,
    owner: str,
    repo: str,
    pr_number: int,
) -> str:
    return _registry_repository_with_identity(
        registry_repository,
        owner=owner,
        repo=repo,
        identity_kind="pr",
        identity_number=pr_number,
    )


def build_registry_image_ref_for_identity(
    registry_repository: str,
    *,
    owner: str,
    repo: str,
    identity_kind: str,
    identity_number: int,
) -> str:
    return _registry_repository_with_identity(
        registry_repository,
        owner=owner,
        repo=repo,
        identity_kind=identity_kind,
        identity_number=identity_number,
    )


def build_registry_version_ref(
    registry_repository: str,
    *,
    owner: str,
    repo: str,
    pr_number: int,
    schema_version: str = DATASET_VERSION,
) -> str:
    repository = build_registry_image_ref(
        registry_repository,
        owner=owner,
        repo=repo,
        pr_number=pr_number,
    )
    return _render_image_ref(
        repository,
        image_schema_tag(schema_version),
    )


def build_registry_version_ref_for_identity(
    registry_repository: str,
    *,
    owner: str,
    repo: str,
    identity_kind: str,
    identity_number: int,
    schema_version: str = DATASET_VERSION,
) -> str:
    repository = build_registry_image_ref_for_identity(
        registry_repository,
        owner=owner,
        repo=repo,
        identity_kind=identity_kind,
        identity_number=identity_number,
    )
    return _render_image_ref(
        repository,
        image_schema_tag(schema_version),
    )


def extract_registry_ref(metadata: dict[str, Any]) -> str | None:
    top_level_ref = metadata.get("image_registry_ref")
    if isinstance(top_level_ref, str) and top_level_ref.strip():
        return top_level_ref.strip()

    fallback_ref = metadata.get("registry_ref")
    if isinstance(fallback_ref, str) and fallback_ref.strip():
        return fallback_ref.strip()

    return None


def ensure_local_image_available(
    local_tag: str,
    *,
    pull_if_missing: bool = True,
    missing_message: str | None = None,
) -> bool:
    if image_exists(local_tag):
        return False

    if not pull_if_missing:
        raise SystemExit(
            missing_message
            or f"Required local Docker image not found: {local_tag}. Re-run the image creation stage first."
        )

    pull_image(
        local_tag,
        local_tag=local_tag,
    )
    return True
