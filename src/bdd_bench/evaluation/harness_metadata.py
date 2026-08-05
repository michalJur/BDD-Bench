from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from bdd_bench.common.selection import matches_repo
from bdd_bench.evaluation.harness_models import MetadataError

_INSTANCE_ID_PATTERN = re.compile(
    r".+-chain-[1-9]\d*-stage-[1-9]\d*-initial-([0-9a-f]{12})-final-([0-9a-f]{12})"
)
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def load_metadata(path: Path) -> dict[str, Any]:
    """Load metadata from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MetadataError("Metadata must be a JSON object.")
    return payload


def _contains_pr_identity(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if (
                normalized == "entry"
                or normalized == "pr"
                or normalized.startswith(("pr_", "pull_request"))
            ):
                return True
            if _contains_pr_identity(child):
                return True
    elif isinstance(value, list):
        return any(_contains_pr_identity(item) for item in value)
    return False


def validate_metadata(data: dict[str, Any]) -> None:
    """Validate the strict post-publication evaluation schema."""
    if _contains_pr_identity(data):
        raise MetadataError("Evaluation metadata must not contain construction or PR identity.")
    required = (
        "instance_id",
        "owner",
        "repo",
        "chain_id",
        "chain_stage_index",
        "initial_commit_sha",
        "final_commit_sha",
        "image_tag",
        "image_digest",
    )
    try:
        values = {field: data[field] for field in required}
        patches = data["patches"]
    except KeyError as error:
        raise MetadataError(f"Metadata missing required field: {error}") from error
    if not isinstance(patches, dict):
        raise MetadataError("patches must be an object.")
    try:
        patches["golden_test_patch"]
        patches["golden_patch"]
        evaluation_test_patch = data.get("evaluation_test_patch")
        if isinstance(evaluation_test_patch, str) and evaluation_test_patch.strip():
            patches[evaluation_test_patch.strip()]
    except KeyError as error:
        raise MetadataError(f"Metadata missing required field: {error}") from error
    text_fields = (field for field in required if field != "chain_stage_index")
    if any(
        not isinstance(values[field], str) or not values[field].strip() for field in text_fields
    ):
        raise MetadataError("Required evaluation identity fields must be non-empty strings.")
    instance_id = values["instance_id"]
    match = _INSTANCE_ID_PATTERN.fullmatch(instance_id)
    if match is None:
        raise MetadataError(f"Invalid instance_id: {instance_id}")
    if not isinstance(values["chain_stage_index"], int) or values["chain_stage_index"] <= 0:
        raise MetadataError("chain_stage_index must be a positive integer.")
    if not instance_id.startswith(f"{values['chain_id']}-stage-{values['chain_stage_index']}-"):
        raise MetadataError("chain identity does not match instance_id.")
    initial_commit = values["initial_commit_sha"]
    final_commit = values["final_commit_sha"]
    if _COMMIT_PATTERN.fullmatch(initial_commit) is None or match.group(1) != initial_commit[:12]:
        raise MetadataError("initial_commit_sha does not match instance_id.")
    if _COMMIT_PATTERN.fullmatch(final_commit) is None or match.group(2) != final_commit[:12]:
        raise MetadataError("final_commit_sha does not match instance_id.")
    if not values["image_tag"].endswith(f":{instance_id}-v1.0.0"):
        raise MetadataError("image_tag does not match instance_id and dataset version.")
    if _DIGEST_PATTERN.fullmatch(values["image_digest"]) is None:
        raise MetadataError("image_digest must be a sha256 digest.")
    if any(not isinstance(path, str) or not path.strip() for path in patches.values()):
        raise MetadataError("Every patch path must be a non-empty string.")


def sanitize_label(label: str) -> str:
    """Sanitize an instance ID for use in file names."""
    return label.replace("/", "_").replace("#", "_")


def immutable_image_ref(data: dict[str, Any]) -> str:
    """Return the digest-pinned image reference from validated metadata."""
    validate_metadata(data)
    repository, separator, _tag = data["image_tag"].rpartition(":")
    if not separator:
        raise MetadataError("image_tag must include a tag.")
    return f"{repository}@{data['image_digest']}"


def find_metadata_files(path: Path) -> list[Path]:
    """Find evaluation metadata JSON files in a directory."""
    if path.is_file():
        return [path]

    metadata_files = []
    for json_file in path.glob("**/*.json"):
        try:
            data = load_metadata(json_file)
            if "image_tag" in data and "instance_id" in data:
                validate_metadata(data)
                metadata_files.append(json_file)
        except (json.JSONDecodeError, MetadataError):
            continue
    return sorted(metadata_files)


def materialize_metadata_from_instances_file(
    *,
    instances_file: Path,
    destination_dir: Path,
    selected_repo: str | None,
    selected_instance_id: str | None = None,
) -> list[Path]:
    """Materialize per-instance metadata from a published evaluation manifest."""
    if not instances_file.is_file():
        return []

    payload = json.loads(instances_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MetadataError("Evaluation manifest must be a JSON object.")
    if "schema_version" in payload or payload.get("dataset_version") != "1.0.0":
        raise MetadataError("Expected published dataset version 1.0.0 without schema_version.")
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise MetadataError("Evaluation manifest must contain an instances array.")
    destination_dir.mkdir(parents=True, exist_ok=True)
    metadata_files: list[Path] = []
    seen_ids: set[str] = set()

    for instance in instances:
        if not isinstance(instance, dict):
            raise MetadataError("Every evaluation instance must be a JSON object.")
        validate_metadata(instance)
        instance_id = instance["instance_id"]
        if instance_id in seen_ids:
            raise MetadataError(f"Duplicate instance_id in evaluation manifest: {instance_id}")
        seen_ids.add(instance_id)
        owner = instance["owner"]
        repo = instance["repo"]
        if not matches_repo(owner, repo, selected_repo):
            continue
        if selected_instance_id is not None and instance_id != selected_instance_id:
            continue
        target = destination_dir / f"{sanitize_label(instance_id)}.json"
        target.write_text(json.dumps(instance, indent=2) + "\n", encoding="utf-8")
        metadata_files.append(target)

    return sorted(metadata_files)


def metadata_matches_selection(
    metadata_path: Path,
    *,
    selected_repo: str | None,
    selected_instance_id: str | None = None,
) -> bool:
    try:
        data = load_metadata(metadata_path)
        validate_metadata(data)
        return matches_repo(data["owner"], data["repo"], selected_repo) and (
            selected_instance_id is None or data["instance_id"] == selected_instance_id
        )
    except Exception:
        return False
