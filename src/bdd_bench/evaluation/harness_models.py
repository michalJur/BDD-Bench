from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


class CheckerError(Exception):
    """Base exception for checker errors."""

    pass


class ConfigurationError(CheckerError):
    """Raised when checker configuration is invalid."""

    pass


class MetadataError(CheckerError):
    """Raised when metadata is missing or invalid."""

    pass


class PatchError(CheckerError):
    """Raised when a patch file is missing or invalid."""

    pass


@dataclass
class EvaluationContext:
    instance_id: str
    container_name: str
    repo: str
    image_tag: str
    log_path: Path


@dataclass
class CheckerMode:
    """Configuration for the checker mode based on provided patches."""

    name: str
    description: str
    use_golden_code: bool
    use_golden_tests: bool
    custom_code_path: Path | None
    custom_test_path: Path | None
    log_dir_name: str
    container_prefix: str

    @property
    def result_dir_name(self) -> str:
        """Directory name for storing results (derived from log_dir_name)."""
        return self.log_dir_name.replace("_logs", "_results")


@dataclass
class EvaluationResult:
    instance_id: str
    agent_name: str
    patch_generated: str = ""
    patch_applied_successfully: bool = False
    all_tests_passed: bool = False
    any_tests_passed: bool = False
    passed_test_count: int = 0
    failed_test_count: int = 0
    failed_tests: list[str] = field(default_factory=list)
    test_exit_code: int = -1
    run_count: int = 0
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    detailed_results: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.patch_applied_successfully and self.all_tests_passed

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["resolved"] = self.resolved
        return result


@dataclass
class EvaluationSummary:
    """Summary of evaluation results across multiple instances."""

    agent_name: str
    total_instances: int
    resolved_instances: int
    failed_patch_apply: int
    failed_tests: int
    errors: int
    resolution_rate: float
    results: list[EvaluationResult]
    skipped_instances: int = 0
    not_parsed_instances: int = 0
    infra_error_instances: int = 0
    infra_denied_empty_patch_instances: int = 0
    missing_patch_after_attempt_instances: int = 0
    missing_patch_not_started_instances: int = 0
    disallowed_test_patch_instances: int = 0
    attempted_instances: int = 0
    attempted_resolved_instances: int = 0
    attempted_resolution_rate: float = 0.0
    chain_blocked_instances: int = 0
    lifecycle_first_failures: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "total_instances": self.total_instances,
            "skipped_instances": self.skipped_instances,
            "not_parsed_instances": self.not_parsed_instances,
            "infra_error_instances": self.infra_error_instances,
            "infra_denied_empty_patch_instances": self.infra_denied_empty_patch_instances,
            "missing_patch_after_attempt_instances": self.missing_patch_after_attempt_instances,
            "missing_patch_not_started_instances": self.missing_patch_not_started_instances,
            "disallowed_test_patch_instances": self.disallowed_test_patch_instances,
            "attempted_instances": self.attempted_instances,
            "attempted_resolved_instances": self.attempted_resolved_instances,
            "attempted_resolution_rate": self.attempted_resolution_rate,
            "chain_blocked_instances": self.chain_blocked_instances,
            "lifecycle_first_failures": self.lifecycle_first_failures,
            "resolved_instances": self.resolved_instances,
            "failed_patch_apply": self.failed_patch_apply,
            "failed_tests": self.failed_tests,
            "errors": self.errors,
            "resolution_rate": self.resolution_rate,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }
