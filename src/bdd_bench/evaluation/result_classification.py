"""Shared classification for evaluation-style result records."""

from __future__ import annotations

from collections.abc import Mapping


CHAIN_BLOCKED_ERROR_PREFIX = "Blocked by release-chain stop:"
INFRA_ERROR_PREFIX = "Infrastructure error: "


def is_parse_error(error: str | None) -> bool:
    """Return whether a result failed while parsing its test output."""
    return error is not None and error.startswith("Failed to parse test results")


def is_disallowed_patch_error(error: str | None) -> bool:
    """Return whether a generated patch modified a forbidden path."""
    if error is None:
        return False
    return error.startswith(
        "Agent patch modifies test files (disallowed by dataset rules):",
    ) or error.startswith(
        "Agent patch modifies disallowed files (test/env/config):",
    )


def is_chain_blocked_error(error: str | None) -> bool:
    """Return whether a lifecycle stage was blocked by an earlier failure."""
    return isinstance(error, str) and error.startswith(CHAIN_BLOCKED_ERROR_PREFIX)


def chain_blocked_reason(error: str | None) -> str:
    """Remove the standard lifecycle-blocked prefix from an error."""
    if not is_chain_blocked_error(error):
        return ""
    return str(error)[len(CHAIN_BLOCKED_ERROR_PREFIX) :].strip()


def is_infra_error(error: str | None) -> bool:
    """Return whether the harness explicitly marked an infrastructure error."""
    return error is not None and error.startswith(INFRA_ERROR_PREFIX)


def classify_evaluation_result(
    *,
    resolved: bool,
    error: str | None,
    patch_applied_successfully: bool | None,
    all_tests_passed: bool | None,
    detailed_results: Mapping[str, object] | None = None,
) -> str:
    """Return the canonical category used by evaluation result summaries."""
    details = detailed_results or {}
    if resolved:
        return "resolved"
    if is_parse_error(error):
        return "not_parsed"
    if details.get("infra_denied_empty_patch", False):
        return "infra_denied_empty_patch"
    if is_infra_error(error):
        return "infra_error"
    if is_disallowed_patch_error(error):
        return "disallowed_patch"
    if is_chain_blocked_error(error):
        return "chain_blocked"
    if error:
        error_text = error.lower()
        if "empty when diffing" in error_text:
            return "empty_patch"
        if "missing patch" in error_text:
            return "missing_patch"
        return "error"
    if patch_applied_successfully is False:
        return "patch_apply_failed"
    if all_tests_passed is False:
        return "tests_failed"
    return "unresolved"
