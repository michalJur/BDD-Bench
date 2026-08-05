from __future__ import annotations

from typing import Any

FLAKY_TEST_FIELDS = ("fluctuating_tests",)


def _collect_tests(raw_tests: Any) -> set[str]:
    return {test.strip() for test in raw_tests if isinstance(test, str) and test.strip()}


def collect_unresolved_flaky_tests(data: dict[str, Any]) -> dict[str, set[str]]:
    unresolved: dict[str, set[str]] = {}

    stats = data["stats"]
    stats_by_repo = stats["fluctuating_tests_by_repo"]
    for repo, tests in stats_by_repo.items():
        parsed_tests = _collect_tests(tests)
        if parsed_tests:
            unresolved.setdefault(repo, set()).update(parsed_tests)

    instances = data["instances"]
    for instance in instances:
        repo = instance["repo"]
        for field in FLAKY_TEST_FIELDS:
            unresolved.setdefault(repo, set()).update(_collect_tests(instance[field]))

    return {repo: tests for repo, tests in sorted(unresolved.items()) if tests}


def build_unresolved_flaky_tests_error(
    unresolved: dict[str, set[str]],
    *,
    context: str,
) -> str:
    details = "; ".join(
        f"{repo}: {', '.join(sorted(tests))}" for repo, tests in sorted(unresolved.items()) if tests
    )
    return (
        f"{context}: unresolved flaky tests remain in final instances ({details}). "
        "Add them to scripts/<repo>/predefined_flaky_tests.json and regenerate."
    )


def ensure_no_unresolved_flaky_tests(data: dict[str, Any], *, context: str) -> None:
    unresolved = collect_unresolved_flaky_tests(data)
    if unresolved:
        raise ValueError(build_unresolved_flaky_tests_error(unresolved, context=context))
