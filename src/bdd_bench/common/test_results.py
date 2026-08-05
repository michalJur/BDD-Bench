"""Parse test-run logs shared by dataset construction and evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import TypedDict


class TestRunResult(TypedDict):
    run_number: int
    log_path: str
    passed_test_count: int
    failed_tests: list[str]
    test_exit_code: int


_EXIT_CODE_TRAILER_PATTERN = re.compile(r"LOG COMPLETED AT", re.IGNORECASE)
_OVERSIZE_LOG_PATTERN = re.compile(r"TEST RUN LOG TOO LONG - STOPPING", re.IGNORECASE)
_EXIT_CODE_LINE_PATTERN = re.compile(
    r"---\s*LOG\s+COMPLETED\s+AT.*?\(exit\s+code:\s*(?P<code>-?\d+)\)\s*---",
    re.IGNORECASE,
)
_PYTEST_SUMMARY_COUNT_PATTERN = re.compile(
    r"\b\d+\s+(?:passed|failed|failures?|errors?|skipped|xfailed|xpassed|rerun)\b",
    re.IGNORECASE,
)


def extract_test_statuses(output: str) -> tuple[list[str], list[str]]:
    """Return passed-count placeholders and concrete failed IDs from behave or pytest logs."""
    normalized_output = output or ""
    behave_summary_pattern = re.compile(
        r"(?P<passed>\d+)\s+scenarios?\s+passed,\s+(?P<failed>\d+)\s+failed",
        re.IGNORECASE,
    )

    def parse_behave_tests() -> tuple[list[str], list[str]] | None:
        summary_match = behave_summary_pattern.search(normalized_output)
        passed_count = int(summary_match.group("passed")) if summary_match else 0
        failed_count = int(summary_match.group("failed")) if summary_match else 0
        failed_lines: list[str] = []
        collecting_failures = False
        summary_line_pattern = re.compile(r"^\d+\s+features\s+passed", re.IGNORECASE)
        failure_line_pattern = re.compile(r"(?P<path>[^:]+:\d+)\s+(?P<title>.+)")
        for line in normalized_output.splitlines():
            stripped = line.strip()
            if not collecting_failures:
                if stripped.lower().startswith("failing scenarios"):
                    collecting_failures = True
                continue
            if not stripped or summary_line_pattern.search(stripped):
                break
            failure_match = failure_line_pattern.match(stripped)
            if failure_match:
                path = failure_match.group("path").strip()
                title = failure_match.group("title").strip()
                failed_lines.append(f"behave:{path}::{title}")
            else:
                failed_lines.append(f"behave:{stripped}")
        if failed_lines and not failed_count:
            failed_count = len(failed_lines)
        if passed_count or failed_count or failed_lines:
            passed = [f"behave:scenario:{index}" for index in range(1, passed_count + 1)]
            return passed, failed_lines
        return None

    def parse_pytest_tests() -> tuple[list[str], list[str]] | None:
        pytest_summary_pattern = re.compile(r"^=+(?P<summary>.+?)=+$", re.MULTILINE)
        summary_count_pattern = re.compile(
            r"(\d+)\s+(passed|failed|failures?|errors?)", re.IGNORECASE
        )
        failed_tests: list[str] = []
        in_summary = False
        summary_header_pattern = re.compile(r"=+\s*short test summary info\s*=+", re.IGNORECASE)
        for line in normalized_output.splitlines():
            if not in_summary:
                if summary_header_pattern.search(line):
                    in_summary = True
                continue
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^[=_]{2,}.*[=_]{2,}$", stripped) and _PYTEST_SUMMARY_COUNT_PATTERN.search(
                stripped
            ):
                break
            failed_match = re.match(
                r"^(FAIL(?:ED)?|ERROR)\s+(?P<name>\S+)", stripped, re.IGNORECASE
            )
            if failed_match:
                failed_tests.append(failed_match.group("name"))

        passed_count = 0
        failed_total = 0
        for match in pytest_summary_pattern.finditer(normalized_output):
            summary = match.group("summary")
            lowered_summary = summary.lower()
            if not any(keyword in lowered_summary for keyword in ("passed", "failed", "error")):
                continue
            counts = summary_count_pattern.findall(summary)
            passed_count = sum(
                int(number) for number, label in counts if label.lower().startswith("pass")
            )
            failed_total = sum(
                int(number)
                for number, label in counts
                if label.lower().startswith(("fail", "error"))
            )
            if passed_count or failed_total:
                break
        if failed_tests and failed_total == 0:
            failed_total = len(failed_tests)
        passed_list = [f"pytest:passed:{index}" for index in range(1, passed_count + 1)]
        if passed_list or failed_tests:
            return passed_list, failed_tests
        return None

    return parse_behave_tests() or parse_pytest_tests() or ([], [])


def log_is_finalized(content: str) -> bool:
    return bool(_EXIT_CODE_TRAILER_PATTERN.search(content.upper().rstrip()))


def log_is_broken(content: str) -> bool:
    return bool(_OVERSIZE_LOG_PATTERN.search(content))


def load_run_result_from_log(log_path: Path, run_number: int) -> TestRunResult | None:
    """Parse a completed, non-truncated test log."""
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        logging.error("Failed to read log %s: %s", log_path, error)
        return None
    if log_is_broken(content):
        logging.error("Skipping broken test log %s", log_path)
        return None
    if not log_is_finalized(content):
        return None
    exit_match = _EXIT_CODE_LINE_PATTERN.search(content)
    try:
        exit_code = int(exit_match.group("code")) if exit_match else 0
    except ValueError:
        exit_code = 0
    passed_tests, failed_tests = extract_test_statuses(content)
    return {
        "run_number": run_number,
        "log_path": str(log_path),
        "passed_test_count": len(passed_tests),
        "failed_tests": failed_tests,
        "test_exit_code": exit_code,
    }
