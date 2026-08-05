from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

MIN_PR_NUMBER_BY_REPO: dict[str, int] = {
    # Hardcoded pipeline rule: keep qutebrowser instances at/after this PR number.
    # Add new per-repo floors here as needed.
    "jrnl-org/jrnl": 0,
    "niccokunzmann/open-web-calendar": 0,
    "partcad/partcad": 0,
    "qutebrowser/qutebrowser": 3000,
}

DEFAULT_TEST_RUN_REPETITIONS = 3
TEST_RUN_REPETITIONS_BY_REPO: dict[str, int] = {
    # Keep lightweight repositories on fewer runs, while preserving extra
    # fluctuation detection for larger suites.
    "jrnl-org/jrnl": 1,
    "niccokunzmann/open-web-calendar": 1,
    "partcad/partcad": 1,
    "qutebrowser/qutebrowser": 1,
    "scanny/python-pptx": 1,
}

DEFAULT_PYTEST_RERUNS = 0
PYTEST_RERUNS_BY_REPO: dict[str, int] = {
    # Per-repo pytest --reruns count (pytest-rerunfailures). 0 = disabled.
    "jrnl-org/jrnl": 0,
    "niccokunzmann/open-web-calendar": 0,
    "partcad/partcad": 0,
    "qutebrowser/qutebrowser": 3,
}
REPO_PR_RANGES_ENV_VAR = "BDD_BENCH_REPO_PR_RANGES"


class _MissingRepoRange:
    pass


_MISSING_REPO_RANGE = _MissingRepoRange()


@dataclass(frozen=True)
class PRRange:
    start: int | None
    end: int | None


def split_repo_selector(selected_repo: str | None) -> tuple[str, ...]:
    if selected_repo is None:
        return ()

    selectors: list[str] = []
    seen: set[str] = set()
    for raw_selector in selected_repo.split(","):
        selector = raw_selector.strip().lower()
        if not selector:
            continue
        if selector in {"all", "*"}:
            return ()
        if selector in seen:
            continue
        seen.add(selector)
        selectors.append(selector)
    return tuple(selectors)


def parse_pr_range(value: str | None) -> PRRange | None:
    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        raise ValueError("PR selector cannot be empty.")

    if ":" in candidate:
        parts = candidate.split(":", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid PR selector '{value}'. Use a single number (e.g. 5231) "
                "or slice-like range (e.g. 4000:6000, 4000:, :6000)."
            )
        start_raw, end_raw = parts
        try:
            start = int(start_raw) if start_raw else None
            end = int(end_raw) if end_raw else None
        except ValueError as error:
            raise ValueError(
                f"Invalid PR selector '{value}'. Use a single number (e.g. 5231) "
                "or slice-like range (e.g. 4000:6000, 4000:, :6000)."
            ) from error
        if start is None and end is None:
            return None
        if start is not None and end is not None and start > end:
            raise ValueError(f"Invalid PR range '{value}'. Start must be <= end for a range.")
        return PRRange(start=start, end=end)

    try:
        number = int(candidate)
    except ValueError as error:
        raise ValueError(
            f"Invalid PR selector '{value}'. Use a single number (e.g. 5231) "
            "or slice-like range (e.g. 4000:6000, 4000:, :6000)."
        ) from error
    return PRRange(start=number, end=number)


def parse_repo_pr_ranges(value: str | None) -> dict[str, PRRange | None]:
    """Parse per-repository source PR ranges.

    Accepted forms:
    - JSON object: {"qutebrowser": ":6679", "jrnl": "all"}
    - Shorthand: qutebrowser=:6679,jrnl=all

    Values use the same syntax as ``--pr``. ``all``, ``*``, ``null``, or an
    empty value means no limit for that repository.
    """
    if value is None:
        return {}
    raw = value.strip()
    if not raw:
        return {}

    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid repo PR ranges JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("Repo PR ranges JSON must be an object.")
        items = payload.items()
    else:
        parsed: dict[str, str] = {}
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                repo, range_text = item.split("=", maxsplit=1)
            elif ":" in item and item.count(":") == 1 and not item.startswith(":"):
                repo, range_text = item.split(":", maxsplit=1)
            else:
                raise ValueError(
                    "Invalid repo PR ranges entry "
                    f"'{item}'. Use repo=range or JSON, e.g. qutebrowser=:6679."
                )
            parsed[repo] = range_text
        items = parsed.items()

    result: dict[str, PRRange | None] = {}
    for raw_repo, raw_range in items:
        if not isinstance(raw_repo, str) or not raw_repo.strip():
            raise ValueError("Repo PR ranges keys must be non-empty strings.")
        repo = raw_repo.strip().lower()
        if raw_range is None:
            result[repo] = None
            continue
        if not isinstance(raw_range, str):
            raw_range = str(raw_range)
        range_text = raw_range.strip()
        if not range_text or range_text.lower() in {"all", "*", "none", "null"}:
            result[repo] = None
            continue
        result[repo] = parse_pr_range(range_text)
    return result


def repo_pr_ranges_from_env() -> dict[str, PRRange | None]:
    return parse_repo_pr_ranges(os.environ.get(REPO_PR_RANGES_ENV_VAR))


def repo_pr_range_for(
    owner: str | None,
    repo: str | None,
    repo_pr_ranges: dict[str, PRRange | None] | None = None,
) -> PRRange | None | _MissingRepoRange:
    if not repo_pr_ranges or not repo:
        return _MISSING_REPO_RANGE

    repo_name = repo.lower()
    repo_key = f"{owner.lower()}/{repo_name}" if owner else None
    for selector, candidate_range in repo_pr_ranges.items():
        if selector == repo_name or (repo_key and selector == repo_key):
            return candidate_range
    return _MISSING_REPO_RANGE


def matches_repo(
    owner: str | None,
    repo: str | None,
    selected_repo: str | None,
) -> bool:
    targets = split_repo_selector(selected_repo)
    if not targets:
        return True
    if not repo:
        return False

    repo_name = repo.lower()
    repo_key = f"{owner.lower()}/{repo_name}" if owner else None
    for target in targets:
        if target == repo_name:
            return True
        if repo_key and target == repo_key:
            return True
    return False


def matches_pr_number(number: int | None, selected_range: PRRange | None) -> bool:
    if selected_range is None:
        return True
    if not isinstance(number, int):
        return False
    if selected_range.start is not None and number < selected_range.start:
        return False
    if selected_range.end is not None and number > selected_range.end:
        return False
    return True


def matches_repo_pr_ranges(
    owner: str | None,
    repo: str | None,
    number: int | None,
    repo_pr_ranges: dict[str, PRRange | None] | None = None,
) -> bool:
    if not repo_pr_ranges:
        return True

    selected_range = repo_pr_range_for(owner, repo, repo_pr_ranges)
    if isinstance(selected_range, _MissingRepoRange):
        return True
    return matches_pr_number(number, selected_range)


def repo_pr_range_number_for_instance(
    instance: dict[str, Any],
    entry: dict[str, Any] | None = None,
) -> int | None:
    """Return the PR number used for per-repo range truncation.

    Release-chain instances represent an edge from a candidate to an anchor.
    The per-repo range is meant to truncate an already-built chain at the last
    included anchor, so prefer the anchor PR when it is available. Plain PR
    entries do not have anchor metadata and fall back to their source PR.
    """
    anchor_number = instance.get("anchor_real_pr_number")
    if isinstance(anchor_number, int):
        return anchor_number
    if entry is None:
        raw_entry = instance.get("entry")
        entry = raw_entry if isinstance(raw_entry, dict) else None
    if not isinstance(entry, dict):
        return None
    return _entry_source_pr_number(entry)


def filter_instances_by_repo_pr_ranges(
    instances: list[dict[str, Any]],
    repo_pr_ranges: dict[str, PRRange | None] | None = None,
) -> list[dict[str, Any]]:
    if not repo_pr_ranges:
        return instances

    selected: list[dict[str, Any]] = []
    stopped_repo_keys: set[str] = set()
    for instance in instances:
        raw_entry = instance.get("entry")
        entry = raw_entry if isinstance(raw_entry, dict) else instance
        owner = entry.get("owner")
        repo = entry.get("repo")
        if not isinstance(owner, str) or not isinstance(repo, str):
            continue

        selected_range = repo_pr_range_for(owner, repo, repo_pr_ranges)
        if isinstance(selected_range, _MissingRepoRange) or selected_range is None:
            selected.append(instance)
            continue

        repo_key = f"{owner.lower()}/{repo.lower()}"
        if repo_key in stopped_repo_keys:
            continue

        anchor_number = instance.get("anchor_real_pr_number")
        if not isinstance(anchor_number, int):
            anchor_number = entry.get("anchor_real_pr_number")
        if isinstance(anchor_number, int) and selected_range.end is not None:
            if selected_range.start is not None and anchor_number < selected_range.start:
                continue
            selected.append(instance)
            if anchor_number == selected_range.end:
                stopped_repo_keys.add(repo_key)
            continue

        range_number = repo_pr_range_number_for_instance(instance, entry)
        if matches_pr_number(range_number, selected_range):
            selected.append(instance)
    return selected


def matches_repo_pr_floor(owner: str | None, repo: str | None, number: int | None) -> bool:
    """Return whether a PR number satisfies hardcoded per-repo minimum thresholds."""
    if not isinstance(number, int):
        return False
    if not repo:
        return False

    repo_candidates: list[str] = [repo.lower()]
    if owner:
        repo_candidates.insert(0, f"{owner.lower()}/{repo.lower()}")

    for candidate in repo_candidates:
        floor = MIN_PR_NUMBER_BY_REPO.get(candidate)
        if floor is not None:
            return number >= floor
    return True


def get_test_run_repetitions(
    owner: str | None,
    repo: str | None,
    *,
    default: int = DEFAULT_TEST_RUN_REPETITIONS,
) -> int:
    """Return the test run count for a repository, falling back to default."""
    if default < 1:
        raise ValueError("Default test run repetitions must be >= 1.")
    if not repo:
        return default

    repo_candidates: list[str] = [repo.lower()]
    if owner:
        repo_candidates.insert(0, f"{owner.lower()}/{repo.lower()}")

    for candidate in repo_candidates:
        repetitions = TEST_RUN_REPETITIONS_BY_REPO.get(candidate)
        if repetitions is not None:
            if repetitions < 1:
                raise ValueError(
                    f"Invalid configured test run repetitions for '{candidate}': {repetitions}"
                )
            return repetitions
    return default


def get_pytest_reruns(
    owner: str | None,
    repo: str | None,
    *,
    default: int = DEFAULT_PYTEST_RERUNS,
) -> int:
    """Return the pytest --reruns count for a repository, falling back to default."""
    if not repo:
        return default

    repo_candidates: list[str] = [repo.lower()]
    if owner:
        repo_candidates.insert(0, f"{owner.lower()}/{repo.lower()}")

    for candidate in repo_candidates:
        reruns = PYTEST_RERUNS_BY_REPO.get(candidate)
        if reruns is not None:
            return reruns
    return default


def _entry_source_pr_number(entry: dict[str, Any]) -> int | None:
    """Return the original PR number, preferring ``real_pr_number`` over ``number``."""
    real = entry.get("real_pr_number")
    if isinstance(real, int):
        return real
    number = entry.get("number")
    return number if isinstance(number, int) else None


def filter_entries(
    entries: list[dict[str, Any]],
    *,
    selected_repo: str | None,
    selected_range: PRRange | None,
    repo_pr_ranges: dict[str, PRRange | None] | None = None,
) -> list[dict[str, Any]]:
    ranges = repo_pr_ranges if repo_pr_ranges is not None else repo_pr_ranges_from_env()
    selected = [
        entry
        for entry in entries
        if matches_repo(entry.get("owner"), entry.get("repo"), selected_repo)
        # Match --pr against the source (GitHub) PR number. Collection entries are
        # keyed by GitHub number directly; post-preselection entries carry an
        # internal dataset id in "number" and the GitHub number in "real_pr_number",
        # so without this a GitHub-space --pr range would filter everything out.
        and matches_pr_number(_entry_source_pr_number(entry), selected_range)
        and matches_repo_pr_floor(
            entry.get("owner"), entry.get("repo"), _entry_source_pr_number(entry)
        )
    ]
    return filter_instances_by_repo_pr_ranges(selected, ranges)
