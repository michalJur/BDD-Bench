from __future__ import annotations

from fnmatch import fnmatch


_REPO_ENV_GLOBS: dict[str, tuple[str, ...]] = {
    # Per-repo glob patterns for files that configure the build/test environment
    # (dependency pins, tox/CI config, Dockerfiles, etc.).  These are neither
    # application code nor test code but must be present so the test runner can
    # set up its environment correctly.
    "qutebrowser": (
        "misc/requirements/**",
        "requirements*.txt",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "tox.ini",
    ),
    "jrnl": (
        "requirements*.txt",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "poetry.lock",
    ),
    "open-web-calendar": (
        "pyproject.toml",
        "requirements/**",
        "tox.ini",
        "behave.ini",
    ),
    "partcad": (
        "requirements*.txt",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "poetry.lock",
    ),
    "python-pptx": (
        "requirements*.txt",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "tox.ini",
    ),
    "python-docx": (
        "requirements*.txt",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "tox.ini",
    ),
}


_REPO_NON_BDD_TEST_GLOBS: dict[str, tuple[str, ...]] = {
    # Per-repo glob patterns for test directories that contain unit tests (not
    # BDD tests).  Files matching these globs are excluded from the test patch
    # so that unit tests are not leaked to the model.
    "qutebrowser": ("tests/unit/**",),
    "jrnl": ("tests/unit/**",),
    # Only the pytest unit-test modules are non-BDD. The rest of
    # open_web_calendar/test/ (rpc.py, api_mocking.py, conftest.py, __init__.py,
    # responses/ fixtures, ...) is test *infrastructure* imported by the behave
    # environment at runtime, so it must flow into the golden test patch — not
    # be dropped from both patches.
    "open-web-calendar": (
        "open_web_calendar/test/test_*.py",
        "open_web_calendar/test/**/test_*.py",
    ),
    "partcad": (
        "partcad/tests/**",
        "partcad-cli/tests/**",
    ),
    # scanny libraries: pytest unit suite lives under tests/; the behave acceptance
    # suite (features/, features/steps/) is the BDD surface that flows to the model.
    "python-pptx": ("tests/**",),
    "python-docx": ("tests/**",),
}


_REPO_TEST_FIXTURE_GLOBS: dict[str, tuple[str, ...]] = {
    # open-web-calendar's selenium behave suite lives under open_web_calendar/features/
    # (feature specs, browser steps, and the calendars/ ICS fixtures the scenarios load).
    "open-web-calendar": (
        "open_web_calendar/features/**",
        "features/**",
    ),
    # PartCAD BDD tests frequently pull data fixtures from examples/*
    # which are part of product code changes and must be present when the
    # golden test patch is applied on top of the initial commit.
    "partcad": (
        "examples/feature_convert/**",
        "examples/feature_convert_part/**",
        "examples/feature_convert_sketch/**",
        "examples/feature_import/**",
        "examples/produce_part_obj/**",
        # Behave step wrappers in PartCAD can shell out via coverage with
        # --rcfile pointing here; keep it in test patch when tests depend on it.
        "dev-tools/coverage.rc",
    ),
    # qutebrowser end-to-end tests can require root test-runner configuration
    # updates (e.g. new pytest markers or tox passenv entries). Keep these in
    # the golden test patch so test collection matches the PR test environment.
    "qutebrowser": (
        "pytest.ini",
        "scripts/dev/run_pylint_on_tests.py",
        "scripts/dev/standardpaths_tester.py",
        "scripts/keytester.py",
        "scripts/testbrowser/**",
    ),
    # jrnl behave suites rely broadly on the features tree (feature specs,
    # steps, and data fixtures), plus helper imports from jrnl/behave_testing.py
    # during tests-only execution.
    "jrnl": (
        "features/**",
        "jrnl/behave_testing.py",
        "tests/bdd/**",
        "tests/lib/**",
        "tests/data/**",
    ),
    # scanny libraries store the behave acceptance-suite fixtures (binary .pptx /
    # .docx samples plus image assets) under features/steps/test_files/. These are
    # loaded at runtime via test_pptx('...')/test_docx('...') and must flow into the
    # golden test patch; otherwise a PR that adds a new fixture leaves it in the code
    # patch and the fixture is absent when the test patch is applied at eval time.
    # (tests/test_files/** stays code/excluded via the tests/** non-BDD rule.)
    "python-pptx": ("features/steps/test_files/**",),
    "python-docx": ("features/steps/test_files/**",),
}


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def _normalize_repo_name(repo: str | None) -> str:
    if not isinstance(repo, str):
        return ""
    normalized = _normalize_path(repo).lower()
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    return normalized


def is_repo_test_fixture_path(path: str, *, repo: str | None) -> bool:
    repo_key = _normalize_repo_name(repo)
    if not repo_key:
        return False

    patterns = _REPO_TEST_FIXTURE_GLOBS.get(repo_key, ())
    if not patterns:
        return False

    lowered = _normalize_path(path).lower()
    normalized_patterns = tuple(_normalize_path(pattern).lower() for pattern in patterns)
    return any(fnmatch(lowered, pattern) for pattern in normalized_patterns)


def is_non_bdd_test_path(path: str, *, repo: str | None = None) -> bool:
    """Return True when a file is in a unit test directory excluded from the test patch."""
    repo_key = _normalize_repo_name(repo)
    if not repo_key:
        return False

    patterns = _REPO_NON_BDD_TEST_GLOBS.get(repo_key, ())
    if not patterns:
        return False

    lowered = _normalize_path(path).lower()
    normalized_patterns = tuple(_normalize_path(pattern).lower() for pattern in patterns)
    return any(fnmatch(lowered, pattern) for pattern in normalized_patterns)


def is_env_path(path: str, *, repo: str | None = None) -> bool:
    """Return True when a file path is an environment/build configuration artifact."""
    repo_key = _normalize_repo_name(repo)
    if not repo_key:
        return False

    patterns = _REPO_ENV_GLOBS.get(repo_key, ())
    if not patterns:
        return False

    lowered = _normalize_path(path).lower()
    normalized_patterns = tuple(_normalize_path(pattern).lower() for pattern in patterns)
    return any(fnmatch(lowered, pattern) for pattern in normalized_patterns)


def is_test_path(path: str, *, repo: str | None = None) -> bool:
    """Return True when a file path should be treated as a test artifact."""
    normalized = _normalize_path(path)
    lowered = normalized.lower()
    parts = [segment for segment in lowered.split("/") if segment]
    filename = parts[-1] if parts else lowered
    stem = filename.rsplit(".", 1)[0]
    directories = parts[:-1]

    if is_non_bdd_test_path(normalized, repo=repo):
        return False

    if is_repo_test_fixture_path(normalized, repo=repo):
        return True

    if lowered.endswith(".feature"):
        return True

    # Treat all files under conventional test directories as test artifacts.
    if any(segment in {"test", "tests", "testing"} for segment in directories):
        return True

    if not lowered.endswith(".py"):
        return False

    # Pytest naming/layout conventions.
    if stem == "conftest" or stem.startswith("test_") or stem.endswith("_test"):
        return True

    # Behave/Gherkin Python support files under a real "features/" directory.
    if "features" in directories:
        return True

    # Non-standard step-definition layouts: only classify if clearly BDD-related.
    bdd_hint_segments = {"features", "test", "tests", "testing", "bdd", "behave"}
    for index, segment in enumerate(directories):
        if segment in {"steps", "step_definitions"} and any(
            hint in bdd_hint_segments for hint in directories[:index]
        ):
            return True

    return False
