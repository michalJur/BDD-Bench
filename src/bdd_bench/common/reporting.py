from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEFAULT_REPORT_PROGRESSION_MODE = "lifecycle"
DEFAULT_REPORT_INCLUDE_HIDDEN_TESTS = True
DEFAULT_REPORT_CONFIG_DESCRIPTION = "lifecycle progression with hidden tests enabled"


def is_default_report_config(config: Mapping[str, Any]) -> bool:
    """Return whether an evaluation config belongs in the default reports.

    The check is intentionally strict: missing execution metadata is not enough
    to establish that a run used the paper's default evaluation protocol.
    """
    execution = config.get("execution")
    if not isinstance(execution, Mapping):
        return False
    return (
        execution.get("progression_mode") == DEFAULT_REPORT_PROGRESSION_MODE
        and execution.get("include_hidden_tests") is DEFAULT_REPORT_INCLUDE_HIDDEN_TESTS
    )
