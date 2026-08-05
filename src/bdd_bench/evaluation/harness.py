from __future__ import annotations

from bdd_bench.evaluation.harness_agents import (
    Agent,
    DummyAgent,
    GoldenAgent,
    MiniSweAgent,
    get_agent,
)
from bdd_bench.evaluation.harness_cli import main, parse_args, print_summary
from bdd_bench.evaluation.harness_core import EvaluationHarness
from bdd_bench.evaluation.harness_metadata import (
    find_metadata_files,
    load_metadata,
    materialize_metadata_from_instances_file,
    metadata_matches_selection,
    sanitize_label,
    validate_metadata,
)
from bdd_bench.evaluation.harness_models import (
    CheckerError,
    CheckerMode,
    ConfigurationError,
    EvaluationResult,
    EvaluationSummary,
    MetadataError,
    PatchError,
)

__all__ = [
    "Agent",
    "DummyAgent",
    "GoldenAgent",
    "MiniSweAgent",
    "get_agent",
    "main",
    "parse_args",
    "print_summary",
    "EvaluationHarness",
    "find_metadata_files",
    "load_metadata",
    "materialize_metadata_from_instances_file",
    "metadata_matches_selection",
    "sanitize_label",
    "validate_metadata",
    "CheckerError",
    "CheckerMode",
    "ConfigurationError",
    "EvaluationResult",
    "EvaluationSummary",
    "MetadataError",
    "PatchError",
]


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()
