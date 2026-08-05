from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from bdd_bench.common.selection import split_repo_selector
from bdd_bench.common.variants import default_dataset_instances_path
from bdd_bench.evaluation.harness_agents import (
    MINI_SWE_DEFAULT_COMMAND_TIMEOUT,
    MINI_SWE_DEFAULT_COST_LIMIT,
    MINI_SWE_DEFAULT_MAX_OUTPUT_TOKENS,
    MINI_SWE_DEFAULT_MODEL_NAME,
    MINI_SWE_DEFAULT_STEP_LIMIT,
    Agent,
    get_agent,
    resolve_default_reasoning_effort,
    resolve_max_output_tokens,
    resolve_mini_swe_agent_version,
)
from bdd_bench.evaluation.harness_core import EvaluationHarness
from bdd_bench.evaluation.harness_metadata import (
    materialize_metadata_from_instances_file,
)
from bdd_bench.evaluation.harness_core import _is_parse_error
from bdd_bench.evaluation.harness_models import EvaluationSummary
from bdd_bench.evaluation.harness_policy import (
    PRIMARY_GENERATION_AGENT,
    artifact_agent_name,
    normalize_agent_name,
    require_supported_generation_agent,
)
from bdd_bench.evaluation.run_config import agent_config_from_run_config


def print_summary(summary: EvaluationSummary) -> None:
    """Print evaluation summary to console."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Agent: {summary.agent_name}")
    print(f"Total instances: {summary.total_instances}")
    print(f"Skipped: {summary.skipped_instances}")
    print(f"Not parsed: {summary.not_parsed_instances}")
    print(f"Resolved: {summary.resolved_instances} ({100 * summary.resolution_rate:.1f}%)")
    print("Infra-denied empty patch instances: " f"{summary.infra_denied_empty_patch_instances}")
    print(f"Failed patch apply: {summary.failed_patch_apply}")
    print(f"Failed tests: {summary.failed_tests}")
    print(f"Disallowed test-patch instances: {summary.disallowed_test_patch_instances}")
    print(f"Infra errors: {summary.infra_error_instances}")
    print(f"Errors: {summary.errors}")
    if summary.chain_blocked_instances or summary.lifecycle_first_failures:
        print(
            "Attempted-stage resolved: "
            f"{summary.attempted_resolved_instances}/{summary.attempted_instances} "
            f"({100 * summary.attempted_resolution_rate:.1f}%)"
        )
        print(f"Blocked by prior lifecycle failure: {summary.chain_blocked_instances}")
        for failure in summary.lifecycle_first_failures:
            category = failure.get("category", "unknown")
            segment_id = failure.get("segment_id", "unknown")
            survived = failure.get("survived_stages", 0)
            total = failure.get("total_reported_stages", 0)
            if category == "all_resolved":
                print(f"  {segment_id}: all {survived}/{total} reported stages resolved")
                continue
            print(
                f"  {segment_id}: stopped at {failure.get('instance_id')} "
                f"after {survived}/{total} stages ({category}), "
                f"blocked later stages {failure.get('blocked_after', 0)}"
            )
    print("-" * 70)

    # Per-instance breakdown
    print("\nPer-Instance Results:")

    def _error_snippet(error: str) -> str:
        compact = " ".join(error.split())
        return f"{compact[:240]}..." if len(compact) > 240 else compact

    for result in summary.results:
        if _is_parse_error(result.error):
            print(f"  ~ {result.instance_id}: NOT PARSED (log parse failed)")
            continue
        if result.detailed_results.get("infra_denied_empty_patch", False):
            print(
                "  ! "
                f"{result.instance_id}: INFRA_DENIED_EMPTY_PATCH "
                "(generation blocked by environment restrictions)"
            )
            continue
        if isinstance(result.error, str) and result.error.startswith(
            "Blocked by release-chain stop:"
        ):
            reason = result.error.split(":", 1)[1].strip()
            print(f"  - {result.instance_id}: CHAIN_BLOCKED ({reason})")
            continue
        status = "RESOLVED" if result.resolved else "FAILED"
        emoji = "✓" if result.resolved else "✗"
        detail = ""
        if result.error:
            detail = f" (Error: {_error_snippet(result.error)})"
        elif not result.patch_applied_successfully:
            detail = " (Patch apply failed)"
        elif not result.all_tests_passed:
            detail = f" ({result.failed_test_count} tests failed)"

        print(f"  {emoji} {result.instance_id}: {status}{detail}")

    print("=" * 70)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate coding agents on BDD-bench instances.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate and evaluate with the secure default agent
    PYTHONPATH=src python -m bdd_bench.evaluation.harness --generate-and-evaluate

    # Select a model while keeping the same mini-swe scaffold
    PYTHONPATH=src python -m bdd_bench.evaluation.harness --model openai/example-model

        """,
    )

    parser.add_argument(
        "--agent",
        type=str,
        default=PRIMARY_GENERATION_AGENT,
        choices=[
            "dummy",
            "golden",
            "mini-swe",
            "mini_swe",
        ],
        help=(
            "Agent identity: mini-swe, or deterministic dummy/golden baselines "
            "(default: mini-swe)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model identifier used by mini-swe with the benchmark's model-specific "
            "reasoning policy. "
            "Defaults to BDD_BENCH_MINI_SWE_MODEL "
            f"or {MINI_SWE_DEFAULT_MODEL_NAME!r}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_evaluation"),
        help="Output root directory for evaluation runs (default: output_evaluation).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Run folder name under --output. Required for modes that evaluate existing patches/logs; "
            "use 'latest' to select the newest run folder. Optional for generation modes "
            "(auto-generated as <timestamp>_<agent> when omitted)."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of test runs per instance (default: 1).",
    )
    parser.add_argument(
        "--instances-file",
        type=Path,
        default=None,
        help="Optional path to a published evaluation manifest.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel workers (default: 1). In release-chain progression, "
            "workers run independent chains; stages within each chain remain sequential."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence console output (only show warnings and errors).",
    )
    parser.add_argument(
        "--live-agent-stream",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Stream runtime-agent command output live to console while generating patches "
            "(mini-swe only)."
        ),
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help=(
            "Repository selector to include. Accepts repo, owner/repo, or a comma-separated "
            "list. "
            "Use 'all' to evaluate the entire dataset."
        ),
    )
    parser.add_argument(
        "--instance-id",
        type=str,
        default=None,
        help="Evaluate one exact published instance ID.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle selected instances before evaluation (deterministic seed: 42).",
    )
    parser.add_argument(
        "--progression-mode",
        type=str,
        choices=["basic", "lifecycle"],
        default="basic",
        help=(
            "Release-chain progression strategy: "
            "'basic' replays golden release baseline per stage; "
            "'lifecycle' additionally carries prior agent stage patches forward "
            "(default: basic)."
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--generate-patches-only",
        action="store_true",
        help="Generate and persist patches without running test evaluation.",
    )
    mode_group.add_argument(
        "--generate-and-evaluate",
        action="store_true",
        help="Generate patches and evaluate each instance as soon as its patch is ready.",
    )
    mode_group.add_argument(
        "--evaluate-existing-patches",
        action="store_true",
        help="Evaluate using patches from the selected run (--run-id).",
    )
    mode_group.add_argument(
        "--run-tests-for-existing-patches",
        action="store_true",
        help="Run tests for patches from the selected run (--run-id) and write evaluation logs.",
    )
    mode_group.add_argument(
        "--evaluate-test-results",
        action="store_true",
        help="Evaluate existing evaluation logs from the selected run (--run-id).",
    )
    mode_group.add_argument(
        "--continue-run",
        action="store_true",
        help=(
            "Continue an existing run selected by --run-id. "
            "Behavior is controlled by --continue-mode."
        ),
    )
    parser.add_argument(
        "--continue-mode",
        type=str,
        choices=["generate-missing", "run-tests-missing", "report-only"],
        default=None,
        help=(
            "Behavior for --continue-run: "
            "'generate-missing' resumes generation/test execution; "
            "'run-tests-missing' skips further patch generation and runs tests only where "
            "evaluation logs are missing; "
            "'report-only' only regenerates summary from currently available artifacts. "
            "Default for --continue-run: generate-missing."
        ),
    )
    return parser.parse_args(argv)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _selected_mode(args: argparse.Namespace) -> str:
    if getattr(args, "continue_run", False):
        return "continue-run"
    if args.generate_patches_only:
        return "generate-patches-only"
    if args.generate_and_evaluate:
        return "generate-and-evaluate"
    if getattr(args, "run_tests_for_existing_patches", False):
        return "run-tests-for-existing-patches"
    if getattr(args, "evaluate_test_results", False):
        return "evaluate-test-results"
    if args.evaluate_existing_patches:
        return "evaluate-existing-patches"
    return "evaluate"


def _new_mini_swe_settings(args: argparse.Namespace) -> dict[str, object]:
    model_name = getattr(args, "model", None)
    if not isinstance(model_name, str) or not model_name.strip():
        model_name = os.environ.get("BDD_BENCH_MINI_SWE_MODEL") or MINI_SWE_DEFAULT_MODEL_NAME
    model_name = model_name.strip()
    return {
        "model_name": model_name,
        "reasoning_effort": resolve_default_reasoning_effort(model_name),
        "step_limit": _env_int(
            "BDD_BENCH_MINI_SWE_MAX_STEPS",
            MINI_SWE_DEFAULT_STEP_LIMIT,
        ),
        "command_timeout_seconds": _env_int(
            "BDD_BENCH_MINI_SWE_COMMAND_TIMEOUT",
            MINI_SWE_DEFAULT_COMMAND_TIMEOUT,
        ),
        "max_output_tokens": (
            MINI_SWE_DEFAULT_MAX_OUTPUT_TOKENS
            if MINI_SWE_DEFAULT_MAX_OUTPUT_TOKENS is not None
            else resolve_max_output_tokens(model_name)
        ),
        "cost_limit": MINI_SWE_DEFAULT_COST_LIMIT,
    }


def _new_run_agent(agent_name: str, args: argparse.Namespace) -> Agent:
    settings = _new_mini_swe_settings(args) if agent_name == "mini-swe" else None
    return get_agent(agent_name, mini_swe_settings=settings)


def _load_existing_artifact_agent(*, output_dir: Path, run_id: str) -> str:
    run_config = _load_existing_run_config(output_dir=output_dir, run_id=run_id)
    try:
        return artifact_agent_name(run_config)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def _normalize_repo_selector(value: str | None) -> str | None:
    selectors = split_repo_selector(value)
    if not selectors:
        return None
    return ",".join(selectors)


def _build_run_config_payload(
    *,
    args: argparse.Namespace,
    mode: str,
    agent_name: str,
    harness: EvaluationHarness,
    agent: Agent | None,
) -> dict[str, object]:
    selection: dict[str, object] = {}
    if args.repo is not None:
        selection["repo"] = args.repo
    if args.instance_id is not None:
        selection["instance_id"] = args.instance_id
    if args.shuffle:
        selection["shuffle"] = True

    payload: dict[str, object] = {
        "mode": mode,
        "run_id": harness.run_id,
        "agent_name": agent_name,
        "instances_file": str(args.instances_file),
        "selection": selection,
        "execution": {
            "runs": int(args.runs),
            "workers": int(getattr(harness, "workers", args.workers if args.workers else 0)),
            "quiet": bool(args.quiet),
            "live_agent_stream": bool(getattr(args, "live_agent_stream", False)),
            "progression_mode": str(getattr(args, "progression_mode", "basic")),
            "include_hidden_tests": bool(getattr(args, "include_hidden_tests", True)),
        },
    }
    if agent_name == "mini-swe":
        resolved = _new_mini_swe_settings(args)
        if agent is not None:
            resolved = {
                "model_name": getattr(agent, "model_name", resolved["model_name"]),
                "reasoning_effort": getattr(
                    agent,
                    "reasoning_effort",
                    resolved["reasoning_effort"],
                ),
                "step_limit": getattr(agent, "step_limit", resolved["step_limit"]),
                "command_timeout_seconds": getattr(
                    agent,
                    "command_timeout",
                    resolved["command_timeout_seconds"],
                ),
                "max_output_tokens": getattr(
                    agent,
                    "max_output_tokens",
                    resolved["max_output_tokens"],
                ),
                "cost_limit": getattr(agent, "cost_limit", resolved["cost_limit"]),
            }
        payload["agent"] = {
            "agent_version": resolve_mini_swe_agent_version(),
            **resolved,
        }
    return payload


def _load_existing_run_config(*, output_dir: Path, run_id: str) -> dict[str, object]:
    config_path = output_dir / run_id / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def _resolve_existing_run_id(*, output_dir: Path, run_id: str) -> str:
    normalized_run_id = run_id.strip()
    if normalized_run_id != "latest":
        return normalized_run_id
    if not output_dir.is_dir():
        raise SystemExit(f"Evaluation output directory not found: {output_dir}")

    candidates = [
        path for path in output_dir.iterdir() if path.is_dir() and (path / "config.json").is_file()
    ]
    if not candidates:
        raise SystemExit(f"No existing evaluation runs found under {output_dir}.")
    latest_run = max(candidates, key=lambda path: (path.stat().st_mtime, path.name))
    return latest_run.name


def _require_config_field(
    payload: dict[str, object],
    *,
    field: str,
    context: str = "run config",
) -> object:
    return payload[field]


def _require_dict_field(
    payload: dict[str, object],
    *,
    field: str,
    context: str = "run config",
) -> dict[str, object]:
    return cast(dict[str, object], payload[field])


def _require_non_empty_string(value: object, *, field_path: str) -> str:
    return cast(str, value).strip()


def _require_optional_string(value: object, *, field_path: str) -> str | None:
    if value is None:
        return None
    return cast(str, value)


def _require_bool(value: object, *, field_path: str) -> bool:
    return cast(bool, value)


def _require_positive_int(value: object, *, field_path: str) -> int:
    return cast(int, value)


def _apply_continue_run_settings(
    *,
    args: argparse.Namespace,
    run_config: dict[str, object],
) -> dict[str, object] | None:
    try:
        configured_agent = artifact_agent_name(run_config)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.agent = configured_agent

    configured_instances_file = _require_non_empty_string(
        _require_config_field(run_config, field="instances_file"),
        field_path="instances_file",
    )
    args.instances_file = Path(configured_instances_file)

    selection = _require_dict_field(run_config, field="selection")
    if "pr" in selection:
        raise SystemExit(
            "This run uses legacy PR selection. Rewrite its artifacts to instance IDs first."
        )
    args.repo = _require_optional_string(selection.get("repo"), field_path="selection.repo")
    args.instance_id = _require_optional_string(
        selection.get("instance_id"),
        field_path="selection.instance_id",
    )
    args.shuffle = _require_bool(selection.get("shuffle", False), field_path="selection.shuffle")

    execution = _require_dict_field(run_config, field="execution")
    args.runs = _require_positive_int(
        _require_config_field(execution, field="runs", context="run config.execution"),
        field_path="execution.runs",
    )
    args.workers = _require_positive_int(
        _require_config_field(execution, field="workers", context="run config.execution"),
        field_path="execution.workers",
    )
    args.quiet = _require_bool(
        _require_config_field(execution, field="quiet", context="run config.execution"),
        field_path="execution.quiet",
    )
    saved_live_stream = _require_bool(
        _require_config_field(
            execution,
            field="live_agent_stream",
            context="run config.execution",
        ),
        field_path="execution.live_agent_stream",
    )
    args.live_agent_stream = saved_live_stream
    saved_progression_mode = execution.get("progression_mode", "basic")
    saved_progression_mode = _require_non_empty_string(
        saved_progression_mode,
        field_path="execution.progression_mode",
    )
    args.progression_mode = saved_progression_mode
    saved_include_hidden_tests = execution.get("include_hidden_tests", True)
    args.include_hidden_tests = cast(bool, saved_include_hidden_tests)

    if configured_agent != "mini-swe":
        return None

    mini_swe = agent_config_from_run_config(run_config)
    return {
        "model_name": _require_non_empty_string(
            _require_config_field(mini_swe, field="model_name", context="run config.mini_swe"),
            field_path="mini_swe.model_name",
        ),
        "step_limit": _require_positive_int(
            _require_config_field(mini_swe, field="step_limit", context="run config.mini_swe"),
            field_path="mini_swe.step_limit",
        ),
        "command_timeout_seconds": _require_positive_int(
            _require_config_field(
                mini_swe,
                field="command_timeout_seconds",
                context="run config.mini_swe",
            ),
            field_path="mini_swe.command_timeout_seconds",
        ),
        "reasoning_effort": mini_swe.get("reasoning_effort"),
        "max_output_tokens": mini_swe.get(
            "max_output_tokens",
            MINI_SWE_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        "cost_limit": mini_swe.get("cost_limit", MINI_SWE_DEFAULT_COST_LIMIT),
    }


def main() -> None:
    """Main entry point."""
    args = parse_args()
    # Hidden bronze tests are mandatory for new runs. Archived continuations may
    # overwrite this from their saved configuration for backward compatibility.
    args.include_hidden_tests = True
    if getattr(args, "instances_file", None) is None:
        args.instances_file = default_dataset_instances_path()
    raw_continue_mode = getattr(args, "continue_mode", None)
    continue_mode = raw_continue_mode or "generate-missing"
    if not getattr(args, "continue_run", False) and raw_continue_mode is not None:
        raise SystemExit("--continue-mode can only be used with --continue-run.")
    continue_mini_swe_settings: dict[str, object] | None = None
    if getattr(args, "continue_run", False):
        run_id = getattr(args, "run_id", None)
        if not run_id:
            raise SystemExit("--run-id is required for --continue-run.")
        run_id = _resolve_existing_run_id(output_dir=args.output, run_id=run_id)
        run_config = _load_existing_run_config(output_dir=args.output, run_id=run_id)
        args.run_id = run_id
        continue_mini_swe_settings = _apply_continue_run_settings(
            args=args,
            run_config=run_config,
        )
        # Continuation should default to progress bars, not INFO log spam.
        args.quiet = True
    mode = _selected_mode(args)
    existing_artifact_modes = {
        "evaluate-existing-patches",
        "run-tests-for-existing-patches",
        "evaluate-test-results",
    }
    if mode in existing_artifact_modes:
        run_id = getattr(args, "run_id", None)
        if not run_id:
            raise SystemExit(f"--run-id is required for --{mode}.")
        run_id = _resolve_existing_run_id(output_dir=args.output, run_id=run_id)
        args.run_id = run_id
        args.agent = _load_existing_artifact_agent(output_dir=args.output, run_id=run_id)

    policy_mode = (
        "continue-generate-missing"
        if mode == "continue-run" and continue_mode == "generate-missing"
        else mode
    )
    try:
        args.agent = require_supported_generation_agent(
            agent_name=args.agent,
            mode=policy_mode,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.repo = _normalize_repo_selector(args.repo)
    if getattr(args, "live_agent_stream", None) is None:
        args.live_agent_stream = False
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    generated_metadata_dir = TemporaryDirectory(prefix="bdd-bench-eval-metadata-")
    try:
        metadata_paths = materialize_metadata_from_instances_file(
            instances_file=args.instances_file,
            destination_dir=Path(generated_metadata_dir.name),
            selected_repo=args.repo,
            selected_instance_id=args.instance_id,
        )

        if args.shuffle:
            random.seed(42)
            random.shuffle(metadata_paths)

        if not metadata_paths:
            logging.error(
                "No valid instances found in %s for the selected filters.",
                args.instances_file,
            )
            return

        logging.info(f"Found {len(metadata_paths)} instance(s) matching selection")
        agent_name = normalize_agent_name(args.agent)
        run_id = getattr(args, "run_id", None)
        reuses_existing_run = mode in {
            "evaluate-existing-patches",
            "run-tests-for-existing-patches",
            "evaluate-test-results",
            "continue-run",
        }
        if reuses_existing_run and not run_id:
            raise SystemExit(
                "--run-id is required for --evaluate-existing-patches, "
                "--run-tests-for-existing-patches, and --evaluate-test-results."
            )
        if reuses_existing_run and run_id:
            run_id = _resolve_existing_run_id(output_dir=args.output, run_id=run_id)
            args.run_id = run_id

        run_model_name: str | None = None
        if not reuses_existing_run and agent_name == "mini-swe":
            configured_model = _new_mini_swe_settings(args)["model_name"]
            run_model_name = cast(str, configured_model)

        # Create harness and run evaluation
        harness = EvaluationHarness(
            output_dir=args.output,
            runs=args.runs,
            cleanup=True,
            instances_file=args.instances_file,
            workers=args.workers,
            quiet=args.quiet,
            live_agent_stream=getattr(args, "live_agent_stream", False),
            agent_name=agent_name,
            model_name=run_model_name,
            run_id=run_id,
            create_run=not reuses_existing_run,
            progression_mode=getattr(args, "progression_mode", "basic"),
            include_hidden_tests=bool(getattr(args, "include_hidden_tests", True)),
        )
        logging.info(f"Using run directory: {harness.run_dir}")

        if getattr(args, "continue_run", False):
            patches_dir = harness.patches_dir_for_agent(agent_name)
            generation: dict[str, int] | None = None
            run_stats: dict[str, int] | None = None
            retry_cleanup: dict[str, object] | None = None
            if continue_mode == "generate-missing":
                agent = get_agent(args.agent, mini_swe_settings=continue_mini_swe_settings)
                logging.info(
                    f"Continuing run {harness.run_id} with agent {agent.name} on "
                    f"{len(metadata_paths)} selected instance(s)."
                )
                retry_cleanup = harness.cleanup_retryable_generation_artifacts(
                    agent_name=agent_name,
                    metadata_paths=metadata_paths,
                )
                # In lifecycle mode, generation must be interleaved with evaluation so
                # a failing prior stage stops downstream patch generation.
                summary, generation = harness.generate_and_evaluate_batch(
                    agent,
                    metadata_paths,
                    skip_existing_patch=True,
                    skip_instances_with_existing_eval_logs=True,
                )
                missing_patch_instances: list[str] = []
                missing_eval_log_instances: list[str] = []
            elif continue_mode == "run-tests-missing":
                logging.info(
                    f"Continuing run {harness.run_id} without patch generation on "
                    f"{len(metadata_paths)} selected instance(s): running tests only where "
                    "evaluation logs are missing."
                )
                run_stats, _missing_patch_instances_after_run = (
                    harness.run_tests_batch_from_patches(
                        agent_name=agent_name,
                        metadata_paths=metadata_paths,
                        patches_dir=patches_dir,
                        skip_missing_patches=True,
                        skip_instances_with_existing_eval_logs=True,
                    )
                )
            else:
                logging.info(
                    f"Generating report only for run {harness.run_id} "
                    f"on {len(metadata_paths)} selected instance(s)."
                )
            if continue_mode != "generate-missing":
                summary, missing_patch_instances, missing_eval_log_instances = (
                    harness.evaluate_test_results_batch_from_patches(
                        agent_name=agent_name,
                        metadata_paths=metadata_paths,
                        patches_dir=patches_dir,
                        skip_missing_patches=True,
                    )
                )

            print("\n" + "=" * 70)
            print("CONTINUE RUN SUMMARY")
            print("=" * 70)
            print(f"Run: {harness.run_id}")
            print(f"Agent: {agent_name}")
            print(f"Continuation mode: {continue_mode}")
            if generation is not None:
                cleaned_count = retry_cleanup["count"] if retry_cleanup is not None else 0
                print(f"Retryable generation artifacts cleaned: {cleaned_count}")
                print(f"Patch generated now: {generation['generated']}")
                print(f"Patch already present: {generation['skipped_existing']}")
                generation_errors = generation.get(
                    "errors",
                    generation.get("generation_errors", 0),
                )
                print(f"Patch generation errors: {generation_errors}")
                if run_stats is None:
                    print("Tests completed now: integrated with generation/evaluation")
            elif run_stats is not None:
                print("Patch generation: skipped (run-tests-missing)")
            else:
                print("Patch/test execution: skipped (report-only)")
            if run_stats is not None:
                print(f"Tests completed now: {run_stats['completed']}")
                print(
                    f"Tests skipped (existing eval logs): "
                    f"{run_stats['skipped_existing_eval_logs']}"
                )
                print(f"Run-stage errors: {run_stats['errors']}")
            print("=" * 70)
            print_summary(summary)
            if missing_patch_instances:
                print(
                    f"\nSkipped {len(missing_patch_instances)} instance(s) with missing patches "
                    f"in {patches_dir}."
                )
            if missing_eval_log_instances:
                print(
                    f"Skipped {len(missing_eval_log_instances)} instance(s) with missing evaluation logs."
                )
            return

        if args.generate_patches_only:
            agent = _new_run_agent(agent_name, args)
            logging.info(f"Using agent for patch generation: {agent.name}")
            harness.write_run_config(
                _build_run_config_payload(
                    args=args,
                    mode=mode,
                    agent_name=agent_name,
                    harness=harness,
                    agent=agent,
                ),
            )
            generation = harness.generate_patches_batch(
                agent,
                metadata_paths,
                skip_existing_patch=True,
            )
            print("\n" + "=" * 70)
            print("PATCH GENERATION SUMMARY")
            print("=" * 70)
            print(f"Agent: {generation['agent_name']}")
            print(f"Total instances: {generation['total_instances']}")
            print(f"Generated patches: {generation['generated']}")
            print(f"Skipped existing: {generation['skipped_existing']}")
            print(f"Errors: {generation['errors']}")
            print("=" * 70)
            return

        if args.generate_and_evaluate:
            agent = _new_run_agent(agent_name, args)
            logging.info(f"Using agent for generate+evaluate: {agent.name}")
            harness.write_run_config(
                _build_run_config_payload(
                    args=args,
                    mode=mode,
                    agent_name=agent_name,
                    harness=harness,
                    agent=agent,
                ),
            )
            summary, generation_stats = harness.generate_and_evaluate_batch(
                agent,
                metadata_paths,
                skip_existing_patch=True,
            )
            print("\n" + "=" * 70)
            print("PATCH GENERATION SUMMARY")
            print("=" * 70)
            print(f"Agent: {agent.name}")
            print(f"Generated patches: {generation_stats['generated']}")
            print(f"Skipped existing: {generation_stats['skipped_existing']}")
            print(f"Generation errors: {generation_stats['generation_errors']}")
            print("=" * 70)
            print_summary(summary)
            return

        if getattr(args, "run_tests_for_existing_patches", False):
            patches_dir = harness.patches_dir_for_agent(agent_name)
            run_stats, missing_patch_instances = harness.run_tests_batch_from_patches(
                agent_name=agent_name,
                metadata_paths=metadata_paths,
                patches_dir=patches_dir,
                skip_missing_patches=True,
            )
            print("\n" + "=" * 70)
            print("TEST RUN SUMMARY")
            print("=" * 70)
            print(f"Agent: {run_stats['agent_name']}")
            print(f"Total instances: {run_stats['total_instances']}")
            print(f"Completed: {run_stats['completed']}")
            print(f"Errors: {run_stats['errors']}")
            print(f"Skipped: {run_stats['skipped_instances']}")
            print("=" * 70)
            if missing_patch_instances:
                print(
                    f"\nSkipped {len(missing_patch_instances)} instance(s) with missing patches "
                    f"in {patches_dir}."
                )
            return

        if getattr(args, "evaluate_test_results", False):
            patches_dir = harness.patches_dir_for_agent(agent_name)
            summary, missing_patch_instances, missing_eval_log_instances = (
                harness.evaluate_test_results_batch_from_patches(
                    agent_name=agent_name,
                    metadata_paths=metadata_paths,
                    patches_dir=patches_dir,
                    skip_missing_patches=True,
                )
            )
            print_summary(summary)
            if missing_patch_instances:
                print(
                    f"\nSkipped {len(missing_patch_instances)} instance(s) with missing patches "
                    f"in {patches_dir}."
                )
            if missing_eval_log_instances:
                print(
                    f"Skipped {len(missing_eval_log_instances)} instance(s) with missing evaluation logs."
                )
            return

        if args.evaluate_existing_patches:
            patches_dir = harness.patches_dir_for_agent(agent_name)
            summary, missing_patch_instances = harness.evaluate_batch_from_patches(
                agent_name=agent_name,
                metadata_paths=metadata_paths,
                patches_dir=patches_dir,
                skip_missing_patches=True,
            )
            print_summary(summary)
            if missing_patch_instances:
                print(
                    f"\nSkipped {len(missing_patch_instances)} instance(s) with missing patches "
                    f"in {patches_dir}."
                )
            return

        agent = _new_run_agent(agent_name, args)
        logging.info(f"Using agent: {agent.name}")
        harness.write_run_config(
            _build_run_config_payload(
                args=args,
                mode=mode,
                agent_name=agent_name,
                harness=harness,
                agent=agent,
            ),
        )

        if len(metadata_paths) == 1:
            result = harness.evaluate_single(agent, metadata_paths[0])
            is_not_parsed = _is_parse_error(result.error)
            summary = EvaluationSummary(
                agent_name=agent.name,
                total_instances=1,
                resolved_instances=1 if result.resolved else 0,
                failed_patch_apply=(
                    1 if not result.patch_applied_successfully and not result.error else 0
                ),
                failed_tests=(
                    1 if result.patch_applied_successfully and not result.all_tests_passed else 0
                ),
                errors=1 if result.error and not is_not_parsed else 0,
                resolution_rate=1.0 if result.resolved else 0.0,
                results=[result],
                skipped_instances=0,
                not_parsed_instances=1 if is_not_parsed else 0,
                attempted_instances=1,
                attempted_resolved_instances=1 if result.resolved else 0,
                attempted_resolution_rate=1.0 if result.resolved else 0.0,
                disallowed_test_patch_instances=(
                    1
                    if result.error
                    and (
                        result.error.startswith(
                            "Agent patch modifies test files (disallowed by dataset rules):"
                        )
                        or result.error.startswith(
                            "Agent patch modifies disallowed files (test/env/config):"
                        )
                    )
                    else 0
                ),
            )
            # Save results for single evaluation
            harness._save_results(summary)
        else:
            summary = harness.evaluate_batch(agent, metadata_paths)

        # Print summary (unless quiet mode is enabled)
        print_summary(summary)
    finally:
        generated_metadata_dir.cleanup()
