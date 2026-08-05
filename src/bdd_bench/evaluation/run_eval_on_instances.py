#!/usr/bin/env python
"""Run evaluation harness on instances from dataset_instances.json.

This utility reads a published evaluation manifest and runs the evaluation
harness on all instances (optionally filtered by correct_instance status).

Usage:
    # Run on all correct instances with dummy agent
    PYTHONPATH=src python -m bdd_bench.evaluation.run_eval_on_instances

    # Run on all instances (including incorrect ones)
    PYTHONPATH=src python -m bdd_bench.evaluation.run_eval_on_instances --include-incorrect

    # Use golden agent for verification
    PYTHONPATH=src python -m bdd_bench.evaluation.run_eval_on_instances --agent golden

    # Custom output directory
    PYTHONPATH=src python -m bdd_bench.evaluation.run_eval_on_instances --output output_evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from bdd_bench.common.variants import default_dataset_instances_path
from bdd_bench.evaluation.harness import EvaluationHarness, get_agent, print_summary
from bdd_bench.evaluation.harness_agents import (
    MINI_SWE_DEFAULT_COMMAND_TIMEOUT,
    MINI_SWE_DEFAULT_COST_LIMIT,
    MINI_SWE_DEFAULT_MAX_OUTPUT_TOKENS,
    MINI_SWE_DEFAULT_MODEL_NAME,
    MINI_SWE_DEFAULT_STEP_LIMIT,
    Agent,
    resolve_default_reasoning_effort,
    resolve_max_output_tokens,
    resolve_mini_swe_agent_version,
)
from bdd_bench.evaluation.harness_metadata import (
    load_metadata,
    materialize_metadata_from_instances_file,
)
from bdd_bench.evaluation.harness_policy import (
    PRIMARY_GENERATION_AGENT,
    artifact_agent_name,
    normalize_agent_name,
    require_supported_generation_agent,
)


def load_instances(
    instances_file: Path,
    *,
    include_incorrect: bool = False,
) -> list[dict]:
    """Load instances from a published evaluation manifest."""
    data = json.loads(instances_file.read_text(encoding="utf-8"))
    instances = data["instances"]
    if include_incorrect:
        return instances
    return [instance for instance in instances if instance["correct_instance"]]


def _instance_id_from_metadata(data: dict) -> str:
    return data["instance_id"]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _new_mini_swe_settings(args: argparse.Namespace) -> dict[str, object]:
    model_name = getattr(args, "model", None)
    if not isinstance(model_name, str) or not model_name.strip():
        model_name = os.environ.get("BDD_BENCH_MINI_SWE_MODEL") or MINI_SWE_DEFAULT_MODEL_NAME
    model_name = model_name.strip()
    return {
        "model_name": model_name,
        "reasoning_effort": resolve_default_reasoning_effort(model_name),
        "step_limit": _env_int("BDD_BENCH_MINI_SWE_MAX_STEPS", MINI_SWE_DEFAULT_STEP_LIMIT),
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


def _selected_mode(args: argparse.Namespace) -> str:
    if args.generate_patches_only:
        return "generate-patches-only"
    if args.generate_and_evaluate:
        return "generate-and-evaluate"
    if args.evaluate_existing_patches:
        return "evaluate-existing-patches"
    return "evaluate"


def _build_run_config_payload(
    *,
    args: argparse.Namespace,
    mode: str,
    agent_name: str,
    harness: EvaluationHarness,
    agent: Agent | None,
) -> dict[str, object]:
    selection: dict[str, object] = {}
    if args.include_incorrect:
        selection["include_incorrect"] = True

    payload: dict[str, object] = {
        "mode": mode,
        "run_id": harness.run_id,
        "agent_name": agent_name,
        "instances_file": str(args.instances_file),
        "selection": selection,
        "execution": {
            "runs": int(args.runs),
            "workers": int(getattr(harness, "workers", args.workers if args.workers else 0)),
            "quiet": bool(getattr(harness, "quiet", False)),
            "live_agent_stream": bool(args.live_agent_stream),
            "progression_mode": str(getattr(args, "progression_mode", "basic")),
            "include_hidden_tests": True,
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


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run evaluation on instances from dataset_instances.json",
    )
    parser.add_argument(
        "--instances-file",
        type=Path,
        default=None,
        help="Optional path to a published evaluation manifest.",
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
            "Model identifier used by the fixed mini-swe scaffold with the "
            "benchmark's model-specific reasoning policy."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_evaluation"),
        help="Output root directory for evaluation runs (default: output_evaluation)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Run folder name under --output. Required for --evaluate-existing-patches. "
            "Optional for generation/evaluation modes."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of test runs per instance (default: 1)",
    )
    parser.add_argument(
        "--include-incorrect",
        action="store_true",
        help="Include instances marked as incorrect",
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
        "--live-agent-stream",
        action="store_true",
        help=(
            "Stream runtime-agent command output live to console while generating patches "
            "(mini-swe only)."
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
    parser.add_argument(
        "--progression-mode",
        type=str,
        choices=["basic", "lifecycle"],
        default="basic",
        help="Release-chain progression mode (default: basic).",
    )
    args = parser.parse_args()
    args.include_hidden_tests = True
    if args.instances_file is None:
        args.instances_file = default_dataset_instances_path()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load instances
    if not args.instances_file.is_file():
        logging.error(f"Instances file not found: {args.instances_file}")
        return

    instances = load_instances(
        args.instances_file,
        include_incorrect=args.include_incorrect,
    )

    if not instances:
        logging.error("No instances found in file")
        return

    logging.info(f"Found {len(instances)} instance(s) to evaluate")

    generated_metadata_dir = TemporaryDirectory(prefix="bdd-bench-eval-metadata-")
    try:
        metadata_by_instance_id: dict[str, Path] = {}
        generated_paths = materialize_metadata_from_instances_file(
            instances_file=args.instances_file,
            destination_dir=Path(generated_metadata_dir.name),
            selected_repo=None,
        )
        for metadata_path in generated_paths:
            data = load_metadata(metadata_path)
            instance_id = _instance_id_from_metadata(data)
            metadata_by_instance_id.setdefault(instance_id, metadata_path)

        # Select metadata files matching target instances
        metadata_paths: list[Path] = []
        missing_instances: list[str] = []

        for instance in instances:
            instance_id = instance["instance_id"]
            selected_metadata_path = metadata_by_instance_id.get(instance_id)
            if selected_metadata_path:
                metadata_paths.append(selected_metadata_path)
                logging.info(f"Found metadata for {instance_id}: {selected_metadata_path}")
            else:
                missing_instances.append(instance_id)
                logging.warning(f"Metadata not found for {instance_id}")

        if missing_instances:
            logging.warning(f"Missing metadata for {len(missing_instances)} instance(s):")
            for instance_id in missing_instances:
                logging.warning(f"  - {instance_id}")

        if not metadata_paths:
            logging.error("No metadata files found for any instances")
            return

        logging.info(f"Found metadata for {len(metadata_paths)}/{len(instances)} instance(s)")
        mode = _selected_mode(args)
        reuses_existing_run = mode == "evaluate-existing-patches"
        if reuses_existing_run and not args.run_id:
            raise SystemExit("--run-id is required for --evaluate-existing-patches.")
        if reuses_existing_run:
            config_path = args.output / args.run_id / "config.json"
            try:
                run_config = json.loads(config_path.read_text(encoding="utf-8"))
                agent_name = artifact_agent_name(run_config)
            except (OSError, json.JSONDecodeError, ValueError) as error:
                raise SystemExit(
                    f"Could not load archived run config {config_path}: {error}"
                ) from error
            args.agent = agent_name
        else:
            try:
                agent_name = require_supported_generation_agent(
                    agent_name=args.agent,
                    mode=mode,
                )
            except ValueError as error:
                raise SystemExit(str(error)) from error
            args.agent = agent_name
        agent_name = normalize_agent_name(agent_name)
        run_model_name: str | None = None
        if not reuses_existing_run and agent_name == "mini-swe":
            configured_model = _new_mini_swe_settings(args)["model_name"]
            run_model_name = str(configured_model)

        # Create harness and run evaluation
        harness = EvaluationHarness(
            output_dir=args.output,
            runs=args.runs,
            cleanup=True,
            instances_file=args.instances_file,
            workers=args.workers,
            live_agent_stream=args.live_agent_stream,
            agent_name=agent_name,
            model_name=run_model_name,
            run_id=args.run_id,
            create_run=not reuses_existing_run,
            progression_mode=getattr(args, "progression_mode", "basic"),
            include_hidden_tests=True,
        )
        logging.info(f"Using run directory: {harness.run_dir}")

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
            print("\nInstances Summary:")
            print(f"  Total in evaluation manifest: {len(instances)}")
            print(f"  With metadata: {len(metadata_paths)}")
            print(f"  Missing metadata: {len(missing_instances)}")
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
            print("\nInstances Summary:")
            print(f"  Total in evaluation manifest: {len(instances)}")
            print(f"  With metadata: {len(metadata_paths)}")
            print(f"  Missing metadata: {len(missing_instances)}")
            print(f"  Evaluated: {summary.total_instances}")
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
            print("\nInstances Summary:")
            print(f"  Total in evaluation manifest: {len(instances)}")
            print(f"  With metadata: {len(metadata_paths)}")
            print(f"  Missing metadata: {len(missing_instances)}")
            print(
                f"  Missing patches (skipped): {len(missing_patch_instances)} "
                f"[dir: {patches_dir}]"
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
        summary = harness.evaluate_batch(agent, metadata_paths)

        # Print summary
        print_summary(summary)

        # Print additional context
        print("\nInstances Summary:")
        print(f"  Total in evaluation manifest: {len(instances)}")
        print(f"  Evaluated: {len(metadata_paths)}")
        print(f"  Missing metadata: {len(missing_instances)}")
        print(f"  Resolved by agent: {summary.resolved_instances}/{len(metadata_paths)}")
        print(f"  Resolution rate: {100 * summary.resolution_rate:.1f}%")
    finally:
        generated_metadata_dir.cleanup()


if __name__ == "__main__":
    main()
