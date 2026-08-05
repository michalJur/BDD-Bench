from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bdd_bench.common.command import determine_worker_count
from bdd_bench.common.flaky_tests import ensure_no_unresolved_flaky_tests
from bdd_bench.common.variants import default_dataset_instances_path
from bdd_bench.evaluation.harness_batch import _HarnessBatch, _is_parse_error
from bdd_bench.evaluation.harness_execution import _HarnessExecution
from bdd_bench.evaluation.harness_paths import _HarnessPaths
from bdd_bench.evaluation.harness_metadata import validate_metadata
from bdd_bench.evaluation.harness_policy import PRIMARY_GENERATION_AGENT, normalize_agent_name
from bdd_bench.evaluation.harness_single import _HarnessSingle

__all__ = ["EvaluationHarness", "_is_parse_error"]


class EvaluationHarness(_HarnessPaths, _HarnessExecution, _HarnessSingle, _HarnessBatch):
    def __init__(
        self,
        output_dir: Path = Path("output_evaluation"),
        runs: int = 1,
        cleanup: bool = True,
        instances_file: Path = default_dataset_instances_path(),
        workers: int | None = 1,
        quiet: bool = False,
        live_agent_stream: bool = False,
        agent_name: str = PRIMARY_GENERATION_AGENT,
        model_name: str | None = None,
        run_id: str | None = None,
        container_name_scope: str | None = None,
        create_run: bool = True,
        progression_mode: str = "basic",
        include_hidden_tests: bool = True,
    ):
        normalized_agent_name = normalize_agent_name(agent_name)
        self.base_output_dir = output_dir
        self.agent_name = normalized_agent_name
        if run_id is None:
            if not create_run:
                raise ValueError(
                    "run_id must be provided when create_run=False "
                    "(required for reusing an existing evaluation run)."
                )
            run_id = self._default_run_id(
                agent_name=normalized_agent_name,
                model_name=model_name,
            )

        self.run_id = run_id
        self.container_name_scope = container_name_scope or run_id
        self.run_dir = self.base_output_dir / run_id
        if create_run:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        elif not self.run_dir.is_dir():
            raise FileNotFoundError(
                f"Evaluation run directory not found: {self.run_dir}. "
                "Provide a valid --run-id or create a run first."
            )

        # Keep output_dir as run-scoped output root for compatibility with existing helpers.
        self.output_dir = self.run_dir
        self.config_path = self.run_dir / "config.json"
        self.runs = runs
        self.cleanup = cleanup
        self.workers = determine_worker_count(workers)
        self.quiet = quiet
        self.live_agent_stream = live_agent_stream
        self.include_hidden_tests = include_hidden_tests
        if progression_mode not in {"basic", "lifecycle"}:
            raise ValueError("progression_mode must be one of: basic, lifecycle.")
        self.progression_mode = progression_mode

        if not instances_file.is_file():
            raise FileNotFoundError(f"Evaluation manifest not found: {instances_file}")
        data = json.loads(instances_file.read_text(encoding="utf-8"))
        if "schema_version" in data or data.get("dataset_version") != "1.0.0":
            raise ValueError("Expected published dataset version 1.0.0 without schema_version.")
        ensure_no_unresolved_flaky_tests(
            data,
            context=f"EvaluationHarness ({instances_file})",
        )

        # Store full instance data for test command building
        self.instances: dict[str, dict[str, Any]] = {}
        instances = data["instances"]
        for inst in instances:
            validate_metadata(inst)
            instance_id = inst["instance_id"]
            if instance_id in self.instances:
                raise ValueError(f"Duplicate instance_id in evaluation manifest: {instance_id}")
            self.instances[instance_id] = inst

        logging.info(f"Loaded data for {len(self.instances)} instances")

    @staticmethod
    def _default_run_id(*, agent_name: str, model_name: str | None = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{agent_name}"
        if isinstance(model_name, str) and model_name.strip():
            model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name.strip())
            model_slug = model_slug.strip("-._") or "model"
            run_id = f"{run_id}_{model_slug}"
        return run_id

    def write_run_config(self, payload: dict[str, Any]) -> Path:
        """Persist run-level configuration in the run root."""
        merged_payload = dict(payload)
        merged_payload.setdefault("schema_version", 2)
        merged_payload.setdefault("run_id", self.run_id)
        merged_payload.setdefault("agent_name", self.agent_name)
        merged_payload.setdefault("written_at_utc", datetime.now(timezone.utc).isoformat())

        self.config_path.write_text(
            json.dumps(merged_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logging.info(f"Run config saved to {self.config_path}")
        return self.config_path
