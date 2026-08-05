from __future__ import annotations

from pathlib import Path

from bdd_bench.common.variants import agent_visible_test_patch_key_for_evaluation
from bdd_bench.evaluation.harness_metadata import sanitize_label


class _HarnessPaths:
    """Path resolution and test-command helpers for EvaluationHarness."""

    def _instance_for_instance_id(
        self,
        instance_id: str,
        *,
        command_instance_id_override: str | None = None,
    ) -> dict:
        command_instance_id = command_instance_id_override or instance_id
        instance = self.instances.get(command_instance_id)  # type: ignore[attr-defined]
        if not instance:
            raise ValueError(f"Instance {command_instance_id} not found in loaded instances")
        return instance

    def _agent_visible_test_patch_key(
        self,
        instance_id: str,
        *,
        metadata_evaluation_test_patch: str | None = None,
        command_instance_id_override: str | None = None,
    ) -> str:
        instance = self._instance_for_instance_id(
            instance_id,
            command_instance_id_override=command_instance_id_override,
        )
        patch_key = metadata_evaluation_test_patch
        if not isinstance(patch_key, str) or not patch_key.strip():
            raw_patch_key = instance.get("evaluation_test_patch")
            patch_key = raw_patch_key if isinstance(raw_patch_key, str) else "golden_test_patch"
        return agent_visible_test_patch_key_for_evaluation(patch_key)

    def patches_dir_for_agent(self, agent_name: str) -> Path:
        """Return the run-scoped patch directory."""
        del agent_name
        return self.output_dir / "patches"  # type: ignore[attr-defined]

    def _agent_patch_path(self, *, agent_name: str, instance_id: str) -> Path:
        patch_dir = self.patches_dir_for_agent(agent_name) / sanitize_label(instance_id)
        patch_dir.mkdir(parents=True, exist_ok=True)
        return patch_dir / "agent_patch.diff"

    def resolve_patch_file(
        self,
        *,
        patches_dir: Path,
        agent_name: str,
        instance_id: str,
    ) -> Path | None:
        """Resolve a patch file for a instance ID from a directory (or direct file)."""
        del agent_name
        if patches_dir.is_file():
            return patches_dir

        safe_instance_id = sanitize_label(instance_id)
        candidate = patches_dir / safe_instance_id / "agent_patch.diff"
        return candidate if candidate.is_file() else None

    def _build_test_command(
        self,
        instance_id: str,
        repo: str,
        *,
        command_instance_id_override: str | None = None,
    ) -> str:
        """
        Load precomputed test command fields prepared in stage 7.
        """
        del repo  # Command is fully precomputed in the evaluation manifest.
        instance = self._instance_for_instance_id(
            instance_id,
            command_instance_id_override=command_instance_id_override,
        )
        raw_eval_patch_key = instance.get("evaluation_test_patch")
        eval_patch_key = (
            raw_eval_patch_key.strip()
            if isinstance(raw_eval_patch_key, str) and raw_eval_patch_key.strip()
            else "golden_test_patch"
        )
        effective_patch_key = self._effective_evaluation_test_patch_key(  # type: ignore[attr-defined]
            eval_patch_key
        )

        if effective_patch_key != eval_patch_key:
            raw_agent_command = instance.get("agent_test_command")
            raw_command_patch_key = instance.get("agent_visible_test_patch_key")
            command_patch_key = (
                raw_command_patch_key.strip()
                if isinstance(raw_command_patch_key, str) and raw_command_patch_key.strip()
                else None
            )
            if isinstance(raw_agent_command, str) and raw_agent_command.strip():
                expected_patch_key = agent_visible_test_patch_key_for_evaluation(eval_patch_key)
                if command_patch_key is None or command_patch_key == expected_patch_key:
                    return raw_agent_command.strip()
                raise ValueError(
                    "Instance "
                    f"{command_instance_id_override or instance_id} "
                    f"has 'agent_test_command' for {command_patch_key}, "
                    f"but hidden-test-disabled evaluation requires {expected_patch_key}. "
                    "Regenerate the published evaluation manifest for this mode."
                )
            raise ValueError(
                "Instance "
                f"{command_instance_id_override or instance_id} "
                "is missing 'agent_test_command' required for hidden-test-disabled evaluation. "
                "Regenerate the published evaluation manifest."
            )

        raw_command = instance.get("test_command")

        if not isinstance(raw_command, str) or not raw_command.strip():
            raise ValueError(
                "Instance "
                f"{command_instance_id_override or instance_id} "
                "is missing 'test_command' in the evaluation manifest."
            )
        return raw_command.strip()

    def _build_agent_test_command(
        self,
        instance_id: str,
        repo: str,
        *,
        metadata_evaluation_test_patch: str | None = None,
        command_instance_id_override: str | None = None,
    ) -> str:
        instance = self._instance_for_instance_id(
            instance_id,
            command_instance_id_override=command_instance_id_override,
        )
        visible_patch_key = self._agent_visible_test_patch_key(
            instance_id,
            metadata_evaluation_test_patch=metadata_evaluation_test_patch,
            command_instance_id_override=command_instance_id_override,
        )
        raw_eval_patch_key = metadata_evaluation_test_patch or instance.get("evaluation_test_patch")
        eval_patch_key = (
            raw_eval_patch_key.strip()
            if isinstance(raw_eval_patch_key, str) and raw_eval_patch_key.strip()
            else "golden_test_patch"
        )
        raw_command = instance.get("agent_test_command")
        raw_command_patch_key = instance.get("agent_visible_test_patch_key")
        command_patch_key = (
            raw_command_patch_key.strip()
            if isinstance(raw_command_patch_key, str) and raw_command_patch_key.strip()
            else None
        )
        if isinstance(raw_command, str) and raw_command.strip():
            if command_patch_key is None or command_patch_key == visible_patch_key:
                return raw_command.strip()
            raise ValueError(
                "Instance "
                f"{command_instance_id_override or instance_id} "
                f"has 'agent_test_command' for {command_patch_key}, "
                f"but the requested visible test patch is {visible_patch_key}. "
                "Regenerate the published evaluation manifest for this mode."
            )
        if visible_patch_key != eval_patch_key:
            raise ValueError(
                "Instance "
                f"{command_instance_id_override or instance_id} "
                "is missing 'agent_test_command' for the requested visible test patch. "
                "Regenerate the published evaluation manifest."
            )
        return self._build_test_command(
            instance_id,
            repo,
            command_instance_id_override=command_instance_id_override,
        )

    def _eval_log_root_for_agent(self, agent_name: str) -> Path:
        del agent_name
        return self.output_dir / "eval_logs"  # type: ignore[attr-defined]

    def _latest_eval_log_dir(
        self,
        *,
        agent_name: str,
        instance_id: str,
    ) -> Path | None:
        instance_log_root = self._eval_log_root_for_agent(agent_name) / sanitize_label(instance_id)
        if not instance_log_root.is_dir():
            return None
        candidates = sorted(path for path in instance_log_root.iterdir() if path.is_dir())
        if not candidates:
            return None
        return candidates[-1]

    def _run_log_paths_from_dir(self, log_dir: Path) -> list[Path]:
        def _run_number(path: Path) -> int:
            stem = path.stem
            if stem.startswith("run_"):
                suffix = stem[4:]
                if suffix.isdigit():
                    return int(suffix)
            return 10**9

        return sorted(log_dir.glob("run_*.log"), key=_run_number)
