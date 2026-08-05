from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def agent_config_from_run_config(run_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return one normalized agent config from current or archived run metadata."""
    agent_name = run_config.get("agent_name")
    normalized: dict[str, Any] = {}
    if isinstance(agent_name, str) and agent_name.strip():
        normalized["agent_name"] = agent_name.strip()

    current = run_config.get("agent")
    if isinstance(current, dict):
        normalized.update(current)
        return normalized

    archived = run_config.get("agent_config")
    if isinstance(archived, dict):
        normalized.update(archived)
        return normalized

    archived_mini_swe = run_config.get("mini_swe")
    if isinstance(archived_mini_swe, dict):
        normalized.update(archived_mini_swe)
    return normalized


def runtime_agent_settings(run_config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract only settings consumed when recreating a model-driven agent."""
    agent_config = agent_config_from_run_config(run_config)
    settings = {
        key: agent_config[key]
        for key in (
            "model_name",
            "step_limit",
            "command_timeout_seconds",
            "reasoning_effort",
            "max_output_tokens",
            "cost_limit",
        )
        if agent_config.get(key) is not None
    }
    return settings or None
