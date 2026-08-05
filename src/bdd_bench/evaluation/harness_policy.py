from __future__ import annotations

from typing import Any

PRIMARY_GENERATION_AGENT = "mini-swe"
BASELINE_GENERATION_AGENTS = frozenset({"dummy", "golden"})
NEW_RUN_GENERATION_AGENTS = frozenset({PRIMARY_GENERATION_AGENT, *BASELINE_GENERATION_AGENTS})

AGENT_NAME_ALIASES = {
    "mini_swe": PRIMARY_GENERATION_AGENT,
}

SUPPORTED_ARTIFACT_AGENTS = frozenset(
    {
        "dummy",
        "golden",
        PRIMARY_GENERATION_AGENT,
    }
)

PATCH_GENERATION_MODES = frozenset(
    {
        "evaluate",
        "generate-patches-only",
        "generate-and-evaluate",
        "continue-generate-missing",
    }
)


def normalize_agent_name(agent_name: str) -> str:
    normalized = agent_name.strip()
    return AGENT_NAME_ALIASES.get(normalized, normalized)


def artifact_agent_name(run_config: dict[str, Any]) -> str:
    raw_agent_name = run_config.get("agent_name")
    if not isinstance(raw_agent_name, str) or not raw_agent_name.strip():
        raise ValueError("Existing run config is missing a non-empty agent_name.")
    agent_name = normalize_agent_name(raw_agent_name)
    if agent_name not in SUPPORTED_ARTIFACT_AGENTS:
        raise ValueError(f"Unsupported agent_name in run config: {agent_name}")
    return agent_name


def require_supported_generation_agent(*, agent_name: str, mode: str) -> str:
    normalized = normalize_agent_name(agent_name)
    if mode in PATCH_GENERATION_MODES and normalized not in NEW_RUN_GENERATION_AGENTS:
        raise ValueError(
            "Patch generation supports only mini-swe and the deterministic "
            f"dummy/golden baselines; got {normalized!r}."
        )
    return normalized
