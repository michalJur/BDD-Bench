from __future__ import annotations

import re

from bdd_bench.common.diff_parser import (
    extract_patch_paths as _extract_patch_paths,
    parse_diff_git_header_paths,
)

_DIFF_GIT_RE = re.compile(r"^diff --git ", re.MULTILINE)


def extract_patch_paths(patch: str) -> list[str]:
    """Extract unique file paths from `diff --git` headers in a unified diff."""
    return _extract_patch_paths(patch)


def filter_patch_excluding_paths(patch: str, excluded_paths: set[str]) -> str:
    """Return a unified diff with file hunks touching ``excluded_paths`` removed."""
    if not excluded_paths:
        return patch

    positions = [match.start() for match in _DIFF_GIT_RE.finditer(patch)]
    if not positions:
        return patch

    kept_chunks: list[str] = []
    normalized_excluded = {path.strip() for path in excluded_paths if path.strip()}
    for index, start in enumerate(positions):
        end = positions[index + 1] if index + 1 < len(positions) else len(patch)
        chunk = patch[start:end]
        first_line = chunk.split("\n", 1)[0]
        parsed = parse_diff_git_header_paths(first_line)
        if parsed is None:
            kept_chunks.append(chunk)
            continue
        chunk_paths: set[str] = set()
        for raw in parsed:
            path = raw
            if path.startswith("a/") or path.startswith("b/"):
                path = path[2:]
            if path and path != "/dev/null":
                chunk_paths.add(path)
        if chunk_paths & normalized_excluded:
            continue
        kept_chunks.append(chunk)

    return "".join(kept_chunks)
