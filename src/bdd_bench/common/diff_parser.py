from __future__ import annotations

import re
import shlex


def _split_diff_git_payload(payload: str) -> tuple[str, str] | None:
    """Split a `diff --git` payload into its left/right raw paths.

    Handles both quoted headers (parsed by shlex) and unquoted headers with
    spaces in file names by splitting on the last ` b/` marker.
    """
    try:
        tokens = shlex.split(payload)
    except ValueError:
        tokens = []
    if len(tokens) == 2:
        return tokens[0], tokens[1]

    separator = " b/"
    sep_index = payload.rfind(separator)
    if sep_index <= 0:
        return None

    left = payload[:sep_index]
    right_suffix = payload[sep_index + len(separator) :]
    if not left or not right_suffix:
        return None
    return left, f"b/{right_suffix}"


def parse_diff_git_header_paths(line: str) -> tuple[str, str] | None:
    """Parse a `diff --git` line and return its raw `a/...` and `b/...` paths."""
    prefix = "diff --git "
    if not line.startswith(prefix):
        return None
    payload = line[len(prefix) :].rstrip("\r\n")
    return _split_diff_git_payload(payload)


_DIFF_GIT_RE = re.compile(r"^diff --git ", re.MULTILINE)


def _split_patch_into_file_diffs(patch: str) -> list[str]:
    """Split a unified diff into per-file chunks (each starting with ``diff --git``)."""
    positions = [m.start() for m in _DIFF_GIT_RE.finditer(patch)]
    if not positions:
        return []
    chunks: list[str] = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(patch)
        chunks.append(patch[start:end])
    return chunks


def _paths_from_chunk(chunk: str) -> list[str]:
    """Return stripped file paths referenced in a single diff chunk."""
    first_line = chunk.split("\n", 1)[0]
    parsed = parse_diff_git_header_paths(first_line)
    if parsed is None:
        return []
    paths: list[str] = []
    for raw in parsed:
        path = raw
        if path.startswith("a/") or path.startswith("b/"):
            path = path[2:]
        if path and path != "/dev/null":
            paths.append(path)
    return paths


def filter_patch_by_extensions(patch: str, extensions: set[str]) -> str:
    """Keep only diff hunks for files whose extension is in *extensions*.

    Each entry in *extensions* should include the leading dot, e.g. ``{".feature"}``.
    Returns the filtered unified-diff text (possibly empty).
    """
    kept: list[str] = []
    for chunk in _split_patch_into_file_diffs(patch):
        paths = _paths_from_chunk(chunk)
        if any(f".{p.rsplit('.', 1)[-1]}" in extensions for p in paths if "." in p):
            kept.append(chunk)
    return "".join(kept)


def extract_patch_paths(patch: str) -> list[str]:
    """Extract unique file paths from unified diff `dif f --git` headers."""
    touched: list[str] = []
    seen: set[str] = set()

    for line in patch.splitlines():
        parsed = parse_diff_git_header_paths(line)
        if parsed is None:
            continue
        for raw in parsed:
            path = raw
            if path.startswith("a/") or path.startswith("b/"):
                path = path[2:]
            if not path or path == "/dev/null" or path in seen:
                continue
            seen.add(path)
            touched.append(path)

    return touched
