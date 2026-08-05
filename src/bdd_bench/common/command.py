from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any


def determine_worker_count(requested: int | None) -> int:
    if requested is not None:
        return max(1, requested)
    cpu_total = os.cpu_count() or 1
    baseline = max(1, cpu_total // 8)
    logging.info(f"Detected {cpu_total} CPU(s); defaulting to {baseline} worker(s).")
    return baseline


def apply_template_replacements(content: str, replacements: dict[str, Any] | None = None) -> str:
    if not replacements:
        return content
    rendered = content
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, str(value))
    return rendered


def load_template(path: Path, replacements: dict[str, Any] | None = None) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"Failed to read template {path}: {error}") from error
    if not replacements:
        return content
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, str(value))
    return content


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    input_bytes: bytes | None = None,
    text: bool = False,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    cmd_input: str | bytes | None = None
    if input_text is not None:
        if input_bytes is not None:
            raise ValueError("Cannot provide both input_text and input_bytes")
        text = True
        cmd_input = input_text
    else:
        cmd_input = input_bytes
    try:
        return subprocess.run(
            command,
            check=True,
            cwd=str(cwd) if cwd is not None else None,
            input=cmd_input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
        )
    except subprocess.CalledProcessError as error:
        logging.error(f"Command failed: {' '.join(command)}")
        if error.stdout:
            logging.error(f"STDOUT:\n{error.stdout}")
        if error.stderr:
            logging.error(f"STDERR:\n{error.stderr}")
        raise
