from __future__ import annotations

import logging
import shlex

from bdd_bench.common.command import run_command


def build_historyless_git_reinit_script(repo_hint: str) -> str:
    """Build a shell script that rewrites repository history to a single baseline commit.

    The script:
    1. Locates the repository root using the provided hint and common fallbacks.
    2. Removes all nested `.git` directories/files inside that root.
    3. Re-initializes git and creates a single baseline commit.
    4. Tags the baseline as `initial-pr-commit`.
    """
    normalized_hint = (repo_hint or "").strip()
    quoted_hint = shlex.quote(normalized_hint)
    return f"""set -euo pipefail
repo_hint={quoted_hint}
candidate_roots=()
if [ -n "$repo_hint" ]; then
  candidate_roots+=("$repo_hint")
  case "$repo_hint" in
    /*) ;;
    *) candidate_roots+=("/$repo_hint") ;;
  esac
  candidate_roots+=("/workspace/$repo_hint" "/repo/$repo_hint")
fi

unique_candidates=()
for candidate in "${{candidate_roots[@]}}"; do
  [ -n "$candidate" ] || continue
  duplicate=0
  for existing in "${{unique_candidates[@]}}"; do
    if [ "$existing" = "$candidate" ]; then
      duplicate=1
      break
    fi
  done
  if [ "$duplicate" -eq 0 ]; then
    unique_candidates+=("$candidate")
  fi
done

repo_root=""
for candidate in "${{unique_candidates[@]}}"; do
  if [ -d "$candidate" ]; then
    resolved=$(git -C "$candidate" rev-parse --show-toplevel 2>/dev/null || true)
    if [ -n "$resolved" ] && [ -d "$resolved" ]; then
      repo_root="$resolved"
      break
    fi
    if [ -e "$candidate/.git" ]; then
      repo_root="$candidate"
      break
    fi
  fi
done

if [ -z "$repo_root" ]; then
  first_git=$(find / -xdev -maxdepth 4 \\( -type d -o -type f \\) -name .git 2>/dev/null | head -n 1 || true)
  if [ -n "$first_git" ]; then
    repo_root=$(dirname "$first_git")
  fi
fi

if [ -z "$repo_root" ]; then
  for candidate in "${{unique_candidates[@]}}"; do
    if [ -d "$candidate" ]; then
      repo_root="$candidate"
      break
    fi
  done
fi

if [ -z "$repo_root" ]; then
  echo "Could not determine repository root for repo_hint=$repo_hint" >&2
  exit 1
fi

find "$repo_root" -mindepth 1 \\( -type d -name .git -o -type f -name .git \\) -exec rm -rf {{}} + 2>/dev/null || true
rm -rf "$repo_root/.git"

git -C "$repo_root" init >/dev/null
git -C "$repo_root" add -A
if [ -n "$(git -C "$repo_root" status --porcelain)" ]; then
  git -C "$repo_root" -c user.name='bdd-bench' -c user.email='bdd-bench@local' \\
    commit --no-gpg-sign -m 'bdd-bench historyless baseline' >/dev/null
fi
if ! git -C "$repo_root" rev-parse --verify HEAD >/dev/null 2>&1; then
  git -C "$repo_root" -c user.name='bdd-bench' -c user.email='bdd-bench@local' \\
    commit --allow-empty --no-gpg-sign -m 'bdd-bench historyless baseline' >/dev/null
fi
git -C "$repo_root" tag -f initial-pr-commit >/dev/null 2>&1 || true

commit_count=$(git -C "$repo_root" rev-list --count --all 2>/dev/null || echo 0)
if [ "$commit_count" -gt 1 ]; then
  echo "History stripping failed for $repo_root (commit_count=$commit_count)" >&2
  exit 1
fi

echo "$repo_root"
"""


def sanitize_container_git_history(*, container_name: str, repo_hint: str) -> str:
    """Ensure the repo inside a running container has no historical commits."""
    completed = run_command(
        [
            "docker",
            "exec",
            container_name,
            "bash",
            "-lc",
            build_historyless_git_reinit_script(repo_hint),
        ],
        text=True,
    )
    stdout = completed.stdout
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
    repo_root = stdout.strip().splitlines()[-1].strip() if stdout else ""
    if not repo_root:
        raise RuntimeError(
            f"Failed to sanitize git history in {container_name}: repository root not reported."
        )
    logging.info(
        "Sanitized git history in container %s at repository root %s",
        container_name,
        repo_root,
    )
    return repo_root
