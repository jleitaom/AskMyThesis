"""
run_info.py — provenance stamp for eval reports.

Every report records the git commit and the settings that produced it, so a number
in evals/ can always be traced back to the code and config behind it.
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args):
    """
    Run a git command in the repo; None if git is unavailable or it fails.
    """
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def run_stamp(config):
    """
    Git SHA, whether tracked files had uncommitted changes, UTC timestamp, and the
    given config (the settings that affect the report's numbers).
    """
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
    }
