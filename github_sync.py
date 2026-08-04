"""Durability sync for the deployed owner path: periodically commits
data/labels.csv and data/learning_curve.csv back to GitHub via the Contents
API, so accumulated swipes survive a Streamlit Cloud container restart even
though the container's local filesystem isn't guaranteed to persist.

Only ever called for the app owner (see app.py's is_owner branch) — testers'
labels are never written to disk in the first place, so there's nothing of
theirs to sync.

Requires GITHUB_TOKEN (a fine-grained PAT scoped to just this repo's
Contents read/write permission) and GITHUB_REPO (e.g.
"asim-aa/spotify-taste-pipeline"). If either is missing, commit_data_files_to_github()
skips gracefully and prints a one-time explanatory message rather than
crashing or spamming on every call.
"""

import base64
import os
from datetime import datetime
from pathlib import Path

import requests

from spotify_client import DATA_DIR

GITHUB_API_BASE = "https://api.github.com"
SYNC_FILES = ["data/labels.csv", "data/learning_curve.csv"]

_warned_missing_token = False


def _github_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_file_sha(repo: str, path: str, token: str):
    """Returns (ok, sha). sha is None if the file doesn't exist on GitHub yet
    (still ok=True — that's a valid state for a first-ever sync)."""
    try:
        resp = requests.get(
            f"{GITHUB_API_BASE}/repos/{repo}/contents/{path}",
            headers=_github_headers(token),
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"[github_sync] Network error fetching {path}: {e}")
        return False, None

    if resp.status_code == 200:
        return True, resp.json()["sha"]
    if resp.status_code == 404:
        return True, None
    print(f"[github_sync] Failed to fetch {path} (status {resp.status_code}): {resp.text[:200]}")
    return False, None


def _put_file(repo: str, path: str, token: str, content_bytes: bytes, sha, message: str) -> bool:
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("ascii"),
    }
    if sha is not None:
        payload["sha"] = sha

    try:
        resp = requests.put(
            f"{GITHUB_API_BASE}/repos/{repo}/contents/{path}",
            headers=_github_headers(token),
            json=payload,
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"[github_sync] Network error committing {path}: {e}")
        return False

    if resp.status_code in (200, 201):
        return True
    if resp.status_code == 409:
        print(
            f"[github_sync] Conflict committing {path}: the file changed on "
            "GitHub since its SHA was read. Skipping this file rather than "
            "overwriting a newer commit — it'll be retried on the next sync."
        )
        return False

    print(f"[github_sync] Failed to commit {path} (status {resp.status_code}): {resp.text[:200]}")
    return False


def commit_data_files_to_github() -> bool:
    """Commits data/labels.csv and data/learning_curve.csv back to GitHub.
    Never raises — returns True only if every file that exists locally was
    committed successfully, False otherwise (with a printed warning), same
    never-crash-the-swipe-loop pattern as spotify_client.unsave_track.
    """
    global _warned_missing_token

    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")

    if not token or not repo:
        if not _warned_missing_token:
            print(
                "[github_sync] GITHUB_TOKEN/GITHUB_REPO not set — skipping "
                "GitHub durability sync. Local writes still happen normally; "
                "they just won't survive a container restart on Streamlit "
                "Cloud. Set both env vars to enable this."
            )
            _warned_missing_token = True
        return False

    all_ok = True
    for rel_path in SYNC_FILES:
        local_path = DATA_DIR / Path(rel_path).name
        if not local_path.exists():
            continue

        ok, sha = _get_file_sha(repo, rel_path, token)
        if not ok:
            all_ok = False
            continue

        content = local_path.read_bytes()
        message = f"Sync {rel_path} from deployed app ({datetime.now().isoformat(timespec='seconds')})"
        if not _put_file(repo, rel_path, token, content, sha, message):
            all_ok = False

    return all_ok
