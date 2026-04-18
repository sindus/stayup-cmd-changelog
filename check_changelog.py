#!/usr/bin/env python3
"""
Stayup — monitors GitHub releases and stores changelogs in PostgreSQL.

For each tracked repository, the script fetches recent GitHub releases.
If no releases exist, it falls back to cloning the repo and reading a
changelog file. New content is stored only when something has changed
since the last run. Entries older than config["retention_days"] are deleted each run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import psycopg2
import requests

# Candidate changelog filenames, checked in priority order.
CHANGELOG_NAMES = [
    "CHANGELOG.md",
    "CHANGELOG",
    "CHANGELOG.txt",
    "CHANGELOG.rst",
    "changelog.md",
    "changelog.txt",
    "CHANGES.md",
    "CHANGES",
    "CHANGES.txt",
    "HISTORY.md",
    "HISTORY.txt",
]

DDL = """
CREATE TABLE IF NOT EXISTS repository (
    id          SERIAL PRIMARY KEY,
    url         TEXT NOT NULL UNIQUE,
    type        TEXT NOT NULL,
    config      JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS connector_changelog (
    id              SERIAL PRIMARY KEY,
    repository_id   INTEGER NOT NULL REFERENCES repository(id),
    version         TEXT,
    content         TEXT NOT NULL,
    datetime        TIMESTAMPTZ,
    executed_at     TIMESTAMPTZ NOT NULL,
    success         BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS log (
    id              SERIAL PRIMARY KEY,
    repository_id   INTEGER,
    error           TEXT NOT NULL,
    executed_at     TIMESTAMPTZ NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def get_db_conn() -> psycopg2.extensions.connection:
    """Return a psycopg2 connection.

    Reads DATABASE_URL first; falls back to individual DB_* environment
    variables (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD).
    """
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return psycopg2.connect(database_url)
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )


def init_db(conn: psycopg2.extensions.connection) -> None:
    """Create tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def upsert_repository(conn: psycopg2.extensions.connection, url: str) -> int:
    """Insert a repository URL if it does not exist yet and return its id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO repository (url, type)
            VALUES (%s, 'changelog')
            ON CONFLICT (url) DO UPDATE SET url = EXCLUDED.url
            RETURNING id
            """,
            (url,),
        )
        row = cur.fetchone()
    conn.commit()
    return row[0]


def get_repositories(conn: psycopg2.extensions.connection) -> list[tuple[int, str, dict]]:
    """Return all tracked repositories of type 'changelog' as a list of (id, url, config) tuples."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, url, config FROM repository WHERE type = 'changelog' ORDER BY id")
        rows = cur.fetchall()
        return [(row[0], row[1], json.loads(row[2]) if isinstance(row[2], str) else (row[2] or {})) for row in rows]


def get_latest_entry(conn: psycopg2.extensions.connection, repository_id: int) -> tuple[str | None, str | None]:
    """Return (version, content) of the most recent successful changelog entry.

    Returns (None, None) if no entry exists yet.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT version, content FROM connector_changelog
            WHERE repository_id = %s AND success = TRUE
            ORDER BY executed_at DESC
            LIMIT 1
            """,
            (repository_id,),
        )
        row = cur.fetchone()
    return (row[0], row[1]) if row else (None, None)


def get_saved_versions(conn: psycopg2.extensions.connection, repository_id: int) -> set[str]:
    """Return the set of all release versions already saved for a repository."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT version FROM connector_changelog
            WHERE repository_id = %s AND version IS NOT NULL
            """,
            (repository_id,),
        )
        return {row[0] for row in cur.fetchall()}


def save_entry(
    conn: psycopg2.extensions.connection,
    repository_id: int,
    version: str | None,
    content: str,
    changelog_date: datetime | None,
    executed_at: datetime,
) -> None:
    """Persist a changelog entry to the database."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO connector_changelog (repository_id, version, content, datetime, executed_at, success)
            VALUES (%s, %s, %s, %s, %s, TRUE)
            """,
            (repository_id, version, content, changelog_date, executed_at),
        )
    conn.commit()


def cleanup_old_entries(conn: psycopg2.extensions.connection, repository_id: int, retention_days: int) -> None:
    """Delete changelog entries for a repository older than retention_days days."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM connector_changelog WHERE repository_id = %s AND executed_at < NOW() - %s * INTERVAL '1 day'",
            (repository_id, retention_days),
        )
    conn.commit()


def save_error(
    conn: psycopg2.extensions.connection,
    repository_id: int | None,
    error: str,
    executed_at: datetime,
) -> None:
    """Persist a retrieval error to the log table."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO log (repository_id, error, executed_at)
            VALUES (%s, %s, %s)
            """,
            (repository_id, error, executed_at),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# GitHub API — releases
# ---------------------------------------------------------------------------


def parse_github_owner_repo(url: str) -> tuple[str, str]:
    """Extract (owner, repo) from a GitHub URL.

    Example: "https://github.com/facebook/react" -> ("facebook", "react")
    """
    parts = url.rstrip("/").split("/")
    return parts[-2], parts[-1]


def get_releases(repo_url: str, limit: int = 5) -> list[tuple[str, str, datetime]]:
    """Fetch the most recent GitHub releases for a repository (newest first).

    Returns an empty list if the repository has no releases or does not exist.
    Raises requests.HTTPError for unexpected API errors.
    Uses the GITHUB_TOKEN environment variable when present to increase
    the rate limit from 60 to 5000 requests per hour.
    """
    owner, repo = parse_github_owner_repo(repo_url)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases"

    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(api_url, headers=headers, params={"per_page": limit}, timeout=10)

    if response.status_code == 404:
        return []
    response.raise_for_status()

    releases = []
    for data in response.json():
        published_at = datetime.fromisoformat(data["published_at"].replace("Z", "+00:00"))
        releases.append((data["tag_name"], data["body"] or "", published_at))
    return releases


# ---------------------------------------------------------------------------
# Fallback — git clone + changelog file
# ---------------------------------------------------------------------------


def clone_repo(repo_url: str, target_dir: str) -> None:
    """Shallow-clone a repository into target_dir.

    Raises RuntimeError if the clone fails.
    GIT_TERMINAL_PROMPT is disabled to prevent interactive credential prompts.
    """
    result = subprocess.run(
        ["git", "clone", "--depth=1", repo_url, target_dir],
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"Clone failed: {result.stderr.strip()}")


def find_changelog(repo_dir: str) -> str | None:
    """Return the path of the first changelog file found in repo_dir, or None."""
    for name in CHANGELOG_NAMES:
        path = os.path.join(repo_dir, name)
        if os.path.isfile(path):
            return path
    return None


def get_changelog_git_date(repo_dir: str, filename: str) -> datetime | None:
    """Return the committer date of the most recent commit touching filename.

    Returns None if git produces no output (e.g. untracked file).
    """
    result = subprocess.run(
        ["git", "log", "-1", "--format=%cI", "--", filename],
        capture_output=True,
        text=True,
        cwd=repo_dir,
    )
    date_str = result.stdout.strip()
    if not date_str:
        return None
    return datetime.fromisoformat(date_str)


def get_changelog_from_repo(repo_url: str) -> tuple[str | None, str, datetime | None]:
    """Clone the repository and read the changelog file.

    Returns (version=None, content, changelog_date).
    Raises RuntimeError if no changelog file is found.
    The temporary clone directory is always removed on exit.
    """
    tmp_dir = tempfile.mkdtemp(prefix="stayup_")
    repo_dir = os.path.join(tmp_dir, "repo")
    try:
        clone_repo(repo_url, repo_dir)
        changelog_path = find_changelog(repo_dir)
        if changelog_path is None:
            raise RuntimeError("No release or changelog file found.")
        changelog_name = os.path.basename(changelog_path)
        changelog_date = get_changelog_git_date(repo_dir, changelog_name)
        with open(changelog_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        return None, content, changelog_date
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def process_repository(
    conn: psycopg2.extensions.connection, repository_id: int, repository_url: str, executed_at: datetime, config: dict
) -> None:
    """Fetch the latest release(s) (or changelog file) for one repository and persist new entries.

    Release-based repos:
    - If no previous entry exists, the latest release is stored as the initial snapshot.
    - Otherwise, iterates through recent GitHub releases (newest first) and saves every
      release not already in the database, stopping when a known version is found.
      At most config["max_iterations"] (default 5) new entries are saved per run.

    File-based repos (no releases):
    - Saves the changelog file content whenever it differs from the last saved entry.

    Any exception is caught, logged to the `log` table, and printed to stderr.
    """
    max_iterations = config.get("max_iterations", 5)
    try:
        releases = get_releases(repository_url, limit=max_iterations)

        if releases:
            saved_versions = get_saved_versions(conn, repository_id)

            if not saved_versions:
                # First run: save only the latest release.
                version, content, changelog_date = releases[0]
                save_entry(conn, repository_id, version, content, changelog_date, executed_at)
            else:
                count = 0
                for version, content, changelog_date in releases:
                    if count >= max_iterations:
                        break
                    if version in saved_versions:
                        break
                    save_entry(conn, repository_id, version, content, changelog_date, executed_at)
                    count += 1
        else:
            version, content, changelog_date = get_changelog_from_repo(repository_url)
            prev_version, prev_content = get_latest_entry(conn, repository_id)
            if prev_content is None:
                save_entry(conn, repository_id, version, content, changelog_date, executed_at)
            elif content != prev_content:
                save_entry(conn, repository_id, version, content, changelog_date, executed_at)

    except Exception as e:
        save_error(conn, repository_id, str(e), executed_at)
        print(f"[{repository_url}] Error: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor GitHub releases and store changelogs.")
    parser.add_argument("--add", metavar="URL", help="Add a repository to track and exit.")
    args = parser.parse_args()

    conn = get_db_conn()
    try:
        init_db(conn)

        if args.add:
            upsert_repository(conn, args.add)
            print(f"Repository added: {args.add}")
            return

        executed_at = datetime.now(tz=timezone.utc)
        repositories = get_repositories(conn)

        if not repositories:
            print("No repositories tracked. Use --add <url> to add one.")
            return

        for repository_id, repository_url, config in repositories:
            process_repository(conn, repository_id, repository_url, executed_at, config)
            cleanup_old_entries(conn, repository_id, config.get("retention_days", 15))

    finally:
        conn.close()


if __name__ == "__main__":
    main()
