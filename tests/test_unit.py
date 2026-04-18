"""Unit tests — no external dependencies (DB, git, network)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from check_changelog import (
    cleanup_old_entries,
    clone_repo,
    find_changelog,
    get_changelog_git_date,
    get_latest_entry,
    get_releases,
    get_repositories,
    get_saved_versions,
    init_db,
    parse_github_owner_repo,
    save_entry,
    save_error,
    upsert_repository,
)

# ---------------------------------------------------------------------------
# parse_github_owner_repo
# ---------------------------------------------------------------------------


class TestParseGithubOwnerRepo:
    def test_standard_url(self):
        assert parse_github_owner_repo("https://github.com/facebook/react") == ("facebook", "react")

    def test_trailing_slash(self):
        assert parse_github_owner_repo("https://github.com/vercel/next.js/") == ("vercel", "next.js")


# ---------------------------------------------------------------------------
# find_changelog
# ---------------------------------------------------------------------------


class TestFindChangelog:
    def test_finds_changelog_md(self, tmp_path):
        (tmp_path / "CHANGELOG.md").write_text("content")
        assert find_changelog(str(tmp_path)).endswith("CHANGELOG.md")

    def test_finds_first_match_in_priority_order(self, tmp_path):
        (tmp_path / "CHANGELOG.md").write_text("a")
        (tmp_path / "changelog.md").write_text("b")
        assert find_changelog(str(tmp_path)).endswith("CHANGELOG.md")

    def test_falls_back_to_other_names(self, tmp_path):
        (tmp_path / "HISTORY.md").write_text("content")
        assert find_changelog(str(tmp_path)).endswith("HISTORY.md")

    def test_returns_none_when_not_found(self, tmp_path):
        assert find_changelog(str(tmp_path)) is None

    def test_ignores_directories_with_changelog_name(self, tmp_path):
        (tmp_path / "CHANGELOG.md").mkdir()
        assert find_changelog(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# clone_repo
# ---------------------------------------------------------------------------


class TestCloneRepo:
    @patch("check_changelog.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        clone_repo("https://example.com/repo", "/tmp/dest")
        args, kwargs = mock_run.call_args
        assert args[0] == ["git", "clone", "--depth=1", "https://example.com/repo", "/tmp/dest"]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"

    @patch("check_changelog.subprocess.run")
    def test_failure_raises_runtime_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=128, stderr="repo not found")
        with pytest.raises(RuntimeError, match="Clone failed"):
            clone_repo("https://example.com/bad", "/tmp/dest")


# ---------------------------------------------------------------------------
# get_changelog_git_date
# ---------------------------------------------------------------------------


class TestGetChangelogGitDate:
    @patch("check_changelog.subprocess.run")
    def test_returns_datetime_on_valid_output(self, mock_run):
        mock_run.return_value = MagicMock(stdout="2024-06-15T12:00:00+00:00\n")
        result = get_changelog_git_date("/repo", "CHANGELOG.md")
        assert result == datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    @patch("check_changelog.subprocess.run")
    def test_returns_none_on_empty_output(self, mock_run):
        mock_run.return_value = MagicMock(stdout="")
        result = get_changelog_git_date("/repo", "CHANGELOG.md")
        assert result is None


# ---------------------------------------------------------------------------
# get_releases
# ---------------------------------------------------------------------------


class TestGetReleases:
    @patch("check_changelog.requests.get")
    def test_returns_list_of_releases(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {"tag_name": "v1.2.3", "body": "- Fix bug", "published_at": "2024-06-15T12:00:00Z"},
                {"tag_name": "v1.2.2", "body": "- Previous", "published_at": "2024-05-01T00:00:00Z"},
            ],
        )
        releases = get_releases("https://github.com/user/repo")
        assert len(releases) == 2
        assert releases[0][0] == "v1.2.3"
        assert releases[0][1] == "- Fix bug"
        assert releases[1][0] == "v1.2.2"

    @patch("check_changelog.requests.get")
    def test_returns_empty_list_on_404(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        result = get_releases("https://github.com/user/repo")
        assert result == []

    @patch("check_changelog.requests.get")
    def test_returns_empty_list_when_no_releases(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        result = get_releases("https://github.com/user/repo")
        assert result == []

    @patch("check_changelog.requests.get")
    def test_empty_body_returns_empty_string(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"tag_name": "v1.0.0", "body": None, "published_at": "2024-01-01T00:00:00Z"}],
        )
        releases = get_releases("https://github.com/user/repo")
        assert releases[0][1] == ""

    @patch("check_changelog.requests.get")
    def test_sends_token_header_when_set(self, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "mytoken")
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"tag_name": "v1.0.0", "body": "content", "published_at": "2024-01-01T00:00:00Z"}],
        )
        get_releases("https://github.com/user/repo")
        headers = mock_get.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer mytoken"

    @patch("check_changelog.requests.get")
    def test_passes_per_page_param(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        get_releases("https://github.com/user/repo", limit=3)
        params = mock_get.call_args[1]["params"]
        assert params["per_page"] == 3


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def make_conn_mock():
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


class TestInitDb:
    def test_executes_ddl_and_commits(self):
        conn, cursor = make_conn_mock()
        init_db(conn)
        assert cursor.execute.call_count == 1  # DDL only
        conn.commit.assert_called_once()


class TestUpsertRepository:
    def test_returns_id(self):
        conn, cursor = make_conn_mock()
        cursor.fetchone.return_value = (42,)
        result = upsert_repository(conn, "https://github.com/user/repo")
        assert result == 42
        sql = cursor.execute.call_args[0][0]
        assert "INSERT INTO repository" in sql
        assert "ON CONFLICT" in sql

    def test_passes_url_as_parameter(self):
        conn, cursor = make_conn_mock()
        cursor.fetchone.return_value = (1,)
        upsert_repository(conn, "https://github.com/user/repo")
        params = cursor.execute.call_args[0][1]
        assert params == ("https://github.com/user/repo",)


class TestGetRepositories:
    def test_returns_list_of_tuples(self):
        conn, cursor = make_conn_mock()
        cursor.fetchall.return_value = [(1, "https://github.com/a/b", {"max_entries": 3}), (2, "https://github.com/c/d", {})]
        result = get_repositories(conn)
        assert result == [(1, "https://github.com/a/b", {"max_entries": 3}), (2, "https://github.com/c/d", {})]

    def test_returns_empty_list_when_no_repos(self):
        conn, cursor = make_conn_mock()
        cursor.fetchall.return_value = []
        result = get_repositories(conn)
        assert result == []

    def test_filters_by_changelog_type(self):
        conn, cursor = make_conn_mock()
        cursor.fetchall.return_value = []
        get_repositories(conn)
        sql = cursor.execute.call_args[0][0]
        assert "type = 'changelog'" in sql


class TestGetLatestEntry:
    def test_returns_version_and_content_when_found(self):
        conn, cursor = make_conn_mock()
        cursor.fetchone.return_value = ("v1.0.0", "release notes")
        version, content = get_latest_entry(conn, 1)
        assert version == "v1.0.0"
        assert content == "release notes"

    def test_returns_none_none_when_no_entry(self):
        conn, cursor = make_conn_mock()
        cursor.fetchone.return_value = None
        version, content = get_latest_entry(conn, 1)
        assert version is None
        assert content is None


class TestGetSavedVersions:
    def test_returns_set_of_versions(self):
        conn, cursor = make_conn_mock()
        cursor.fetchall.return_value = [("v1.0.0",), ("v1.1.0",)]
        result = get_saved_versions(conn, 1)
        assert result == {"v1.0.0", "v1.1.0"}

    def test_returns_empty_set_when_no_entries(self):
        conn, cursor = make_conn_mock()
        cursor.fetchall.return_value = []
        result = get_saved_versions(conn, 1)
        assert result == set()


class TestSaveEntry:
    def test_inserts_with_version_and_commits(self):
        conn, cursor = make_conn_mock()
        executed_at = datetime.now(tz=timezone.utc)
        save_entry(conn, 1, "v1.0.0", "## v1.0\n- init", None, executed_at)
        cursor.execute.assert_called_once()
        conn.commit.assert_called_once()
        params = cursor.execute.call_args[0][1]
        assert params[0] == 1  # repository_id
        assert params[1] == "v1.0.0"  # version
        assert params[2] == "## v1.0\n- init"  # content
        assert params[4] == executed_at

    def test_success_flag_in_sql(self):
        conn, cursor = make_conn_mock()
        save_entry(conn, 1, None, "content", None, datetime.now(tz=timezone.utc))
        sql = cursor.execute.call_args[0][0]
        assert "TRUE" in sql

    def test_no_diff_column_in_sql(self):
        conn, cursor = make_conn_mock()
        save_entry(conn, 1, None, "content", None, datetime.now(tz=timezone.utc))
        sql = cursor.execute.call_args[0][0]
        assert "diff" not in sql.lower()


class TestSaveError:
    def test_inserts_error_and_commits(self):
        conn, cursor = make_conn_mock()
        executed_at = datetime.now(tz=timezone.utc)
        save_error(conn, 5, "something went wrong", executed_at)
        cursor.execute.assert_called_once()
        conn.commit.assert_called_once()
        params = cursor.execute.call_args[0][1]
        assert params == (5, "something went wrong", executed_at)

    def test_accepts_none_repository_id(self):
        conn, cursor = make_conn_mock()
        save_error(conn, None, "error", datetime.now(tz=timezone.utc))
        params = cursor.execute.call_args[0][1]
        assert params[0] is None


class TestCleanupOldEntries:
    def test_executes_delete_and_commits(self):
        conn, cursor = make_conn_mock()
        cleanup_old_entries(conn, 1, 15)
        cursor.execute.assert_called_once()
        conn.commit.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        assert "DELETE FROM connector_changelog" in sql

    def test_uses_repository_id_and_retention_days(self):
        conn, cursor = make_conn_mock()
        cleanup_old_entries(conn, 7, 30)
        params = cursor.execute.call_args[0][1]
        assert params == (7, 30)
