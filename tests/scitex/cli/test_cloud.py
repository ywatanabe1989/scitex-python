#!/usr/bin/env python3
"""Tests for scitex.cli.cloud - Cloud/Git CLI commands."""

import os
import socket
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scitex.cli.cloud import (
    SyncLock,
    check_large_files,
    check_workspace_sync_status,
    cloud,
    is_in_workspace,
)


class TestCloudGroup:
    """Tests for the cloud command group."""

    def test_cloud_help(self):
        """Test that cloud help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["--help"])
        assert result.exit_code == 0
        assert "Cloud/Git operations" in result.output

    def test_cloud_short_help(self):
        """Test that -h also shows help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["-h"])
        assert result.exit_code == 0
        assert "Cloud/Git operations" in result.output

    def test_cloud_has_subcommands(self):
        """Test that all expected subcommands are registered."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["--help"])
        expected_commands = [
            "login",
            "clone",
            "create",
            "list",
            "search",
            "delete",
            "fork",
            "pr",
            "issue",
            "push",
            "pull",
            "status",
            "enrich",
        ]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in cloud help"


class TestIsInWorkspace:
    """Tests for workspace detection."""

    def test_in_workspace_via_env_var(self):
        """Test workspace detection via SCITEX_WORKSPACE env var."""
        with patch.dict(os.environ, {"SCITEX_WORKSPACE": "1"}):
            assert is_in_workspace() is True

    def test_in_workspace_via_hostname(self):
        """Test workspace detection via hostname pattern."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("socket.gethostname", return_value="scitex-workspace-abc123"):
                assert is_in_workspace() is True

    def test_in_workspace_via_marker_file(self):
        """Test workspace detection via marker file."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("socket.gethostname", return_value="regular-host"):
                with patch("pathlib.Path.exists", return_value=True):
                    assert is_in_workspace() is True

    def test_not_in_workspace(self):
        """Test detection when NOT in workspace."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove SCITEX_WORKSPACE if it exists
            os.environ.pop("SCITEX_WORKSPACE", None)
            with patch("socket.gethostname", return_value="my-laptop"):
                with patch("pathlib.Path.exists", return_value=False):
                    assert is_in_workspace() is False


class TestCheckWorkspaceSyncStatus:
    """Tests for workspace sync status checking."""

    def test_uncommitted_changes(self):
        """Test detection of uncommitted changes."""
        with patch("subprocess.run") as mock_run:
            # First call for git status returns uncommitted changes
            mock_result = MagicMock()
            mock_result.stdout = "M file.py\n"
            mock_run.return_value = mock_result

            needs_sync, message = check_workspace_sync_status()
            assert needs_sync is True
            assert "Uncommitted changes" in message

    def test_unpushed_commits(self):
        """Test detection of unpushed commits."""
        with patch("subprocess.run") as mock_run:

            def run_side_effect(*args, **kwargs):
                result = MagicMock()
                cmd = args[0]
                if "status" in cmd:
                    result.stdout = ""  # No uncommitted changes
                elif "rev-list" in cmd:
                    result.stdout = "3\n"  # 3 unpushed commits
                    result.returncode = 0
                return result

            mock_run.side_effect = run_side_effect

            needs_sync, message = check_workspace_sync_status()
            assert needs_sync is True
            assert "Unpushed commits" in message

    def test_synced(self):
        """Test detection when workspace is synced."""
        with patch("subprocess.run") as mock_run:

            def run_side_effect(*args, **kwargs):
                result = MagicMock()
                cmd = args[0]
                if "status" in cmd:
                    result.stdout = ""  # No uncommitted changes
                elif "rev-list" in cmd:
                    result.stdout = "0\n"  # No unpushed commits
                    result.returncode = 0
                return result

            mock_run.side_effect = run_side_effect

            needs_sync, message = check_workspace_sync_status()
            assert needs_sync is False
            assert "Synced" in message


class TestCheckLargeFiles:
    """Tests for large file detection."""

    def test_no_large_files(self):
        """Test when no large files exist."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = ""
            mock_run.return_value = mock_result

            large_files = check_large_files(threshold_mb=100)
            assert large_files == []

    def test_large_files_detected(self):
        """Test detection of large files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a "large" file (we'll use a lower threshold for testing)
            large_file = Path(tmpdir) / "large_file.bin"
            # Create file with 10KB of data
            large_file.write_bytes(b"x" * 10 * 1024)

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = "large_file.bin\n"
                mock_run.return_value = mock_result

                with patch("os.getcwd", return_value=tmpdir):
                    # Use very low threshold (0.001 MB = 1KB)
                    large_files = check_large_files(threshold_mb=0.001)
                    assert len(large_files) == 1
                    assert large_files[0][0] == "large_file.bin"
                    assert large_files[0][1] > 0.001


class TestSyncLock:
    """Tests for the SyncLock class."""

    def test_lock_acquisition(self):
        """Test that lock can be acquired."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "test.lock"
            lock = SyncLock(lock_path=str(lock_path), timeout=5)

            with lock:
                # Lock should be acquired
                assert lock.lock_file is not None
                # Lock file should exist
                assert lock_path.exists()

    def test_lock_release(self):
        """Test that lock is released after context exit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "test.lock"
            lock = SyncLock(lock_path=str(lock_path), timeout=5)

            with lock:
                pass  # Just enter and exit

            # After exit, lock file should still exist but be releasable
            # We can acquire a new lock
            lock2 = SyncLock(lock_path=str(lock_path), timeout=1)
            with lock2:
                assert True  # Should be able to acquire


class TestCloudLogin:
    """Tests for cloud login command."""

    def test_login_help(self):
        """Test login command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["login", "--help"])
        assert result.exit_code == 0
        assert "Login to SciTeX Cloud" in result.output


class TestCloudClone:
    """Tests for cloud clone command."""

    def test_clone_help(self):
        """Test clone command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["clone", "--help"])
        assert result.exit_code == 0
        assert "Clone a repository" in result.output

    def test_clone_missing_argument(self):
        """Test clone command without required argument."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["clone"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output


class TestCloudCreate:
    """Tests for cloud create command."""

    def test_create_help(self):
        """Test create command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["create", "--help"])
        assert result.exit_code == 0
        assert "Create a new repository" in result.output


class TestCloudList:
    """Tests for cloud list command."""

    def test_list_help(self):
        """Test list command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["list", "--help"])
        assert result.exit_code == 0
        assert "List repositories" in result.output


class TestCloudSearch:
    """Tests for cloud search command."""

    def test_search_help(self):
        """Test search command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["search", "--help"])
        assert result.exit_code == 0
        assert "Search for repositories" in result.output


class TestCloudPR:
    """Tests for cloud PR subcommand group."""

    def test_pr_help(self):
        """Test PR command group help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["pr", "--help"])
        assert result.exit_code == 0
        assert "Pull request operations" in result.output

    def test_pr_create_help(self):
        """Test PR create command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["pr", "create", "--help"])
        assert result.exit_code == 0
        assert "Create a pull request" in result.output

    def test_pr_list_help(self):
        """Test PR list command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["pr", "list", "--help"])
        assert result.exit_code == 0
        assert "List pull requests" in result.output


class TestCloudIssue:
    """Tests for cloud issue subcommand group."""

    def test_issue_help(self):
        """Test issue command group help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["issue", "--help"])
        assert result.exit_code == 0
        assert "Issue operations" in result.output

    def test_issue_create_help(self):
        """Test issue create command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["issue", "create", "--help"])
        assert result.exit_code == 0
        assert "Create an issue" in result.output

    def test_issue_create_requires_title(self):
        """Test issue create requires title."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["issue", "create"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()


class TestCloudPushPullStatus:
    """Tests for push, pull, and status commands."""

    def test_push_help(self):
        """Test push command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["push", "--help"])
        assert result.exit_code == 0
        assert "Push local changes" in result.output

    def test_pull_help(self):
        """Test pull command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["pull", "--help"])
        assert result.exit_code == 0
        assert "Pull workspace changes" in result.output

    def test_status_help(self):
        """Test status command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["status", "--help"])
        assert result.exit_code == 0
        assert "Show repository status" in result.output

    def test_push_blocked_in_workspace(self):
        """Test push is blocked when in workspace."""
        runner = CliRunner()
        with patch("scitex.cli.cloud.is_in_workspace", return_value=True):
            result = runner.invoke(cloud, ["push"])
            assert result.exit_code == 1
            assert "inside a SciTeX workspace" in result.output

    def test_pull_blocked_in_workspace(self):
        """Test pull is blocked when in workspace."""
        runner = CliRunner()
        with patch("scitex.cli.cloud.is_in_workspace", return_value=True):
            result = runner.invoke(cloud, ["pull"])
            assert result.exit_code == 1
            assert "inside a SciTeX workspace" in result.output

    def test_status_blocked_in_workspace(self):
        """Test status is blocked when in workspace."""
        runner = CliRunner()
        with patch("scitex.cli.cloud.is_in_workspace", return_value=True):
            result = runner.invoke(cloud, ["status"])
            assert result.exit_code == 1
            assert "inside a SciTeX workspace" in result.output


class TestCloudEnrich:
    """Tests for cloud enrich command."""

    def test_enrich_help(self):
        """Test enrich command help."""
        runner = CliRunner()
        result = runner.invoke(cloud, ["enrich", "--help"])
        assert result.exit_code == 0
        assert "Enrich BibTeX file" in result.output

    def test_enrich_missing_api_key(self):
        """Test enrich command without API key."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False) as f:
            f.write(b"@article{test,}")
            f.flush()

            result = runner.invoke(
                cloud, ["enrich", "-i", f.name, "-o", "/tmp/output.bib"]
            )
            assert result.exit_code == 1
            assert "API key required" in result.output

            os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
