#!/usr/bin/env python3
# File: tests/scitex/security/test_github.py

"""Tests for scitex.security.github module."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex.security.github import (
    GitHubSecurityError,
    _run_gh_command,
    check_gh_auth,
    check_github_alerts,
    format_alerts_report,
    get_code_scanning_alerts,
    get_dependabot_alerts,
    get_latest_alerts_file,
    get_secret_alerts,
    save_alerts_to_file,
)


class TestGitHubSecurityError:
    """Tests for GitHubSecurityError exception."""

    def test_can_raise_error(self):
        """Test that GitHubSecurityError can be raised."""
        with pytest.raises(GitHubSecurityError):
            raise GitHubSecurityError("Test error")

    def test_error_message(self):
        """Test that error message is preserved."""
        with pytest.raises(GitHubSecurityError, match="Custom error message"):
            raise GitHubSecurityError("Custom error message")

    def test_inherits_from_exception(self):
        """Test that GitHubSecurityError inherits from Exception."""
        error = GitHubSecurityError("test")
        assert isinstance(error, Exception)


class TestRunGhCommand:
    """Tests for _run_gh_command function."""

    @patch("subprocess.run")
    def test_successful_command(self, mock_run):
        """Test successful gh command execution."""
        mock_run.return_value = MagicMock(stdout="command output", returncode=0)
        result = _run_gh_command(["auth", "status"])
        assert result == "command output"
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_command_with_args(self, mock_run):
        """Test gh command with arguments."""
        mock_run.return_value = MagicMock(stdout="output", returncode=0)
        _run_gh_command(["api", "/repos/owner/repo"])
        call_args = mock_run.call_args
        assert call_args[0][0] == ["gh", "api", "/repos/owner/repo"]

    @patch("subprocess.run")
    def test_command_failure_raises_error(self, mock_run):
        """Test that CalledProcessError raises GitHubSecurityError."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "gh", stderr="error message"
        )
        with pytest.raises(GitHubSecurityError, match="GitHub CLI error"):
            _run_gh_command(["auth", "status"])

    @patch("subprocess.run")
    def test_gh_not_found_raises_error(self, mock_run):
        """Test that FileNotFoundError raises GitHubSecurityError."""
        mock_run.side_effect = FileNotFoundError("gh not found")
        with pytest.raises(GitHubSecurityError, match="GitHub CLI .* not found"):
            _run_gh_command(["auth", "status"])


class TestCheckGhAuth:
    """Tests for check_gh_auth function."""

    @patch("subprocess.run")
    def test_authenticated_returns_true(self, mock_run):
        """Test that authenticated user returns True."""
        mock_run.return_value = MagicMock(returncode=0)
        assert check_gh_auth() is True

    @patch("subprocess.run")
    def test_not_authenticated_returns_false(self, mock_run):
        """Test that non-authenticated user returns False."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "gh")
        assert check_gh_auth() is False

    @patch("subprocess.run")
    def test_gh_not_found_returns_false(self, mock_run):
        """Test that missing gh CLI returns False."""
        mock_run.side_effect = FileNotFoundError()
        assert check_gh_auth() is False


class TestGetSecretAlerts:
    """Tests for get_secret_alerts function."""

    @patch("scitex.security.github._run_gh_command")
    def test_returns_parsed_alerts(self, mock_run):
        """Test that alerts are parsed from JSON."""
        alert_data = {"state": "open", "secretType": "API Key", "url": "http://test"}
        mock_run.return_value = json.dumps(alert_data)
        alerts = get_secret_alerts("owner/repo")
        assert len(alerts) == 1
        assert alerts[0]["state"] == "open"

    @patch("scitex.security.github._run_gh_command")
    def test_empty_output_returns_empty_list(self, mock_run):
        """Test that empty output returns empty list."""
        mock_run.return_value = ""
        alerts = get_secret_alerts()
        assert alerts == []

    @patch("scitex.security.github._run_gh_command")
    def test_whitespace_output_returns_empty_list(self, mock_run):
        """Test that whitespace-only output returns empty list."""
        mock_run.return_value = "   \n   "
        alerts = get_secret_alerts()
        assert alerts == []

    @patch("scitex.security.github._run_gh_command")
    def test_multiple_alerts(self, mock_run):
        """Test parsing multiple line-delimited JSON alerts."""
        alert1 = json.dumps({"state": "open", "secretType": "Key1", "url": "url1"})
        alert2 = json.dumps({"state": "closed", "secretType": "Key2", "url": "url2"})
        mock_run.return_value = f"{alert1}\n{alert2}"
        alerts = get_secret_alerts()
        assert len(alerts) == 2

    @patch("scitex.security.github._run_gh_command")
    def test_error_returns_empty_list(self, mock_run):
        """Test that GitHubSecurityError returns empty list."""
        mock_run.side_effect = GitHubSecurityError("API error")
        alerts = get_secret_alerts()
        assert alerts == []

    @patch("scitex.security.github._run_gh_command")
    def test_custom_repo_path(self, mock_run):
        """Test that custom repo modifies API path."""
        mock_run.return_value = ""
        get_secret_alerts("myorg/myrepo")
        call_args = mock_run.call_args[0][0]
        assert "/repos/myorg/myrepo/" in call_args[1]


class TestGetDependabotAlerts:
    """Tests for get_dependabot_alerts function."""

    @patch("scitex.security.github._run_gh_command")
    def test_returns_parsed_alerts(self, mock_run):
        """Test that dependabot alerts are parsed correctly."""
        alert_data = {
            "state": "open",
            "severity": "high",
            "summary": "Vulnerability",
            "package": "test-pkg",
            "cve": "CVE-2024-1234",
            "url": "http://test",
            "created_at": "2024-01-01",
        }
        mock_run.return_value = json.dumps(alert_data)
        alerts = get_dependabot_alerts()
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "high"

    @patch("scitex.security.github._run_gh_command")
    def test_empty_output_returns_empty_list(self, mock_run):
        """Test that empty output returns empty list."""
        mock_run.return_value = ""
        alerts = get_dependabot_alerts()
        assert alerts == []

    @patch("scitex.security.github._run_gh_command")
    def test_error_returns_empty_list(self, mock_run):
        """Test that errors return empty list."""
        mock_run.side_effect = GitHubSecurityError("error")
        alerts = get_dependabot_alerts()
        assert alerts == []

    @patch("scitex.security.github._run_gh_command")
    def test_custom_repo_path(self, mock_run):
        """Test that custom repo modifies API path."""
        mock_run.return_value = ""
        get_dependabot_alerts("org/repo")
        call_args = mock_run.call_args[0][0]
        assert "/repos/org/repo/" in call_args[1]


class TestGetCodeScanningAlerts:
    """Tests for get_code_scanning_alerts function."""

    @patch("scitex.security.github._run_gh_command")
    def test_returns_parsed_alerts(self, mock_run):
        """Test that code scanning alerts are parsed correctly."""
        alert_data = {
            "state": "open",
            "severity": "error",
            "description": "SQL Injection",
            "location": "app.py",
            "line": 42,
            "url": "http://test",
            "created_at": "2024-01-01",
        }
        mock_run.return_value = json.dumps(alert_data)
        alerts = get_code_scanning_alerts()
        assert len(alerts) == 1
        assert alerts[0]["description"] == "SQL Injection"

    @patch("scitex.security.github._run_gh_command")
    def test_empty_output_returns_empty_list(self, mock_run):
        """Test that empty output returns empty list."""
        mock_run.return_value = ""
        alerts = get_code_scanning_alerts()
        assert alerts == []

    @patch("scitex.security.github._run_gh_command")
    def test_error_returns_empty_list(self, mock_run):
        """Test that errors return empty list."""
        mock_run.side_effect = GitHubSecurityError("error")
        alerts = get_code_scanning_alerts()
        assert alerts == []


class TestCheckGitHubAlerts:
    """Tests for check_github_alerts function."""

    @patch("scitex.security.github.check_gh_auth")
    def test_not_authenticated_raises_error(self, mock_auth):
        """Test that unauthenticated user raises GitHubSecurityError."""
        mock_auth.return_value = False
        with pytest.raises(GitHubSecurityError, match="Not authenticated"):
            check_github_alerts()

    @patch("scitex.security.github.get_code_scanning_alerts")
    @patch("scitex.security.github.get_dependabot_alerts")
    @patch("scitex.security.github.get_secret_alerts")
    @patch("scitex.security.github.check_gh_auth")
    def test_returns_all_alert_types(
        self, mock_auth, mock_secrets, mock_dependabot, mock_code
    ):
        """Test that all alert types are returned."""
        mock_auth.return_value = True
        mock_secrets.return_value = [{"type": "secret"}]
        mock_dependabot.return_value = [{"type": "dependabot"}]
        mock_code.return_value = [{"type": "code"}]

        result = check_github_alerts()
        assert "secrets" in result
        assert "dependabot" in result
        assert "code_scanning" in result
        assert len(result["secrets"]) == 1
        assert len(result["dependabot"]) == 1
        assert len(result["code_scanning"]) == 1

    @patch("scitex.security.github.get_code_scanning_alerts")
    @patch("scitex.security.github.get_dependabot_alerts")
    @patch("scitex.security.github.get_secret_alerts")
    @patch("scitex.security.github.check_gh_auth")
    def test_passes_repo_to_functions(
        self, mock_auth, mock_secrets, mock_dependabot, mock_code
    ):
        """Test that repo parameter is passed to all functions."""
        mock_auth.return_value = True
        mock_secrets.return_value = []
        mock_dependabot.return_value = []
        mock_code.return_value = []

        check_github_alerts("test/repo")
        mock_secrets.assert_called_with("test/repo")
        mock_dependabot.assert_called_with("test/repo")
        mock_code.assert_called_with("test/repo")


class TestFormatAlertsReport:
    """Tests for format_alerts_report function."""

    def test_empty_alerts_report(self):
        """Test report with no alerts."""
        alerts = {"secrets": [], "dependabot": [], "code_scanning": []}
        report = format_alerts_report(alerts)
        assert "GitHub Security Alerts Report" in report
        assert "No open secret scanning alerts" in report
        assert "No open Dependabot alerts" in report
        assert "No open code scanning alerts" in report
        assert "Total open alerts: 0" in report
        assert "No open security alerts" in report

    def test_report_with_open_secrets(self):
        """Test report with open secret alerts."""
        alerts = {
            "secrets": [
                {
                    "state": "open",
                    "secretType": "AWS Key",
                    "url": "http://example.com",
                    "path": "config.py",
                    "line": 10,
                    "createdAt": "2024-01-01",
                }
            ],
            "dependabot": [],
            "code_scanning": [],
        }
        report = format_alerts_report(alerts)
        assert "AWS Key" in report
        assert "config.py:10" in report
        assert "Total open alerts: 1" in report

    def test_report_with_open_dependabot(self):
        """Test report with open Dependabot alerts."""
        alerts = {
            "secrets": [],
            "dependabot": [
                {
                    "state": "open",
                    "severity": "high",
                    "summary": "XSS vulnerability",
                    "package": "lodash",
                    "cve": "CVE-2024-9999",
                    "url": "http://example.com",
                }
            ],
            "code_scanning": [],
        }
        report = format_alerts_report(alerts)
        assert "HIGH" in report
        assert "XSS vulnerability" in report
        assert "lodash" in report
        assert "CVE-2024-9999" in report

    def test_report_with_open_code_scanning(self):
        """Test report with open code scanning alerts."""
        alerts = {
            "secrets": [],
            "dependabot": [],
            "code_scanning": [
                {
                    "state": "open",
                    "severity": "error",
                    "description": "SQL Injection vulnerability",
                    "location": "app.py",
                    "line": 50,
                    "url": "http://example.com",
                }
            ],
        }
        report = format_alerts_report(alerts)
        assert "ERROR" in report
        assert "SQL Injection" in report
        assert "app.py:50" in report

    def test_report_filters_closed_alerts(self):
        """Test that closed alerts are not counted as open."""
        alerts = {
            "secrets": [{"state": "closed", "secretType": "Key", "url": "http://x"}],
            "dependabot": [],
            "code_scanning": [],
        }
        report = format_alerts_report(alerts)
        assert "Total open alerts: 0" in report

    def test_report_with_missing_optional_fields(self):
        """Test report handles missing optional fields gracefully."""
        alerts = {
            "secrets": [{"state": "open", "secretType": "Key", "url": "http://x"}],
            "dependabot": [
                {
                    "state": "open",
                    "summary": "Bug",
                    "package": "pkg",
                    "url": "http://x",
                }
            ],
            "code_scanning": [
                {"state": "open", "description": "Issue", "url": "http://x"}
            ],
        }
        report = format_alerts_report(alerts)
        assert "Total open alerts: 3" in report

    def test_report_with_action_required(self):
        """Test that action required message appears when alerts exist."""
        alerts = {
            "secrets": [{"state": "open", "secretType": "Key", "url": "http://x"}],
            "dependabot": [],
            "code_scanning": [],
        }
        report = format_alerts_report(alerts)
        assert "ACTION REQUIRED" in report


class TestSaveAlertsToFile:
    """Tests for save_alerts_to_file function."""

    def test_saves_report_to_file(self, tmp_path):
        """Test that report is saved to file."""
        alerts = {"secrets": [], "dependabot": [], "code_scanning": []}
        output_file = save_alerts_to_file(alerts, tmp_path)
        assert output_file.exists()
        content = output_file.read_text()
        assert "GitHub Security Alerts Report" in content

    def test_creates_output_directory(self, tmp_path):
        """Test that output directory is created if not exists."""
        output_dir = tmp_path / "new" / "nested" / "dir"
        alerts = {"secrets": [], "dependabot": [], "code_scanning": []}
        output_file = save_alerts_to_file(alerts, output_dir)
        assert output_dir.exists()
        assert output_file.exists()

    def test_creates_latest_symlink(self, tmp_path):
        """Test that latest symlink is created."""
        alerts = {"secrets": [], "dependabot": [], "code_scanning": []}
        save_alerts_to_file(alerts, tmp_path)
        latest_link = tmp_path / "security-latest.txt"
        assert latest_link.exists()
        assert latest_link.is_symlink()

    def test_symlink_updates_on_second_save(self, tmp_path):
        """Test that symlink is updated on second save."""
        alerts = {"secrets": [], "dependabot": [], "code_scanning": []}
        first_file = save_alerts_to_file(alerts, tmp_path)
        second_file = save_alerts_to_file(alerts, tmp_path)
        latest_link = tmp_path / "security-latest.txt"
        # Symlink should point to second file
        assert latest_link.resolve() == second_file

    def test_no_symlink_when_disabled(self, tmp_path):
        """Test that symlink is not created when disabled."""
        alerts = {"secrets": [], "dependabot": [], "code_scanning": []}
        save_alerts_to_file(alerts, tmp_path, create_symlink=False)
        latest_link = tmp_path / "security-latest.txt"
        assert not latest_link.exists()

    def test_filename_contains_timestamp(self, tmp_path):
        """Test that filename contains timestamp."""
        alerts = {"secrets": [], "dependabot": [], "code_scanning": []}
        output_file = save_alerts_to_file(alerts, tmp_path)
        # Filename format: security-YYYYMMDD_HHMMSS.txt
        assert output_file.name.startswith("security-")
        assert output_file.suffix == ".txt"


class TestGetLatestAlertsFile:
    """Tests for get_latest_alerts_file function."""

    def test_returns_none_for_empty_dir(self, tmp_path):
        """Test that None is returned for empty directory."""
        result = get_latest_alerts_file(tmp_path)
        assert result is None

    def test_returns_symlink_when_exists(self, tmp_path):
        """Test that symlink is returned when it exists."""
        # Create a file and symlink
        real_file = tmp_path / "security-20240101_120000.txt"
        real_file.write_text("test content")
        latest_link = tmp_path / "security-latest.txt"
        latest_link.symlink_to(real_file.name)

        result = get_latest_alerts_file(tmp_path)
        assert result == latest_link

    def test_returns_most_recent_file_as_fallback(self, tmp_path):
        """Test that most recent file is returned when no symlink."""
        # Create multiple files
        file1 = tmp_path / "security-20240101_100000.txt"
        file2 = tmp_path / "security-20240102_100000.txt"
        file3 = tmp_path / "security-20240103_100000.txt"
        file1.write_text("old")
        file2.write_text("middle")
        file3.write_text("newest")

        result = get_latest_alerts_file(tmp_path)
        # Should return file with latest timestamp in name (sorted reverse)
        assert result.name == "security-20240103_100000.txt"

    def test_ignores_non_security_files(self, tmp_path):
        """Test that non-security files are ignored."""
        other_file = tmp_path / "other-file.txt"
        other_file.write_text("not a security file")

        result = get_latest_alerts_file(tmp_path)
        assert result is None

    def test_uses_default_directory_when_none(self, tmp_path, monkeypatch):
        """Test that default directory is used when None."""
        monkeypatch.chdir(tmp_path)
        # This will look for ./logs/security which won't exist
        result = get_latest_alerts_file(None)
        assert result is None


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
