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

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/security/github.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: ~/proj/scitex-code/src/scitex/security/github.py
# 
# """
# GitHub Security Alerts Module
# 
# Fetches and processes security alerts from GitHub.
# """
# 
# import json
# import subprocess
# from datetime import datetime
# from pathlib import Path
# from typing import Dict, List, Optional
# 
# 
# class GitHubSecurityError(Exception):
#     """Raised when GitHub security operations fail."""
# 
#     pass
# 
# 
# def _run_gh_command(args: List[str]) -> str:
#     """Run GitHub CLI command and return output."""
#     try:
#         result = subprocess.run(
#             ["gh"] + args,
#             capture_output=True,
#             text=True,
#             check=True,
#         )
#         return result.stdout
#     except subprocess.CalledProcessError as e:
#         raise GitHubSecurityError(f"GitHub CLI error: {e.stderr}")
#     except FileNotFoundError:
#         raise GitHubSecurityError(
#             "GitHub CLI (gh) not found. Install: https://cli.github.com/"
#         )
# 
# 
# def check_gh_auth() -> bool:
#     """Check if GitHub CLI is authenticated."""
#     try:
#         subprocess.run(
#             ["gh", "auth", "status"],
#             capture_output=True,
#             check=True,
#         )
#         return True
#     except (subprocess.CalledProcessError, FileNotFoundError):
#         return False
# 
# 
# def get_secret_alerts(repo: Optional[str] = None) -> List[Dict]:
#     """
#     Get secret scanning alerts.
# 
#     Args:
#         repo: Repository in format 'owner/repo'. If None, uses current repo.
# 
#     Returns:
#         List of secret scanning alerts
#     """
#     try:
#         # Use GitHub REST API for secret scanning
#         api_path = "/repos/:owner/:repo/secret-scanning/alerts"
#         if repo:
#             owner, repo_name = repo.split("/")
#             api_path = f"/repos/{owner}/{repo_name}/secret-scanning/alerts"
# 
#         output = _run_gh_command(
#             [
#                 "api",
#                 api_path,
#                 "--paginate",
#                 "--jq",
#                 ".[] | {state, secretType: .secret_type_display_name, "
#                 "url: .html_url, "
#                 "createdAt: .created_at, "
#                 "path: .first_location_detected.path, "
#                 "line: .first_location_detected.start_line}",
#             ]
#         )
# 
#         if not output.strip():
#             return []
# 
#         # Parse line-delimited JSON
#         alerts = []
#         for line in output.strip().split("\n"):
#             if line.strip():
#                 alerts.append(json.loads(line))
#         return alerts
#     except GitHubSecurityError:
#         return []
# 
# 
# def get_dependabot_alerts(repo: Optional[str] = None) -> List[Dict]:
#     """
#     Get Dependabot vulnerability alerts.
# 
#     Args:
#         repo: Repository in format 'owner/repo'. If None, uses current repo.
# 
#     Returns:
#         List of Dependabot alerts
#     """
#     try:
#         # Use GitHub API to get Dependabot alerts
#         api_path = "/repos/:owner/:repo/dependabot/alerts"
#         if repo:
#             owner, repo_name = repo.split("/")
#             api_path = f"/repos/{owner}/{repo_name}/dependabot/alerts"
# 
#         output = _run_gh_command(
#             [
#                 "api",
#                 api_path,
#                 "--paginate",
#                 "--jq",
#                 ".[] | {state, severity: .security_advisory.severity, "
#                 "summary: .security_advisory.summary, "
#                 "package: .dependency.package.name, "
#                 "cve: .security_advisory.cve_id, "
#                 "url: .html_url, "
#                 "created_at: .created_at}",
#             ]
#         )
# 
#         if not output.strip():
#             return []
# 
#         # Parse line-delimited JSON
#         alerts = []
#         for line in output.strip().split("\n"):
#             if line.strip():
#                 alerts.append(json.loads(line))
#         return alerts
#     except GitHubSecurityError:
#         return []
# 
# 
# def get_code_scanning_alerts(repo: Optional[str] = None) -> List[Dict]:
#     """
#     Get code scanning alerts.
# 
#     Args:
#         repo: Repository in format 'owner/repo'. If None, uses current repo.
# 
#     Returns:
#         List of code scanning alerts
#     """
#     try:
#         # Use GitHub API to get code scanning alerts
#         api_path = "/repos/:owner/:repo/code-scanning/alerts"
#         if repo:
#             owner, repo_name = repo.split("/")
#             api_path = f"/repos/{owner}/{repo_name}/code-scanning/alerts"
# 
#         output = _run_gh_command(
#             [
#                 "api",
#                 api_path,
#                 "--paginate",
#                 "--jq",
#                 ".[] | {state, severity: .rule.severity, "
#                 "description: .rule.description, "
#                 "location: .most_recent_instance.location.path, "
#                 "line: .most_recent_instance.location.start_line, "
#                 "url: .html_url, "
#                 "created_at: .created_at}",
#             ]
#         )
# 
#         if not output.strip():
#             return []
# 
#         # Parse line-delimited JSON
#         alerts = []
#         for line in output.strip().split("\n"):
#             if line.strip():
#                 alerts.append(json.loads(line))
#         return alerts
#     except GitHubSecurityError:
#         return []
# 
# 
# def check_github_alerts(repo: Optional[str] = None) -> Dict[str, List[Dict]]:
#     """
#     Check all GitHub security alerts.
# 
#     Args:
#         repo: Repository in format 'owner/repo'. If None, uses current repo.
# 
#     Returns:
#         Dictionary with keys: 'secrets', 'dependabot', 'code_scanning'
# 
#     Raises:
#         GitHubSecurityError: If GitHub CLI is not installed or not authenticated
#     """
#     if not check_gh_auth():
#         raise GitHubSecurityError(
#             "Not authenticated with GitHub CLI. Run: gh auth login"
#         )
# 
#     return {
#         "secrets": get_secret_alerts(repo),
#         "dependabot": get_dependabot_alerts(repo),
#         "code_scanning": get_code_scanning_alerts(repo),
#     }
# 
# 
# def format_alerts_report(alerts: Dict[str, List[Dict]]) -> str:
#     """
#     Format alerts into a readable text report.
# 
#     Args:
#         alerts: Dictionary of alerts from check_github_alerts()
# 
#     Returns:
#         Formatted text report
#     """
#     lines = []
#     lines.append("=" * 50)
#     lines.append("GitHub Security Alerts Report")
#     lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     lines.append("=" * 50)
#     lines.append("")
# 
#     # Secret scanning alerts
#     lines.append("### SECRET SCANNING ALERTS ###")
#     lines.append("")
#     secrets = [s for s in alerts["secrets"] if s.get("state") == "open"]
#     if secrets:
#         for alert in secrets:
#             lines.append(f"- [{alert['state']}] {alert['secretType']}")
#             path = alert.get("path", "N/A")
#             line_num = alert.get("line", "")
#             if path != "N/A" and line_num:
#                 lines.append(f"  Location: {path}:{line_num}")
#             lines.append(f"  Created: {alert.get('createdAt', 'N/A')}")
#             lines.append(f"  URL: {alert['url']}")
#             lines.append("")
#     else:
#         lines.append("No open secret scanning alerts")
#         lines.append("")
# 
#     lines.append("=" * 50)
#     lines.append("")
# 
#     # Dependabot alerts
#     lines.append("### DEPENDABOT VULNERABILITY ALERTS ###")
#     lines.append("")
#     dependabot = [d for d in alerts["dependabot"] if d.get("state") == "open"]
#     if dependabot:
#         for alert in dependabot:
#             severity = alert.get("severity", "unknown").upper()
#             lines.append(f"- [{alert['state']}] {severity}: {alert['summary']}")
#             lines.append(f"  Package: {alert['package']}")
#             lines.append(f"  CVE: {alert.get('cve') or 'N/A'}")
#             lines.append(f"  URL: {alert['url']}")
#             lines.append("")
#     else:
#         lines.append("No open Dependabot alerts")
#         lines.append("")
# 
#     lines.append("=" * 50)
#     lines.append("")
# 
#     # Code scanning alerts
#     lines.append("### CODE SCANNING ALERTS ###")
#     lines.append("")
#     code_scanning = [c for c in alerts["code_scanning"] if c.get("state") == "open"]
#     if code_scanning:
#         for alert in code_scanning:
#             severity = alert.get("severity", "unknown").upper()
#             lines.append(f"- [{alert['state']}] {severity}: {alert['description']}")
#             location = alert.get("location", "N/A")
#             line_num = alert.get("line", "")
#             if line_num:
#                 location = f"{location}:{line_num}"
#             lines.append(f"  Location: {location}")
#             lines.append(f"  URL: {alert['url']}")
#             lines.append("")
#     else:
#         lines.append("No open code scanning alerts")
#         lines.append("")
# 
#     lines.append("=" * 50)
#     lines.append("")
# 
#     # Summary
#     total = len(secrets) + len(dependabot) + len(code_scanning)
#     lines.append("### SUMMARY ###")
#     lines.append("")
#     lines.append(f"Total open alerts: {total}")
#     lines.append(f"  - Secrets: {len(secrets)}")
#     lines.append(f"  - Dependabot: {len(dependabot)}")
#     lines.append(f"  - Code Scanning: {len(code_scanning)}")
#     lines.append("")
# 
#     if total > 0:
#         lines.append("⚠️  ACTION REQUIRED: Security issues found!")
#     else:
#         lines.append("✓ No open security alerts")
# 
#     return "\n".join(lines)
# 
# 
# def save_alerts_to_file(
#     alerts: Dict[str, List[Dict]],
#     output_dir: Optional[Path] = None,
#     create_symlink: bool = True,
# ) -> Path:
#     """
#     Save alerts to a timestamped file.
# 
#     Args:
#         alerts: Dictionary of alerts from check_github_alerts()
#         output_dir: Directory to save file. Defaults to ./logs/security
#         create_symlink: If True, create 'security-latest.txt' symlink
# 
#     Returns:
#         Path to saved file
#     """
#     if output_dir is None:
#         output_dir = Path.cwd() / "logs" / "security"
#     else:
#         output_dir = Path(output_dir)
# 
#     output_dir.mkdir(parents=True, exist_ok=True)
# 
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_file = output_dir / f"security-{timestamp}.txt"
# 
#     report = format_alerts_report(alerts)
#     output_file.write_text(report)
# 
#     # Create symlink to latest
#     if create_symlink:
#         latest_link = output_dir / "security-latest.txt"
#         if latest_link.exists() or latest_link.is_symlink():
#             latest_link.unlink()
#         latest_link.symlink_to(output_file.name)
# 
#     return output_file
# 
# 
# def get_latest_alerts_file(security_dir: Optional[Path] = None) -> Optional[Path]:
#     """
#     Get path to the latest security alerts file.
# 
#     Args:
#         security_dir: Directory containing security files. Defaults to ./logs/security
# 
#     Returns:
#         Path to latest file, or None if not found
#     """
#     if security_dir is None:
#         security_dir = Path.cwd() / "logs" / "security"
#     else:
#         security_dir = Path(security_dir)
# 
#     latest_link = security_dir / "security-latest.txt"
#     if latest_link.exists():
#         return latest_link
# 
#     # Fallback: find most recent file
#     files = sorted(security_dir.glob("security-*.txt"), reverse=True)
#     return files[0] if files else None

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/security/github.py
# --------------------------------------------------------------------------------
