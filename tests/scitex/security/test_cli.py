#!/usr/bin/env python3
# File: tests/scitex/security/test_cli.py

"""Tests for scitex.security.cli module."""

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex.security.cli import check_command, latest_command, main
from scitex.security.github import GitHubSecurityError


class TestCheckCommand:
    """Tests for check_command function."""

    @patch("scitex.security.cli.format_alerts_report")
    @patch("scitex.security.cli.check_github_alerts")
    def test_no_alerts_exits_zero(self, mock_check, mock_format, capsys):
        """Test that no alerts exits with code 0."""
        mock_check.return_value = {"secrets": [], "dependabot": [], "code_scanning": []}
        mock_format.return_value = "No alerts"

        with pytest.raises(SystemExit) as exc_info:
            check_command()
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "No security alerts found" in captured.out

    @patch("scitex.security.cli.format_alerts_report")
    @patch("scitex.security.cli.check_github_alerts")
    def test_open_alerts_exits_one(self, mock_check, mock_format, capsys):
        """Test that open alerts exits with code 1."""
        mock_check.return_value = {
            "secrets": [{"state": "open"}],
            "dependabot": [],
            "code_scanning": [],
        }
        mock_format.return_value = "Found alerts"

        with pytest.raises(SystemExit) as exc_info:
            check_command()
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Found 1 open security alert" in captured.out

    @patch("scitex.security.cli.format_alerts_report")
    @patch("scitex.security.cli.check_github_alerts")
    def test_passes_repo_argument(self, mock_check, mock_format):
        """Test that repo argument is passed to check_github_alerts."""
        mock_check.return_value = {"secrets": [], "dependabot": [], "code_scanning": []}
        mock_format.return_value = "Report"

        with pytest.raises(SystemExit):
            check_command(repo="owner/repo")

        mock_check.assert_called_once_with("owner/repo")

    @patch("scitex.security.cli.save_alerts_to_file")
    @patch("scitex.security.cli.format_alerts_report")
    @patch("scitex.security.cli.check_github_alerts")
    def test_save_option_calls_save_function(self, mock_check, mock_format, mock_save):
        """Test that --save option calls save_alerts_to_file."""
        mock_check.return_value = {"secrets": [], "dependabot": [], "code_scanning": []}
        mock_format.return_value = "Report"
        mock_save.return_value = Path("/tmp/security-test.txt")

        with pytest.raises(SystemExit):
            check_command(save=True)

        mock_save.assert_called_once()

    @patch("scitex.security.cli.save_alerts_to_file")
    @patch("scitex.security.cli.format_alerts_report")
    @patch("scitex.security.cli.check_github_alerts")
    def test_output_dir_passed_to_save(self, mock_check, mock_format, mock_save):
        """Test that output_dir is passed to save_alerts_to_file."""
        mock_check.return_value = {"secrets": [], "dependabot": [], "code_scanning": []}
        mock_format.return_value = "Report"
        mock_save.return_value = Path("/custom/dir/security.txt")

        with pytest.raises(SystemExit):
            check_command(save=True, output_dir="/custom/dir")

        call_args = mock_save.call_args
        assert call_args[0][1] == Path("/custom/dir")

    @patch("scitex.security.cli.check_github_alerts")
    def test_github_security_error_exits_one(self, mock_check, capsys):
        """Test that GitHubSecurityError exits with code 1."""
        mock_check.side_effect = GitHubSecurityError("Auth failed")

        with pytest.raises(SystemExit) as exc_info:
            check_command()
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "ERROR" in captured.err
        assert "Auth failed" in captured.err

    @patch("scitex.security.cli.format_alerts_report")
    @patch("scitex.security.cli.check_github_alerts")
    def test_counts_multiple_alert_types(self, mock_check, mock_format, capsys):
        """Test that alerts from all types are counted."""
        mock_check.return_value = {
            "secrets": [{"state": "open"}],
            "dependabot": [{"state": "open"}, {"state": "open"}],
            "code_scanning": [{"state": "open"}],
        }
        mock_format.return_value = "Report"

        with pytest.raises(SystemExit) as exc_info:
            check_command()
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Found 4 open security alert" in captured.out

    @patch("scitex.security.cli.format_alerts_report")
    @patch("scitex.security.cli.check_github_alerts")
    def test_ignores_closed_alerts(self, mock_check, mock_format, capsys):
        """Test that closed alerts are not counted."""
        mock_check.return_value = {
            "secrets": [{"state": "closed"}],
            "dependabot": [{"state": "dismissed"}],
            "code_scanning": [],
        }
        mock_format.return_value = "Report"

        with pytest.raises(SystemExit) as exc_info:
            check_command()
        assert exc_info.value.code == 0


class TestLatestCommand:
    """Tests for latest_command function."""

    @patch("scitex.security.cli.get_latest_alerts_file")
    def test_displays_file_content(self, mock_get_latest, tmp_path, capsys):
        """Test that file content is displayed."""
        test_file = tmp_path / "security-latest.txt"
        test_file.write_text("Security Report Content")
        mock_get_latest.return_value = test_file

        # No exit on success (falls through)
        latest_command(str(tmp_path))

        captured = capsys.readouterr()
        assert "Security Report Content" in captured.out

    @patch("scitex.security.cli.get_latest_alerts_file")
    def test_no_file_exits_one(self, mock_get_latest, capsys):
        """Test that missing file exits with code 1."""
        mock_get_latest.return_value = None

        with pytest.raises(SystemExit) as exc_info:
            latest_command()
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "No security alerts files found" in captured.out

    @patch("scitex.security.cli.get_latest_alerts_file")
    def test_file_content_printed(self, mock_get_latest, tmp_path, capsys):
        """Test that file content is printed to stdout."""
        test_file = tmp_path / "security-20240101.txt"
        test_file.write_text("Report content here")
        mock_get_latest.return_value = test_file

        # No exit on success (falls through)
        latest_command()

        captured = capsys.readouterr()
        assert "Report content here" in captured.out

    @patch("scitex.security.cli.get_latest_alerts_file")
    def test_passes_security_dir(self, mock_get_latest, tmp_path):
        """Test that security_dir is passed to get_latest_alerts_file."""
        mock_get_latest.return_value = None

        with pytest.raises(SystemExit):
            latest_command(security_dir=str(tmp_path))

        mock_get_latest.assert_called_once_with(Path(str(tmp_path)))

    @patch("scitex.security.cli.get_latest_alerts_file")
    def test_exception_exits_one(self, mock_get_latest, capsys):
        """Test that exceptions exit with code 1."""
        mock_get_latest.side_effect = Exception("File error")

        with pytest.raises(SystemExit) as exc_info:
            latest_command()
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "ERROR" in captured.err


class TestMain:
    """Tests for main function."""

    def test_no_command_prints_help(self, capsys, monkeypatch):
        """Test that no command prints help and exits."""
        monkeypatch.setattr(sys, "argv", ["scitex-security"])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch("scitex.security.cli.check_command")
    def test_check_command_called(self, mock_check, monkeypatch):
        """Test that 'check' subcommand calls check_command."""
        monkeypatch.setattr(sys, "argv", ["scitex-security", "check"])
        mock_check.side_effect = SystemExit(0)

        with pytest.raises(SystemExit):
            main()

        mock_check.assert_called_once_with(None, False, None)

    @patch("scitex.security.cli.check_command")
    def test_check_with_repo(self, mock_check, monkeypatch):
        """Test that --repo is passed to check_command."""
        monkeypatch.setattr(
            sys, "argv", ["scitex-security", "check", "--repo", "org/repo"]
        )
        mock_check.side_effect = SystemExit(0)

        with pytest.raises(SystemExit):
            main()

        mock_check.assert_called_once_with("org/repo", False, None)

    @patch("scitex.security.cli.check_command")
    def test_check_with_save(self, mock_check, monkeypatch):
        """Test that --save is passed to check_command."""
        monkeypatch.setattr(sys, "argv", ["scitex-security", "check", "--save"])
        mock_check.side_effect = SystemExit(0)

        with pytest.raises(SystemExit):
            main()

        mock_check.assert_called_once_with(None, True, None)

    @patch("scitex.security.cli.check_command")
    def test_check_with_output_dir(self, mock_check, monkeypatch):
        """Test that --output-dir is passed to check_command."""
        monkeypatch.setattr(
            sys, "argv", ["scitex-security", "check", "--output-dir", "/custom/path"]
        )
        mock_check.side_effect = SystemExit(0)

        with pytest.raises(SystemExit):
            main()

        mock_check.assert_called_once_with(None, False, "/custom/path")

    @patch("scitex.security.cli.check_command")
    def test_check_with_all_options(self, mock_check, monkeypatch):
        """Test check command with all options."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "scitex-security",
                "check",
                "--repo",
                "owner/repo",
                "--save",
                "--output-dir",
                "/out",
            ],
        )
        mock_check.side_effect = SystemExit(0)

        with pytest.raises(SystemExit):
            main()

        mock_check.assert_called_once_with("owner/repo", True, "/out")

    @patch("scitex.security.cli.latest_command")
    def test_latest_command_called(self, mock_latest, monkeypatch):
        """Test that 'latest' subcommand calls latest_command."""
        monkeypatch.setattr(sys, "argv", ["scitex-security", "latest"])

        main()

        mock_latest.assert_called_once_with(None)

    @patch("scitex.security.cli.latest_command")
    def test_latest_with_dir(self, mock_latest, monkeypatch):
        """Test that --dir is passed to latest_command."""
        monkeypatch.setattr(
            sys, "argv", ["scitex-security", "latest", "--dir", "/logs/security"]
        )

        main()

        mock_latest.assert_called_once_with("/logs/security")


class TestMainIntegration:
    """Integration tests for main function."""

    @patch("scitex.security.cli.format_alerts_report")
    @patch("scitex.security.cli.check_github_alerts")
    def test_full_check_flow(self, mock_check, mock_format, monkeypatch, capsys):
        """Test full check command flow."""
        monkeypatch.setattr(sys, "argv", ["scitex-security", "check"])
        mock_check.return_value = {"secrets": [], "dependabot": [], "code_scanning": []}
        mock_format.return_value = "Empty report"

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "Checking GitHub security alerts" in captured.out

    @patch("scitex.security.cli.get_latest_alerts_file")
    def test_full_latest_flow(self, mock_get_latest, tmp_path, monkeypatch, capsys):
        """Test full latest command flow."""
        test_file = tmp_path / "security.txt"
        test_file.write_text("Latest report")
        mock_get_latest.return_value = test_file
        monkeypatch.setattr(sys, "argv", ["scitex-security", "latest"])

        main()

        captured = capsys.readouterr()
        assert "Latest report" in captured.out


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
