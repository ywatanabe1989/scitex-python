#!/usr/bin/env python3
"""Tests for scitex.cli.security - Security CLI commands."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scitex.cli.security import security


class TestSecurityGroup:
    """Tests for the security command group."""

    def test_security_help(self):
        """Test that security help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(security, ["--help"])
        assert result.exit_code == 0
        assert "Security utilities" in result.output

    def test_security_has_subcommands(self):
        """Test that all expected subcommands are registered."""
        runner = CliRunner()
        result = runner.invoke(security, ["--help"])
        expected_commands = ["check", "latest"]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in security help"


class TestSecurityCheck:
    """Tests for the security check command."""

    def test_check_help(self):
        """Test check command help."""
        runner = CliRunner()
        result = runner.invoke(security, ["check", "--help"])
        assert result.exit_code == 0
        assert "Check GitHub security alerts" in result.output

    def test_check_no_alerts(self):
        """Test check command when no alerts are found."""
        runner = CliRunner()
        with patch("scitex.cli.security.check_github_alerts") as mock_check:
            mock_check.return_value = {
                "dependabot": [],
                "code_scanning": [],
                "secret_scanning": [],
            }

            with patch("scitex.cli.security.format_alerts_report") as mock_format:
                mock_format.return_value = "No security alerts found."

                result = runner.invoke(security, ["check"])
                assert result.exit_code == 0
                assert "No security alerts found" in result.output

    def test_check_with_alerts(self):
        """Test check command when alerts are found."""
        runner = CliRunner()
        with patch("scitex.cli.security.check_github_alerts") as mock_check:
            mock_check.return_value = {
                "dependabot": [{"state": "open", "package": "lodash"}],
                "code_scanning": [],
                "secret_scanning": [],
            }

            with patch("scitex.cli.security.format_alerts_report") as mock_format:
                mock_format.return_value = "1 dependabot alert found."

                result = runner.invoke(security, ["check"])
                assert result.exit_code == 1  # Exit with error when alerts found
                assert "Found 1 open security alert" in result.output

    def test_check_with_repo_option(self):
        """Test check command with --repo option."""
        runner = CliRunner()
        with patch("scitex.cli.security.check_github_alerts") as mock_check:
            mock_check.return_value = {
                "dependabot": [],
                "code_scanning": [],
                "secret_scanning": [],
            }

            with patch("scitex.cli.security.format_alerts_report") as mock_format:
                mock_format.return_value = "No alerts."

                result = runner.invoke(security, ["check", "--repo", "owner/repo"])
                assert result.exit_code == 0
                mock_check.assert_called_with("owner/repo")

    def test_check_with_save_option(self):
        """Test check command with --save option."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.cli.security.check_github_alerts") as mock_check:
                mock_check.return_value = {
                    "dependabot": [],
                    "code_scanning": [],
                    "secret_scanning": [],
                }

                with patch("scitex.cli.security.format_alerts_report") as mock_format:
                    mock_format.return_value = "No alerts."

                    with patch("scitex.cli.security.save_alerts_to_file") as mock_save:
                        output_path = Path(tmpdir) / "report.txt"
                        mock_save.return_value = output_path

                        result = runner.invoke(security, ["check", "--save"])
                        assert result.exit_code == 0
                        assert "Report saved to" in result.output

    def test_check_with_output_dir(self):
        """Test check command with --output-dir option."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.cli.security.check_github_alerts") as mock_check:
                mock_check.return_value = {
                    "dependabot": [],
                    "code_scanning": [],
                    "secret_scanning": [],
                }

                with patch("scitex.cli.security.format_alerts_report") as mock_format:
                    mock_format.return_value = "No alerts."

                    with patch("scitex.cli.security.save_alerts_to_file") as mock_save:
                        output_path = Path(tmpdir) / "report.txt"
                        mock_save.return_value = output_path

                        result = runner.invoke(
                            security, ["check", "--save", "--output-dir", tmpdir]
                        )
                        assert result.exit_code == 0

    def test_check_github_error(self):
        """Test check command when GitHub API fails."""
        runner = CliRunner()
        with patch("scitex.cli.security.check_github_alerts") as mock_check:
            from scitex.security import GitHubSecurityError

            mock_check.side_effect = GitHubSecurityError("API rate limit exceeded")

            result = runner.invoke(security, ["check"])
            assert result.exit_code == 1
            assert "ERROR" in result.output
            assert "API rate limit exceeded" in result.output


class TestSecurityLatest:
    """Tests for the security latest command."""

    def test_latest_help(self):
        """Test latest command help."""
        runner = CliRunner()
        result = runner.invoke(security, ["latest", "--help"])
        assert result.exit_code == 0
        assert "Show the latest security alerts file" in result.output

    def test_latest_no_files(self):
        """Test latest command when no alert files exist."""
        runner = CliRunner()
        with patch("scitex.cli.security.get_latest_alerts_file") as mock_get:
            mock_get.return_value = None

            result = runner.invoke(security, ["latest"])
            assert result.exit_code == 1
            assert "No security alerts files found" in result.output

    def test_latest_with_file(self):
        """Test latest command when alert file exists."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Security Report\n=============\nNo alerts found.")
            f.flush()

            with patch("scitex.cli.security.get_latest_alerts_file") as mock_get:
                mock_get.return_value = Path(f.name)

                result = runner.invoke(security, ["latest"])
                assert result.exit_code == 0
                assert "Security Report" in result.output

            os.unlink(f.name)

    def test_latest_with_dir_option(self):
        """Test latest command with --dir option."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a security report file
            report_path = Path(tmpdir) / "security-2025-01-01.txt"
            report_path.write_text("Test report content")

            with patch("scitex.cli.security.get_latest_alerts_file") as mock_get:
                mock_get.return_value = report_path

                result = runner.invoke(security, ["latest", "--dir", tmpdir])
                assert result.exit_code == 0
                assert "Test report content" in result.output


class TestSecurityAlertTypes:
    """Tests for different alert types."""

    def test_dependabot_alerts(self):
        """Test handling of dependabot alerts."""
        runner = CliRunner()
        with patch("scitex.cli.security.check_github_alerts") as mock_check:
            mock_check.return_value = {
                "dependabot": [
                    {"state": "open", "package": "lodash", "severity": "high"},
                    {"state": "open", "package": "axios", "severity": "medium"},
                ],
                "code_scanning": [],
                "secret_scanning": [],
            }

            with patch("scitex.cli.security.format_alerts_report") as mock_format:
                mock_format.return_value = "2 dependabot alerts"

                result = runner.invoke(security, ["check"])
                assert result.exit_code == 1
                assert "Found 2 open security alert" in result.output

    def test_code_scanning_alerts(self):
        """Test handling of code scanning alerts."""
        runner = CliRunner()
        with patch("scitex.cli.security.check_github_alerts") as mock_check:
            mock_check.return_value = {
                "dependabot": [],
                "code_scanning": [
                    {"state": "open", "rule": "sql-injection"},
                ],
                "secret_scanning": [],
            }

            with patch("scitex.cli.security.format_alerts_report") as mock_format:
                mock_format.return_value = "1 code scanning alert"

                result = runner.invoke(security, ["check"])
                assert result.exit_code == 1

    def test_secret_scanning_alerts(self):
        """Test handling of secret scanning alerts."""
        runner = CliRunner()
        with patch("scitex.cli.security.check_github_alerts") as mock_check:
            mock_check.return_value = {
                "dependabot": [],
                "code_scanning": [],
                "secret_scanning": [
                    {"state": "open", "secret_type": "api_key"},
                ],
            }

            with patch("scitex.cli.security.format_alerts_report") as mock_format:
                mock_format.return_value = "1 secret scanning alert"

                result = runner.invoke(security, ["check"])
                assert result.exit_code == 1

    def test_mixed_alerts(self):
        """Test handling of mixed alert types."""
        runner = CliRunner()
        with patch("scitex.cli.security.check_github_alerts") as mock_check:
            mock_check.return_value = {
                "dependabot": [{"state": "open"}],
                "code_scanning": [{"state": "open"}],
                "secret_scanning": [{"state": "open"}],
            }

            with patch("scitex.cli.security.format_alerts_report") as mock_format:
                mock_format.return_value = "3 alerts total"

                result = runner.invoke(security, ["check"])
                assert result.exit_code == 1
                assert "Found 3 open security alert" in result.output


class TestSecurityIntegration:
    """Integration tests for security commands."""

    def test_check_and_save_flow(self):
        """Test checking and saving alerts in one command."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.cli.security.check_github_alerts") as mock_check:
                mock_check.return_value = {
                    "dependabot": [],
                    "code_scanning": [],
                    "secret_scanning": [],
                }

                with patch("scitex.cli.security.format_alerts_report") as mock_format:
                    mock_format.return_value = "All clear!"

                    with patch("scitex.cli.security.save_alerts_to_file") as mock_save:
                        report_path = Path(tmpdir) / "security.txt"
                        report_path.parent.mkdir(parents=True, exist_ok=True)
                        report_path.write_text("All clear!")
                        mock_save.return_value = report_path

                        result = runner.invoke(
                            security, ["check", "--save", "--output-dir", tmpdir]
                        )
                        assert result.exit_code == 0
                        assert "Report saved" in result.output


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
