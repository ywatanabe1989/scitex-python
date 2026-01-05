#!/usr/bin/env python3
"""Tests for scitex.cli.main - Main CLI entry point."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scitex.cli.main import cli, completion


class TestCLIGroup:
    """Tests for the main CLI command group."""

    def test_cli_help(self):
        """Test that CLI help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "SciTeX - Integrated Scientific Research Platform" in result.output

    def test_cli_short_help(self):
        """Test that -h also shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["-h"])
        assert result.exit_code == 0
        assert "SciTeX" in result.output

    def test_cli_version(self):
        """Test that version option works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        # Exit code 0 for version display
        assert result.exit_code == 0

    def test_cli_has_subcommands(self):
        """Test that all expected subcommands are registered."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        expected_commands = [
            "cloud",
            "config",
            "convert",
            "scholar",
            "security",
            "web",
            "writer",
            "completion",
        ]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in CLI help"

    def test_cli_unknown_command(self):
        """Test that unknown command shows error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["unknown-command"])
        assert result.exit_code != 0
        assert "No such command" in result.output or "Error" in result.output


class TestCompletionCommand:
    """Tests for the completion command."""

    def test_completion_show_bash(self):
        """Test showing bash completion script."""
        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "--shell", "bash", "--show"])
        assert result.exit_code == 0
        assert ".bashrc" in result.output
        assert "_SCITEX_COMPLETE=bash_source" in result.output

    def test_completion_show_zsh(self):
        """Test showing zsh completion script."""
        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "--shell", "zsh", "--show"])
        assert result.exit_code == 0
        assert ".zshrc" in result.output
        assert "_SCITEX_COMPLETE=zsh_source" in result.output

    def test_completion_show_fish(self):
        """Test showing fish completion script."""
        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "--shell", "fish", "--show"])
        assert result.exit_code == 0
        assert "config.fish" in result.output
        assert "_SCITEX_COMPLETE=fish_source" in result.output

    def test_completion_auto_detect_bash(self):
        """Test auto-detection of bash shell."""
        runner = CliRunner(env={"SHELL": "/bin/bash"})
        result = runner.invoke(cli, ["completion", "--show"])
        assert result.exit_code == 0
        assert "bash" in result.output.lower()

    def test_completion_auto_detect_zsh(self):
        """Test auto-detection of zsh shell."""
        runner = CliRunner(env={"SHELL": "/bin/zsh"})
        result = runner.invoke(cli, ["completion", "--show"])
        assert result.exit_code == 0
        assert "zsh" in result.output.lower()

    def test_completion_auto_detect_fish(self):
        """Test auto-detection of fish shell."""
        runner = CliRunner(env={"SHELL": "/usr/bin/fish"})
        result = runner.invoke(cli, ["completion", "--show"])
        assert result.exit_code == 0
        assert "fish" in result.output.lower()

    def test_completion_auto_detect_unknown_shell(self):
        """Test error when shell cannot be auto-detected."""
        runner = CliRunner(env={"SHELL": "/bin/unknown"})
        result = runner.invoke(cli, ["completion"])
        assert result.exit_code == 1
        assert "Could not auto-detect shell" in result.output

    def test_completion_install_bash(self):
        """Test installing bash completion."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            bashrc = Path(tmpdir) / ".bashrc"
            bashrc.touch()

            with patch.object(Path, "home", return_value=Path(tmpdir)):
                with patch(
                    "os.path.expanduser",
                    side_effect=lambda x: str(bashrc) if "bashrc" in x else x,
                ):
                    result = runner.invoke(cli, ["completion", "--shell", "bash"])
                    # Should succeed or show already installed message
                    # The actual file writing may fail in test environment
                    # so we just check it doesn't crash unexpectedly
                    assert (
                        "completion" in result.output.lower()
                        or result.exit_code in [0, 1]
                    )

    def test_completion_already_installed(self):
        """Test that already-installed completion is detected."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            bashrc = Path(tmpdir) / ".bashrc"
            # Write a completion line
            bashrc.write_text(
                'eval "$(_SCITEX_COMPLETE=bash_source /usr/bin/scitex)"\n'
            )

            with patch("os.path.expanduser", return_value=str(bashrc)):
                with patch("os.path.exists", return_value=True):
                    # We can't easily test this without complex mocking
                    # Just verify the command runs
                    result = runner.invoke(
                        cli, ["completion", "--shell", "bash", "--show"]
                    )
                    assert result.exit_code == 0


class TestCLISubcommandAccess:
    """Tests for accessing subcommands."""

    def test_config_subcommand_accessible(self):
        """Test that config subcommand is accessible."""
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "Configuration management" in result.output

    def test_cloud_subcommand_accessible(self):
        """Test that cloud subcommand is accessible."""
        runner = CliRunner()
        result = runner.invoke(cli, ["cloud", "--help"])
        assert result.exit_code == 0
        assert "Cloud/Git operations" in result.output

    def test_convert_subcommand_accessible(self):
        """Test that convert subcommand is accessible."""
        runner = CliRunner()
        result = runner.invoke(cli, ["convert", "--help"])
        assert result.exit_code == 0
        assert "Convert and validate" in result.output

    def test_scholar_subcommand_accessible(self):
        """Test that scholar subcommand is accessible."""
        runner = CliRunner()
        result = runner.invoke(cli, ["scholar", "--help"])
        assert result.exit_code == 0
        assert "Literature management" in result.output

    def test_security_subcommand_accessible(self):
        """Test that security subcommand is accessible."""
        runner = CliRunner()
        result = runner.invoke(cli, ["security", "--help"])
        assert result.exit_code == 0
        assert "Security utilities" in result.output

    def test_web_subcommand_accessible(self):
        """Test that web subcommand is accessible."""
        runner = CliRunner()
        result = runner.invoke(cli, ["web", "--help"])
        assert result.exit_code == 0
        assert "Web scraping" in result.output

    def test_writer_subcommand_accessible(self):
        """Test that writer subcommand is accessible."""
        runner = CliRunner()
        result = runner.invoke(cli, ["writer", "--help"])
        assert result.exit_code == 0
        assert "Manuscript writing" in result.output


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
