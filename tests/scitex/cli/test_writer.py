#!/usr/bin/env python3
"""Tests for scitex.cli.writer - Thin wrapper delegating to scitex-writer."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from scitex.cli.writer import writer


class TestWriterThinWrapper:
    """Tests for the writer thin wrapper command."""

    def test_writer_help(self):
        """Test that writer help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(writer, ["--help"])
        assert result.exit_code == 0
        assert "Manuscript writing" in result.output
        assert "delegates to scitex-writer" in result.output

    def test_writer_short_help(self):
        """Test that -h also shows help."""
        runner = CliRunner()
        result = runner.invoke(writer, ["-h"])
        assert result.exit_code == 0
        assert "Manuscript writing" in result.output

    def test_writer_lists_scitex_writer_commands(self):
        """Test that help lists scitex-writer commands."""
        runner = CliRunner()
        result = runner.invoke(writer, ["--help"])
        assert result.exit_code == 0
        # Check that scitex-writer commands are mentioned
        expected_commands = ["compile", "bib", "tables", "figures", "guidelines"]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not mentioned in help"

    @patch("subprocess.call")
    def test_writer_delegates_to_scitex_writer(self, mock_call):
        """Test that writer delegates commands to scitex-writer CLI."""
        mock_call.return_value = 0
        runner = CliRunner()

        # Test delegation with a command
        result = runner.invoke(writer, ["guidelines", "list"])

        # Verify subprocess.call was invoked with scitex-writer
        mock_call.assert_called_once()
        args = mock_call.call_args[0][0]
        assert args[0] == "scitex-writer"
        assert "guidelines" in args
        assert "list" in args

    @patch("subprocess.call")
    def test_writer_passes_all_args(self, mock_call):
        """Test that all arguments are passed through."""
        mock_call.return_value = 0
        runner = CliRunner()

        result = runner.invoke(writer, ["compile", "manuscript", "--draft"])

        mock_call.assert_called_once()
        args = mock_call.call_args[0][0]
        assert args == ["scitex-writer", "compile", "manuscript", "--draft"]

    @patch("subprocess.call")
    def test_writer_returns_subprocess_exit_code(self, mock_call):
        """Test that exit code from scitex-writer is returned."""
        mock_call.return_value = 1
        runner = CliRunner()

        result = runner.invoke(writer, ["bib", "list", "./nonexistent"])

        assert result.exit_code == 1


class TestWriterPackageCheck:
    """Tests for scitex-writer package availability check."""

    def test_has_writer_pkg_flag(self):
        """Test that HAS_WRITER_PKG flag is available."""
        from scitex.cli.writer import HAS_WRITER_PKG

        # Should be True since scitex-writer is installed
        assert HAS_WRITER_PKG is True

    @patch("scitex.cli.writer.HAS_WRITER_PKG", False)
    def test_error_when_package_not_installed(self):
        """Test error message when scitex-writer not installed."""
        # Need to reimport to get the patched version
        from scitex.cli import writer as writer_module

        runner = CliRunner()
        with patch.object(writer_module, "HAS_WRITER_PKG", False):
            result = runner.invoke(writer_module.writer, ["guidelines", "list"])
            assert result.exit_code == 1
            assert "scitex-writer" in result.output
            assert "pip install" in result.output


# EOF
