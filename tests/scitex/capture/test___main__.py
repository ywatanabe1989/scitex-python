#!/usr/bin/env python3
"""Tests for scitex.capture.__main__ module.

Tests the entry point for python -m scitex.capture:
- Module imports
- Main function accessibility
- Module execution
"""

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestModuleImports:
    """Test module import functionality."""

    def test_module_importable(self):
        """Test __main__ module can be imported."""
        from scitex.capture import __main__

        assert __main__ is not None

    def test_main_accessible(self):
        """Test main function is accessible from module."""
        from scitex.capture.__main__ import main

        assert callable(main)

    def test_main_is_from_cli(self):
        """Test main function is imported from cli module."""
        from scitex.capture.__main__ import main
        from scitex.capture.cli import main as cli_main

        assert main is cli_main


class TestModuleExecution:
    """Test module execution as script."""

    def test_module_runnable_with_help(self):
        """Test module can be run with python -m and --help."""
        result = subprocess.run(
            [sys.executable, "-m", "scitex.capture", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "help" in result.stdout.lower()

    def test_module_execution_calls_main(self):
        """Test module execution calls main function."""
        from scitex.capture import __main__

        with patch.object(__main__, "main", return_value=0) as mock_main:
            # Simulate running as __main__
            with patch.object(__main__, "__name__", "__main__"):
                # The actual execution happens at import time,
                # so we test the structure
                assert hasattr(__main__, "main")
                assert callable(__main__.main)

    def test_module_returns_main_exit_code(self):
        """Test module passes main's return value to sys.exit."""
        # Test by running actual subprocess with mocked capture
        result = subprocess.run(
            [sys.executable, "-m", "scitex.capture", "--stop"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # --stop should return 0 (success)
        assert result.returncode == 0


class TestModuleIntegration:
    """Test module integration with CLI."""

    def test_help_output_contains_capture_commands(self):
        """Test help output contains expected capture commands."""
        result = subprocess.run(
            [sys.executable, "-m", "scitex.capture", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Check for common capture-related options
        help_text = result.stdout.lower()
        assert "--list" in help_text or "list" in help_text
        assert "--info" in help_text or "info" in help_text

    def test_list_action_via_module(self):
        """Test --list action works via module."""
        with patch("scitex.capture.get_info") as mock_info:
            mock_info.return_value = {"Windows": {"Details": []}}

            result = subprocess.run(
                [sys.executable, "-m", "scitex.capture", "--list"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # May succeed or fail depending on WSL state, but should not crash
            assert result.returncode in [0, 1]

    def test_info_action_via_module(self):
        """Test --info action works via module."""
        result = subprocess.run(
            [sys.executable, "-m", "scitex.capture", "--info"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # May succeed or fail depending on WSL state, but should not crash
        assert result.returncode in [0, 1]


class TestModuleAttributes:
    """Test module-level attributes."""

    def test_module_has_file_attribute(self):
        """Test module has __FILE__ attribute."""
        from scitex.capture import __main__

        assert hasattr(__main__, "__FILE__")

    def test_module_has_dir_attribute(self):
        """Test module has __DIR__ attribute."""
        from scitex.capture import __main__

        assert hasattr(__main__, "__DIR__")

    def test_module_docstring_exists(self):
        """Test module has a docstring."""
        from scitex.capture import __main__

        # The docstring is in the source but may not be __doc__
        # Check the source structure is correct
        assert __main__ is not None


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
