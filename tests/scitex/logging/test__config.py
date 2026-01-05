#!/usr/bin/env python3
"""Tests for scitex.logging._config module."""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestSetLevel:
    """Test set_level function."""

    def setup_method(self):
        """Reset logging state before each test."""
        import scitex.logging._config as config_module
        from scitex.logging._config import _GLOBAL_LEVEL

        config_module._GLOBAL_LEVEL = None

    def test_set_level_with_string_debug(self):
        """Test setting level with 'debug' string."""
        from scitex.logging._config import get_level, set_level

        set_level("debug")
        assert get_level() == logging.DEBUG

    def test_set_level_with_string_info(self):
        """Test setting level with 'info' string."""
        from scitex.logging._config import get_level, set_level

        set_level("info")
        assert get_level() == logging.INFO

    def test_set_level_with_string_warning(self):
        """Test setting level with 'warning' string."""
        from scitex.logging._config import get_level, set_level

        set_level("warning")
        assert get_level() == logging.WARNING

    def test_set_level_with_string_error(self):
        """Test setting level with 'error' string."""
        from scitex.logging._config import get_level, set_level

        set_level("error")
        assert get_level() == logging.ERROR

    def test_set_level_with_string_critical(self):
        """Test setting level with 'critical' string."""
        from scitex.logging._config import get_level, set_level

        set_level("critical")
        assert get_level() == logging.CRITICAL

    def test_set_level_with_integer(self):
        """Test setting level with integer constant."""
        from scitex.logging._config import get_level, set_level

        set_level(logging.WARNING)
        assert get_level() == logging.WARNING

    def test_set_level_case_insensitive(self):
        """Test that level string is case insensitive."""
        from scitex.logging._config import get_level, set_level

        set_level("DEBUG")
        assert get_level() == logging.DEBUG

        set_level("Info")
        assert get_level() == logging.INFO

    def test_set_level_with_custom_success(self):
        """Test setting level with 'success' custom level."""
        from scitex.logging._config import get_level, set_level
        from scitex.logging._levels import SUCCESS

        set_level("success")
        assert get_level() == SUCCESS

    def test_set_level_with_custom_fail(self):
        """Test setting level with 'fail' custom level."""
        from scitex.logging._config import get_level, set_level
        from scitex.logging._levels import FAIL

        set_level("fail")
        assert get_level() == FAIL


class TestGetLevel:
    """Test get_level function."""

    def test_get_level_returns_global_level_when_set(self):
        """Test get_level returns the globally set level."""
        from scitex.logging._config import get_level, set_level

        set_level(logging.ERROR)
        assert get_level() == logging.ERROR

    def test_get_level_returns_root_logger_level_when_global_not_set(self):
        """Test get_level returns root logger level when global not set."""
        import scitex.logging._config as config_module
        from scitex.logging._config import get_level

        config_module._GLOBAL_LEVEL = None
        root_level = logging.getLogger().level
        assert get_level() == root_level


class TestFileLogging:
    """Test enable_file_logging and is_file_logging_enabled functions."""

    def setup_method(self):
        """Reset file logging state before each test."""
        import scitex.logging._config as config_module

        config_module._FILE_LOGGING_ENABLED = True

    def test_file_logging_enabled_by_default(self):
        """Test that file logging is enabled by default."""
        from scitex.logging._config import is_file_logging_enabled

        assert is_file_logging_enabled() is True

    def test_enable_file_logging_true(self):
        """Test enabling file logging."""
        from scitex.logging._config import enable_file_logging, is_file_logging_enabled

        enable_file_logging(True)
        assert is_file_logging_enabled() is True

    def test_enable_file_logging_false(self):
        """Test disabling file logging."""
        from scitex.logging._config import enable_file_logging, is_file_logging_enabled

        enable_file_logging(False)
        assert is_file_logging_enabled() is False

    def test_enable_file_logging_toggle(self):
        """Test toggling file logging on and off."""
        from scitex.logging._config import enable_file_logging, is_file_logging_enabled

        enable_file_logging(False)
        assert is_file_logging_enabled() is False
        enable_file_logging(True)
        assert is_file_logging_enabled() is True


class TestConfigure:
    """Test configure function."""

    def setup_method(self):
        """Reset logging state before each test."""
        # Clear existing handlers
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    def teardown_method(self):
        """Clean up after tests."""
        # Restore default state
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    def test_configure_with_console_only(self):
        """Test configure with only console output."""
        from scitex.logging._config import configure

        configure(
            level="info", enable_console=True, enable_file=False, capture_prints=False
        )

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)

    def test_configure_sets_level(self):
        """Test that configure sets the log level."""
        from scitex.logging._config import configure, get_level

        configure(
            level="warning",
            enable_console=True,
            enable_file=False,
            capture_prints=False,
        )
        assert get_level() == logging.WARNING

    def test_configure_with_string_level(self):
        """Test configure with string level argument."""
        from scitex.logging._config import configure, get_level

        configure(
            level="debug", enable_console=True, enable_file=False, capture_prints=False
        )
        assert get_level() == logging.DEBUG

    def test_configure_clears_existing_handlers(self):
        """Test that configure clears existing handlers before adding new ones."""
        from scitex.logging._config import configure

        # Add some handlers first
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        assert len(root.handlers) >= 2

        # Configure should clear them
        configure(
            level="info", enable_console=True, enable_file=False, capture_prints=False
        )
        assert len(root.handlers) == 1

    def test_configure_with_file_logging(self):
        """Test configure with file logging enabled."""
        from scitex.logging._config import configure

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            configure(
                level="info",
                log_file=log_file,
                enable_file=True,
                enable_console=True,
                capture_prints=False,
            )

            root = logging.getLogger()
            assert len(root.handlers) == 2  # Console + File

    def test_configure_no_handlers_when_both_disabled(self):
        """Test configure with both console and file disabled."""
        from scitex.logging._config import configure

        configure(
            level="info", enable_console=False, enable_file=False, capture_prints=False
        )

        root = logging.getLogger()
        assert len(root.handlers) == 0


class TestGetLogPath:
    """Test get_log_path function."""

    def test_get_log_path_returns_none_when_no_file_handler(self):
        """Test get_log_path returns None when no file handler exists."""
        from scitex.logging._config import configure, get_log_path

        configure(
            level="info", enable_console=True, enable_file=False, capture_prints=False
        )
        assert get_log_path() is None

    def test_get_log_path_returns_path_when_file_handler_exists(self):
        """Test get_log_path returns the log file path when file handler exists."""
        from scitex.logging._config import configure, get_log_path

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            configure(
                level="info",
                log_file=log_file,
                enable_file=True,
                enable_console=False,
                capture_prints=False,
            )

            result = get_log_path()
            assert result is not None
            assert log_file in result


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
