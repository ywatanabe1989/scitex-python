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
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_config.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-08-21 21:41:37 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/log/_config.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """Configuration and setup for SciTeX logging."""
# 
# import logging
# from typing import Optional, Union
# 
# from ._handlers import create_console_handler, create_file_handler, get_default_log_path
# from ._levels import CRITICAL, DEBUG, ERROR, FAIL, INFO, SUCCESS, WARNING
# from ._logger import setup_logger_class
# 
# # Global level variable
# _GLOBAL_LEVEL = None
# _FILE_LOGGING_ENABLED = True  # Enable file logging by default
# 
# 
# def set_level(level: Union[str, int]):
#     """Set global log level for all SciTeX loggers."""
#     global _GLOBAL_LEVEL
# 
#     level_map = {
#         "debug": DEBUG,
#         "info": INFO,
#         "warning": WARNING,
#         "error": ERROR,
#         "critical": CRITICAL,
#         "success": SUCCESS,
#         "fail": FAIL,
#     }
# 
#     if isinstance(level, str):
#         level = level_map.get(level.lower(), level)
# 
#     _GLOBAL_LEVEL = level
#     logging.getLogger().setLevel(level)
# 
#     # Update all existing handlers
#     for handler in logging.getLogger().handlers:
#         handler.setLevel(level)
# 
# 
# def get_level():
#     """Get current global log level."""
#     return _GLOBAL_LEVEL or logging.getLogger().level
# 
# 
# def enable_file_logging(enabled: bool = True):
#     """Enable or disable file logging globally."""
#     global _FILE_LOGGING_ENABLED
#     _FILE_LOGGING_ENABLED = enabled
# 
# 
# def is_file_logging_enabled():
#     """Check if file logging is enabled."""
#     return _FILE_LOGGING_ENABLED
# 
# 
# def configure(
#     level: Union[str, int] = "info",
#     log_file: Optional[str] = None,
#     enable_file: bool = True,
#     enable_console: bool = True,
#     capture_prints: bool = True,
#     max_file_size: int = 10 * 1024 * 1024,
#     backup_count: int = 5,
# ):
#     """Configure logging for SciTeX with both console and file output.
# 
#     Args:
#         level: Log level (string or logging constant)
#         log_file: Path to log file (default: ~/.scitex/logs/scitex-YYYY-MM-DD.log)
#         enable_file: Whether to enable file logging
#         enable_console: Whether to enable console logging
#         capture_prints: Whether to capture print() statements to logs
#         max_file_size: Maximum size of log file before rotation (default: 10MB)
#         backup_count: Number of backup files to keep (default: 5)
#     """
#     # Setup custom logger class
#     setup_logger_class()
# 
#     # Convert level if string
#     level_map = {
#         "debug": DEBUG,
#         "info": INFO,
#         "warning": WARNING,
#         "error": ERROR,
#         "critical": CRITICAL,
#         "success": SUCCESS,
#         "fail": FAIL,
#     }
# 
#     if isinstance(level, str):
#         level = level_map.get(level.lower(), level)
# 
#     # Set global level
#     set_level(level)
# 
#     # Clear any existing handlers
#     root_logger = logging.getLogger()
#     for handler in root_logger.handlers[:]:
#         root_logger.removeHandler(handler)
# 
#     handlers = []
# 
#     # Add console handler if enabled
#     if enable_console:
#         console_handler = create_console_handler(level)
#         handlers.append(console_handler)
# 
#     # Add file handler if enabled
#     if enable_file and is_file_logging_enabled():
#         if log_file is None:
#             log_file = get_default_log_path()
# 
#         file_handler = create_file_handler(
#             log_file, level, max_bytes=max_file_size, backup_count=backup_count
#         )
#         handlers.append(file_handler)
# 
#     # Configure basic logging with our handlers
#     logging.basicConfig(
#         level=level,
#         handlers=handlers,
#         force=True,  # Force reconfiguration
#     )
# 
#     # Enable print capture if requested
#     if capture_prints:
#         from ._print_capture import enable_print_capture
# 
#         enable_print_capture()
# 
# 
# def get_log_path():
#     """Get the current log file path."""
#     for handler in logging.getLogger().handlers:
#         if hasattr(handler, "baseFilename"):
#             return handler.baseFilename
#     return None
# 
# 
# __all__ = [
#     "set_level",
#     "get_level",
#     "enable_file_logging",
#     "is_file_logging_enabled",
#     "configure",
#     "get_log_path",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_config.py
# --------------------------------------------------------------------------------
