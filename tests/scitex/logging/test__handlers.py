#!/usr/bin/env python3
"""Tests for scitex.logging._handlers module."""

import logging
import logging.handlers
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest


class TestCreateConsoleHandler:
    """Test create_console_handler function."""

    def test_create_console_handler_returns_stream_handler(self):
        """Test that create_console_handler returns a StreamHandler."""
        from scitex.logging._handlers import create_console_handler

        handler = create_console_handler()
        assert isinstance(handler, logging.StreamHandler)

    def test_create_console_handler_default_level(self):
        """Test that default level is INFO."""
        from scitex.logging._handlers import create_console_handler

        handler = create_console_handler()
        assert handler.level == logging.INFO

    def test_create_console_handler_custom_level(self):
        """Test create_console_handler with custom level."""
        from scitex.logging._handlers import create_console_handler

        handler = create_console_handler(level=logging.DEBUG)
        assert handler.level == logging.DEBUG

    def test_create_console_handler_has_scitex_formatter(self):
        """Test that handler has SciTeXConsoleFormatter."""
        from scitex.logging._formatters import SciTeXConsoleFormatter
        from scitex.logging._handlers import create_console_handler

        handler = create_console_handler()
        assert isinstance(handler.formatter, SciTeXConsoleFormatter)


class TestCreateFileHandler:
    """Test create_file_handler function."""

    def test_create_file_handler_returns_rotating_handler(self):
        """Test that create_file_handler returns a RotatingFileHandler."""
        from scitex.logging._handlers import create_file_handler

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            handler = create_file_handler(log_path)
            assert isinstance(handler, logging.handlers.RotatingFileHandler)
            handler.close()

    def test_create_file_handler_creates_directory(self):
        """Test that file handler creates parent directories."""
        from scitex.logging._handlers import create_file_handler

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "nested", "dir", "test.log")
            handler = create_file_handler(log_path)
            assert os.path.isdir(os.path.dirname(log_path))
            handler.close()

    def test_create_file_handler_default_level(self):
        """Test that default level is INFO."""
        from scitex.logging._handlers import create_file_handler

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            handler = create_file_handler(log_path)
            assert handler.level == logging.INFO
            handler.close()

    def test_create_file_handler_custom_level(self):
        """Test create_file_handler with custom level."""
        from scitex.logging._handlers import create_file_handler

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            handler = create_file_handler(log_path, level=logging.DEBUG)
            assert handler.level == logging.DEBUG
            handler.close()

    def test_create_file_handler_custom_max_bytes(self):
        """Test create_file_handler with custom max_bytes."""
        from scitex.logging._handlers import create_file_handler

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            handler = create_file_handler(log_path, max_bytes=1024)
            assert handler.maxBytes == 1024
            handler.close()

    def test_create_file_handler_custom_backup_count(self):
        """Test create_file_handler with custom backup_count."""
        from scitex.logging._handlers import create_file_handler

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            handler = create_file_handler(log_path, backup_count=10)
            assert handler.backupCount == 10
            handler.close()

    def test_create_file_handler_has_scitex_formatter(self):
        """Test that handler has SciTeXFileFormatter."""
        from scitex.logging._formatters import SciTeXFileFormatter
        from scitex.logging._handlers import create_file_handler

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            handler = create_file_handler(log_path)
            assert isinstance(handler.formatter, SciTeXFileFormatter)
            handler.close()

    def test_create_file_handler_writes_logs(self):
        """Test that file handler actually writes logs."""
        from scitex.logging._handlers import create_file_handler

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            handler = create_file_handler(log_path)

            logger = logging.getLogger("test_file_handler")
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)

            logger.info("Test message")
            handler.flush()
            handler.close()

            with open(log_path) as f:
                content = f.read()
            assert "Test message" in content

            logger.removeHandler(handler)


class TestGetDefaultLogPath:
    """Test get_default_log_path function."""

    def test_get_default_log_path_returns_string(self):
        """Test that get_default_log_path returns a string."""
        from scitex.logging._handlers import get_default_log_path

        path = get_default_log_path()
        assert isinstance(path, str)

    def test_get_default_log_path_contains_scitex(self):
        """Test that path contains 'scitex' in the filename."""
        from scitex.logging._handlers import get_default_log_path

        path = get_default_log_path()
        assert "scitex" in os.path.basename(path).lower()

    def test_get_default_log_path_contains_date(self):
        """Test that path contains current date."""
        from scitex.logging._handlers import get_default_log_path

        path = get_default_log_path()
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in path

    def test_get_default_log_path_has_log_extension(self):
        """Test that path has .log extension."""
        from scitex.logging._handlers import get_default_log_path

        path = get_default_log_path()
        assert path.endswith(".log")

    def test_get_default_log_path_contains_logs_dir(self):
        """Test that path contains 'logs' directory."""
        from scitex.logging._handlers import get_default_log_path

        path = get_default_log_path()
        assert "logs" in path


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
