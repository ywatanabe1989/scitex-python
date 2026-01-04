#!/usr/bin/env python3
"""Tests for scitex.logging._print_capture module."""

import logging
import os
import sys
from io import StringIO

import pytest


class TestPrintCapture:
    """Test PrintCapture class."""

    def teardown_method(self):
        """Ensure stdout is restored after each test."""
        from scitex.logging._print_capture import disable_print_capture

        disable_print_capture()

    def test_print_capture_init(self):
        """Test PrintCapture initialization."""
        from scitex.logging._print_capture import PrintCapture

        capture = PrintCapture()
        assert capture.capturing is False
        assert capture.original_stdout is sys.stdout

    def test_print_capture_init_custom_logger(self):
        """Test PrintCapture with custom logger name."""
        from scitex.logging._print_capture import PrintCapture

        capture = PrintCapture(logger_name="custom.logger")
        assert capture.logger.name == "custom.logger"

    def test_print_capture_start_capture(self):
        """Test starting capture."""
        from scitex.logging._print_capture import PrintCapture

        capture = PrintCapture()
        original = sys.stdout

        capture.start_capture()
        assert capture.capturing is True
        assert sys.stdout is capture

        capture.stop_capture()
        assert sys.stdout is original

    def test_print_capture_stop_capture(self):
        """Test stopping capture."""
        from scitex.logging._print_capture import PrintCapture

        capture = PrintCapture()
        original = sys.stdout

        capture.start_capture()
        capture.stop_capture()

        assert capture.capturing is False
        assert sys.stdout is original

    def test_print_capture_double_start(self):
        """Test that double start is safe."""
        from scitex.logging._print_capture import PrintCapture

        capture = PrintCapture()
        capture.start_capture()
        capture.start_capture()  # Should not raise
        assert capture.capturing is True
        capture.stop_capture()

    def test_print_capture_double_stop(self):
        """Test that double stop is safe."""
        from scitex.logging._print_capture import PrintCapture

        capture = PrintCapture()
        capture.stop_capture()  # Should not raise when not capturing
        capture.stop_capture()  # Should not raise

    def test_print_capture_write_to_stdout(self):
        """Test that write goes to original stdout."""
        from scitex.logging._print_capture import PrintCapture

        buffer = StringIO()
        capture = PrintCapture()
        capture.original_stdout = buffer

        capture.write("test message")
        assert "test message" in buffer.getvalue()

    def test_print_capture_flush(self):
        """Test flush method."""
        from scitex.logging._print_capture import PrintCapture

        capture = PrintCapture()
        # Should not raise
        capture.flush()

    def test_print_capture_isatty(self):
        """Test isatty method."""
        from scitex.logging._print_capture import PrintCapture

        capture = PrintCapture()
        # Should return same as original stdout
        result = capture.isatty()
        assert isinstance(result, bool)

    def test_print_capture_context_manager(self):
        """Test PrintCapture as context manager."""
        from scitex.logging._print_capture import PrintCapture

        original = sys.stdout
        capture = PrintCapture()

        with capture as c:
            assert c is capture
            assert capture.capturing is True
            assert sys.stdout is capture

        assert capture.capturing is False
        assert sys.stdout is original


class TestGlobalPrintCapture:
    """Test global print capture functions."""

    def teardown_method(self):
        """Ensure print capture is disabled after each test."""
        from scitex.logging._print_capture import disable_print_capture

        disable_print_capture()
        # Reset the global state
        import scitex.logging._print_capture as pc_module

        pc_module._print_capture = None

    def test_enable_print_capture(self):
        """Test enable_print_capture function."""
        from scitex.logging._print_capture import (
            enable_print_capture,
            is_print_capture_enabled,
        )

        assert is_print_capture_enabled() is False
        enable_print_capture()
        assert is_print_capture_enabled() is True

    def test_disable_print_capture(self):
        """Test disable_print_capture function."""
        from scitex.logging._print_capture import (
            disable_print_capture,
            enable_print_capture,
            is_print_capture_enabled,
        )

        enable_print_capture()
        assert is_print_capture_enabled() is True

        disable_print_capture()
        assert is_print_capture_enabled() is False

    def test_is_print_capture_enabled_false_initially(self):
        """Test that print capture is disabled initially."""
        from scitex.logging._print_capture import is_print_capture_enabled

        assert is_print_capture_enabled() is False

    def test_enable_print_capture_custom_logger(self):
        """Test enable_print_capture with custom logger name."""
        from scitex.logging._print_capture import enable_print_capture

        # Should not raise
        enable_print_capture(logger_name="custom.print.logger")

    def test_enable_print_capture_idempotent(self):
        """Test that multiple enable calls are safe."""
        from scitex.logging._print_capture import (
            enable_print_capture,
            is_print_capture_enabled,
        )

        enable_print_capture()
        enable_print_capture()  # Should be safe
        assert is_print_capture_enabled() is True


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
