#!/usr/bin/env python3
"""Tests for scitex.logging._Tee module."""

import os
import sys
import tempfile
from io import StringIO

import pytest


class TestTee:
    """Test Tee class."""

    def test_tee_init_stdout(self):
        """Test Tee initialization with stdout."""
        from scitex.logging._Tee import Tee

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            tee = Tee(sys.stdout, log_path, verbose=False)

            assert tee._stream is sys.stdout
            assert tee._log_path == log_path
            assert tee._is_stderr is False
            tee.close()

    def test_tee_init_stderr(self):
        """Test Tee initialization with stderr."""
        from scitex.logging._Tee import Tee

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            tee = Tee(sys.stderr, log_path, verbose=False)

            assert tee._stream is sys.stderr
            assert tee._is_stderr is True
            tee.close()

    def test_tee_creates_log_file(self):
        """Test that Tee creates the log file."""
        from scitex.logging._Tee import Tee

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            tee = Tee(sys.stdout, log_path, verbose=False)

            assert os.path.exists(log_path)
            tee.close()

    def test_tee_write_to_stream_and_file(self):
        """Test that Tee writes to both stream and file."""
        from scitex.logging._Tee import Tee

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            buffer = StringIO()
            tee = Tee(buffer, log_path, verbose=False)

            tee.write("test message")
            tee.flush()

            # Check stream
            assert "test message" in buffer.getvalue()

            # Check file
            tee.close()
            with open(log_path) as f:
                content = f.read()
            assert "test message" in content

    def test_tee_stderr_filters_progress_bars(self):
        """Test that stderr Tee filters progress bar patterns."""
        from scitex.logging._Tee import Tee

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            buffer = StringIO()
            # Create a mock stderr
            tee = Tee(buffer, log_path, verbose=False)
            tee._is_stderr = True

            # Regular message should be logged
            tee.write("error message\n")

            # Progress bar pattern should be filtered
            tee.write("  50%|████      [A")

            tee.close()
            with open(log_path) as f:
                content = f.read()

            assert "error message" in content
            # Progress bar should be filtered for stderr
            assert "50%" not in content

    def test_tee_flush(self):
        """Test Tee flush method."""
        from scitex.logging._Tee import Tee

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            tee = Tee(sys.stdout, log_path, verbose=False)

            tee.write("test")
            tee.flush()  # Should not raise
            tee.close()

    def test_tee_isatty(self):
        """Test Tee isatty method."""
        from scitex.logging._Tee import Tee

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            tee = Tee(sys.stdout, log_path, verbose=False)

            # Should return same as stream
            result = tee.isatty()
            assert isinstance(result, bool)
            tee.close()

    def test_tee_fileno(self):
        """Test Tee fileno method."""
        from scitex.logging._Tee import Tee

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            tee = Tee(sys.stdout, log_path, verbose=False)

            # Should return stream's fileno
            result = tee.fileno()
            assert isinstance(result, int)
            tee.close()

    def test_tee_buffer_property(self):
        """Test Tee buffer property."""
        from scitex.logging._Tee import Tee

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            tee = Tee(sys.stdout, log_path, verbose=False)

            # Should return stream's buffer
            buffer = tee.buffer
            assert buffer is sys.stdout.buffer
            tee.close()

    def test_tee_close(self):
        """Test Tee close method."""
        from scitex.logging._Tee import Tee

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            tee = Tee(sys.stdout, log_path, verbose=False)

            tee.close()
            assert tee._log_file is None

    def test_tee_close_idempotent(self):
        """Test that Tee close can be called multiple times."""
        from scitex.logging._Tee import Tee

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            tee = Tee(sys.stdout, log_path, verbose=False)

            tee.close()
            tee.close()  # Should not raise

    def test_tee_handles_invalid_log_path(self):
        """Test Tee handles invalid log path gracefully."""
        from scitex.logging._Tee import Tee

        # Path that cannot be created
        tee = Tee(sys.stdout, "/nonexistent/deep/path/test.log", verbose=False)
        assert tee._log_file is None


class TestTeeFunction:
    """Test tee() function."""

    def test_tee_function_returns_tuple(self):
        """Test that tee() returns a tuple of Tee objects."""
        from scitex.logging._Tee import Tee, tee

        with tempfile.TemporaryDirectory() as tmpdir:
            result = tee(sys, sdir=tmpdir, verbose=False)

            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], Tee)
            assert isinstance(result[1], Tee)

            result[0].close()
            result[1].close()

    def test_tee_function_creates_log_files(self):
        """Test that tee() creates log files."""
        from scitex.logging._Tee import tee

        with tempfile.TemporaryDirectory() as tmpdir:
            stdout_tee, stderr_tee = tee(sys, sdir=tmpdir, verbose=False)

            logs_dir = os.path.join(tmpdir, "logs")
            stdout_log = os.path.join(logs_dir, "stdout.log")
            stderr_log = os.path.join(logs_dir, "stderr.log")

            assert os.path.exists(stdout_log)
            assert os.path.exists(stderr_log)

            stdout_tee.close()
            stderr_tee.close()

    def test_tee_function_creates_logs_directory(self):
        """Test that tee() creates logs subdirectory."""
        from scitex.logging._Tee import tee

        with tempfile.TemporaryDirectory() as tmpdir:
            stdout_tee, stderr_tee = tee(sys, sdir=tmpdir, verbose=False)

            logs_dir = os.path.join(tmpdir, "logs")
            assert os.path.isdir(logs_dir)

            stdout_tee.close()
            stderr_tee.close()


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
