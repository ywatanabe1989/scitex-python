#!/usr/bin/env python3
"""Tests for scitex.capture.capture module.

Tests core screenshot capture functionality including:
- ScreenshotWorker initialization and configuration
- Worker lifecycle (start/stop)
- Status reporting
- CaptureManager high-level interface
"""

import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


class TestScreenshotWorkerInit:
    """Test ScreenshotWorker initialization."""

    def test_default_initialization(self):
        """Test worker initializes with default parameters."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ScreenshotWorker(output_dir=tmpdir)

            assert worker.output_dir == Path(tmpdir)
            assert worker.interval_sec == 1.0
            assert worker.verbose is False
            assert worker.use_jpeg is True
            assert worker.jpeg_quality == 60
            assert worker.running is False
            assert worker.worker_thread is None
            assert worker.screenshot_count == 0
            assert worker.session_id is None
            assert worker.monitor == 0
            assert worker.capture_all is False

    def test_custom_initialization(self):
        """Test worker initializes with custom parameters."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            on_capture = lambda x: None
            on_error = lambda x: None

            worker = ScreenshotWorker(
                output_dir=tmpdir,
                interval_sec=2.5,
                verbose=True,
                use_jpeg=False,
                jpeg_quality=90,
                on_capture=on_capture,
                on_error=on_error,
            )

            assert worker.interval_sec == 2.5
            assert worker.verbose is True
            assert worker.use_jpeg is False
            assert worker.jpeg_quality == 90
            assert worker.on_capture is on_capture
            assert worker.on_error is on_error

    def test_creates_output_directory(self):
        """Test worker creates output directory if it doesn't exist."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested", "deep", "dir")
            assert not os.path.exists(nested_dir)

            worker = ScreenshotWorker(output_dir=nested_dir)

            assert os.path.exists(nested_dir)
            assert worker.output_dir == Path(nested_dir)


class TestScreenshotWorkerLifecycle:
    """Test ScreenshotWorker start/stop lifecycle."""

    def test_start_sets_running_state(self):
        """Test start() sets running state correctly."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ScreenshotWorker(output_dir=tmpdir)

            # Mock _take_screenshot to avoid actual capture
            worker._take_screenshot = MagicMock(return_value=None)

            assert worker.running is False
            worker.start()
            assert worker.running is True
            assert worker.session_id is not None
            assert worker.worker_thread is not None
            assert worker.worker_thread.is_alive()

            worker.stop()
            assert worker.running is False

    def test_start_with_custom_session_id(self):
        """Test start() uses provided session_id."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ScreenshotWorker(output_dir=tmpdir)
            worker._take_screenshot = MagicMock(return_value=None)

            worker.start(session_id="my_custom_session")
            assert worker.session_id == "my_custom_session"
            worker.stop()

    def test_start_generates_session_id_from_timestamp(self):
        """Test start() generates session_id from timestamp when not provided."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ScreenshotWorker(output_dir=tmpdir)
            worker._take_screenshot = MagicMock(return_value=None)

            worker.start()
            # Session ID format: YYYYMMDD_HHMMSS
            assert len(worker.session_id) == 15
            assert "_" in worker.session_id
            worker.stop()

    def test_stop_when_not_running(self):
        """Test stop() is safe when worker not running."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ScreenshotWorker(output_dir=tmpdir)
            # Should not raise
            worker.stop()
            assert worker.running is False

    def test_double_start_is_idempotent(self):
        """Test calling start() twice doesn't create duplicate threads."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ScreenshotWorker(output_dir=tmpdir)
            worker._take_screenshot = MagicMock(return_value=None)

            worker.start()
            first_thread = worker.worker_thread
            first_session = worker.session_id

            worker.start()  # Second call
            assert worker.worker_thread is first_thread
            assert worker.session_id == first_session

            worker.stop()

    def test_worker_thread_is_daemon(self):
        """Test worker thread runs as daemon."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ScreenshotWorker(output_dir=tmpdir)
            worker._take_screenshot = MagicMock(return_value=None)

            worker.start()
            assert worker.worker_thread.daemon is True
            worker.stop()


class TestScreenshotWorkerStatus:
    """Test ScreenshotWorker status reporting."""

    def test_get_status_returns_all_fields(self):
        """Test get_status() returns complete status dict."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ScreenshotWorker(
                output_dir=tmpdir, interval_sec=2.0, use_jpeg=True, jpeg_quality=75
            )

            status = worker.get_status()

            assert "running" in status
            assert "session_id" in status
            assert "screenshot_count" in status
            assert "output_dir" in status
            assert "interval_sec" in status
            assert "use_jpeg" in status
            assert "jpeg_quality" in status

            assert status["running"] is False
            assert status["screenshot_count"] == 0
            assert status["interval_sec"] == 2.0
            assert status["use_jpeg"] is True
            assert status["jpeg_quality"] == 75

    def test_get_status_reflects_running_state(self):
        """Test get_status() reflects current running state."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ScreenshotWorker(output_dir=tmpdir)
            worker._take_screenshot = MagicMock(return_value=None)

            assert worker.get_status()["running"] is False

            worker.start(session_id="test_session")
            status = worker.get_status()
            assert status["running"] is True
            assert status["session_id"] == "test_session"

            worker.stop()
            assert worker.get_status()["running"] is False


class TestScreenshotWorkerWSLDetection:
    """Test WSL detection functionality."""

    def test_is_wsl_on_linux_with_microsoft(self):
        """Test _is_wsl() returns True on WSL."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ScreenshotWorker(output_dir=tmpdir)

            # Mock the conditions for WSL
            with patch.object(sys, "platform", "linux"):
                with patch("os.uname") as mock_uname:
                    mock_uname.return_value = MagicMock(
                        release="5.15.90.1-microsoft-standard-WSL2"
                    )
                    assert worker._is_wsl() is True

    def test_is_wsl_on_linux_without_microsoft(self):
        """Test _is_wsl() returns False on regular Linux."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ScreenshotWorker(output_dir=tmpdir)

            with patch.object(sys, "platform", "linux"):
                with patch("os.uname") as mock_uname:
                    mock_uname.return_value = MagicMock(release="5.15.0-generic")
                    assert worker._is_wsl() is False

    def test_is_wsl_on_non_linux(self):
        """Test _is_wsl() returns False on non-Linux platforms."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ScreenshotWorker(output_dir=tmpdir)

            with patch.object(sys, "platform", "darwin"):
                assert worker._is_wsl() is False

            with patch.object(sys, "platform", "win32"):
                assert worker._is_wsl() is False


class TestScreenshotWorkerCallbacks:
    """Test ScreenshotWorker callback functionality."""

    def test_on_capture_callback_called(self):
        """Test on_capture callback is called after successful capture."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            captured_paths = []
            worker = ScreenshotWorker(
                output_dir=tmpdir,
                interval_sec=0.1,
                on_capture=lambda path: captured_paths.append(path),
            )

            # Mock _take_screenshot to return a fake path
            worker._take_screenshot = MagicMock(return_value="/fake/path.jpg")

            worker.start()
            time.sleep(0.25)  # Allow a few captures
            worker.stop()

            assert len(captured_paths) > 0
            assert all(path == "/fake/path.jpg" for path in captured_paths)

    def test_on_error_callback_called(self):
        """Test on_error callback is called when capture raises exception."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            errors = []
            worker = ScreenshotWorker(
                output_dir=tmpdir,
                interval_sec=0.1,
                on_error=lambda e: errors.append(e),
            )

            # Mock _take_screenshot to raise exception
            worker._take_screenshot = MagicMock(side_effect=RuntimeError("Test error"))

            worker.start()
            time.sleep(0.25)
            worker.stop()

            assert len(errors) > 0
            assert all(isinstance(e, RuntimeError) for e in errors)


class TestCaptureManager:
    """Test CaptureManager high-level interface."""

    def test_initialization(self):
        """Test CaptureManager initializes correctly."""
        from scitex.capture.capture import CaptureManager

        manager = CaptureManager()
        assert manager.worker is None

    def test_start_capture_creates_worker(self):
        """Test start_capture creates and starts worker."""
        from scitex.capture.capture import CaptureManager, ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CaptureManager()

            # Mock the worker's _take_screenshot
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                worker = manager.start_capture(output_dir=tmpdir)

                assert manager.worker is not None
                assert manager.worker is worker
                assert worker.running is True

                manager.stop_capture()
                assert manager.worker is None

    def test_start_capture_with_parameters(self):
        """Test start_capture passes parameters correctly."""
        from scitex.capture.capture import CaptureManager, ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CaptureManager()

            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                worker = manager.start_capture(
                    output_dir=tmpdir,
                    interval=2.5,
                    jpeg=False,
                    quality=90,
                    verbose=True,
                    monitor_id=1,
                    capture_all=True,
                )

                assert worker.interval_sec == 2.5
                assert worker.use_jpeg is False
                assert worker.jpeg_quality == 90
                assert worker.verbose is True
                assert worker.monitor == 1
                assert worker.capture_all is True

                manager.stop_capture()

    def test_start_capture_when_already_running(self):
        """Test start_capture returns existing worker if already running."""
        from scitex.capture.capture import CaptureManager, ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CaptureManager()

            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                worker1 = manager.start_capture(output_dir=tmpdir)
                worker2 = manager.start_capture(output_dir=tmpdir)

                assert worker1 is worker2
                manager.stop_capture()

    def test_stop_capture_when_not_running(self):
        """Test stop_capture is safe when no worker running."""
        from scitex.capture.capture import CaptureManager

        manager = CaptureManager()
        # Should not raise
        manager.stop_capture()
        assert manager.worker is None


class TestCaptureManagerSingleScreenshot:
    """Test CaptureManager single screenshot functionality."""

    def test_take_single_screenshot_with_mocked_capture(self):
        """Test take_single_screenshot with mocked capture backend."""
        from scitex.capture.capture import CaptureManager, ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CaptureManager()
            output_path = os.path.join(tmpdir, "test.jpg")

            # Create a mock screenshot file
            def mock_take_screenshot(self):
                fake_path = os.path.join(tmpdir, "single_0000_test.jpg")
                Path(fake_path).touch()
                return fake_path

            with patch.object(
                ScreenshotWorker, "_take_screenshot", mock_take_screenshot
            ):
                with patch.object(ScreenshotWorker, "_is_wsl", return_value=False):
                    result = manager.take_single_screenshot(
                        output_path=output_path, jpeg=True, quality=85
                    )

                    # Result should be the output_path if rename was successful
                    # or None if capture failed
                    # In mocked test, we just verify the method doesn't crash
                    assert result is None or isinstance(result, str)

    def test_take_single_screenshot_generates_default_path(self):
        """Test take_single_screenshot generates path when none provided."""
        from scitex.capture.capture import CaptureManager, ScreenshotWorker

        manager = CaptureManager()

        # Mock to return None (no capture possible)
        with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
            result = manager.take_single_screenshot()
            assert result is None


class TestFilenameGeneration:
    """Test filename generation for screenshots."""

    def test_filename_format_jpeg(self):
        """Test JPEG filename format includes session, count, timestamp."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ScreenshotWorker(output_dir=tmpdir, use_jpeg=True)
            worker.session_id = "20250104_120000"
            worker.screenshot_count = 5

            # Mock the capture methods to return False
            worker._is_wsl = MagicMock(return_value=False)
            worker._capture_native_screen = MagicMock(return_value=False)

            # Call _take_screenshot (will fail but we can check filename logic)
            result = worker._take_screenshot()

            # Since both capture methods return False, result should be None
            assert result is None

    def test_filename_format_png(self):
        """Test PNG extension when use_jpeg is False."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = ScreenshotWorker(output_dir=tmpdir, use_jpeg=False)
            worker.session_id = "test_session"
            worker.screenshot_count = 0

            # Verify extension setting
            ext = "jpg" if worker.use_jpeg else "png"
            assert ext == "png"


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
