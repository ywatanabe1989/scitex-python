#!/usr/bin/env python3
"""Tests for scitex.capture.utils module.

Tests utility functions for screen capture:
- capture() function (main API)
- take_screenshot() simple interface
- start_monitor()/stop_monitor() for continuous capture
- Cache management
- Category detection
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCaptureFunction:
    """Test the main capture() function."""

    def test_capture_returns_none_on_failure(self):
        """Test capture returns None when capture fails."""
        from scitex.capture.capture import ScreenshotWorker

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                with patch.object(ScreenshotWorker, "_is_wsl", return_value=False):
                    from scitex.capture.utils import capture

                    result = capture(
                        path=os.path.join(tmpdir, "test.jpg"),
                        verbose=False,
                        auto_categorize=False,
                    )
                    assert result is None

    def test_capture_with_message(self):
        """Test capture with message parameter."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.utils import capture

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                with patch.object(ScreenshotWorker, "_is_wsl", return_value=False):
                    result = capture(
                        message="test message",
                        path=os.path.join(tmpdir, "test.jpg"),
                        verbose=False,
                    )
                    assert result is None


class TestCaptureMonitorSettings:
    """Test capture function monitor settings."""

    def test_capture_with_monitor_id(self):
        """Test capture with specific monitor_id."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.utils import capture

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                with patch.object(ScreenshotWorker, "_is_wsl", return_value=False):
                    result = capture(
                        path=os.path.join(tmpdir, "test.jpg"),
                        monitor_id=1,
                        verbose=False,
                    )
                    assert result is None

    def test_capture_all_monitors(self):
        """Test capture with capture_all=True."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.utils import capture

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                with patch.object(ScreenshotWorker, "_is_wsl", return_value=False):
                    result = capture(
                        path=os.path.join(tmpdir, "test.jpg"),
                        capture_all=True,
                        verbose=False,
                    )
                    assert result is None

    def test_capture_all_shorthand(self):
        """Test capture with all=True shorthand."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.utils import capture

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                with patch.object(ScreenshotWorker, "_is_wsl", return_value=False):
                    result = capture(
                        path=os.path.join(tmpdir, "test.jpg"),
                        all=True,
                        verbose=False,
                    )
                    assert result is None


class TestCaptureURLCapture:
    """Test URL capture functionality."""

    def test_capture_url_returns_none_on_failure(self):
        """Test URL capture returns None when all methods fail."""
        from scitex.capture.utils import capture

        # Mock playwright import to fail
        with patch.dict(sys.modules, {"playwright.sync_api": None}):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = capture(
                    url="http://localhost:8000",
                    path=os.path.join(tmpdir, "test.jpg"),
                    verbose=False,
                )
                # Returns None when playwright not available and not on WSL
                assert result is None


class TestCaptureAppCapture:
    """Test app-specific capture functionality."""

    def test_capture_app_not_found(self):
        """Test capture when specified app is not found."""
        from scitex.capture.utils import _manager, capture

        with patch.object(
            _manager, "get_info", return_value={"Windows": {"Details": []}}
        ):
            result = capture(
                app="nonexistent_app",
                verbose=False,
            )
            assert result is None

    def test_capture_app_found_but_capture_fails(self):
        """Test capture when app is found but capture fails."""
        from scitex.capture.utils import _manager, capture

        mock_windows = {
            "Windows": {
                "Details": [
                    {
                        "ProcessName": "chrome",
                        "Title": "Google Chrome",
                        "Handle": 12345,
                    }
                ]
            }
        }

        with patch.object(_manager, "get_info", return_value=mock_windows):
            with patch.object(_manager, "capture_window", return_value=None):
                result = capture(
                    app="chrome",
                    verbose=False,
                )
                assert result is None


class TestTakeScreenshot:
    """Test take_screenshot simple interface."""

    def test_take_screenshot_returns_none_on_failure(self):
        """Test take_screenshot returns None when capture fails."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.utils import take_screenshot

        with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
            result = take_screenshot()
            assert result is None

    def test_take_screenshot_with_custom_path(self):
        """Test take_screenshot with custom output path."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.utils import take_screenshot

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "custom.jpg")

            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                result = take_screenshot(output_path=output_path)
                assert result is None

    def test_take_screenshot_with_quality(self):
        """Test take_screenshot with quality parameter."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.utils import take_screenshot

        with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
            result = take_screenshot(jpeg=True, quality=50)
            assert result is None


class TestStartStopMonitor:
    """Test start_monitor and stop_monitor functions."""

    def test_start_monitor_returns_worker(self):
        """Test start_monitor returns a ScreenshotWorker."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.utils import start_monitor, stop_monitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                worker = start_monitor(
                    output_dir=tmpdir,
                    interval=0.5,
                    verbose=False,
                )

                assert isinstance(worker, ScreenshotWorker)
                assert worker.running is True

                stop_monitor()
                assert worker.running is False

    def test_start_monitor_with_callbacks(self):
        """Test start_monitor with on_capture and on_error callbacks."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.utils import start_monitor, stop_monitor

        with tempfile.TemporaryDirectory() as tmpdir:
            captures = []
            errors = []

            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                worker = start_monitor(
                    output_dir=tmpdir,
                    on_capture=lambda p: captures.append(p),
                    on_error=lambda e: errors.append(e),
                    verbose=False,
                )

                assert worker.on_capture is not None
                assert worker.on_error is not None

                stop_monitor()

    def test_start_monitor_with_monitor_settings(self):
        """Test start_monitor with monitor_id and capture_all."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.utils import start_monitor, stop_monitor

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                worker = start_monitor(
                    output_dir=tmpdir,
                    monitor_id=2,
                    capture_all=True,
                    verbose=False,
                )

                assert worker.monitor == 2
                assert worker.capture_all is True

                stop_monitor()

    def test_stop_monitor_when_not_started(self):
        """Test stop_monitor is safe when not started."""
        from scitex.capture.utils import stop_monitor

        # Should not raise
        stop_monitor()


class TestCacheSizeManagement:
    """Test cache size management functionality."""

    def test_manage_cache_size_does_nothing_under_limit(self):
        """Test cache management doesn't delete files under limit."""
        from scitex.capture.utils import _manage_cache_size

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create small test files
            for i in range(5):
                (cache_dir / f"test_{i}.jpg").write_bytes(b"x" * 100)

            _manage_cache_size(cache_dir, max_size_gb=1.0)

            # All files should still exist
            assert len(list(cache_dir.glob("*.jpg"))) == 5

    def test_manage_cache_size_deletes_old_files(self):
        """Test cache management deletes oldest files when over limit."""
        from scitex.capture.utils import _manage_cache_size

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create files with different modification times
            for i in range(5):
                file_path = cache_dir / f"test_{i}.jpg"
                file_path.write_bytes(b"x" * 1024 * 1024)  # 1MB each
                time.sleep(0.1)  # Ensure different mtimes

            # Set limit to ~2MB (should delete 3 oldest files)
            _manage_cache_size(cache_dir, max_size_gb=0.000002)

            # Some files should be deleted
            remaining = list(cache_dir.glob("*.jpg"))
            assert len(remaining) < 5

    def test_manage_cache_size_nonexistent_dir(self):
        """Test cache management handles nonexistent directory."""
        from scitex.capture.utils import _manage_cache_size

        # Should not raise
        _manage_cache_size(Path("/nonexistent/path"), max_size_gb=1.0)

    def test_manage_cache_size_handles_png_files(self):
        """Test cache management also handles PNG files."""
        from scitex.capture.utils import _manage_cache_size

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create PNG files
            for i in range(3):
                (cache_dir / f"test_{i}.png").write_bytes(b"x" * 100)

            _manage_cache_size(cache_dir, max_size_gb=1.0)

            # All PNG files should still exist
            assert len(list(cache_dir.glob("*.png"))) == 3


class TestCategoryDetection:
    """Test screenshot category detection."""

    def test_detect_category_returns_stdout_by_default(self):
        """Test _detect_category returns 'stdout' by default."""
        from scitex.capture.utils import _detect_category

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.jpg")

            try:
                from PIL import Image

                img = Image.new("RGB", (10, 10), color="white")
                img.save(test_file)

                result = _detect_category(test_file)
                assert result == "stdout"
            except ImportError:
                pytest.skip("PIL not available")

    def test_detect_category_detects_red_as_error(self):
        """Test _detect_category detects red-dominant images as error."""
        from scitex.capture.utils import _detect_category

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.jpg")

            try:
                from PIL import Image

                # Create mostly red image (>5% red pixels)
                img = Image.new("RGB", (100, 100), color=(255, 50, 50))
                img.save(test_file)

                result = _detect_category(test_file)
                assert result == "error"
            except ImportError:
                pytest.skip("PIL not available")

    def test_detect_category_detects_error_from_filename(self):
        """Test _detect_category detects error keywords in filename."""
        from scitex.capture.utils import _detect_category

        # Without a valid image, it falls back to filename-based detection
        result = _detect_category("/path/to/error_screenshot.jpg")
        assert result == "stderr"

        result = _detect_category("/path/to/fail_test.jpg")
        assert result == "stderr"

    def test_detect_category_detects_warning_from_filename(self):
        """Test _detect_category detects warning keywords in filename."""
        from scitex.capture.utils import _detect_category

        result = _detect_category("/path/to/warning_dialog.jpg")
        assert result == "stderr"

    def test_detect_category_handles_missing_file(self):
        """Test _detect_category handles missing file gracefully."""
        from scitex.capture.utils import _detect_category

        # Nonexistent file should return stdout (default)
        result = _detect_category("/nonexistent/file.jpg")
        assert result == "stdout"


class TestExceptionContextDetection:
    """Test exception context detection."""

    def test_is_in_exception_context_false_normally(self):
        """Test _is_in_exception_context returns False normally."""
        from scitex.capture.utils import _is_in_exception_context

        assert _is_in_exception_context() is False

    def test_is_in_exception_context_true_in_except(self):
        """Test _is_in_exception_context returns True in except block."""
        from scitex.capture.utils import _is_in_exception_context

        try:
            raise ValueError("Test error")
        except ValueError:
            assert _is_in_exception_context() is True


class TestMessageMetadata:
    """Test message metadata functionality."""

    def test_add_message_metadata_with_pil(self):
        """Test _add_message_metadata adds EXIF metadata with PIL."""
        from scitex.capture.utils import _add_message_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.jpg")

            try:
                from PIL import Image

                img = Image.new("RGB", (10, 10), color="white")
                img.save(test_file)

                _add_message_metadata(test_file, "Test message")

                # Verify file still exists
                assert os.path.exists(test_file)
            except ImportError:
                pytest.skip("PIL not available")

    def test_add_message_metadata_creates_text_file_fallback(self):
        """Test _add_message_metadata creates text file when PIL fails."""
        from scitex.capture.utils import _add_message_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.jpg")
            Path(test_file).touch()

            # Mock PIL to fail
            with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
                _add_message_metadata(test_file, "Test message")

            # Text file should be created
            txt_file = Path(test_file).with_suffix(".txt")
            assert txt_file.exists()
            content = txt_file.read_text()
            assert "Test message" in content


class TestModuleExports:
    """Test module exports."""

    def test_all_exports_accessible(self):
        """Test all __all__ exports are accessible."""
        from scitex.capture import utils

        for name in utils.__all__:
            assert hasattr(utils, name)

    def test_capture_is_exported(self):
        """Test capture function is exported."""
        from scitex.capture.utils import capture

        assert callable(capture)

    def test_take_screenshot_is_exported(self):
        """Test take_screenshot function is exported."""
        from scitex.capture.utils import take_screenshot

        assert callable(take_screenshot)

    def test_start_stop_monitor_exported(self):
        """Test start_monitor and stop_monitor are exported."""
        from scitex.capture.utils import start_monitor, stop_monitor

        assert callable(start_monitor)
        assert callable(stop_monitor)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
