#!/usr/bin/env python3
# Time-stamp: "2025-08-01 12:15:00"
# Author: Yusuke Watanabe
# File: test_screenshot_capturer.py

"""Tests for screenshot capture functionality."""

import asyncio
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")

from scitex.scholar.utils import ScreenshotCapturer


class MockPage:
    """Mock Playwright page for testing."""

    def __init__(self, url="https://example.com", title="Example Page"):
        self.url = url
        self._title = title
        self.viewport_size = {"width": 1920, "height": 1080}
        self._screenshot_count = 0

    async def title(self):
        return self._title

    async def screenshot(self, path: str, full_page: bool = True, timeout: int = 10000):
        """Mock screenshot capture."""
        # Create a dummy file to simulate screenshot
        Path(path).touch()
        self._screenshot_count += 1

    async def evaluate(self, script: str):
        """Mock JavaScript evaluation."""
        if "contentType" in script:
            return "text/html"
        return None

    async def add_style_tag(self, content: str):
        """Mock style injection."""
        pass


@pytest.mark.asyncio
async def test_screenshot_capturer_init():
    """Test ScreenshotCapturer initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        capturer = ScreenshotCapturer(Path(tmpdir) / "screenshots")
        assert capturer.screenshot_dir.exists()
        assert capturer.screenshot_dir.is_dir()


@pytest.mark.asyncio
async def test_capture_on_failure():
    """Test screenshot capture on failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        capturer = ScreenshotCapturer(Path(tmpdir))
        page = MockPage()

        error_info = {
            "error": "PDF not found",
            "status_code": 404,
            "timestamp": datetime.now().isoformat()
        }

        screenshot_path = await capturer.capture_on_failure(
            page, error_info, "10.1234/test.doi"
        )

        assert screenshot_path is not None
        assert screenshot_path.exists()
        assert screenshot_path.suffix == ".png"

        # Check info file was created
        info_file = screenshot_path.with_suffix(".txt")
        assert info_file.exists()
        assert "PDF not found" in info_file.read_text()


@pytest.mark.asyncio
async def test_capture_workflow():
    """Test workflow stage screenshot capture."""
    with tempfile.TemporaryDirectory() as tmpdir:
        capturer = ScreenshotCapturer(Path(tmpdir))
        page = MockPage()

        screenshot_path = await capturer.capture_workflow(
            page,
            "pre_auth",
            "10.5678/workflow.test",
            {"session": "active", "cookies": 5}
        )

        assert screenshot_path is not None
        assert "pre_auth" in screenshot_path.name
        assert screenshot_path.exists()

        # Check additional info was saved
        info_file = screenshot_path.with_suffix(".txt")
        assert info_file.exists()
        assert "pre_auth Stage Information" in info_file.read_text()
        assert "session: active" in info_file.read_text()


@pytest.mark.asyncio
async def test_capture_comparison():
    """Test comparison screenshot with highlighting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        capturer = ScreenshotCapturer(Path(tmpdir))
        page = MockPage()

        screenshot_path = await capturer.capture_comparison(
            page,
            "a.pdf-link",
            "10.9999/compare.test"
        )

        assert screenshot_path is not None
        assert "comparison" in screenshot_path.name
        assert screenshot_path.exists()


def test_cleanup_old_screenshots():
    """Test cleanup of old screenshots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        capturer = ScreenshotCapturer(Path(tmpdir))

        # Create some test files
        old_file = capturer.screenshot_dir / "old_screenshot.png"
        old_file.touch()
        old_txt = capturer.screenshot_dir / "old_screenshot.txt"
        old_txt.touch()

        new_file = capturer.screenshot_dir / "new_screenshot.png"
        new_file.touch()

        # Make old files older than 7 days
        old_time = datetime.now().timestamp() - (8 * 24 * 60 * 60)
        import os
        os.utime(old_file, (old_time, old_time))
        os.utime(old_txt, (old_time, old_time))

        # Run cleanup
        deleted = capturer.cleanup_old_screenshots(days=7)

        assert deleted == 1
        assert not old_file.exists()
        assert not old_txt.exists()
        assert new_file.exists()


@pytest.mark.asyncio
async def test_screenshot_with_special_characters():
    """Test screenshot naming with special characters in identifier."""
    with tempfile.TemporaryDirectory() as tmpdir:
        capturer = ScreenshotCapturer(Path(tmpdir))
        page = MockPage()

        # DOI with slashes and colons
        doi = "10.1038/s41586-020-2832-5:supplementary"

        screenshot_path = await capturer.capture_on_failure(
            page, {"error": "test"}, doi
        )

        assert screenshot_path is not None
        assert "/" not in screenshot_path.name
        assert ":" not in screenshot_path.name


@pytest.mark.asyncio
async def test_screenshot_capture_error_handling():
    """Test error handling during screenshot capture."""
    with tempfile.TemporaryDirectory() as tmpdir:
        capturer = ScreenshotCapturer(Path(tmpdir))

        # Create a page that raises an error
        class ErrorPage(MockPage):
            async def screenshot(self, *args, **kwargs):
                raise Exception("Screenshot failed")

        page = ErrorPage()

        # Should return None on error
        screenshot_path = await capturer.capture_on_failure(
            page, {"error": "test"}, "error.test"
        )

        assert screenshot_path is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
