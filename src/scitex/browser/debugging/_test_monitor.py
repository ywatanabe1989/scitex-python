#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-09
# File: /home/ywatanabe/proj/scitex-code/src/scitex/browser/debugging/_test_monitor.py

"""
Test monitoring with periodic screenshots using scitex.capture.

Provides automated visual monitoring during E2E tests:
- Periodic screenshots at configurable intervals
- Integration with scitex.capture for WSL/Windows support
- GIF generation from test sessions
- Pytest fixture integration

Usage in conftest.py:
    from scitex.browser.debugging import (
        create_test_monitor,
        TestMonitor,
    )

    @pytest.fixture
    def test_monitor():
        monitor = TestMonitor(interval=2.0, verbose=True)
        monitor.start()
        yield monitor
        monitor.stop()
        monitor.create_gif()  # Optional: create GIF from screenshots
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from scitex.config import get_paths

if TYPE_CHECKING:
    from playwright.sync_api import Page


class TestMonitor:
    """
    Monitor E2E tests with periodic screenshots using scitex.capture.

    Captures screenshots at regular intervals during test execution,
    allowing visual inspection of test progress and debugging.
    """

    def __init__(
        self,
        output_dir: str | Path = None,
        interval: float = 2.0,
        quality: int = 70,
        verbose: bool = False,
        test_name: str = None,
    ):
        """
        Initialize test monitor.

        Args:
            output_dir: Directory for screenshots (default: $SCITEX_DIR/test_monitor)
            interval: Seconds between screenshots (default: 2.0)
            quality: JPEG quality 1-100 (default: 70)
            verbose: Print capture messages
            test_name: Optional test name for session identification
        """
        self.output_dir = get_paths().resolve("test_monitor", output_dir)
        self.interval = interval
        self.quality = quality
        self.verbose = verbose
        self.test_name = test_name
        self.session_id = None
        self._worker = None
        self._capture_manager = None

    def start(self, test_name: str = None) -> str:
        """
        Start periodic screenshot capture.

        Args:
            test_name: Optional test name to include in session

        Returns:
            Session ID for this capture session
        """
        try:
            from scitex.capture import CaptureManager
        except ImportError:
            if self.verbose:
                print("[TestMonitor] scitex.capture not available")
            return None

        # Generate session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = test_name or self.test_name or "test"
        safe_name = (
            name.replace("::", "_").replace("[", "_").replace("]", "").replace("/", "_")
        )
        self.session_id = f"{timestamp}_{safe_name}"

        # Create session directory
        session_dir = self.output_dir / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Start capture
        self._capture_manager = CaptureManager()
        self._worker = self._capture_manager.start_capture(
            output_dir=str(session_dir),
            interval=self.interval,
            jpeg=True,
            quality=self.quality,
            verbose=self.verbose,
        )

        if self.verbose:
            print(f"[TestMonitor] Started: {session_dir} (interval: {self.interval}s)")

        return self.session_id

    def stop(self) -> dict:
        """
        Stop screenshot capture.

        Returns:
            Status dict with session info
        """
        if self._capture_manager:
            self._capture_manager.stop_capture()

        status = self.get_status()

        if self.verbose and self._worker:
            print(f"[TestMonitor] Stopped: {self._worker.screenshot_count} screenshots")

        return status

    def get_status(self) -> dict:
        """Get current monitor status."""
        if self._worker:
            return self._worker.get_status()
        return {
            "running": False,
            "session_id": self.session_id,
            "output_dir": str(self.output_dir),
        }

    def take_snapshot(self, message: str = None) -> Optional[str]:
        """
        Take an immediate snapshot (in addition to periodic captures).

        Args:
            message: Optional message to include in filename

        Returns:
            Path to saved screenshot
        """
        try:
            from scitex.capture import snap

            return snap(message=message, output_dir=str(self.output_dir))
        except ImportError:
            return None

    def create_gif(
        self, duration: float = 0.5, output_path: str = None
    ) -> Optional[str]:
        """
        Create GIF from captured screenshots.

        Args:
            duration: Duration per frame in seconds
            output_path: Output path for GIF (auto-generated if None)

        Returns:
            Path to created GIF
        """
        if not self.session_id:
            return None

        try:
            from scitex.capture import create_gif_from_session

            session_dir = self.output_dir / self.session_id
            if output_path is None:
                output_path = str(session_dir / f"{self.session_id}.gif")

            return create_gif_from_session(
                session_id=self.session_id,
                output_path=output_path,
                duration=duration,
            )
        except ImportError:
            if self.verbose:
                print("[TestMonitor] GIF creation requires scitex.capture")
            return None

    def get_screenshots(self) -> list:
        """Get list of captured screenshot paths."""
        if not self.session_id:
            return []

        session_dir = self.output_dir / self.session_id
        if not session_dir.exists():
            return []

        return sorted(session_dir.glob("*.jpg")) + sorted(session_dir.glob("*.png"))


def create_test_monitor_fixture(
    output_dir: str | Path = None,
    interval: float = 2.0,
    auto_gif: bool = False,
):
    """
    Create a pytest fixture for test monitoring.

    Usage in conftest.py:
        from scitex.browser.debugging import create_test_monitor_fixture

        test_monitor = create_test_monitor_fixture(interval=2.0, auto_gif=True)

    Args:
        output_dir: Directory for screenshots
        interval: Seconds between screenshots
        auto_gif: Create GIF automatically on test completion

    Returns:
        A pytest fixture function
    """
    import pytest

    @pytest.fixture
    def test_monitor(request):
        """Pytest fixture for visual test monitoring."""
        monitor = TestMonitor(
            output_dir=output_dir,
            interval=interval,
            verbose=True,
            test_name=request.node.nodeid,
        )
        monitor.start()
        yield monitor
        monitor.stop()
        if auto_gif:
            gif_path = monitor.create_gif()
            if gif_path:
                print(f"[TestMonitor] GIF created: {gif_path}")

    return test_monitor


# Convenience function for quick monitoring
def monitor_test(
    test_func=None,
    interval: float = 2.0,
    auto_gif: bool = False,
):
    """
    Decorator for monitoring tests with periodic screenshots.

    Usage:
        @monitor_test(interval=1.0, auto_gif=True)
        def test_my_feature(page):
            # test code...

    Args:
        test_func: Test function (for use without parentheses)
        interval: Seconds between screenshots
        auto_gif: Create GIF on completion
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = TestMonitor(
                interval=interval,
                verbose=True,
                test_name=func.__name__,
            )
            monitor.start()
            try:
                return func(*args, **kwargs)
            finally:
                monitor.stop()
                if auto_gif:
                    monitor.create_gif()

        return wrapper

    if test_func is not None:
        return decorator(test_func)
    return decorator


# EOF
