#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-18 09:55:53 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/capture/session.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/capture/session.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------


class Session:
    """Context manager for CAM session with automatic start/stop."""

    def __init__(
        self,
        output_dir: str = "~/.scitex/capture/",
        interval: float = 1.0,
        jpeg: bool = True,
        quality: int = 60,
        on_capture=None,
        on_error=None,
        verbose: bool = True,
        monitor_id: int = 0,
        capture_all: bool = False,
    ):
        """Initialize session parameters."""
        self.output_dir = output_dir
        self.interval = interval
        self.jpeg = jpeg
        self.quality = quality
        self.on_capture = on_capture
        self.on_error = on_error
        self.verbose = verbose
        self.monitor_id = monitor_id
        self.capture_all = capture_all
        self.worker = None

    def __enter__(self):
        """Start monitoring when entering context."""
        from .utils import start_monitor

        self.worker = start_monitor(
            output_dir=self.output_dir,
            interval=self.interval,
            jpeg=self.jpeg,
            quality=self.quality,
            on_capture=self.on_capture,
            on_error=self.on_error,
            verbose=self.verbose,
            monitor_id=self.monitor_id,
            capture_all=self.capture_all,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring when exiting context."""
        from .utils import stop_monitor

        stop_monitor()
        return False


def session(**kwargs):
    """Create a new session context manager."""
    return Session(**kwargs)


# EOF
