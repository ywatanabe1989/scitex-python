#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scitex.capture - AI's Camera
A lightweight, intuitive screen capture library optimized for WSL and Windows.

Features:
- Windows host screen capture from WSL
- Multi-monitor support
- JPEG compression for smaller file sizes
- Continuous monitoring with configurable intervals
- Human-readable timestamps
- Thread-safe operation

Usage:
    from scitex import capture

    # Single screenshot
    capture.snap("debug message")

    # Multi-monitor
    capture.snap(capture_all=True)

    # Continuous monitoring
    capture.start()
    # ... do work ...
    capture.stop()
"""

from .utils import capture, start_monitor, stop_monitor
from .gif import (
    create_gif_from_session,
    create_gif_from_files,
    create_gif_from_pattern,
    create_gif_from_latest_session,
)
from .session import session
from .capture import CaptureManager

# Global manager for monitor enumeration
_manager = CaptureManager()


def get_info():
    """Get comprehensive display info (monitors, windows, virtual desktops)."""
    return _manager.get_info()


# Simpler, clearer aliases
get_info = get_info  # Primary: simple and clear
list_windows = get_info  # Alternative: focus on windows
get_display_info = get_info  # Legacy


def capture_window(window_handle: int, output_path: str = None):
    """
    Capture a specific window by its handle.

    Args:
        window_handle: Window handle from get_info()['Windows']['Details']
        output_path: Optional path to save screenshot

    Returns:
        Path to saved screenshot

    Examples:
        >>> from scitex import capture
        >>> info = capture.get_info()
        >>> windows = info['Windows']['Details']
        >>> if windows:
        >>>     handle = windows[0]['Handle']
        >>>     path = capture.capture_window(handle)
    """
    return _manager.capture_window(window_handle, output_path)


# Convenience aliases - these are the main public API
snap = capture  # Primary: natural camera action
take = capture  # Alternative: "take a picture"
cpt = capture  # Legacy: backwards compatibility
start = start_monitor
stop = stop_monitor

# GIF creation aliases
gif = create_gif_from_latest_session  # Primary: simple
make_gif = create_gif_from_latest_session  # Alternative

__version__ = "0.2.1"
__author__ = "Yusuke Watanabe"
__email__ = "Yusuke.Watanabe@scitex.ai"

# Only expose the essential functions
__all__ = [
    "capture",
    "snap",  # Primary API
    "take",  # Alternative API
    "cpt",  # Legacy
    "start",
    "stop",
    "session",
    "get_info",  # Primary: get all display info
    "list_windows",  # Alternative: focus on windows
    "get_info",  # Legacy
    "get_display_info",  # Legacy
    "capture_window",
    "create_gif_from_session",
    "create_gif_from_files",
    "create_gif_from_pattern",
    "create_gif_from_latest_session",
]
