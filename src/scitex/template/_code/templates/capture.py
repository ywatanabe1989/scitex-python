#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/template/_templates/capture.py
# ----------------------------------------

"""Template for stx.capture screenshot module usage."""

TEMPLATE = {
    "name": "Capture Module",
    "description": "stx.capture usage for screenshots, monitoring, and screen recording",
    "filename": "capture_script.py",
    "priority": 6,
    "content": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: {timestamp}
# File: {filepath}

"""
stx.capture - Screenshot & Screen Recording
===========================================

stx.capture provides:
- Single and multi-monitor screenshots
- Window-specific capture
- Continuous monitoring with timestamps
- Animated GIF creation from sequences
- Error detection in screenshots

Usage Patterns
--------------
1. MCP: capture_* tools for Claude Code integration
2. Python API: stx.capture.* functions
3. Useful for: debugging, documentation, testing
"""

import scitex as stx

# ============================================================
# Pattern 1: Basic Screenshots (via MCP)
# ============================================================

"""
MCP Tools for Claude Code:
--------------------------

# Capture current monitor
capture_capture_screenshot()

# Capture specific monitor
capture_capture_screenshot(monitor_id=1)

# Capture all monitors
capture_capture_screenshot(all=True)

# Capture specific window by app name
capture_capture_screenshot(app="Firefox")

# Get screenshot as base64 (for analysis)
capture_capture_screenshot(return_base64=True)

# Capture with message annotation
capture_capture_screenshot(message="Before fix")
"""

# ============================================================
# Pattern 2: Window Capture
# ============================================================

"""
Window-Specific Capture:
------------------------

# List all visible windows
capture_list_windows()

# Capture window by handle
capture_capture_window(window_handle=12345)

# Get system info (monitors, desktops)
capture_get_info()
"""

# ============================================================
# Pattern 3: Continuous Monitoring
# ============================================================

"""
Screen Monitoring:
------------------

# Start continuous capture (1 screenshot/second)
capture_start_monitoring(interval=1.0, monitor_id=0)

# Start with all monitors
capture_start_monitoring(capture_all=True, interval=0.5)

# Get monitoring status
capture_get_monitoring_status()

# Stop monitoring
capture_stop_monitoring()

# List monitoring sessions
capture_list_sessions(limit=10)
"""

# ============================================================
# Pattern 4: GIF Creation
# ============================================================

"""
Animated GIF Creation:
----------------------

# Create GIF from monitoring session
capture_create_gif(session_id="xxx", duration=0.5)

# Create GIF from specific images
capture_create_gif(
    image_paths=["/path/1.png", "/path/2.png"],
    output_path="animation.gif",
    duration=0.3
)

# Create GIF with frame limit
capture_create_gif(
    session_id="xxx",
    max_frames=50,
    optimize=True
)
"""

# ============================================================
# Pattern 5: Python API Usage
# ============================================================

def python_api_example():
    """Direct Python API for screenshots."""
    from scitex.capture import (
        capture_screenshot,
        start_monitoring,
        stop_monitoring,
        create_gif,
        list_windows,
    )

    # Take a screenshot
    result = capture_screenshot(
        monitor_id=0,
        quality=85
    )
    print(f"Saved to: {{result['path']}}")

    # List windows
    windows = list_windows()
    for win in windows:
        print(f"Window: {{win['title']}} (handle: {{win['handle']}})")

    # Start monitoring
    start_monitoring(interval=1.0)

    # ... do something ...

    # Stop and create GIF
    stop_monitoring()
    create_gif(duration=0.5, output_path="recording.gif")

# ============================================================
# Pattern 6: With @stx.session (Documentation)
# ============================================================

@stx.session
def main(
    interval=2.0,
    duration=10,
    CONFIG=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Capture session for documentation."""
    from scitex.capture import (
        capture_screenshot,
        start_monitoring,
        stop_monitoring,
        create_gif,
    )
    import time

    # Take initial screenshot
    result = capture_screenshot(
        message="Initial state"
    )
    stx.io.save(result, CONFIG.SDIR_OUT / "initial.json")

    # Start monitoring
    logger.info(f"Starting {{duration}}s monitoring...")
    start_monitoring(
        interval=interval,
        output_dir=str(CONFIG.SDIR_OUT / "frames")
    )

    time.sleep(duration)

    stop_monitoring()

    # Create GIF
    logger.info("Creating GIF...")
    create_gif(
        output_path=str(CONFIG.SDIR_OUT / "recording.gif"),
        duration=0.5
    )

    logger.info(f"Recording saved to {{CONFIG.SDIR_OUT}}")
    return 0

# ============================================================
# Pattern 7: Error Analysis
# ============================================================

"""
Screenshot Analysis:
--------------------

# Analyze screenshot for errors
capture_analyze_screenshot(path="/path/to/screenshot.png")

# Returns error indicators like:
# - Red error dialogs
# - Error text patterns
# - Warning indicators
"""

# ============================================================
# MCP Tools Reference
# ============================================================

"""
MCP Tools:
----------

Screenshots:
- capture_capture_screenshot(monitor_id, all, app, url, quality, return_base64)
- capture_capture_window(window_handle, output_path)
- capture_list_windows()
- capture_get_info()

Monitoring:
- capture_start_monitoring(interval, monitor_id, capture_all)
- capture_stop_monitoring()
- capture_get_monitoring_status()
- capture_list_sessions(limit)

Processing:
- capture_create_gif(session_id, image_paths, duration, max_frames)
- capture_analyze_screenshot(path)

Cache:
- capture_list_recent_screenshots(limit, category)
- capture_clear_cache(max_size_gb)
"""

if __name__ == "__main__":
    main()
''',
}

__all__ = ["TEMPLATE"]

# EOF
