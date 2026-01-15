#!/usr/bin/env python3
# Timestamp: 2026-01-15
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_tools/capture.py
"""Capture module tools for FastMCP unified server."""

from __future__ import annotations

import json
from typing import Optional


def _json(data: dict) -> str:
    return json.dumps(data, indent=2, default=str)


def register_capture_tools(mcp) -> None:
    """Register capture tools with FastMCP server."""

    @mcp.tool()
    async def capture_capture_screenshot(
        monitor_id: int = 0,
        all: bool = False,
        quality: int = 85,
        message: Optional[str] = None,
        return_base64: bool = False,
        url: Optional[str] = None,
        app: Optional[str] = None,
    ) -> str:
        """[capture] Capture screenshot - monitor, window, browser, or everything."""
        from scitex.capture._mcp.handlers import capture_screenshot_handler

        result = await capture_screenshot_handler(
            monitor_id=monitor_id,
            all=all,
            quality=quality,
            message=message,
            return_base64=return_base64,
            url=url,
            app=app,
        )
        return _json(result)

    @mcp.tool()
    async def capture_start_monitoring(
        interval: float = 1.0,
        quality: int = 60,
        monitor_id: int = 0,
        capture_all: bool = False,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> str:
        """[capture] Start continuous screenshot monitoring."""
        from scitex.capture._mcp.handlers import start_monitoring_handler

        result = await start_monitoring_handler(
            interval=interval,
            quality=quality,
            monitor_id=monitor_id,
            capture_all=capture_all,
            output_dir=output_dir,
            verbose=verbose,
        )
        return _json(result)

    @mcp.tool()
    async def capture_stop_monitoring() -> str:
        """[capture] Stop continuous screenshot monitoring."""
        from scitex.capture._mcp.handlers import stop_monitoring_handler

        result = await stop_monitoring_handler()
        return _json(result)

    @mcp.tool()
    async def capture_get_monitoring_status() -> str:
        """[capture] Get current monitoring status and statistics."""
        from scitex.capture._mcp.handlers import get_monitoring_status_handler

        result = await get_monitoring_status_handler()
        return _json(result)

    @mcp.tool()
    async def capture_analyze_screenshot(path: str) -> str:
        """[capture] Analyze a screenshot for error indicators."""
        from scitex.capture._mcp.handlers import analyze_screenshot_handler

        result = await analyze_screenshot_handler(path=path)
        return _json(result)

    @mcp.tool()
    async def capture_list_recent_screenshots(
        limit: int = 10, category: str = "all"
    ) -> str:
        """[capture] List recent screenshots from cache."""
        from scitex.capture._mcp.handlers import list_recent_screenshots_handler

        result = await list_recent_screenshots_handler(limit=limit, category=category)
        return _json(result)

    @mcp.tool()
    async def capture_clear_cache(
        clear_all: bool = False, max_size_gb: float = 1.0
    ) -> str:
        """[capture] Clear screenshot cache or manage cache size."""
        from scitex.capture._mcp.handlers import clear_cache_handler

        result = await clear_cache_handler(clear_all=clear_all, max_size_gb=max_size_gb)
        return _json(result)

    @mcp.tool()
    async def capture_create_gif(
        session_id: Optional[str] = None,
        image_paths: Optional[list] = None,
        pattern: Optional[str] = None,
        output_path: Optional[str] = None,
        duration: float = 0.5,
        max_frames: Optional[int] = None,
        optimize: bool = True,
    ) -> str:
        """[capture] Create an animated GIF from screenshots."""
        from scitex.capture._mcp.handlers import create_gif_handler

        result = await create_gif_handler(
            session_id=session_id,
            image_paths=image_paths,
            pattern=pattern,
            output_path=output_path,
            duration=duration,
            max_frames=max_frames,
            optimize=optimize,
        )
        return _json(result)

    @mcp.tool()
    async def capture_list_sessions(limit: int = 10) -> str:
        """[capture] List available monitoring sessions."""
        from scitex.capture._mcp.handlers import list_sessions_handler

        result = await list_sessions_handler(limit=limit)
        return _json(result)

    @mcp.tool()
    async def capture_get_info() -> str:
        """[capture] Enumerate monitors, desktops, and windows."""
        from scitex.capture._mcp.handlers import get_info_handler

        result = await get_info_handler()
        return _json(result)

    @mcp.tool()
    async def capture_list_windows() -> str:
        """[capture] List all visible windows with handles."""
        from scitex.capture._mcp.handlers import list_windows_handler

        result = await list_windows_handler()
        return _json(result)

    @mcp.tool()
    async def capture_capture_window(
        window_handle: int,
        output_path: Optional[str] = None,
        quality: int = 85,
    ) -> str:
        """[capture] Capture a specific window by its handle."""
        from scitex.capture._mcp.handlers import capture_window_handler

        result = await capture_window_handler(
            window_handle=window_handle,
            output_path=output_path,
            quality=quality,
        )
        return _json(result)


# EOF
