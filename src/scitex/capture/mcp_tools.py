#!/usr/bin/env python3
"""
MCP tool definitions for SciTeX Capture.

This module contains the tool handler implementations that are
registered with the MCP server.
"""

import asyncio
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from scitex import capture

from .grid import draw_cursor_overlay, draw_grid_overlay


def get_capture_dir() -> Path:
    """Get the screenshot capture directory."""
    import os
    import shutil

    SCITEX_BASE_DIR = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
    new_dir = SCITEX_BASE_DIR / "capture"
    old_dir = Path.home() / ".cache" / "cammy"

    new_dir.mkdir(parents=True, exist_ok=True)

    if old_dir.exists():
        new_screenshots = list(new_dir.glob("*.jpg"))
        if not new_screenshots or len(new_screenshots) == 0:
            try:
                for img in old_dir.glob("*.jpg"):
                    shutil.move(str(img), str(new_dir / img.name))
            except Exception:
                pass

    return new_dir


class OverlayTools:
    """Tools for adding overlays to screenshots."""

    @staticmethod
    async def add_cursor_overlay(
        image_path: str,
        cursor_x: Optional[int] = None,
        cursor_y: Optional[int] = None,
        output_path: Optional[str] = None,
        monitor_offset_y: int = 1080,
    ) -> Dict[str, Any]:
        """Add cursor position marker overlay to a screenshot."""
        try:
            loop = asyncio.get_event_loop()

            cursor_pos = None
            if cursor_x is not None and cursor_y is not None:
                cursor_pos = (cursor_x, cursor_y)

            result_path = await loop.run_in_executor(
                None,
                lambda: draw_cursor_overlay(
                    filepath=image_path,
                    cursor_pos=cursor_pos,
                    output_path=output_path,
                    monitor_offset_y=monitor_offset_y,
                ),
            )

            return {
                "success": True,
                "path": result_path,
                "message": f"Cursor overlay added to {result_path}",
                "cursor_position": cursor_pos,
                "monitor_offset_y": monitor_offset_y,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def add_grid_overlay(
        image_path: str,
        grid_spacing: int = 100,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add coordinate grid overlay to a screenshot."""
        try:
            loop = asyncio.get_event_loop()

            result_path = await loop.run_in_executor(
                None,
                lambda: draw_grid_overlay(
                    filepath=image_path,
                    grid_spacing=grid_spacing,
                    output_path=output_path,
                ),
            )

            return {
                "success": True,
                "path": result_path,
                "message": f"Grid overlay ({grid_spacing}px) added to {result_path}",
                "grid_spacing": grid_spacing,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


class CaptureTools:
    """Tools for screenshot capture operations."""

    @staticmethod
    async def capture_screenshot(
        message=None,
        monitor_id=0,
        all=False,
        app=None,
        url=None,
        quality=85,
        return_base64=False,
    ) -> Dict[str, Any]:
        """Capture a screenshot."""
        try:
            loop = asyncio.get_event_loop()

            def do_capture():
                return capture.snap(
                    message=message,
                    quality=quality,
                    monitor_id=monitor_id,
                    all=all,
                    app=app,
                    url=url,
                    verbose=True,
                )

            path = await loop.run_in_executor(None, do_capture)

            if not path:
                return {"success": False, "error": "Failed to capture screenshot"}

            category = "stderr" if "-stderr.jpg" in path else "stdout"

            result = {
                "success": True,
                "path": path,
                "category": category,
                "message": f"Screenshot saved to {path}",
                "timestamp": datetime.now().isoformat(),
            }

            if return_base64 and path:
                with open(path, "rb") as f:
                    result["base64"] = base64.b64encode(f.read()).decode()

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def capture_window_tool(
        window_handle: int, output_path: str = None, quality: int = 85
    ) -> Dict[str, Any]:
        """Capture a specific window by handle."""
        try:
            loop = asyncio.get_event_loop()
            path = await loop.run_in_executor(
                None, capture.capture_window, window_handle, output_path
            )

            if path:
                return {
                    "success": True,
                    "path": path,
                    "window_handle": window_handle,
                    "message": f"Window captured to {path}",
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to capture window {window_handle}",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}


class InfoTools:
    """Tools for getting system information."""

    @staticmethod
    async def get_info() -> Dict[str, Any]:
        """Enumerate all monitors and virtual desktops."""
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, capture.get_info)

            return {
                "success": True,
                "monitors": info.get("Monitors", {}),
                "virtual_desktops": info.get("VirtualDesktops", {}),
                "windows": info.get("Windows", {}),
                "timestamp": info.get("Timestamp", ""),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def list_windows() -> Dict[str, Any]:
        """List all visible windows."""
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, capture.get_info)

            windows = info.get("Windows", {})
            window_list = windows.get("Details", [])

            formatted_windows = []
            for win in window_list:
                formatted_windows.append(
                    {
                        "handle": win.get("Handle"),
                        "title": win.get("Title"),
                        "process_name": win.get("ProcessName"),
                        "process_id": win.get("ProcessId"),
                    }
                )

            return {
                "success": True,
                "windows": formatted_windows,
                "count": len(formatted_windows),
                "visible_count": windows.get("VisibleCount", 0),
                "message": f"Found {len(formatted_windows)} windows",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
