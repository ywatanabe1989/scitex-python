#!/usr/bin/env python3
# Timestamp: 2026-01-15
# File: src/scitex/capture/_mcp/handlers.py
# ----------------------------------------

"""MCP tool handlers for scitex-capture."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime
from pathlib import Path

__all__ = [
    "capture_screenshot_handler",
    "start_monitoring_handler",
    "stop_monitoring_handler",
    "get_monitoring_status_handler",
    "analyze_screenshot_handler",
    "list_recent_screenshots_handler",
    "clear_cache_handler",
    "create_gif_handler",
    "list_sessions_handler",
    "get_info_handler",
    "list_windows_handler",
    "capture_window_handler",
]


def _get_capture_dir() -> Path:
    """Get the capture output directory."""
    import os

    base_dir = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
    capture_dir = base_dir / "capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    return capture_dir


# Monitoring state (module-level singleton)
_monitoring_active = False
_monitoring_worker = None


async def capture_screenshot_handler(
    message: str | None = None,
    monitor_id: int = 0,
    all: bool = False,
    app: str | None = None,
    url: str | None = None,
    quality: int = 85,
    return_base64: bool = False,
) -> dict:
    """Capture screenshot with optional overlays."""
    try:
        from scitex import capture

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


async def start_monitoring_handler(
    interval: float = 1.0,
    monitor_id: int = 0,
    capture_all: bool = False,
    output_dir: str | None = None,
    quality: int = 60,
    verbose: bool = True,
) -> dict:
    """Start continuous screenshot monitoring."""
    global _monitoring_active, _monitoring_worker

    if _monitoring_active:
        return {"success": False, "message": "Monitoring already active"}

    try:
        from scitex import capture

        loop = asyncio.get_event_loop()

        def start():
            return capture.start_monitor(
                output_dir=output_dir or str(_get_capture_dir()),
                interval=interval,
                jpeg=True,
                quality=quality,
                verbose=verbose,
                monitor_id=monitor_id,
                capture_all=capture_all,
            )

        _monitoring_worker = await loop.run_in_executor(None, start)
        _monitoring_active = True

        return {
            "success": True,
            "message": f"Started monitoring with {interval}s interval",
            "interval": interval,
            "monitor_id": monitor_id,
            "capture_all": capture_all,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def stop_monitoring_handler() -> dict:
    """Stop continuous screenshot monitoring."""
    global _monitoring_active, _monitoring_worker

    if not _monitoring_active:
        return {"success": False, "message": "Monitoring not active"}

    try:
        from scitex import capture

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, capture.stop)

        stats = {}
        if _monitoring_worker:
            stats = {
                "screenshots_taken": getattr(_monitoring_worker, "screenshot_count", 0),
                "session_id": getattr(_monitoring_worker, "session_id", None),
            }

        _monitoring_active = False
        _monitoring_worker = None

        return {"success": True, "message": "Monitoring stopped", **stats}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_monitoring_status_handler() -> dict:
    """Get current monitoring status and statistics."""
    global _monitoring_active, _monitoring_worker

    status = {
        "success": True,
        "active": _monitoring_active,
        "cache_dir": str(_get_capture_dir()),
    }

    if _monitoring_active and _monitoring_worker:
        status.update(
            {
                "screenshots_taken": getattr(_monitoring_worker, "screenshot_count", 0),
                "session_id": getattr(_monitoring_worker, "session_id", None),
            }
        )

    cache_dir = _get_capture_dir()
    if cache_dir.exists():
        jpg_files = list(cache_dir.glob("*.jpg"))
        total_size = sum(f.stat().st_size for f in jpg_files)
        status["cache_size_mb"] = round(total_size / (1024 * 1024), 2)
        status["screenshot_count"] = len(jpg_files)

    return status


async def analyze_screenshot_handler(path: str) -> dict:
    """Analyze screenshot for error indicators."""
    try:
        from ..utils import _detect_category

        loop = asyncio.get_event_loop()
        category = await loop.run_in_executor(None, _detect_category, path)

        path_obj = Path(path)
        if not path_obj.exists():
            return {"success": False, "error": f"File not found: {path}"}

        return {
            "success": True,
            "path": path,
            "category": category,
            "is_error": category == "stderr",
            "size_kb": round(path_obj.stat().st_size / 1024, 2),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def list_recent_screenshots_handler(
    limit: int = 10,
    category: str = "all",
) -> dict:
    """List recent screenshots from cache."""
    try:
        cache_dir = _get_capture_dir()
        if not cache_dir.exists():
            return {"success": True, "screenshots": [], "count": 0}

        screenshots = list(cache_dir.glob("*.jpg"))

        if category == "stdout":
            screenshots = [s for s in screenshots if "-stdout.jpg" in s.name]
        elif category == "stderr":
            screenshots = [s for s in screenshots if "-stderr.jpg" in s.name]

        screenshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        screenshots = screenshots[:limit]

        result_list = []
        for s in screenshots:
            cat = "stderr" if "-stderr.jpg" in s.name else "stdout"
            result_list.append(
                {
                    "filename": s.name,
                    "path": str(s),
                    "category": cat,
                    "size_kb": round(s.stat().st_size / 1024, 2),
                }
            )

        return {
            "success": True,
            "screenshots": result_list,
            "count": len(result_list),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def clear_cache_handler(
    max_size_gb: float = 1.0,
    clear_all: bool = False,
) -> dict:
    """Clear screenshot cache or manage cache size."""
    try:
        cache_dir = _get_capture_dir()
        if not cache_dir.exists():
            return {"success": True, "message": "Cache does not exist"}

        if clear_all:
            removed = 0
            for s in cache_dir.glob("*.jpg"):
                s.unlink()
                removed += 1
            return {"success": True, "removed_count": removed}
        else:
            from ..utils import _manage_cache_size

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _manage_cache_size, cache_dir, max_size_gb)
            return {"success": True, "max_size_gb": max_size_gb}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def create_gif_handler(
    session_id: str | None = None,
    image_paths: list[str] | None = None,
    pattern: str | None = None,
    output_path: str | None = None,
    duration: float = 0.5,
    optimize: bool = True,
    max_frames: int | None = None,
) -> dict:
    """Create an animated GIF from screenshots."""
    try:
        from ..gif import GifCreator

        creator = GifCreator()
        loop = asyncio.get_event_loop()
        capture_dir = str(_get_capture_dir())

        if session_id:
            if session_id == "latest":
                result = await loop.run_in_executor(
                    None,
                    creator.create_gif_from_recent_session,
                    capture_dir,
                    duration,
                    optimize,
                    max_frames,
                )
            else:
                result = await loop.run_in_executor(
                    None,
                    creator.create_gif_from_session,
                    session_id,
                    output_path,
                    capture_dir,
                    duration,
                    optimize,
                    max_frames,
                )
        elif image_paths:
            if not output_path:
                output_path = str(
                    _get_capture_dir() / f"gif_{datetime.now():%Y%m%d_%H%M%S}.gif"
                )
            result = await loop.run_in_executor(
                None,
                creator.create_gif_from_files,
                image_paths,
                output_path,
                duration,
                optimize,
            )
        elif pattern:
            result = await loop.run_in_executor(
                None,
                creator.create_gif_from_pattern,
                pattern,
                output_path,
                duration,
                optimize,
                max_frames,
            )
        else:
            return {
                "success": False,
                "error": "Specify session_id, image_paths, or pattern",
            }

        if result:
            return {"success": True, "path": result, "duration": duration}
        return {"success": False, "error": "No images found to create GIF"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def list_sessions_handler(limit: int = 10) -> dict:
    """List available monitoring sessions."""
    try:
        from ..gif import GifCreator

        creator = GifCreator()
        loop = asyncio.get_event_loop()
        sessions = await loop.run_in_executor(
            None, creator.get_recent_sessions, str(_get_capture_dir())
        )
        return {
            "success": True,
            "sessions": sessions[:limit],
            "count": min(len(sessions), limit),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_info_handler() -> dict:
    """Get monitor and window information."""
    try:
        from scitex import capture

        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, capture.get_info)
        return {
            "success": True,
            "monitors": info.get("Monitors", {}),
            "windows": info.get("Windows", {}),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def list_windows_handler() -> dict:
    """List all visible windows."""
    try:
        from scitex import capture

        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, capture.get_info)
        windows = info.get("Windows", {}).get("Details", [])
        formatted = [
            {
                "handle": w.get("Handle"),
                "title": w.get("Title"),
                "process": w.get("ProcessName"),
            }
            for w in windows
        ]
        return {"success": True, "windows": formatted, "count": len(formatted)}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def capture_window_handler(
    window_handle: int,
    output_path: str | None = None,
    quality: int = 85,
) -> dict:
    """Capture a specific window by handle."""
    try:
        from scitex import capture

        loop = asyncio.get_event_loop()
        path = await loop.run_in_executor(
            None, capture.capture_window, window_handle, output_path
        )
        if path:
            return {"success": True, "path": path, "window_handle": window_handle}
        return {
            "success": False,
            "error": f"Failed to capture window {window_handle}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# EOF
