#!/usr/bin/env python3
"""MCP tool handlers for SciTeX Capture."""

import asyncio
import base64
from datetime import datetime
from pathlib import Path

from scitex import capture

from .mcp_utils import get_capture_dir


class CaptureHandlers:
    """Handlers for capture-related tools."""

    @staticmethod
    async def capture_screenshot(
        message=None,
        monitor_id=0,
        all=False,
        app=None,
        url=None,
        quality=85,
        return_base64=False,
        grid_overlay=False,
        cursor_overlay=False,
        grid_spacing=100,
    ):
        """Capture screenshot with optional overlays."""
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
                return {"success": False, "error": "Failed to capture"}

            # Apply overlays if requested
            if grid_overlay or cursor_overlay:
                from .grid import draw_cursor_overlay, draw_grid_overlay

                # Determine capture mode for coordinate mapping
                capture_mode = "all" if all else str(monitor_id)

                if grid_overlay:
                    path = draw_grid_overlay(path, grid_spacing=grid_spacing)
                if cursor_overlay:
                    path = draw_cursor_overlay(
                        path, output_path=path, capture_mode=capture_mode
                    )

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
    async def capture_window(window_handle: int, output_path=None, quality=85):
        """Capture window by handle."""
        try:
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


class MonitoringHandlers:
    """Handlers for monitoring tools."""

    def __init__(self):
        self.monitoring_active = False
        self.monitoring_worker = None

    async def start_monitoring(
        self,
        interval=1.0,
        monitor_id=0,
        capture_all=False,
        output_dir=None,
        quality=60,
        verbose=True,
    ):
        """Start continuous monitoring."""
        if self.monitoring_active:
            return {"success": False, "message": "Already active"}

        try:
            loop = asyncio.get_event_loop()

            def start():
                return capture.start_monitor(
                    output_dir=output_dir or "~/.scitex/capture/",
                    interval=interval,
                    jpeg=True,
                    quality=quality,
                    verbose=verbose,
                    monitor_id=monitor_id,
                    capture_all=capture_all,
                )

            self.monitoring_worker = await loop.run_in_executor(None, start)
            self.monitoring_active = True

            return {
                "success": True,
                "message": f"Started monitoring with {interval}s interval",
                "interval": interval,
                "monitor_id": monitor_id,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def stop_monitoring(self):
        """Stop monitoring."""
        if not self.monitoring_active:
            return {"success": False, "message": "Not active"}

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, capture.stop)

            stats = {}
            if self.monitoring_worker:
                stats = {
                    "screenshots_taken": self.monitoring_worker.screenshot_count,
                    "session_id": self.monitoring_worker.session_id,
                }

            self.monitoring_active = False
            self.monitoring_worker = None

            return {"success": True, "message": "Stopped", **stats}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_status(self):
        """Get monitoring status."""
        status = {"active": self.monitoring_active, "cache_dir": str(get_capture_dir())}

        if self.monitoring_active and self.monitoring_worker:
            status.update(
                {
                    "screenshots_taken": self.monitoring_worker.screenshot_count,
                    "session_id": self.monitoring_worker.session_id,
                }
            )

        cache_dir = get_capture_dir()
        if cache_dir.exists():
            total_size = sum(f.stat().st_size for f in cache_dir.glob("*.jpg"))
            status["cache_size_mb"] = round(total_size / (1024 * 1024), 2)
            status["screenshot_count"] = len(list(cache_dir.glob("*.jpg")))

        return status


class UtilityHandlers:
    """Handlers for utility tools."""

    @staticmethod
    async def analyze_screenshot(path: str):
        """Analyze screenshot."""
        try:
            from .utils import _detect_category

            loop = asyncio.get_event_loop()
            category = await loop.run_in_executor(None, _detect_category, path)

            path_obj = Path(path)
            if not path_obj.exists():
                return {"success": False, "error": f"Not found: {path}"}

            return {
                "success": True,
                "path": path,
                "category": category,
                "is_error": category == "stderr",
                "size_kb": round(path_obj.stat().st_size / 1024, 2),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def list_recent_screenshots(limit=10, category="all"):
        """List recent screenshots."""
        try:
            cache_dir = get_capture_dir()
            if not cache_dir.exists():
                return {"success": True, "screenshots": []}

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

    @staticmethod
    async def clear_cache(max_size_gb=1.0, clear_all=False):
        """Clear cache."""
        try:
            cache_dir = get_capture_dir()
            if not cache_dir.exists():
                return {"success": True, "message": "Cache does not exist"}

            if clear_all:
                removed = sum(1 for s in cache_dir.glob("*.jpg") if s.unlink() or True)
                return {"success": True, "removed_count": removed}
            else:
                from .utils import _manage_cache_size

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, _manage_cache_size, cache_dir, max_size_gb
                )
                return {"success": True, "max_size_gb": max_size_gb}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_info():
        """Get system info."""
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, capture.get_info)
            return {
                "success": True,
                "monitors": info.get("Monitors", {}),
                "windows": info.get("Windows", {}),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def list_windows():
        """List windows."""
        try:
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


class GifHandlers:
    """Handlers for GIF tools."""

    @staticmethod
    async def create_gif(
        session_id=None,
        image_paths=None,
        pattern=None,
        output_path=None,
        duration=0.5,
        optimize=True,
        max_frames=None,
    ):
        """Create GIF from screenshots."""
        try:
            from .gif import GifCreator

            creator = GifCreator()
            loop = asyncio.get_event_loop()

            if session_id:
                if session_id == "latest":
                    result = await loop.run_in_executor(
                        None,
                        creator.create_gif_from_recent_session,
                        "~/.scitex/capture",
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
                        "~/.scitex/capture",
                        duration,
                        optimize,
                        max_frames,
                    )
            elif image_paths:
                if not output_path:
                    output_path = (
                        f"~/.scitex/capture/gif_{datetime.now():%Y%m%d_%H%M%S}.gif"
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
            return {"success": False, "error": "No images found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def list_sessions(limit=10):
        """List sessions."""
        try:
            from .gif import GifCreator

            creator = GifCreator()
            loop = asyncio.get_event_loop()
            sessions = await loop.run_in_executor(
                None, creator.get_recent_sessions, "~/.scitex/capture"
            )
            return {
                "success": True,
                "sessions": sessions[:limit],
                "count": min(len(sessions), limit),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
