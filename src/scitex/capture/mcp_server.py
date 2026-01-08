#!/usr/bin/env python3
# Timestamp: "2025-10-17 03:24:58 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/capture/mcp_server.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = "./src/scitex/capture/mcp_server.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
MCP Server for SciTeX Capture - Screen Capture for Python
Provides screenshot capture capabilities via Model Context Protocol.
"""

import asyncio
import base64
from datetime import datetime
from pathlib import Path

# Graceful MCP dependency handling
try:
    import mcp.types as types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    types = None  # type: ignore
    Server = None  # type: ignore
    NotificationOptions = None  # type: ignore
    InitializationOptions = None  # type: ignore
    stdio_server = None  # type: ignore

# Directory configuration
import os
import shutil

from scitex import capture

# Use SCITEX_DIR environment variable if set, otherwise default to ~/.scitex
SCITEX_BASE_DIR = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
SCITEX_CAPTURE_DIR = SCITEX_BASE_DIR / "capture"
LEGACY_CAPTURE_DIR = Path.home() / ".cache" / "cammy"


def get_capture_dir() -> Path:
    """
    Get the screenshot capture directory.
    Uses $SCITEX_DIR/capture if SCITEX_DIR is set, otherwise ~/.scitex/capture.
    Migrates from legacy location (~/.cache/cammy) if needed.

    Returns:
        Path to $SCITEX_DIR/capture or ~/.scitex/capture (migrating from ~/.cache/cammy if needed)
    """
    new_dir = SCITEX_CAPTURE_DIR
    old_dir = LEGACY_CAPTURE_DIR

    # Create new directory if it doesn't exist
    new_dir.mkdir(parents=True, exist_ok=True)

    # Migrate from old location if exists and new is empty
    if old_dir.exists():
        new_screenshots = list(new_dir.glob("*.jpg"))
        if not new_screenshots or len(new_screenshots) == 0:
            # Move files from old to new location
            try:
                for img in old_dir.glob("*.jpg"):
                    shutil.move(str(img), str(new_dir / img.name))
                print(f"Migrated screenshots from {old_dir} to {new_dir}")
            except Exception as e:
                print(f"Warning: Could not migrate some files: {e}")

    return new_dir


class CaptureServer:
    def __init__(self):
        self.server = Server("scitex-capture-server")
        self.monitoring_active = False
        self.monitoring_worker = None
        self.setup_handlers()

    def setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools():
            return [
                types.Tool(
                    name="capture_screenshot",
                    description="Capture screenshot - monitor, window, browser, or everything including Windows screens from WSL",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Optional message to include in filename",
                            },
                            "monitor_id": {
                                "type": "integer",
                                "description": "Monitor number (0-based, default: 0 for primary monitor)",
                                "default": 0,
                            },
                            "all": {
                                "type": "boolean",
                                "description": "Capture all monitors (shorthand)",
                                "default": False,
                            },
                            "app": {
                                "type": "string",
                                "description": "App name to capture (e.g., 'chrome', 'code')",
                            },
                            "url": {
                                "type": "string",
                                "description": "URL to capture (e.g., '127.0.0.1:8000' or 'http://localhost:3000')",
                            },
                            "quality": {
                                "type": "integer",
                                "description": "JPEG quality (1-100, default: 85)",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 85,
                            },
                            "return_base64": {
                                "type": "boolean",
                                "description": "Return screenshot as base64 string",
                                "default": False,
                            },
                        },
                    },
                ),
                types.Tool(
                    name="start_monitoring",
                    description="Start continuous screenshot monitoring at regular intervals",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "interval": {
                                "type": "number",
                                "description": "Seconds between captures (default: 1.0)",
                                "minimum": 0.1,
                                "default": 1.0,
                            },
                            "monitor_id": {
                                "type": "integer",
                                "description": "Monitor number (0-based, default: 0 for primary monitor)",
                                "default": 0,
                            },
                            "capture_all": {
                                "type": "boolean",
                                "description": "Capture all monitors combined into single image (overrides monitor_id)",
                                "default": False,
                            },
                            "output_dir": {
                                "type": "string",
                                "description": "Directory for screenshots (default: ~/.scitex/capture)",
                            },
                            "quality": {
                                "type": "integer",
                                "description": "JPEG quality (1-100, default: 60)",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 60,
                            },
                            "verbose": {
                                "type": "boolean",
                                "description": "Show capture messages",
                                "default": True,
                            },
                        },
                    },
                ),
                types.Tool(
                    name="stop_monitoring",
                    description="Stop continuous screenshot monitoring",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="get_monitoring_status",
                    description="Get current monitoring status and statistics",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="analyze_screenshot",
                    description="Analyze a screenshot for error indicators (stdout/stderr categorization)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to screenshot to analyze",
                            }
                        },
                        "required": ["path"],
                    },
                ),
                types.Tool(
                    name="list_recent_screenshots",
                    description="List recent screenshots from cache",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of screenshots to list",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 100,
                            },
                            "category": {
                                "type": "string",
                                "description": "Filter by category (stdout/stderr)",
                                "enum": ["stdout", "stderr", "all"],
                                "default": "all",
                            },
                        },
                    },
                ),
                types.Tool(
                    name="clear_cache",
                    description="Clear screenshot cache or manage cache size",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "max_size_gb": {
                                "type": "number",
                                "description": "Keep cache under this size in GB (removes oldest files)",
                                "minimum": 0.001,
                                "default": 1.0,
                            },
                            "clear_all": {
                                "type": "boolean",
                                "description": "Remove all cached screenshots",
                                "default": False,
                            },
                        },
                    },
                ),
                types.Tool(
                    name="create_gif",
                    description="Create an animated GIF from screenshots to summarize sessions or workflows",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Session ID to create GIF from (e.g., '20250823_104523'). Use 'latest' for most recent session.",
                            },
                            "image_paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of image file paths to create GIF from (alternative to session_id)",
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Glob pattern for images to include (alternative to session_id/image_paths)",
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Output GIF file path (auto-generated if not specified)",
                            },
                            "duration": {
                                "type": "number",
                                "description": "Duration per frame in seconds (default: 0.5)",
                                "minimum": 0.1,
                                "maximum": 5.0,
                                "default": 0.5,
                            },
                            "optimize": {
                                "type": "boolean",
                                "description": "Optimize GIF for smaller file size (default: true)",
                                "default": True,
                            },
                            "max_frames": {
                                "type": "integer",
                                "description": "Maximum number of frames to include (default: no limit)",
                                "minimum": 1,
                                "maximum": 100,
                            },
                        },
                    },
                ),
                types.Tool(
                    name="list_sessions",
                    description="List available monitoring sessions that can be converted to GIFs",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of sessions to list (default: 10)",
                                "minimum": 1,
                                "maximum": 50,
                                "default": 10,
                            }
                        },
                    },
                ),
                types.Tool(
                    name="get_info",
                    description="Enumerate all monitors, virtual desktops, and visible windows",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                types.Tool(
                    name="list_windows",
                    description="List all visible windows with their handles and process names",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                types.Tool(
                    name="capture_window",
                    description="Capture a specific window by its handle",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "window_handle": {
                                "type": "integer",
                                "description": "Window handle from list_windows",
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Optional output path for screenshot",
                            },
                            "quality": {
                                "type": "integer",
                                "description": "JPEG quality (1-100, default: 85)",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 85,
                            },
                        },
                        "required": ["window_handle"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            if name == "capture_screenshot":
                return await self.capture_screenshot(**arguments)
            elif name == "start_monitoring":
                return await self.start_monitoring(**arguments)
            elif name == "stop_monitoring":
                return await self.stop_monitoring()
            elif name == "get_monitoring_status":
                return await self.get_monitoring_status()
            elif name == "analyze_screenshot":
                return await self.analyze_screenshot(**arguments)
            elif name == "list_recent_screenshots":
                return await self.list_recent_screenshots(**arguments)
            elif name == "clear_cache":
                return await self.clear_cache(**arguments)
            elif name == "create_gif":
                return await self.create_gif(**arguments)
            elif name == "list_sessions":
                return await self.list_sessions(**arguments)
            elif name == "get_info":
                return await self.get_info_tool()
            elif name == "list_windows":
                return await self.list_windows_tool()
            elif name == "capture_window":
                return await self.capture_window_tool(**arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

        # Provide screenshots as resources
        @self.server.list_resources()
        async def handle_list_resources():
            cache_dir = get_capture_dir()
            if not cache_dir.exists():
                return []

            resources = []
            # Get last 20 screenshots, sorted by modification time
            screenshots = sorted(
                cache_dir.glob("*.jpg"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:20]

            for img_file in screenshots:
                # Parse category from filename
                category = (
                    "stdout"
                    if "-stdout.jpg" in img_file.name
                    else ("stderr" if "-stderr.jpg" in img_file.name else "unknown")
                )

                mtime = datetime.fromtimestamp(img_file.stat().st_mtime)
                resources.append(
                    types.Resource(
                        uri=f"screenshot://{img_file.name}",
                        name=img_file.name,
                        description=f"{category} screenshot from {mtime.strftime('%Y-%m-%d %H:%M:%S')}",
                        mimeType="image/jpeg",
                    )
                )
            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: str):
            if uri.startswith("screenshot://"):
                filename = uri.replace("screenshot://", "")
                filepath = get_capture_dir() / filename

                if filepath.exists():
                    with open(filepath, "rb") as f:
                        content = base64.b64encode(f.read()).decode()

                    return types.ResourceContent(
                        uri=uri, mimeType="image/jpeg", content=content
                    )
                else:
                    raise ValueError(f"Screenshot not found: {filename}")

    async def capture_screenshot(
        self,
        message=None,
        monitor_id=0,
        all=False,
        app=None,
        url=None,
        quality=85,
        return_base64=False,
    ):
        """Capture a screenshot - monitor, window, browser, or everything."""
        try:
            # Run in thread pool since capture is sync
            loop = asyncio.get_event_loop()

            # Use capture.snap which now handles all, app, url parameters
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
                return {
                    "success": False,
                    "error": "Failed to capture screenshot",
                }

            # Determine category from filename
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
            return {"success": False, "message": "Monitoring already active"}

        try:
            loop = asyncio.get_event_loop()

            # Use a lambda to pass the monitor parameters correctly
            def start_with_monitor():
                return capture.start_monitor(
                    output_dir=output_dir or "~/.scitex/capture/",
                    interval=interval,
                    jpeg=True,
                    quality=quality,
                    on_capture=None,
                    on_error=None,
                    verbose=verbose,
                    monitor_id=monitor_id,
                    capture_all=capture_all,
                )

            self.monitoring_worker = await loop.run_in_executor(
                None, start_with_monitor
            )

            self.monitoring_active = True

            return {
                "success": True,
                "message": f"Started monitoring with {interval}s interval on monitor {monitor_id}",
                "output_dir": output_dir or "~/.scitex/capture/",
                "interval": interval,
                "monitor_id": monitor_id,
                "capture_all": capture_all,
                "quality": quality,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.monitoring_active:
            return {"success": False, "message": "Monitoring not active"}

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, capture.stop)

            # Get stats from worker
            stats = {}
            if self.monitoring_worker:
                stats = {
                    "screenshots_taken": self.monitoring_worker.screenshot_count,
                    "session_id": self.monitoring_worker.session_id,
                    "output_dir": str(self.monitoring_worker.output_dir),
                }

            self.monitoring_active = False
            self.monitoring_worker = None

            return {"success": True, "message": "Monitoring stopped", **stats}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_monitoring_status(self):
        """Get monitoring status."""
        status = {
            "active": self.monitoring_active,
            "cache_dir": str(get_capture_dir()),
        }

        if self.monitoring_active and self.monitoring_worker:
            status.update(
                {
                    "screenshots_taken": self.monitoring_worker.screenshot_count,
                    "session_id": self.monitoring_worker.session_id,
                    "interval": self.monitoring_worker.interval_sec,
                    "output_dir": str(self.monitoring_worker.output_dir),
                    "running": self.monitoring_worker.running,
                }
            )

        # Get cache size
        cache_dir = get_capture_dir()
        if cache_dir.exists():
            total_size = sum(f.stat().st_size for f in cache_dir.glob("*.jpg"))
            status["cache_size_mb"] = round(total_size / (1024 * 1024), 2)
            status["screenshot_count"] = len(list(cache_dir.glob("*.jpg")))

        return status

    async def analyze_screenshot(self, path: str):
        """Analyze screenshot for errors/warnings."""
        try:
            # Use scitex.capture's internal detection
            from .utils import _detect_category

            loop = asyncio.get_event_loop()
            category = await loop.run_in_executor(None, _detect_category, path)

            # Get file info
            path_obj = Path(path)
            if not path_obj.exists():
                return {"success": False, "error": f"File not found: {path}"}

            return {
                "success": True,
                "path": path,
                "category": category,
                "is_error": category == "stderr",
                "size_kb": round(path_obj.stat().st_size / 1024, 2),
                "modified": datetime.fromtimestamp(
                    path_obj.stat().st_mtime
                ).isoformat(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_recent_screenshots(self, limit=10, category="all"):
        """List recent screenshots from cache."""
        try:
            cache_dir = get_capture_dir()
            if not cache_dir.exists():
                return {
                    "success": True,
                    "screenshots": [],
                    "message": "Cache directory does not exist",
                }

            # Get all screenshots
            screenshots = list(cache_dir.glob("*.jpg"))

            # Filter by category if specified
            if category == "stdout":
                screenshots = [s for s in screenshots if "-stdout.jpg" in s.name]
            elif category == "stderr":
                screenshots = [s for s in screenshots if "-stderr.jpg" in s.name]

            # Sort by modification time (newest first)
            screenshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Limit results
            screenshots = screenshots[:limit]

            # Build result
            result_list = []
            for screenshot in screenshots:
                cat = "stderr" if "-stderr.jpg" in screenshot.name else "stdout"
                result_list.append(
                    {
                        "filename": screenshot.name,
                        "path": str(screenshot),
                        "category": cat,
                        "size_kb": round(screenshot.stat().st_size / 1024, 2),
                        "modified": datetime.fromtimestamp(
                            screenshot.stat().st_mtime
                        ).isoformat(),
                    }
                )

            return {
                "success": True,
                "screenshots": result_list,
                "count": len(result_list),
                "total_in_cache": len(list(cache_dir.glob("*.jpg"))),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def clear_cache(self, max_size_gb=1.0, clear_all=False):
        """Clear or manage cache size."""
        try:
            cache_dir = get_capture_dir()
            if not cache_dir.exists():
                return {
                    "success": True,
                    "message": "Cache directory does not exist",
                }

            if clear_all:
                # Remove all screenshots
                removed = 0
                for screenshot in cache_dir.glob("*.jpg"):
                    try:
                        screenshot.unlink()
                        removed += 1
                    except:
                        pass

                return {
                    "success": True,
                    "message": f"Removed {removed} screenshots",
                    "removed_count": removed,
                }
            else:
                # Use scitex.capture's cache management
                from .utils import _manage_cache_size

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, _manage_cache_size, cache_dir, max_size_gb
                )

                # Get new cache size
                total_size = sum(f.stat().st_size for f in cache_dir.glob("*.jpg"))

                return {
                    "success": True,
                    "message": f"Cache managed to stay under {max_size_gb}GB",
                    "cache_size_mb": round(total_size / (1024 * 1024), 2),
                    "max_size_gb": max_size_gb,
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def create_gif(
        self,
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

            # Determine which creation method to use
            if session_id:
                if session_id == "latest":
                    # Use most recent session
                    result_path = await loop.run_in_executor(
                        None,
                        creator.create_gif_from_recent_session,
                        "~/.scitex/capture",
                        duration,
                        optimize,
                        max_frames,
                    )
                else:
                    # Use specific session
                    result_path = await loop.run_in_executor(
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
                # Use specific image paths
                if not output_path:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"~/.scitex/capture/custom_gif_{timestamp}.gif"

                result_path = await loop.run_in_executor(
                    None,
                    creator.create_gif_from_files,
                    image_paths,
                    output_path,
                    duration,
                    optimize,
                )
            elif pattern:
                # Use glob pattern
                result_path = await loop.run_in_executor(
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
                    "error": "Must specify either session_id, image_paths, or pattern",
                }

            if result_path:
                # Get file info
                path_obj = Path(result_path)
                file_size = path_obj.stat().st_size / 1024  # KB

                return {
                    "success": True,
                    "path": result_path,
                    "size_kb": round(file_size, 2),
                    "message": f"GIF created successfully: {result_path}",
                    "duration_per_frame": duration,
                    "optimized": optimize,
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create GIF - no suitable images found",
                }

        except ImportError:
            return {
                "success": False,
                "error": "PIL (Pillow) is required for GIF creation. Install with: pip install Pillow",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_sessions(self, limit=10):
        """List available monitoring sessions."""
        try:
            from .gif import GifCreator

            creator = GifCreator()
            loop = asyncio.get_event_loop()

            sessions = await loop.run_in_executor(
                None, creator.get_recent_sessions, "~/.scitex/capture"
            )

            # Limit results
            sessions = sessions[:limit]

            # Get details for each session
            session_details = []
            cache_dir = get_capture_dir()

            for session_id in sessions:
                # Count screenshots in session
                jpg_files = list(cache_dir.glob(f"{session_id}_*.jpg"))
                png_files = list(cache_dir.glob(f"{session_id}_*.png"))

                if not jpg_files and not png_files:
                    continue

                files = jpg_files + png_files
                files.sort()

                if files:
                    # Get session info
                    first_file = files[0]
                    last_file = files[-1]
                    total_size = sum(f.stat().st_size for f in files)

                    session_details.append(
                        {
                            "session_id": session_id,
                            "screenshot_count": len(files),
                            "first_screenshot": first_file.name,
                            "last_screenshot": last_file.name,
                            "total_size_kb": round(total_size / 1024, 2),
                            "start_time": datetime.fromtimestamp(
                                first_file.stat().st_mtime
                            ).isoformat(),
                            "end_time": datetime.fromtimestamp(
                                last_file.stat().st_mtime
                            ).isoformat(),
                        }
                    )

            return {
                "success": True,
                "sessions": session_details,
                "count": len(session_details),
                "message": f"Found {len(session_details)} monitoring sessions",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_info_tool(self):
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

    async def list_windows_tool(self):
        """List all visible windows."""
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, capture.get_info)

            windows = info.get("Windows", {})
            window_list = windows.get("Details", [])

            # Format for easy use
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
                "message": f"Found {len(formatted_windows)} windows on current virtual desktop",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def capture_window_tool(
        self, window_handle: int, output_path: str = None, quality: int = 85
    ):
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


async def _run_server():
    """Run the MCP server (internal)."""
    server = CaptureServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="scitex-capture",
                server_version="0.2.1",
                capabilities=server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main():
    """Main entry point for the MCP server."""
    if not MCP_AVAILABLE:
        import sys

        print("=" * 60)
        print("MCP Server 'scitex-capture' requires the 'mcp' package.")
        print()
        print("Install with:")
        print("  pip install mcp")
        print()
        print("Or install scitex with MCP support:")
        print("  pip install scitex[mcp]")
        print("=" * 60)
        sys.exit(1)

    asyncio.run(_run_server())


if __name__ == "__main__":
    main()

# EOF
