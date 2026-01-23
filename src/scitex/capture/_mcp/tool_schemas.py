#!/usr/bin/env python3
# Timestamp: 2026-01-15
# File: src/scitex/capture/_mcp/tool_schemas.py
# ----------------------------------------

"""Tool schemas for the scitex-capture MCP server."""

from __future__ import annotations

import mcp.types as types

__all__ = ["get_tool_schemas"]


def get_tool_schemas() -> list[types.Tool]:
    """Return all tool schemas for the MCP server."""
    return [
        # Capture tools
        types.Tool(
            name="capture_screenshot",
            description=(
                "Capture screenshot - monitor, window, browser, or everything "
                "including Windows screens from WSL"
            ),
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
                        "default": 1,
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
                    },
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
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
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
                        "default": 1,
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
                        "maximum": 5,
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
                    },
                },
            },
        ),
        types.Tool(
            name="get_info",
            description="Enumerate all monitors, virtual desktops, and visible windows",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="list_windows",
            description="List all visible windows with their handles and process names",
            inputSchema={"type": "object", "properties": {}},
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


# EOF
