#!/usr/bin/env python3
"""MCP tool definitions for SciTeX Capture."""

import mcp.types as types


def get_capture_tools():
    """Return capture-related tool definitions."""
    return [
        types.Tool(
            name="capture_screenshot",
            description="Capture screenshot with optional grid/cursor overlay",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message for filename",
                    },
                    "monitor_id": {"type": "integer", "default": 0},
                    "all": {
                        "type": "boolean",
                        "default": False,
                        "description": "All monitors",
                    },
                    "app": {"type": "string", "description": "App name to capture"},
                    "url": {"type": "string", "description": "URL to capture"},
                    "quality": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 85,
                    },
                    "return_base64": {"type": "boolean", "default": False},
                    "grid_overlay": {
                        "type": "boolean",
                        "default": False,
                        "description": "Add grid",
                    },
                    "cursor_overlay": {
                        "type": "boolean",
                        "default": False,
                        "description": "Add cursor",
                    },
                    "grid_spacing": {"type": "integer", "default": 25},
                },
            },
        ),
        types.Tool(
            name="capture_window",
            description="Capture a specific window by handle",
            inputSchema={
                "type": "object",
                "properties": {
                    "window_handle": {
                        "type": "integer",
                        "description": "Window handle",
                    },
                    "output_path": {"type": "string"},
                    "quality": {"type": "integer", "default": 85},
                },
                "required": ["window_handle"],
            },
        ),
    ]


def get_monitoring_tools():
    """Return monitoring tool definitions."""
    return [
        types.Tool(
            name="start_monitoring",
            description="Start continuous screenshot monitoring",
            inputSchema={
                "type": "object",
                "properties": {
                    "interval": {"type": "number", "minimum": 0.1, "default": 1.0},
                    "monitor_id": {"type": "integer", "default": 0},
                    "capture_all": {"type": "boolean", "default": False},
                    "output_dir": {"type": "string"},
                    "quality": {"type": "integer", "default": 60},
                    "verbose": {"type": "boolean", "default": True},
                },
            },
        ),
        types.Tool(
            name="stop_monitoring",
            description="Stop continuous monitoring",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_monitoring_status",
            description="Get monitoring status",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


def get_utility_tools():
    """Return utility tool definitions."""
    return [
        types.Tool(
            name="analyze_screenshot",
            description="Analyze screenshot for errors",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        types.Tool(
            name="list_recent_screenshots",
            description="List recent screenshots",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "category": {
                        "type": "string",
                        "enum": ["stdout", "stderr", "all"],
                        "default": "all",
                    },
                },
            },
        ),
        types.Tool(
            name="clear_cache",
            description="Clear screenshot cache",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_size_gb": {"type": "number", "minimum": 0.001, "default": 1.0},
                    "clear_all": {"type": "boolean", "default": False},
                },
            },
        ),
        types.Tool(
            name="get_info",
            description="Get monitor and window info",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="list_windows",
            description="List visible windows",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


def get_gif_tools():
    """Return GIF tool definitions."""
    return [
        types.Tool(
            name="create_gif",
            description="Create GIF from screenshots",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "image_paths": {"type": "array", "items": {"type": "string"}},
                    "pattern": {"type": "string"},
                    "output_path": {"type": "string"},
                    "duration": {"type": "number", "default": 0.5},
                    "optimize": {"type": "boolean", "default": True},
                    "max_frames": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            },
        ),
        types.Tool(
            name="list_sessions",
            description="List monitoring sessions",
            inputSchema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 10}},
            },
        ),
    ]


def get_all_tools():
    """Return all tool definitions."""
    return (
        get_capture_tools()
        + get_monitoring_tools()
        + get_utility_tools()
        + get_gif_tools()
    )
