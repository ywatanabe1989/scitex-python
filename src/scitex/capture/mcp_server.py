#!/usr/bin/env python3
"""MCP Server for SciTeX Capture - Thin orchestrator."""

import asyncio
import base64
from datetime import datetime

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from .mcp_handlers import (
    CaptureHandlers,
    GifHandlers,
    MonitoringHandlers,
    UtilityHandlers,
)
from .mcp_tool_defs import get_all_tools
from .mcp_utils import get_capture_dir


class CaptureServer:
    """MCP Server for screenshot capture with overlay support."""

    def __init__(self):
        self.server = Server("scitex-capture-server")
        self.monitoring = MonitoringHandlers()
        self.setup_handlers()

    def setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools():
            return get_all_tools()

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            handlers = {
                "capture_screenshot": CaptureHandlers.capture_screenshot,
                "capture_window": CaptureHandlers.capture_window,
                "start_monitoring": self.monitoring.start_monitoring,
                "stop_monitoring": self.monitoring.stop_monitoring,
                "get_monitoring_status": self.monitoring.get_status,
                "analyze_screenshot": UtilityHandlers.analyze_screenshot,
                "list_recent_screenshots": UtilityHandlers.list_recent_screenshots,
                "clear_cache": UtilityHandlers.clear_cache,
                "get_info": UtilityHandlers.get_info,
                "list_windows": UtilityHandlers.list_windows,
                "create_gif": GifHandlers.create_gif,
                "list_sessions": GifHandlers.list_sessions,
            }
            if name in handlers:
                return await handlers[name](**arguments)
            raise ValueError(f"Unknown tool: {name}")

        @self.server.list_resources()
        async def handle_list_resources():
            cache_dir = get_capture_dir()
            if not cache_dir.exists():
                return []

            screenshots = sorted(
                cache_dir.glob("*.jpg"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:20]

            resources = []
            for img in screenshots:
                category = "stderr" if "-stderr.jpg" in img.name else "stdout"
                mtime = datetime.fromtimestamp(img.stat().st_mtime)
                resources.append(
                    types.Resource(
                        uri=f"screenshot://{img.name}",
                        name=img.name,
                        description=f"{category} from {mtime:%Y-%m-%d %H:%M:%S}",
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
                raise ValueError(f"Not found: {filename}")


async def main():
    """Main entry point."""
    server = CaptureServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="scitex-capture",
                server_version="0.3.0",
                capabilities=server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
