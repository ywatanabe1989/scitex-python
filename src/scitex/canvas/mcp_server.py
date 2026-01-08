#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/canvas/mcp_server.py
# ----------------------------------------

"""
MCP Server for SciTeX canvas - Multi-panel figure composition.

Provides tools for:
- Creating canvas workspaces
- Adding/removing panels
- Exporting composed figures
- Managing multi-panel layouts
"""

from __future__ import annotations

import asyncio

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

__all__ = ["CanvasServer", "main", "MCP_AVAILABLE"]


class CanvasServer:
    """MCP Server for Multi-panel Figure Composition."""

    def __init__(self):
        self.server = Server("scitex-canvas")
        self.setup_handlers()

    def setup_handlers(self):
        """Set up MCP server handlers."""
        from ._mcp_handlers import (
            add_panel_handler,
            canvas_exists_handler,
            create_canvas_handler,
            export_canvas_handler,
            list_canvases_handler,
            list_panels_handler,
            remove_panel_handler,
        )
        from ._mcp_tool_schemas import get_tool_schemas

        @self.server.list_tools()
        async def handle_list_tools():
            return get_tool_schemas()

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            if name == "create_canvas":
                return await self._wrap_result(create_canvas_handler(**arguments))

            elif name == "add_panel":
                return await self._wrap_result(add_panel_handler(**arguments))

            elif name == "list_panels":
                return await self._wrap_result(list_panels_handler(**arguments))

            elif name == "remove_panel":
                return await self._wrap_result(remove_panel_handler(**arguments))

            elif name == "export_canvas":
                return await self._wrap_result(export_canvas_handler(**arguments))

            elif name == "list_canvases":
                return await self._wrap_result(list_canvases_handler(**arguments))

            elif name == "canvas_exists":
                return await self._wrap_result(canvas_exists_handler(**arguments))

            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _wrap_result(self, coro):
        """Wrap handler result as MCP TextContent."""
        import json

        try:
            result = await coro
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str),
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"success": False, "error": str(e)}, indent=2),
                )
            ]


async def _run_server():
    """Run the MCP server (internal)."""
    server = CanvasServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="scitex-canvas",
                server_version="0.1.0",
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
        print("MCP Server 'scitex-canvas' requires the 'mcp' package.")
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
