#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/diagram/mcp_server.py
# ----------------------------------------

"""
MCP Server for SciTeX diagram - Paper-optimized diagram generation.

Provides tools for:
- Creating diagrams from YAML specs
- Compiling to Mermaid/Graphviz formats
- Using workflow/decision/pipeline presets
- Splitting large diagrams for publication layouts
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

__all__ = ["DiagramServer", "main", "MCP_AVAILABLE"]


class DiagramServer:
    """MCP Server for Paper-Optimized Diagram Generation."""

    def __init__(self):
        self.server = Server("scitex-diagram")
        self.setup_handlers()

    def setup_handlers(self):
        """Set up MCP server handlers."""
        from ._mcp_handlers import (
            compile_graphviz_handler,
            compile_mermaid_handler,
            create_diagram_handler,
            get_paper_modes_handler,
            get_preset_handler,
            list_presets_handler,
            split_diagram_handler,
        )
        from ._mcp_tool_schemas import get_tool_schemas

        @self.server.list_tools()
        async def handle_list_tools():
            return get_tool_schemas()

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            if name == "create_diagram":
                return await self._wrap_result(create_diagram_handler(**arguments))

            elif name == "compile_mermaid":
                return await self._wrap_result(compile_mermaid_handler(**arguments))

            elif name == "compile_graphviz":
                return await self._wrap_result(compile_graphviz_handler(**arguments))

            elif name == "list_presets":
                return await self._wrap_result(list_presets_handler())

            elif name == "get_preset":
                return await self._wrap_result(get_preset_handler(**arguments))

            elif name == "split_diagram":
                return await self._wrap_result(split_diagram_handler(**arguments))

            elif name == "get_paper_modes":
                return await self._wrap_result(get_paper_modes_handler())

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
    server = DiagramServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="scitex-diagram",
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
        print("MCP Server 'scitex-diagram' requires the 'mcp' package.")
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
