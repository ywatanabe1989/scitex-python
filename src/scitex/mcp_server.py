#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/mcp_server.py
# ----------------------------------------

"""
Unified MCP Server for SciTeX - All modules in one server.

Aggregates tools from all SciTeX MCP modules with prefixed names:
- audio_*: Text-to-speech
- capture_*: Screenshot capture
- scholar_*: Literature management
- stats_*: Statistical analysis
- template_*: Project scaffolding
- plt_*: Publication plotting
- canvas_*: Multi-panel figures
- diagram_*: Diagram generation
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

__all__ = ["UnifiedServer", "main"]

# Module registry: (prefix, schema_getter, handler_module)
MODULES = [
    ("audio", "scitex.audio._mcp_tool_schemas", "scitex.audio._mcp_handlers"),
    ("capture", "scitex.capture._mcp_tool_schemas", "scitex.capture._mcp_handlers"),
    ("scholar", "scitex.scholar._mcp_tool_schemas", "scitex.scholar._mcp_handlers"),
    ("stats", "scitex.stats._mcp_tool_schemas", "scitex.stats._mcp_handlers"),
    ("template", "scitex.template._mcp_tool_schemas", "scitex.template._mcp_handlers"),
    ("plt", "scitex.plt._mcp_tool_schemas", "scitex.plt._mcp_handlers"),
    ("canvas", "scitex.canvas._mcp_tool_schemas", "scitex.canvas._mcp_handlers"),
    ("diagram", "scitex.diagram._mcp_tool_schemas", "scitex.diagram._mcp_handlers"),
]


class UnifiedServer:
    """Unified MCP Server aggregating all SciTeX modules."""

    def __init__(self):
        self.server = Server("scitex-mcp-server")
        self.tools: list[types.Tool] = []
        self.handlers: dict[str, Callable] = {}
        self._load_modules()
        self._setup_handlers()

    def _load_modules(self):
        """Load tool schemas and handlers from all modules."""
        import importlib

        for prefix, schema_module, handler_module in MODULES:
            try:
                # Load tool schemas
                schema_mod = importlib.import_module(schema_module)
                tools = schema_mod.get_tool_schemas()

                # Load handlers
                handler_mod = importlib.import_module(handler_module)

                for tool in tools:
                    # Create prefixed tool name
                    prefixed_name = f"{prefix}_{tool.name}"

                    # Create new tool with prefixed name
                    prefixed_tool = types.Tool(
                        name=prefixed_name,
                        description=f"[{prefix}] {tool.description}",
                        inputSchema=tool.inputSchema,
                    )
                    self.tools.append(prefixed_tool)

                    # Map handler
                    handler_name = f"{tool.name}_handler"
                    if hasattr(handler_mod, handler_name):
                        self.handlers[prefixed_name] = getattr(
                            handler_mod, handler_name
                        )

            except ImportError as e:
                # Module not available, skip it
                print(f"Skipping {prefix} module: {e}")
            except Exception as e:
                print(f"Error loading {prefix} module: {e}")

    def _setup_handlers(self):
        """Set up MCP server handlers."""

        @self.server.list_tools()
        async def handle_list_tools():
            return self.tools

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            if name not in self.handlers:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {"success": False, "error": f"Unknown tool: {name}"},
                            indent=2,
                        ),
                    )
                ]

            return await self._wrap_result(self.handlers[name](**arguments))

    async def _wrap_result(self, coro):
        """Wrap handler result as MCP TextContent."""
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


async def main():
    """Run the unified MCP server."""
    server = UnifiedServer()
    print(f"Loaded {len(server.tools)} tools from {len(MODULES)} modules")

    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="scitex-mcp-server",
                server_version="0.1.0",
                capabilities=server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())


# EOF
