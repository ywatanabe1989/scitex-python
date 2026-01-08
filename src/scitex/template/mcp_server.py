#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/template/mcp_server.py
# ----------------------------------------

"""
MCP Server for SciTeX Template - Project Scaffolding Framework

Provides tools for:
- Listing available project templates
- Getting detailed template information
- Creating new projects from templates
- Managing git initialization strategies
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

__all__ = ["TemplateServer", "main", "MCP_AVAILABLE"]


class TemplateServer:
    """MCP Server for Project Template Management."""

    def __init__(self):
        self.server = Server("scitex-template")
        self.setup_handlers()

    def setup_handlers(self):
        """Set up MCP server handlers."""
        from ._mcp_handlers import (
            clone_template_handler,
            get_template_info_handler,
            list_git_strategies_handler,
            list_templates_handler,
        )
        from ._mcp_tool_schemas import get_tool_schemas

        @self.server.list_tools()
        async def handle_list_tools():
            return get_tool_schemas()

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            # List templates
            if name == "list_templates":
                return await self._wrap_result(list_templates_handler())

            # Get template info
            elif name == "get_template_info":
                return await self._wrap_result(get_template_info_handler(**arguments))

            # Clone template
            elif name == "clone_template":
                return await self._wrap_result(clone_template_handler(**arguments))

            # List git strategies
            elif name == "list_git_strategies":
                return await self._wrap_result(list_git_strategies_handler())

            else:
                raise ValueError(f"Unknown tool: {name}")

        @self.server.list_resources()
        async def handle_list_resources():
            """List available template resources."""
            resources = [
                types.Resource(
                    uri="template://templates",
                    name="Available Templates",
                    description="List of all available project templates",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="template://git-strategies",
                    name="Git Strategies",
                    description="Available git initialization strategies",
                    mimeType="application/json",
                ),
            ]
            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: str):
            """Read a template resource."""
            import json

            if uri == "template://templates":
                result = await list_templates_handler()
                return types.TextResourceContents(
                    uri=uri,
                    mimeType="application/json",
                    text=json.dumps(result, indent=2),
                )

            elif uri == "template://git-strategies":
                result = await list_git_strategies_handler()
                return types.TextResourceContents(
                    uri=uri,
                    mimeType="application/json",
                    text=json.dumps(result, indent=2),
                )

            else:
                raise ValueError(f"Unknown resource URI: {uri}")

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
    server = TemplateServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="scitex-template",
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
        print("MCP Server 'scitex-template' requires the 'mcp' package.")
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
