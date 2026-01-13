# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/mcp_server.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2026-01-08
# # File: src/scitex/plt/mcp_server.py
# # ----------------------------------------
# 
# """
# MCP Server for SciTeX plt - Publication-quality plotting.
# 
# Provides tools for:
# - Style configuration (get/set publication styles)
# - Style presets (journal-specific configurations)
# - Figure cropping (auto-crop whitespace)
# - DPI management
# - Color palette access
# """
# 
# from __future__ import annotations
# 
# import asyncio
# 
# # Graceful MCP dependency handling
# try:
#     import mcp.types as types
#     from mcp.server import NotificationOptions, Server
#     from mcp.server.models import InitializationOptions
#     from mcp.server.stdio import stdio_server
# 
#     MCP_AVAILABLE = True
# except ImportError:
#     MCP_AVAILABLE = False
#     types = None  # type: ignore
#     Server = None  # type: ignore
#     NotificationOptions = None  # type: ignore
#     InitializationOptions = None  # type: ignore
#     stdio_server = None  # type: ignore
# 
# __all__ = ["PltServer", "main", "MCP_AVAILABLE"]
# 
# 
# class PltServer:
#     """MCP Server for Publication-quality Plotting."""
# 
#     def __init__(self):
#         self.server = Server("scitex-plt")
#         self.setup_handlers()
# 
#     def setup_handlers(self):
#         """Set up MCP server handlers."""
#         from ._mcp.handlers import (
#             add_panel_label_handler,
#             add_significance_handler,
#             close_figure_handler,
#             create_figure_handler,
#             crop_figure_handler,
#             get_color_palette_handler,
#             get_dpi_settings_handler,
#             get_style_handler,
#             list_presets_handler,
#             plot_bar_handler,
#             plot_box_handler,
#             plot_line_handler,
#             plot_scatter_handler,
#             plot_violin_handler,
#             save_figure_handler,
#             set_style_handler,
#         )
#         from ._mcp.tool_schemas import get_tool_schemas
# 
#         @self.server.list_tools()
#         async def handle_list_tools():
#             return get_tool_schemas()
# 
#         @self.server.call_tool()
#         async def handle_call_tool(name: str, arguments: dict):
#             # Style tools
#             if name == "get_style":
#                 return await self._wrap_result(get_style_handler())
#             elif name == "set_style":
#                 return await self._wrap_result(set_style_handler(**arguments))
#             elif name == "list_presets":
#                 return await self._wrap_result(list_presets_handler())
#             elif name == "crop_figure":
#                 return await self._wrap_result(crop_figure_handler(**arguments))
#             elif name == "get_dpi_settings":
#                 return await self._wrap_result(get_dpi_settings_handler())
#             elif name == "get_color_palette":
#                 return await self._wrap_result(get_color_palette_handler(**arguments))
#             # Plotting tools
#             elif name == "create_figure":
#                 return await self._wrap_result(create_figure_handler(**arguments))
#             elif name == "plot_bar":
#                 return await self._wrap_result(plot_bar_handler(**arguments))
#             elif name == "plot_scatter":
#                 return await self._wrap_result(plot_scatter_handler(**arguments))
#             elif name == "plot_line":
#                 return await self._wrap_result(plot_line_handler(**arguments))
#             elif name == "plot_box":
#                 return await self._wrap_result(plot_box_handler(**arguments))
#             elif name == "plot_violin":
#                 return await self._wrap_result(plot_violin_handler(**arguments))
#             elif name == "add_significance":
#                 return await self._wrap_result(add_significance_handler(**arguments))
#             elif name == "add_panel_label":
#                 return await self._wrap_result(add_panel_label_handler(**arguments))
#             elif name == "save_figure":
#                 return await self._wrap_result(save_figure_handler(**arguments))
#             elif name == "close_figure":
#                 return await self._wrap_result(close_figure_handler(**arguments))
#             else:
#                 raise ValueError(f"Unknown tool: {name}")
# 
#         @self.server.list_resources()
#         async def handle_list_resources():
#             """List available plt resources."""
#             resources = [
#                 types.Resource(
#                     uri="plt://style/current",
#                     name="Current Style",
#                     description="Current publication style configuration",
#                     mimeType="application/json",
#                 ),
#                 types.Resource(
#                     uri="plt://presets",
#                     name="Style Presets",
#                     description="Available journal style presets",
#                     mimeType="application/json",
#                 ),
#                 types.Resource(
#                     uri="plt://colors",
#                     name="Color Palette",
#                     description="SciTeX color palette",
#                     mimeType="application/json",
#                 ),
#             ]
#             return resources
# 
#         @self.server.read_resource()
#         async def handle_read_resource(uri: str):
#             """Read a plt resource."""
#             import json
# 
#             if uri == "plt://style/current":
#                 result = await get_style_handler()
#                 return types.TextResourceContents(
#                     uri=uri,
#                     mimeType="application/json",
#                     text=json.dumps(result, indent=2),
#                 )
# 
#             elif uri == "plt://presets":
#                 result = await list_presets_handler()
#                 return types.TextResourceContents(
#                     uri=uri,
#                     mimeType="application/json",
#                     text=json.dumps(result, indent=2),
#                 )
# 
#             elif uri == "plt://colors":
#                 result = await get_color_palette_handler()
#                 return types.TextResourceContents(
#                     uri=uri,
#                     mimeType="application/json",
#                     text=json.dumps(result, indent=2),
#                 )
# 
#             else:
#                 raise ValueError(f"Unknown resource URI: {uri}")
# 
#     async def _wrap_result(self, coro):
#         """Wrap handler result as MCP TextContent."""
#         import json
# 
#         try:
#             result = await coro
#             return [
#                 types.TextContent(
#                     type="text",
#                     text=json.dumps(result, indent=2, default=str),
#                 )
#             ]
#         except Exception as e:
#             return [
#                 types.TextContent(
#                     type="text",
#                     text=json.dumps({"success": False, "error": str(e)}, indent=2),
#                 )
#             ]
# 
# 
# async def _run_server():
#     """Run the MCP server (internal)."""
#     server = PltServer()
#     async with stdio_server() as (read_stream, write_stream):
#         await server.server.run(
#             read_stream,
#             write_stream,
#             InitializationOptions(
#                 server_name="scitex-plt",
#                 server_version="0.1.0",
#                 capabilities=server.server.get_capabilities(
#                     notification_options=NotificationOptions(),
#                     experimental_capabilities={},
#                 ),
#             ),
#         )
# 
# 
# def main():
#     """Main entry point for the MCP server."""
#     if not MCP_AVAILABLE:
#         import sys
# 
#         print("=" * 60)
#         print("MCP Server 'scitex-plt' requires the 'mcp' package.")
#         print()
#         print("Install with:")
#         print("  pip install mcp")
#         print()
#         print("Or install scitex with MCP support:")
#         print("  pip install scitex[mcp]")
#         print("=" * 60)
#         sys.exit(1)
# 
#     asyncio.run(_run_server())
# 
# 
# if __name__ == "__main__":
#     main()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/mcp_server.py
# --------------------------------------------------------------------------------
