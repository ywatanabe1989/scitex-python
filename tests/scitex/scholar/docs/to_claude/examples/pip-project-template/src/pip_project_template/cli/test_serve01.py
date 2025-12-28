# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/src/pip_project_template/cli/serve01.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-08-27 01:15:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/pip-project-template/src/cli/serve01.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# __FILE__ = (
#     "./pip-project-template/src/cli/serve01.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# """Serve01 command to start MCP Server 01."""
# 
# import argparse
# 
# 
# def create_parser():
#     """Create parser for serve01 command."""
#     parser = argparse.ArgumentParser(description="Start MCP Server 01", add_help=False)
#     parser.add_argument(
#         "--port", type=int, default=8081, help="Port to serve on (default: 8081)"
#     )
#     parser.add_argument(
#         "--host", type=str, default="localhost", help="Host to serve on (default: localhost)"
#     )
#     parser.add_argument(
#         "--transport", 
#         choices=["stdio", "http", "sse"],
#         default="stdio",
#         help="Transport protocol (default: stdio)"
#     )
#     return parser
# 
# 
# def main(args=None):
#     """Execute serve01 command."""
#     parser = create_parser()
#     parsed = parser.parse_args(args)
# 
#     from ..mcp_servers.McpServer01 import run_server
# 
#     print(f"Starting FastMCP Server 01 ({parsed.transport.upper()}) on {parsed.host}:{parsed.port}")
#     
#     # Use specified transport
#     if parsed.transport == "stdio":
#         print("Using STDIO transport")
#         run_server(transport="stdio")
#     elif parsed.transport == "http":
#         print(f"Using HTTP transport at http://{parsed.host}:{parsed.port}/mcp") 
#         run_server(transport="http", host=parsed.host, port=parsed.port)
#     elif parsed.transport == "sse":
#         print(f"Using SSE transport at http://{parsed.host}:{parsed.port}")
#         run_server(transport="sse", host=parsed.host, port=parsed.port)
#     
#     return 0
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/src/pip_project_template/cli/serve01.py
# --------------------------------------------------------------------------------
