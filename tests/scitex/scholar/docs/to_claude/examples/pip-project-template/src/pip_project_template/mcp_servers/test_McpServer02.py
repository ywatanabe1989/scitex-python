# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/src/pip_project_template/mcp_servers/McpServer02.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-08-27 01:43:41 (ywatanabe)"
# # File: /home/ywatanabe/proj/pip-project-template/src/mcp_servers/McpServer02.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# __FILE__ = (
#     "./pip-project-template/src/mcp_servers/McpServer02.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """FastMCP Server 02 implementation with enhanced features."""
# 
# from typing import Any, Dict, List
# 
# from fastmcp import Context, FastMCP
# 
# from ..core._Calculator import Calculator
# 
# # Create FastMCP server instance
# mcp = FastMCP("mcp-server-02")
# 
# # Initialize calculator
# calculator = Calculator()
# 
# 
# @mcp.tool
# async def calculate_advanced(
#     a: float, b: float, operation: str = "add", ctx: Context = None
# ) -> Dict[str, Any]:
#     """Perform mathematical calculations with logging.
# 
#     Args:
#         a: First number
#         b: Second number
#         operation: Operation to perform (add or multiply)
#         ctx: FastMCP context for logging
# 
#     Returns:
#         Dictionary with calculation results
#     """
#     if ctx:
#         await ctx.info(
#             f"Performing {operation} operation: {a} {operation} {b}"
#         )
# 
#     result = calculator.calculate(a, b, operation)
# 
#     if ctx:
#         await ctx.info(f"Result calculated: {result}")
# 
#     return {
#         "success": True,
#         "result": result,
#         "operation": operation,
#         "inputs": {"a": a, "b": b},
#         "server": "McpServer02",
#     }
# 
# 
# @mcp.tool
# def batch_calculate(operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Perform multiple calculations in batch.
# 
#     Args:
#         operations: List of operations, each with 'a', 'b', and 'operation' keys
# 
#     Returns:
#         List of calculation results
#     """
#     results = []
#     for op in operations:
#         try:
#             a = float(op.get("a", 0))
#             b = float(op.get("b", 0))
#             operation = op.get("operation", "add")
# 
#             result = calculator.calculate(a, b, operation)
#             results.append(
#                 {
#                     "success": True,
#                     "result": result,
#                     "operation": operation,
#                     "inputs": {"a": a, "b": b},
#                 }
#             )
#         except Exception as e:
#             results.append({"success": False, "error": str(e), "inputs": op})
# 
#     return results
# 
# 
# @mcp.tool
# def get_server_info() -> Dict[str, Any]:
#     """Get enhanced information about this MCP server.
# 
#     Returns:
#         Dictionary with server information
#     """
#     return {
#         "server": "McpServer02",
#         "version": "0.1.0",
#         "framework": "FastMCP",
#         "description": "Enhanced mathematical calculation server with batch processing",
#         "capabilities": [
#             "single_calculation",
#             "batch_calculation",
#             "async_operations",
#             "logging",
#         ],
#     }
# 
# 
# @mcp.resource("server://metrics")
# def server_metrics():
#     """Server performance metrics resource."""
#     return {
#         "status": "running",
#         "server": "McpServer02",
#         "tools_available": 3,
#         "capabilities": ["batch_processing", "async_operations"],
#         "framework": "FastMCP",
#     }
# 
# 
# @mcp.resource("calculations://history")
# def calculation_history():
#     """Historical calculation data resource."""
#     return {
#         "recent_operations": [
#             {"operation": "add", "count": 42},
#             {"operation": "multiply", "count": 28},
#         ],
#         "total_calculations": 70,
#         "server": "McpServer02",
#     }
# 
# 
# def run_server(
#     transport: str = "stdio", host: str = "localhost", port: int = 8082
# ):
#     """Run the MCP server with specified transport."""
#     if transport == "stdio":
#         mcp.run(transport="stdio")
#     elif transport == "http":
#         mcp.run(transport="http", host=host, port=port, path="/mcp")
#     elif transport == "sse":
#         mcp.run(transport="sse", host=host, port=port)
#     else:
#         raise ValueError(f"Unsupported transport: {transport}")
# 
# 
# def main():
#     """Run MCP server in STDIO mode."""
#     run_server()
# 
# 
# if __name__ == "__main__":
#     main()
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/src/pip_project_template/mcp_servers/McpServer02.py
# --------------------------------------------------------------------------------
