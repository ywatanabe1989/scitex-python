# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/src/pip_project_template/mcp_servers/McpServer01.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-08-27 01:43:35 (ywatanabe)"
# # File: /home/ywatanabe/proj/pip-project-template/src/mcp_servers/McpServer01.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# __FILE__ = (
#     "./pip-project-template/src/mcp_servers/McpServer01.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """FastMCP Server 01 implementation."""
# 
# from typing import Any, Dict
# 
# from fastmcp import FastMCP
# 
# from ..core._Calculator import Calculator
# 
# # Create FastMCP server instance
# mcp = FastMCP("mcp-server-01")
# 
# # Initialize calculator
# calculator = Calculator()
# 
# 
# @mcp.tool
# def add_numbers(a: float, b: float) -> str:
#     """Add two numbers.
# 
#     Args:
#         a: First number
#         b: Second number
# 
#     Returns:
#         String representation of the result
#     """
#     result = calculator.calculate(a, b, "add")
#     return str(result)
# 
# 
# @mcp.tool
# def multiply_numbers(a: float, b: float) -> str:
#     """Multiply two numbers.
# 
#     Args:
#         a: First number
#         b: Second number
# 
#     Returns:
#         String representation of the result
#     """
#     result = calculator.calculate(a, b, "multiply")
#     return str(result)
# 
# 
# @mcp.tool
# def batch_calculate(operations: list) -> Dict[str, Any]:
#     """Perform multiple calculations in batch.
# 
#     Args:
#         operations: List of operations, each with 'a', 'b', and 'operation' keys
# 
#     Returns:
#         Dictionary with batch results
#     """
#     results = []
#     for i, op in enumerate(operations):
#         try:
#             a = float(op.get('a', 0))
#             b = float(op.get('b', 0))
#             operation = op.get('operation', 'add')
#             
#             result = calculator.calculate(a, b, operation)
#                 
#             results.append({
#                 "index": i,
#                 "result": result,
#                 "operation": operation,
#                 "inputs": {"a": a, "b": b}
#             })
#         except Exception as e:
#             results.append({
#                 "index": i,
#                 "error": str(e),
#                 "operation": op.get('operation', 'unknown'),
#                 "inputs": op
#             })
#     
#     return {
#         "success": True,
#         "batch_size": len(operations),
#         "results": results
#     }
# 
# 
# @mcp.resource("server://status")
# def server_status():
#     """Server status resource."""
#     return {"status": "running", "server": "McpServer01", "tools_available": 2}
# 
# 
# def run_server(
#     transport: str = "stdio", host: str = "localhost", port: int = 8081
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/src/pip_project_template/mcp_servers/McpServer01.py
# --------------------------------------------------------------------------------
