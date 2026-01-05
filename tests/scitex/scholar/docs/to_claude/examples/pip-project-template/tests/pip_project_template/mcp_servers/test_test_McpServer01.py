# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/tests/pip_project_template/mcp_servers/test_McpServer01.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Test file for src/mcp_servers/McpServer01.py
# 
# import pytest
# import sys
# from pathlib import Path
# from unittest.mock import patch, MagicMock
# 
# # Add src to path
# sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
# 
# try:
#     from fastmcp import Client
#     FASTMCP_AVAILABLE = True
# except ImportError:
#     FASTMCP_AVAILABLE = False
# 
# import pip_project_template.mcp_servers.McpServer01 as mcp_module
# 
# 
# class TestMcpServer01:
#     """Test suite for mcp_servers.McpServer01"""
# 
#     def test_import(self):
#         """Test that module imports successfully."""
#         assert mcp_module is not None
#         assert hasattr(mcp_module, 'mcp')
#         assert hasattr(mcp_module, 'calculator')
# 
#     def test_mcp_instance(self):
#         """Test that FastMCP instance is created."""
#         assert mcp_module.mcp is not None
#         assert hasattr(mcp_module.mcp, 'tool')  # FastMCP instance should have tool decorator
# 
#     def test_calculator_instance(self):
#         """Test that calculator is properly initialized."""
#         from pip_project_template.core._Calculator import Calculator
#         assert isinstance(mcp_module.calculator, Calculator)
# 
#     def test_calculator_add_operation(self):
#         """Test calculator add operation via module calculator."""
#         result = mcp_module.calculator.calculate(5.0, 3.0, "add")
#         assert result == 8.0
# 
#     def test_calculator_multiply_operation(self):
#         """Test calculator multiply operation via module calculator."""
#         result = mcp_module.calculator.calculate(4.0, 2.5, "multiply")
#         assert result == 10.0
# 
#     def test_calculator_negative_numbers(self):
#         """Test calculator with negative numbers via module calculator."""
#         result = mcp_module.calculator.calculate(-5.0, 3.0, "add")
#         assert result == -2.0
# 
#     def test_mcp_instance_has_tools_and_resources(self):
#         """Test that MCP instance has tools and resources."""
#         # Test that we can access the mcp instance and it has expected methods
#         assert hasattr(mcp_module.mcp, 'tool')
#         assert hasattr(mcp_module.mcp, 'resource')
#         assert hasattr(mcp_module.mcp, 'run')
#     
#     @pytest.mark.asyncio
#     @pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not available for testing")
#     async def test_mcp_tools_via_client(self):
#         """Test MCP tools using FastMCP Client for proper in-memory testing."""
#         # Use FastMCP's in-memory testing approach
#         async with Client(mcp_module.mcp) as client:
#             # List available tools
#             tools = await client.list_tools()
#             tool_names = [tool.name for tool in tools]
#             
#             # Verify expected tools are available
#             expected_tools = ['add_numbers', 'multiply_numbers', 'batch_calculate']
#             for expected_tool in expected_tools:
#                 assert expected_tool in tool_names, f"Tool {expected_tool} not found in {tool_names}"
#     
#     @pytest.mark.asyncio
#     @pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not available for testing")
#     async def test_add_numbers_tool(self):
#         """Test add_numbers tool via FastMCP Client."""
#         async with Client(mcp_module.mcp) as client:
#             result = await client.call_tool('add_numbers', {'a': 5.0, 'b': 3.0})
#             # FastMCP tools return content
#             assert '8.0' in str(result.content[0].text) or str(result.content[0].text) == '8.0'
#     
#     @pytest.mark.asyncio
#     @pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not available for testing")
#     async def test_multiply_numbers_tool(self):
#         """Test multiply_numbers tool via FastMCP Client."""
#         async with Client(mcp_module.mcp) as client:
#             result = await client.call_tool('multiply_numbers', {'a': 4.0, 'b': 2.5})
#             assert '10.0' in str(result.content[0].text) or str(result.content[0].text) == '10.0'
# 
#     def test_run_server_invalid_transport(self):
#         """Test run_server with invalid transport raises ValueError."""
#         with pytest.raises(ValueError) as exc_info:
#             mcp_module.run_server("invalid")
#         assert "Unsupported transport: invalid" in str(exc_info.value)
# 
#     def test_run_server_function_exists(self):
#         """Test that run_server function exists and is callable."""
#         assert hasattr(mcp_module, 'run_server')
#         assert callable(mcp_module.run_server)
#     
#     @pytest.mark.asyncio
#     @pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not available for testing")
#     async def test_mcp_resources(self):
#         """Test MCP resources via FastMCP Client."""
#         async with Client(mcp_module.mcp) as client:
#             # List available resources
#             resources = await client.list_resources()
#             resource_uris = [resource.uri for resource in resources]
#             
#             # Check if any resources are defined (project may or may not have resources)
#             # This test verifies that resource listing works without requiring specific resources
#             assert isinstance(resource_uris, list)  # Should return a list even if empty
# 
#     @pytest.mark.asyncio
#     @pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not available for testing")
#     async def test_batch_calculate_tool(self):
#         """Test batch_calculate tool via FastMCP Client."""
#         async with Client(mcp_module.mcp) as client:
#             operations = [
#                 {"a": 5.0, "b": 3.0, "operation": "add"},
#                 {"a": 4.0, "b": 2.0, "operation": "multiply"}
#             ]
#             result = await client.call_tool('batch_calculate', {'operations': operations})
#             result_data = result.content[0].text
#             
#             # Verify that the batch calculation worked correctly
#             assert '"success":true' in result_data
#             assert '"batch_size":2' in result_data
#             assert '"result":8.0' in result_data  # Should find 8.0 results
# 
#     @pytest.mark.asyncio
#     @pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not available for testing")
#     async def test_batch_calculate_error_handling(self):
#         """Test batch_calculate tool handles invalid operations."""
#         async with Client(mcp_module.mcp) as client:
#             operations = [
#                 {"a": 5.0, "b": 3.0, "operation": "add"},
#                 {"a": 4.0, "b": 2.0, "operation": "invalid_operation"}
#             ]
#             result = await client.call_tool('batch_calculate', {'operations': operations})
#             result_data = result.content[0].text
#             
#             # Should contain error information for invalid operation
#             assert '"error"' in result_data
# 
#     @pytest.mark.asyncio
#     @pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not available for testing")
#     async def test_server_status_resource(self):
#         """Test server://status resource via FastMCP Client."""
#         async with Client(mcp_module.mcp) as client:
#             resources = await client.list_resources()
#             resource_uris = [str(resource.uri) for resource in resources]
#             
#             # Verify server://status resource exists
#             assert "server://status" in resource_uris
#             
#             # Read the resource
#             status_resource = await client.read_resource("server://status")
#             
#             # FastMCP resource response contains content directly
#             if hasattr(status_resource, 'contents'):
#                 resource_data = status_resource.contents[0].text
#             else:
#                 # Handle direct content response
#                 resource_data = str(status_resource)
#             
#             # Verify the status resource contains expected data
#             assert '"status":"running"' in resource_data or "status" in resource_data
#             assert '"server":"McpServer01"' in resource_data or "McpServer01" in resource_data
#             assert '"tools_available":2' in resource_data or "tools_available" in resource_data
# 
# 
#     def test_run_server_stdio_transport(self):
#         """Test run_server with stdio transport."""
#         with patch.object(mcp_module.mcp, 'run') as mock_run:
#             mcp_module.run_server("stdio")
#             mock_run.assert_called_once_with(transport="stdio")
# 
#     def test_run_server_http_transport(self):
#         """Test run_server with http transport."""
#         with patch.object(mcp_module.mcp, 'run') as mock_run:
#             mcp_module.run_server("http", host="127.0.0.1", port=9999)
#             mock_run.assert_called_once_with(transport="http", host="127.0.0.1", port=9999, path="/mcp")
# 
#     def test_run_server_sse_transport(self):
#         """Test run_server with sse transport."""
#         with patch.object(mcp_module.mcp, 'run') as mock_run:
#             mcp_module.run_server("sse", host="0.0.0.0", port=8888)
#             mock_run.assert_called_once_with(transport="sse", host="0.0.0.0", port=8888)
# 
#     def test_main_function_execution(self):
#         """Test main function execution with mocked run_server."""
#         with patch.object(mcp_module, 'run_server') as mock_run:
#             mcp_module.main()
#             mock_run.assert_called_once()
# 
#     def test_module_main_execution_block(self):
#         """Test module execution as script."""
#         import subprocess
#         import sys
#         
#         # Test the module can be executed as a script
#         # This will hit the if __name__ == "__main__": main() block
#         result = subprocess.run([
#             sys.executable, "-m", "pip_project_template.mcp_servers.McpServer01", "--help"
#         ], capture_output=True, text=True, timeout=5)
#         
#         # Should either work with help or show that it can be executed
#         # (may fail because the server tries to run, but that's expected)
#         assert result.returncode in [0, 1, 2]  # Any of these is acceptable for a server
# 
# 
# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/tests/pip_project_template/mcp_servers/test_McpServer01.py
# --------------------------------------------------------------------------------
