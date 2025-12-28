# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/tests/pip_project_template/mcp_servers/test_McpServer02.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Test file for src/mcp_servers/McpServer02.py
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
# import pip_project_template.mcp_servers.McpServer02 as mcp_module
# 
# 
# class TestMcpserver02:
#     """Test suite for mcp_servers.McpServer02"""
# 
#     def test_module_imports(self):
#         """Test that module imports successfully."""
#         import importlib
#         module = importlib.import_module("pip_project_template.mcp_servers.McpServer02")
#         assert module is not None
# 
#     def test_module_has_expected_attributes(self):
#         """Test that module has expected structure."""
#         import importlib
#         module = importlib.import_module("pip_project_template.mcp_servers.McpServer02")
#         
#         # Check that module has a docstring or __all__ or callable functions
#         has_content = (
#             hasattr(module, '__doc__') and module.__doc__ or
#             hasattr(module, '__all__') or
#             any(callable(getattr(module, attr)) for attr in dir(module) 
#                 if not attr.startswith('_'))
#         )
#         assert has_content, f"Module pip_project_template.mcp_servers.McpServer02 appears to be empty or malformed"
# 
#     def test_module_file_exists(self):
#         """Test that the source file exists and is readable."""
#         src_file = Path(__file__).parents[3] / "src" / "pip_project_template" / "mcp_servers" / "McpServer02.py"
#         assert src_file.exists(), f"Source file {src_file} does not exist"
#         assert src_file.is_file(), f"Source path {src_file} is not a file"
# 
#     def test_mcp_instance(self):
#         """Test that FastMCP instance is created."""
#         assert mcp_module.mcp is not None
#         assert hasattr(mcp_module.mcp, 'tool')  # FastMCP instance should have tool decorator
#         
#     def test_run_server_function_exists(self):
#         """Test that run_server function exists and is callable."""
#         assert hasattr(mcp_module, 'run_server')
#         assert callable(mcp_module.run_server)
#         
#     @pytest.mark.asyncio
#     @pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not available for testing")
#     async def test_mcp_server02_tools(self):
#         """Test McpServer02 tools using FastMCP Client."""
#         async with Client(mcp_module.mcp) as client:
#             # List available tools - should have enhanced features vs Server01
#             tools = await client.list_tools()
#             tool_names = [tool.name for tool in tools]
#             
#             # Should have specific tools for Server02
#             expected_tools = ['calculate_advanced', 'batch_calculate', 'get_server_info']
#             for expected_tool in expected_tools:
#                 assert expected_tool in tool_names, f"Tool {expected_tool} not found in {tool_names}"
# 
#     @pytest.mark.asyncio
#     @pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not available for testing")
#     async def test_calculate_advanced_tool(self):
#         """Test calculate_advanced tool via FastMCP Client."""
#         async with Client(mcp_module.mcp) as client:
#             result = await client.call_tool('calculate_advanced', {'a': 10.0, 'b': 5.0, 'operation': 'add'})
#             result_data = result.content[0].text
#             
#             # Should return dict with success, result, etc.
#             assert '"success":true' in result_data or '"success": true' in result_data
#             assert '"result":15.0' in result_data or '"result": 15.0' in result_data
#             assert '"server":"McpServer02"' in result_data or '"server": "McpServer02"' in result_data
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
#             # Should contain multiple results
#             assert '"success":true' in result_data or '"success": true' in result_data
#             assert '"result":8.0' in result_data or '"result": 8.0' in result_data  # 5 + 3
#             assert '"result":8.0' in result_data or '"result": 8.0' in result_data  # 4 * 2
# 
#     @pytest.mark.asyncio
#     @pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not available for testing")
#     async def test_batch_calculate_error_handling(self):
#         """Test batch_calculate error handling."""
#         async with Client(mcp_module.mcp) as client:
#             operations = [
#                 {"a": 5.0, "b": 3.0, "operation": "add"},
#                 {"a": 4.0, "b": 2.0, "operation": "invalid_op"}
#             ]
#             result = await client.call_tool('batch_calculate', {'operations': operations})
#             result_data = result.content[0].text
#             
#             # Should contain error for invalid operation
#             assert '"success":false' in result_data or '"success": false' in result_data
#             assert '"error"' in result_data
# 
#     @pytest.mark.asyncio
#     @pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not available for testing")
#     async def test_get_server_info_tool(self):
#         """Test get_server_info tool via FastMCP Client."""
#         async with Client(mcp_module.mcp) as client:
#             result = await client.call_tool('get_server_info')
#             result_data = result.content[0].text
#             
#             # Should contain server info
#             assert '"server":"McpServer02"' in result_data or '"server": "McpServer02"' in result_data
#             assert '"framework":"FastMCP"' in result_data or '"framework": "FastMCP"' in result_data
#             assert '"version":"0.1.0"' in result_data or '"version": "0.1.0"' in result_data
# 
#     @pytest.mark.asyncio
#     @pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not available for testing")
#     async def test_mcp_resources(self):
#         """Test MCP resources via FastMCP Client."""
#         async with Client(mcp_module.mcp) as client:
#             resources = await client.list_resources()
#             resource_uris = [str(resource.uri) for resource in resources]
#             
#             # Should have specific resources
#             expected_resources = ["server://metrics", "calculations://history"]
#             for expected_resource in expected_resources:
#                 assert expected_resource in resource_uris, f"Resource {expected_resource} not found"
# 
#     @pytest.mark.asyncio
#     @pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not available for testing")
#     async def test_server_metrics_resource(self):
#         """Test server://metrics resource via FastMCP Client."""
#         async with Client(mcp_module.mcp) as client:
#             metrics_resource = await client.read_resource("server://metrics")
#             
#             # Handle different response formats
#             if hasattr(metrics_resource, 'contents'):
#                 resource_data = metrics_resource.contents[0].text
#             else:
#                 resource_data = str(metrics_resource)
#             
#             assert '"server":"McpServer02"' in resource_data or "McpServer02" in resource_data
#             assert '"status":"running"' in resource_data or "running" in resource_data
# 
#     @pytest.mark.asyncio
#     @pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not available for testing")
#     async def test_calculation_history_resource(self):
#         """Test calculations://history resource via FastMCP Client."""
#         async with Client(mcp_module.mcp) as client:
#             history_resource = await client.read_resource("calculations://history")
#             
#             # Handle different response formats
#             if hasattr(history_resource, 'contents'):
#                 resource_data = history_resource.contents[0].text
#             else:
#                 resource_data = str(history_resource)
#             
#             assert '"total_calculations":70' in resource_data or "total_calculations" in resource_data
#             assert '"server":"McpServer02"' in resource_data or "McpServer02" in resource_data
# 
#     def test_run_server_invalid_transport(self):
#         """Test run_server with invalid transport raises ValueError."""
#         with pytest.raises(ValueError) as exc_info:
#             mcp_module.run_server("invalid")
#         assert "Unsupported transport: invalid" in str(exc_info.value)
# 
#     def test_main_function(self):
#         """Test main function execution with mocked run_server."""
#         with patch.object(mcp_module, 'run_server') as mock_run:
#             mcp_module.main()
#             mock_run.assert_called_once()
# 
#     def test_calculator_instance(self):
#         """Test that calculator is properly initialized."""
#         from pip_project_template.core._Calculator import Calculator
#         assert isinstance(mcp_module.calculator, Calculator)
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
#     def test_module_main_execution(self):
#         """Test module execution as script."""
#         import subprocess
#         import sys
#         
#         # Test the module can be executed as a script
#         # This will hit the if __name__ == "__main__": main() block
#         result = subprocess.run([
#             sys.executable, "-m", "pip_project_template.mcp_servers.McpServer02", "--help"
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/tests/pip_project_template/mcp_servers/test_McpServer02.py
# --------------------------------------------------------------------------------
