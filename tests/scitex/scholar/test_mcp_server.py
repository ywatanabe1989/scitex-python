#!/usr/bin/env python3
"""Tests for Scholar MCP server and tool schemas."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestMCPAvailable:
    """Tests for MCP availability handling."""

    def test_mcp_available_is_boolean(self):
        """MCP_AVAILABLE should be a boolean."""
        from scitex.scholar.mcp_server import MCP_AVAILABLE

        assert isinstance(MCP_AVAILABLE, bool)

    @pytest.mark.skipif(
        not pytest.importorskip("mcp", reason="MCP not installed"),
        reason="MCP package required",
    )
    def test_mcp_available_true_when_installed(self):
        """MCP_AVAILABLE should be True when mcp is installed."""
        from scitex.scholar.mcp_server import MCP_AVAILABLE

        assert MCP_AVAILABLE is True


class TestGetScholarDir:
    """Tests for get_scholar_dir function."""

    def test_returns_path(self):
        """get_scholar_dir should return a Path object."""
        from scitex.scholar.mcp_server import get_scholar_dir

        result = get_scholar_dir()
        assert isinstance(result, Path)

    def test_creates_directory(self):
        """get_scholar_dir should create the directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test_scitex"
            with patch.dict(os.environ, {"SCITEX_DIR": str(test_path)}):
                # Need to reload the module to pick up the new env var
                import importlib

                import scitex.scholar.mcp_server as mcp_mod

                importlib.reload(mcp_mod)
                result = mcp_mod.get_scholar_dir()
                assert result.parent.exists()


class TestToolSchemas:
    """Tests for MCP tool schemas."""

    @pytest.fixture
    def tool_schemas(self):
        """Get tool schemas."""
        try:
            from scitex.scholar._mcp.tool_schemas import get_tool_schemas

            return get_tool_schemas()
        except ImportError:
            pytest.skip("MCP not installed")

    def test_returns_list(self, tool_schemas):
        """get_tool_schemas should return a list."""
        assert isinstance(tool_schemas, list)

    def test_has_tools(self, tool_schemas):
        """Should have at least one tool."""
        assert len(tool_schemas) > 0

    def test_each_tool_has_name(self, tool_schemas):
        """Each tool should have a name."""
        for tool in tool_schemas:
            assert hasattr(tool, "name")
            assert tool.name is not None
            assert len(tool.name) > 0

    def test_each_tool_has_description(self, tool_schemas):
        """Each tool should have a description."""
        for tool in tool_schemas:
            assert hasattr(tool, "description")
            assert tool.description is not None
            assert len(tool.description) > 0

    def test_each_tool_has_input_schema(self, tool_schemas):
        """Each tool should have an input schema."""
        for tool in tool_schemas:
            assert hasattr(tool, "inputSchema")
            assert tool.inputSchema is not None
            assert isinstance(tool.inputSchema, dict)

    def test_input_schema_has_type(self, tool_schemas):
        """Each input schema should specify type."""
        for tool in tool_schemas:
            assert "type" in tool.inputSchema
            assert tool.inputSchema["type"] == "object"

    def test_input_schema_has_properties(self, tool_schemas):
        """Each input schema should have properties dict."""
        for tool in tool_schemas:
            assert "properties" in tool.inputSchema
            assert isinstance(tool.inputSchema["properties"], dict)

    def test_core_tools_present(self, tool_schemas):
        """Core tools should be present."""
        tool_names = [t.name for t in tool_schemas]

        expected_tools = [
            "search_papers",
            "resolve_dois",
            "enrich_bibtex",
            "download_pdf",
            "download_pdfs_batch",
            "get_library_status",
            "parse_bibtex",
            "validate_pdfs",
            "resolve_openurls",
            "authenticate",
            "check_auth_status",
            "logout",
            "export_papers",
            "create_project",
            "list_projects",
            "add_papers_to_project",
            "parse_pdf_content",
        ]

        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"


class TestToolSchemaProperties:
    """Tests for specific tool schema properties."""

    @pytest.fixture
    def tool_schemas(self):
        """Get tool schemas as dict by name."""
        try:
            from scitex.scholar._mcp.tool_schemas import get_tool_schemas

            schemas = get_tool_schemas()
            return {t.name: t for t in schemas}
        except ImportError:
            pytest.skip("MCP not installed")

    def test_search_papers_has_query(self, tool_schemas):
        """search_papers should have required query parameter."""
        tool = tool_schemas["search_papers"]
        assert "query" in tool.inputSchema["properties"]
        assert "required" in tool.inputSchema
        assert "query" in tool.inputSchema["required"]

    def test_search_papers_search_mode_enum(self, tool_schemas):
        """search_papers search_mode should have valid enum values."""
        tool = tool_schemas["search_papers"]
        search_mode = tool.inputSchema["properties"]["search_mode"]
        assert "enum" in search_mode
        assert set(search_mode["enum"]) == {"local", "external", "both"}

    def test_download_pdf_has_doi(self, tool_schemas):
        """download_pdf should have required doi parameter."""
        tool = tool_schemas["download_pdf"]
        assert "doi" in tool.inputSchema["properties"]
        assert "doi" in tool.inputSchema["required"]

    def test_download_pdf_auth_method_enum(self, tool_schemas):
        """download_pdf auth_method should have valid enum values."""
        tool = tool_schemas["download_pdf"]
        auth_method = tool.inputSchema["properties"]["auth_method"]
        assert "enum" in auth_method
        assert "openathens" in auth_method["enum"]
        assert "none" in auth_method["enum"]

    def test_authenticate_has_method(self, tool_schemas):
        """authenticate should have required method parameter."""
        tool = tool_schemas["authenticate"]
        assert "method" in tool.inputSchema["properties"]
        assert "method" in tool.inputSchema["required"]

    def test_authenticate_method_enum(self, tool_schemas):
        """authenticate method should have valid enum values."""
        tool = tool_schemas["authenticate"]
        method = tool.inputSchema["properties"]["method"]
        assert "enum" in method
        assert set(method["enum"]) == {"openathens", "shibboleth", "ezproxy"}

    def test_export_papers_format_enum(self, tool_schemas):
        """export_papers format should have valid enum values."""
        tool = tool_schemas["export_papers"]
        fmt = tool.inputSchema["properties"]["format"]
        assert "enum" in fmt
        assert set(fmt["enum"]) == {"bibtex", "ris", "json", "csv"}

    def test_parse_pdf_content_mode_enum(self, tool_schemas):
        """parse_pdf_content mode should have valid enum values."""
        tool = tool_schemas["parse_pdf_content"]
        mode = tool.inputSchema["properties"]["mode"]
        assert "enum" in mode
        expected_modes = {
            "text",
            "sections",
            "tables",
            "images",
            "metadata",
            "pages",
            "scientific",
            "full",
        }
        assert set(mode["enum"]) == expected_modes

    def test_create_project_has_project_name(self, tool_schemas):
        """create_project should have required project_name parameter."""
        tool = tool_schemas["create_project"]
        assert "project_name" in tool.inputSchema["properties"]
        assert "project_name" in tool.inputSchema["required"]


class TestScholarServer:
    """Tests for ScholarServer class."""

    @pytest.fixture
    def mock_mcp(self):
        """Mock MCP module."""
        with patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.types": MagicMock()}):
            yield

    @pytest.mark.skipif(
        not pytest.importorskip("mcp", reason="MCP not installed"),
        reason="MCP package required",
    )
    def test_scholar_server_init(self):
        """ScholarServer should initialize without errors."""
        from scitex.scholar.mcp_server import ScholarServer

        server = ScholarServer()
        assert server is not None
        assert server.server is not None

    @pytest.mark.skipif(
        not pytest.importorskip("mcp", reason="MCP not installed"),
        reason="MCP package required",
    )
    def test_scholar_server_lazy_loads_scholar(self):
        """ScholarServer should lazy-load Scholar instance."""
        from scitex.scholar.mcp_server import ScholarServer

        server = ScholarServer()
        assert server._scholar_instance is None


class TestToolNameConsistency:
    """Tests to ensure tool names are consistent across server."""

    @pytest.mark.skipif(
        not pytest.importorskip("mcp", reason="MCP not installed"),
        reason="MCP package required",
    )
    def test_all_schema_tools_have_handlers(self):
        """All tools in schemas should have corresponding handlers in server."""
        from scitex.scholar._mcp.tool_schemas import get_tool_schemas

        tool_names = {t.name for t in get_tool_schemas()}

        # These tools should be handled in call_tool
        expected_handlers = {
            "search_papers",
            "resolve_dois",
            "enrich_bibtex",
            "download_pdf",
            "download_pdfs_batch",
            "get_library_status",
            "parse_bibtex",
            "validate_pdfs",
            "resolve_openurls",
            "authenticate",
            "check_auth_status",
            "logout",
            "export_papers",
            "create_project",
            "list_projects",
            "add_papers_to_project",
            "parse_pdf_content",
            # Job handlers
            "fetch_papers",
            "list_jobs",
            "get_job_status",
            "start_job",
            "cancel_job",
            "get_job_result",
        }

        # All expected handlers should be in tool names
        for expected in expected_handlers:
            assert expected in tool_names, f"Missing schema for handler: {expected}"


class TestPropertyTypes:
    """Tests for property type validation."""

    @pytest.fixture
    def tool_schemas(self):
        """Get tool schemas as dict by name."""
        try:
            from scitex.scholar._mcp.tool_schemas import get_tool_schemas

            return get_tool_schemas()
        except ImportError:
            pytest.skip("MCP not installed")

    def test_integer_properties_have_type(self, tool_schemas):
        """Integer properties should have type: integer."""
        for tool in tool_schemas:
            for prop_name, prop_def in tool.inputSchema["properties"].items():
                # Skip booleans (bool is subclass of int in Python)
                if "default" in prop_def and isinstance(prop_def["default"], bool):
                    continue
                if "default" in prop_def and isinstance(prop_def["default"], int):
                    assert prop_def.get("type") in ("integer", "number"), (
                        f"{tool.name}.{prop_name} default is int but type is "
                        f"{prop_def.get('type')}"
                    )

    def test_boolean_properties_have_type(self, tool_schemas):
        """Boolean properties should have type: boolean."""
        for tool in tool_schemas:
            for prop_name, prop_def in tool.inputSchema["properties"].items():
                if "default" in prop_def and isinstance(prop_def["default"], bool):
                    assert prop_def.get("type") == "boolean", (
                        f"{tool.name}.{prop_name} default is bool but type is "
                        f"{prop_def.get('type')}"
                    )

    def test_array_properties_have_items(self, tool_schemas):
        """Array properties should have items definition."""
        for tool in tool_schemas:
            for prop_name, prop_def in tool.inputSchema["properties"].items():
                if prop_def.get("type") == "array":
                    assert "items" in prop_def, (
                        f"{tool.name}.{prop_name} is array but missing items"
                    )


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
