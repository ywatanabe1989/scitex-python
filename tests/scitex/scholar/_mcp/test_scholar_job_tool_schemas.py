#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: tests/scitex/scholar/_mcp/test_job_tool_schemas.py
# ----------------------------------------

"""Tests for MCP job tool schemas."""

from __future__ import annotations

import pytest

from scitex.scholar._mcp.job_tool_schemas import get_job_tool_schemas


class TestGetJobToolSchemas:
    """Tests for get_job_tool_schemas function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        schemas = get_job_tool_schemas()
        assert isinstance(schemas, list)

    def test_returns_six_tools(self):
        """Test that exactly 6 job tools are returned."""
        schemas = get_job_tool_schemas()
        assert len(schemas) == 6

    def test_all_have_required_fields(self):
        """Test all tools have name, description, inputSchema."""
        schemas = get_job_tool_schemas()
        for tool in schemas:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "inputSchema")

    def test_tool_names(self):
        """Test expected tool names are present."""
        schemas = get_job_tool_schemas()
        names = {tool.name for tool in schemas}
        expected = {
            "fetch_papers",
            "list_jobs",
            "get_job_status",
            "start_job",
            "cancel_job",
            "get_job_result",
        }
        assert names == expected


class TestFetchPapersSchema:
    """Tests for fetch_papers tool schema."""

    def test_has_papers_property(self):
        """Test papers array property exists."""
        schemas = get_job_tool_schemas()
        fetch = next(t for t in schemas if t.name == "fetch_papers")
        props = fetch.inputSchema["properties"]
        assert "papers" in props
        assert props["papers"]["type"] == "array"

    def test_has_bibtex_path_property(self):
        """Test bibtex_path property exists."""
        schemas = get_job_tool_schemas()
        fetch = next(t for t in schemas if t.name == "fetch_papers")
        props = fetch.inputSchema["properties"]
        assert "bibtex_path" in props
        assert props["bibtex_path"]["type"] == "string"

    def test_has_async_mode_property(self):
        """Test async_mode property exists with default True."""
        schemas = get_job_tool_schemas()
        fetch = next(t for t in schemas if t.name == "fetch_papers")
        props = fetch.inputSchema["properties"]
        assert "async_mode" in props
        assert props["async_mode"]["type"] == "boolean"
        assert props["async_mode"]["default"] is True

    def test_has_browser_mode_enum(self):
        """Test browser_mode has valid enum values."""
        schemas = get_job_tool_schemas()
        fetch = next(t for t in schemas if t.name == "fetch_papers")
        props = fetch.inputSchema["properties"]
        assert "browser_mode" in props
        assert props["browser_mode"]["enum"] == ["stealth", "interactive"]


class TestListJobsSchema:
    """Tests for list_jobs tool schema."""

    def test_has_status_filter(self):
        """Test status filter property exists."""
        schemas = get_job_tool_schemas()
        list_jobs = next(t for t in schemas if t.name == "list_jobs")
        props = list_jobs.inputSchema["properties"]
        assert "status" in props
        assert "enum" in props["status"]

    def test_has_limit_property(self):
        """Test limit property with default."""
        schemas = get_job_tool_schemas()
        list_jobs = next(t for t in schemas if t.name == "list_jobs")
        props = list_jobs.inputSchema["properties"]
        assert "limit" in props
        assert props["limit"]["default"] == 20


class TestJobIdRequiredSchemas:
    """Tests for tools requiring job_id."""

    @pytest.mark.parametrize(
        "tool_name",
        ["get_job_status", "start_job", "cancel_job", "get_job_result"],
    )
    def test_job_id_required(self, tool_name):
        """Test job_id is required for job management tools."""
        schemas = get_job_tool_schemas()
        tool = next(t for t in schemas if t.name == tool_name)
        assert "required" in tool.inputSchema
        assert "job_id" in tool.inputSchema["required"]

    @pytest.mark.parametrize(
        "tool_name",
        ["get_job_status", "start_job", "cancel_job", "get_job_result"],
    )
    def test_has_job_id_property(self, tool_name):
        """Test job_id property exists."""
        schemas = get_job_tool_schemas()
        tool = next(t for t in schemas if t.name == tool_name)
        props = tool.inputSchema["properties"]
        assert "job_id" in props
        assert props["job_id"]["type"] == "string"


class TestToolDescriptions:
    """Tests for tool descriptions."""

    def test_fetch_papers_description(self):
        """Test fetch_papers has meaningful description."""
        schemas = get_job_tool_schemas()
        fetch = next(t for t in schemas if t.name == "fetch_papers")
        assert "async" in fetch.description.lower()
        assert "job_id" in fetch.description.lower()

    def test_list_jobs_description(self):
        """Test list_jobs mentions background jobs."""
        schemas = get_job_tool_schemas()
        list_jobs = next(t for t in schemas if t.name == "list_jobs")
        assert "background" in list_jobs.description.lower()

    def test_get_job_status_description(self):
        """Test get_job_status mentions progress."""
        schemas = get_job_tool_schemas()
        status = next(t for t in schemas if t.name == "get_job_status")
        assert "progress" in status.description.lower()


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])


# EOF
