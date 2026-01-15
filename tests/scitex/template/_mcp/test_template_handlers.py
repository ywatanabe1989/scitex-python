#!/usr/bin/env python3
"""Tests for template MCP handlers."""

import pytest


class TestListTemplatesHandler:
    """Tests for list_templates_handler."""

    @pytest.mark.asyncio
    async def test_list_templates_returns_dict(self):
        """Test that handler returns dict with success key."""
        from scitex.template._mcp.handlers import list_templates_handler

        result = await list_templates_handler()
        assert isinstance(result, dict)
        assert "success" in result

    @pytest.mark.asyncio
    async def test_list_templates_contains_templates(self):
        """Test that result contains templates list."""
        from scitex.template._mcp.handlers import list_templates_handler

        result = await list_templates_handler()
        if result.get("success"):
            assert "templates" in result


class TestGetTemplateInfoHandler:
    """Tests for get_template_info_handler."""

    @pytest.mark.asyncio
    async def test_get_template_info_research(self):
        """Test getting info for research template."""
        from scitex.template._mcp.handlers import get_template_info_handler

        result = await get_template_info_handler("research")
        assert isinstance(result, dict)
        assert "success" in result

    @pytest.mark.asyncio
    async def test_get_template_info_pip_project(self):
        """Test getting info for pip_project template."""
        from scitex.template._mcp.handlers import get_template_info_handler

        result = await get_template_info_handler("pip_project")
        assert isinstance(result, dict)
        assert "success" in result

    @pytest.mark.asyncio
    async def test_get_template_info_invalid(self):
        """Test getting info for invalid template."""
        from scitex.template._mcp.handlers import get_template_info_handler

        result = await get_template_info_handler("nonexistent_template")
        assert isinstance(result, dict)
        assert "success" in result


class TestListGitStrategiesHandler:
    """Tests for list_git_strategies_handler."""

    @pytest.mark.asyncio
    async def test_list_git_strategies(self):
        """Test listing git strategies."""
        from scitex.template._mcp.handlers import list_git_strategies_handler

        result = await list_git_strategies_handler()
        assert isinstance(result, dict)
        assert "success" in result

    @pytest.mark.asyncio
    async def test_list_git_strategies_contains_strategies(self):
        """Test that result contains strategies."""
        from scitex.template._mcp.handlers import list_git_strategies_handler

        result = await list_git_strategies_handler()
        if result.get("success"):
            assert "strategies" in result


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
