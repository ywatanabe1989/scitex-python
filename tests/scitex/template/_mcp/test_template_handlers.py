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


class TestGetCodeTemplateHandler:
    """Tests for get_code_template_handler."""

    @pytest.mark.asyncio
    async def test_get_code_template_session(self):
        """Test getting session code template."""
        from scitex.template._mcp.handlers import get_code_template_handler

        result = await get_code_template_handler("session")
        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "content" in result

    @pytest.mark.asyncio
    async def test_get_code_template_all(self):
        """Test getting all code templates combined."""
        from scitex.template._mcp.handlers import get_code_template_handler

        result = await get_code_template_handler("all")
        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "content" in result

    @pytest.mark.asyncio
    async def test_get_code_template_invalid(self):
        """Test getting invalid code template."""
        from scitex.template._mcp.handlers import get_code_template_handler

        result = await get_code_template_handler("nonexistent")
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_code_template_with_filepath(self):
        """Test getting code template with custom filepath."""
        from scitex.template._mcp.handlers import get_code_template_handler

        result = await get_code_template_handler("session", filepath="custom_script.py")
        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "custom_script.py" in result.get("content", "")

    @pytest.mark.asyncio
    async def test_get_code_template_module_usage(self):
        """Test getting module usage templates."""
        from scitex.template._mcp.handlers import get_code_template_handler

        for template_id in [
            "plt",
            "stats",
            "scholar",
            "audio",
            "capture",
            "diagram",
            "canvas",
            "writer",
        ]:
            result = await get_code_template_handler(template_id)
            assert result.get("success") is True, f"Failed for template: {template_id}"
            assert "content" in result


class TestListCodeTemplatesHandler:
    """Tests for list_code_templates_handler."""

    @pytest.mark.asyncio
    async def test_list_code_templates(self):
        """Test listing code templates."""
        from scitex.template._mcp.handlers import list_code_templates_handler

        result = await list_code_templates_handler()
        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "templates" in result

    @pytest.mark.asyncio
    async def test_list_code_templates_contains_expected(self):
        """Test that list contains expected templates."""
        from scitex.template._mcp.handlers import list_code_templates_handler

        result = await list_code_templates_handler()
        if result.get("success"):
            template_ids = [t["id"] for t in result["templates"]]
            expected = ["session", "io", "config", "plt", "stats", "scholar"]
            for tid in expected:
                assert tid in template_ids, f"Template '{tid}' not in list"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
