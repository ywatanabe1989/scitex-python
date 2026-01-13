#!/usr/bin/env python3
# Timestamp: "2026-01-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/ui/test__mcp_handlers.py

"""Tests for scitex.ui MCP handlers."""

from __future__ import annotations

import pytest


class TestNotifyHandler:
    """Tests for notify_handler."""

    @pytest.mark.asyncio
    async def test_notify_handler_returns_dict(self):
        from scitex.ui._mcp.handlers import notify_handler

        result = await notify_handler(message="Test message")
        assert isinstance(result, dict)
        assert "success" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_notify_handler_with_backend(self):
        from scitex.ui._mcp.handlers import notify_handler

        result = await notify_handler(message="Test", backend="audio")
        assert result["backends_used"] == ["audio"]

    @pytest.mark.asyncio
    async def test_notify_handler_with_multiple_backends(self):
        from scitex.ui._mcp.handlers import notify_handler

        result = await notify_handler(message="Test", backends=["audio", "emacs"])
        assert result["backends_used"] == ["audio", "emacs"]
        assert result["total_count"] == 2

    @pytest.mark.asyncio
    async def test_notify_handler_invalid_backend(self):
        from scitex.ui._mcp.handlers import notify_handler

        result = await notify_handler(message="Test", backend="invalid_backend_xyz")
        assert result["success"] is False
        assert any(
            "Unknown backend" in r.get("error", "") for r in result.get("results", [])
        )

    @pytest.mark.asyncio
    async def test_notify_handler_with_title(self):
        from scitex.ui._mcp.handlers import notify_handler

        result = await notify_handler(
            message="Test message", title="Test Title", backend="audio"
        )
        assert result["title"] == "Test Title"

    @pytest.mark.asyncio
    async def test_notify_handler_with_level(self):
        from scitex.ui._mcp.handlers import notify_handler

        result = await notify_handler(message="Test", level="warning", backend="audio")
        assert result["level"] == "warning"

    @pytest.mark.asyncio
    async def test_notify_handler_invalid_level_defaults_to_info(self):
        from scitex.ui._mcp.handlers import notify_handler

        result = await notify_handler(
            message="Test", level="invalid_level", backend="audio"
        )
        # Should still succeed (defaults to INFO)
        assert "level" in result


class TestNotifyByLevelHandler:
    """Tests for notify_by_level_handler."""

    @pytest.mark.asyncio
    async def test_notify_by_level_returns_dict(self):
        from scitex.ui._mcp.handlers import notify_by_level_handler

        result = await notify_by_level_handler(message="Test")
        assert isinstance(result, dict)
        assert "success" in result

    @pytest.mark.asyncio
    async def test_notify_by_level_uses_config_backends(self):
        from scitex.ui._mcp.handlers import notify_by_level_handler

        result = await notify_by_level_handler(message="Test", level="info")
        # Should use backends configured for info level
        assert "backends_used" in result
        assert isinstance(result["backends_used"], list)

    @pytest.mark.asyncio
    async def test_notify_by_level_warning(self):
        from scitex.ui._mcp.handlers import notify_by_level_handler

        result = await notify_by_level_handler(message="Test", level="warning")
        assert "backends_used" in result


class TestListBackendsHandler:
    """Tests for list_backends_handler."""

    @pytest.mark.asyncio
    async def test_list_backends_returns_dict(self):
        from scitex.ui._mcp.handlers import list_backends_handler

        result = await list_backends_handler()
        assert isinstance(result, dict)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_list_backends_contains_backends(self):
        from scitex.ui._mcp.handlers import list_backends_handler

        result = await list_backends_handler()
        assert "backends" in result
        assert isinstance(result["backends"], list)
        assert len(result["backends"]) > 0

    @pytest.mark.asyncio
    async def test_list_backends_includes_expected_backends(self):
        from scitex.ui._mcp.handlers import list_backends_handler

        result = await list_backends_handler()
        backend_names = [b["name"] for b in result["backends"]]
        assert "audio" in backend_names
        assert "email" in backend_names
        assert "desktop" in backend_names
        assert "emacs" in backend_names

    @pytest.mark.asyncio
    async def test_list_backends_has_counts(self):
        from scitex.ui._mcp.handlers import list_backends_handler

        result = await list_backends_handler()
        assert "total_count" in result
        assert "available_count" in result
        assert result["total_count"] >= result["available_count"]


class TestAvailableBackendsHandler:
    """Tests for available_backends_handler."""

    @pytest.mark.asyncio
    async def test_available_backends_returns_dict(self):
        from scitex.ui._mcp.handlers import available_backends_handler

        result = await available_backends_handler()
        assert isinstance(result, dict)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_available_backends_returns_list(self):
        from scitex.ui._mcp.handlers import available_backends_handler

        result = await available_backends_handler()
        assert "available_backends" in result
        assert isinstance(result["available_backends"], list)

    @pytest.mark.asyncio
    async def test_available_backends_includes_audio(self):
        from scitex.ui._mcp.handlers import available_backends_handler

        result = await available_backends_handler()
        # Audio should always be available
        assert "audio" in result["available_backends"]


class TestGetConfigHandler:
    """Tests for get_config_handler."""

    @pytest.mark.asyncio
    async def test_get_config_returns_dict(self):
        from scitex.ui._mcp.handlers import get_config_handler

        result = await get_config_handler()
        assert isinstance(result, dict)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_get_config_has_config_section(self):
        from scitex.ui._mcp.handlers import get_config_handler

        result = await get_config_handler()
        assert "config" in result
        config = result["config"]
        assert "default_backend" in config
        assert "backend_priority" in config
        assert "level_backends" in config

    @pytest.mark.asyncio
    async def test_get_config_level_backends_structure(self):
        from scitex.ui._mcp.handlers import get_config_handler

        result = await get_config_handler()
        level_backends = result["config"]["level_backends"]
        assert "info" in level_backends
        assert "warning" in level_backends
        assert "error" in level_backends
        assert "critical" in level_backends

    @pytest.mark.asyncio
    async def test_get_config_has_first_available(self):
        from scitex.ui._mcp.handlers import get_config_handler

        result = await get_config_handler()
        assert "first_available" in result["config"]

    @pytest.mark.asyncio
    async def test_get_config_has_timeouts(self):
        from scitex.ui._mcp.handlers import get_config_handler

        result = await get_config_handler()
        assert "timeouts" in result["config"]


class TestToolSchemas:
    """Tests for MCP tool schemas."""

    def test_get_tool_schemas_returns_list(self):
        from scitex.ui._mcp.tool_schemas import get_tool_schemas

        schemas = get_tool_schemas()
        assert isinstance(schemas, list)

    def test_tool_schemas_have_names(self):
        from scitex.ui._mcp.tool_schemas import get_tool_schemas

        schemas = get_tool_schemas()
        names = [s.name for s in schemas]
        assert "notify" in names
        assert "notify_by_level" in names
        assert "list_notification_backends" in names
        assert "available_notification_backends" in names
        assert "get_notification_config" in names

    def test_notify_schema_has_required_params(self):
        from scitex.ui._mcp.tool_schemas import get_tool_schemas

        schemas = get_tool_schemas()
        notify_schema = next(s for s in schemas if s.name == "notify")
        assert notify_schema.inputSchema is not None
        # Message should be required
        assert "message" in notify_schema.inputSchema.get("required", [])


class TestMCPServerModule:
    """Tests for MCP server module."""

    def test_mcp_available_flag_exists(self):
        from scitex.ui.mcp_server import MCP_AVAILABLE

        assert isinstance(MCP_AVAILABLE, bool)

    def test_notify_server_class_exists(self):
        from scitex.ui.mcp_server import NotifyServer

        assert NotifyServer is not None

    def test_main_function_exists(self):
        from scitex.ui.mcp_server import main

        assert callable(main)


# EOF
