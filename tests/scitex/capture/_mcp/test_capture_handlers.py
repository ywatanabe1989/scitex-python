#!/usr/bin/env python3
"""Tests for capture MCP handlers."""

import pytest


class TestCaptureScreenshotHandler:
    """Tests for capture_screenshot_handler."""

    @pytest.mark.asyncio
    async def test_capture_screenshot_returns_dict(self):
        """Test that handler returns dict with success key."""
        from scitex.capture._mcp.handlers import capture_screenshot_handler

        result = await capture_screenshot_handler(return_base64=False)
        assert isinstance(result, dict)
        assert "success" in result

    @pytest.mark.asyncio
    async def test_capture_screenshot_with_base64(self):
        """Test base64 return option."""
        from scitex.capture._mcp.handlers import capture_screenshot_handler

        result = await capture_screenshot_handler(return_base64=True)
        assert isinstance(result, dict)
        assert "success" in result


class TestMonitoringHandlers:
    """Tests for monitoring-related handlers."""

    @pytest.mark.asyncio
    async def test_get_monitoring_status(self):
        """Test monitoring status handler."""
        from scitex.capture._mcp.handlers import get_monitoring_status_handler

        result = await get_monitoring_status_handler()
        assert isinstance(result, dict)
        assert "success" in result

    @pytest.mark.asyncio
    async def test_stop_monitoring_when_not_running(self):
        """Test stop monitoring when not running."""
        from scitex.capture._mcp.handlers import stop_monitoring_handler

        result = await stop_monitoring_handler()
        assert isinstance(result, dict)


class TestListRecentScreenshotsHandler:
    """Tests for list_recent_screenshots_handler."""

    @pytest.mark.asyncio
    async def test_list_recent_screenshots_default(self):
        """Test listing screenshots with defaults."""
        from scitex.capture._mcp.handlers import list_recent_screenshots_handler

        result = await list_recent_screenshots_handler()
        assert isinstance(result, dict)
        assert "success" in result

    @pytest.mark.asyncio
    async def test_list_recent_screenshots_with_limit(self):
        """Test listing screenshots with custom limit."""
        from scitex.capture._mcp.handlers import list_recent_screenshots_handler

        result = await list_recent_screenshots_handler(limit=5)
        assert isinstance(result, dict)
        assert "success" in result


class TestListSessionsHandler:
    """Tests for list_sessions_handler."""

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        """Test listing monitoring sessions."""
        from scitex.capture._mcp.handlers import list_sessions_handler

        result = await list_sessions_handler(limit=10)
        assert isinstance(result, dict)
        assert "success" in result


class TestGetInfoHandler:
    """Tests for get_info_handler."""

    @pytest.mark.asyncio
    async def test_get_info(self):
        """Test getting system info."""
        from scitex.capture._mcp.handlers import get_info_handler

        result = await get_info_handler()
        assert isinstance(result, dict)
        assert "success" in result


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
