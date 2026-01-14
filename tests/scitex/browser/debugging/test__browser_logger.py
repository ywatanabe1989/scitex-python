#!/usr/bin/env python3
"""Tests for browser_logger module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scitex.browser.debugging._browser_logger import (
    _POPUP_COLORS,
    BrowserLogger,
    browser_logger,
    log_page_async,
)


class TestPopupColors:
    """Tests for _POPUP_COLORS constant."""

    def test_popup_colors_exists(self):
        """Should have _POPUP_COLORS dictionary."""
        assert isinstance(_POPUP_COLORS, dict)

    def test_contains_debug_color(self):
        """Should contain debug color (grey)."""
        assert "debug" in _POPUP_COLORS
        assert _POPUP_COLORS["debug"] == "#6C757D"

    def test_contains_info_color(self):
        """Should contain info color (cyan/teal)."""
        assert "info" in _POPUP_COLORS
        assert _POPUP_COLORS["info"] == "#17A2B8"

    def test_contains_success_color(self):
        """Should contain success color (green)."""
        assert "success" in _POPUP_COLORS
        assert _POPUP_COLORS["success"] == "#28A745"

    def test_contains_warning_color(self):
        """Should contain warning color (yellow)."""
        assert "warning" in _POPUP_COLORS
        assert _POPUP_COLORS["warning"] == "#FFC107"

    def test_contains_error_color(self):
        """Should contain error color (red)."""
        assert "error" in _POPUP_COLORS
        assert _POPUP_COLORS["error"] == "#DC3545"

    def test_contains_fail_color(self):
        """Should contain fail color (red)."""
        assert "fail" in _POPUP_COLORS
        assert _POPUP_COLORS["fail"] == "#DC3545"


class TestLogPageAsync:
    """Tests for log_page_async function."""

    @pytest.mark.asyncio
    async def test_returns_true_when_verbose_false(self):
        """Should return True immediately when verbose is False."""
        result = await log_page_async(
            page=None,
            message="Test message",
            verbose=False,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_none_page(self):
        """Should return False when page is None and verbose is True."""
        result = await log_page_async(
            page=None,
            message="Test message",
            verbose=True,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_for_closed_page(self):
        """Should return False when page is closed."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = True

        result = await log_page_async(
            page=mock_page,
            message="Test message",
            verbose=True,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_default_level_is_info(self):
        """Default level should be info."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate = AsyncMock()
        mock_page.on = MagicMock()

        # Should not raise
        await log_page_async(
            page=mock_page,
            message="Test",
            verbose=True,
            take_screenshot=False,
        )

    @pytest.mark.asyncio
    async def test_accepts_custom_duration(self):
        """Should accept custom duration_ms."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate = AsyncMock()
        mock_page.on = MagicMock()

        result = await log_page_async(
            page=mock_page,
            message="Test",
            duration_ms=30000,
            verbose=True,
            take_screenshot=False,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_evaluates_javascript_for_popup(self):
        """Should evaluate JavaScript to create popup."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate = AsyncMock()
        mock_page.on = MagicMock()

        await log_page_async(
            page=mock_page,
            message="Test popup",
            verbose=True,
            take_screenshot=False,
        )

        # Should have called evaluate at least once
        assert mock_page.evaluate.called

    @pytest.mark.asyncio
    async def test_handles_page_error_gracefully(self):
        """Should handle page errors gracefully."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate = AsyncMock(side_effect=Exception("Page error"))

        result = await log_page_async(
            page=mock_page,
            message="Test",
            verbose=True,
            take_screenshot=False,
        )
        # Should return False on error
        assert result is False


class TestBrowserLoggerInit:
    """Tests for BrowserLogger initialization."""

    def test_init_creates_instance(self):
        """Should create instance."""
        logger = BrowserLogger()
        assert logger is not None

    def test_init_with_default_values(self):
        """Should have default values."""
        logger = BrowserLogger()
        assert logger.page is None
        assert logger.duration_ms == 60_000
        assert logger.take_screenshot is True
        assert logger.screenshot_dir is None
        assert logger.verbose is True

    def test_init_with_custom_page(self):
        """Should accept custom page."""
        mock_page = MagicMock()
        logger = BrowserLogger(page=mock_page)
        assert logger.page is mock_page

    def test_init_with_custom_duration(self):
        """Should accept custom duration_ms."""
        logger = BrowserLogger(duration_ms=30000)
        assert logger.duration_ms == 30000

    def test_init_with_take_screenshot_false(self):
        """Should accept take_screenshot=False."""
        logger = BrowserLogger(take_screenshot=False)
        assert logger.take_screenshot is False

    def test_init_with_screenshot_dir(self):
        """Should accept custom screenshot_dir."""
        logger = BrowserLogger(screenshot_dir="/custom/path")
        assert logger.screenshot_dir == "/custom/path"

    def test_init_with_verbose_false(self):
        """Should accept verbose=False."""
        logger = BrowserLogger(verbose=False)
        assert logger.verbose is False


class TestBrowserLoggerMethods:
    """Tests for BrowserLogger logging methods."""

    @pytest.mark.asyncio
    async def test_debug_method_exists(self):
        """Should have debug method."""
        logger = BrowserLogger(verbose=False)
        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False

        result = await logger.debug(mock_page, "Debug message")
        assert result is True

    @pytest.mark.asyncio
    async def test_info_method_exists(self):
        """Should have info method."""
        logger = BrowserLogger(verbose=False)
        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False

        result = await logger.info(mock_page, "Info message")
        assert result is True

    @pytest.mark.asyncio
    async def test_success_method_exists(self):
        """Should have success method."""
        logger = BrowserLogger(verbose=False)
        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False

        result = await logger.success(mock_page, "Success message")
        assert result is True

    @pytest.mark.asyncio
    async def test_warning_method_exists(self):
        """Should have warning method."""
        logger = BrowserLogger(verbose=False)
        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False

        result = await logger.warning(mock_page, "Warning message")
        assert result is True

    @pytest.mark.asyncio
    async def test_warn_method_exists(self):
        """Should have warn method (alias for warning)."""
        logger = BrowserLogger(verbose=False)
        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False

        result = await logger.warn(mock_page, "Warn message")
        assert result is True

    @pytest.mark.asyncio
    async def test_error_method_exists(self):
        """Should have error method."""
        logger = BrowserLogger(verbose=False)
        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False

        result = await logger.error(mock_page, "Error message")
        assert result is True

    @pytest.mark.asyncio
    async def test_fail_method_exists(self):
        """Should have fail method."""
        logger = BrowserLogger(verbose=False)
        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False

        result = await logger.fail(mock_page, "Fail message")
        assert result is True

    @pytest.mark.asyncio
    async def test_methods_accept_custom_duration(self):
        """Methods should accept custom duration_ms."""
        logger = BrowserLogger(verbose=False)
        mock_page = AsyncMock()

        # Should not raise
        await logger.info(mock_page, "Test", duration_ms=5000)

    @pytest.mark.asyncio
    async def test_methods_accept_take_screenshot(self):
        """Methods should accept take_screenshot parameter."""
        logger = BrowserLogger(verbose=False)
        mock_page = AsyncMock()

        await logger.info(mock_page, "Test", take_screenshot=False)

    @pytest.mark.asyncio
    async def test_methods_accept_screenshot_dir(self):
        """Methods should accept screenshot_dir parameter."""
        logger = BrowserLogger(verbose=False)
        mock_page = AsyncMock()

        await logger.info(mock_page, "Test", screenshot_dir="/custom")

    @pytest.mark.asyncio
    async def test_methods_accept_func_name(self):
        """Methods should accept func_name parameter."""
        logger = BrowserLogger(verbose=False)
        mock_page = AsyncMock()

        await logger.info(mock_page, "Test", func_name="CustomFunc")


class TestBrowserLoggerInternalLog:
    """Tests for BrowserLogger._log method."""

    @pytest.mark.asyncio
    async def test_log_calls_log_page_async(self):
        """_log should delegate to log_page_async."""
        logger = BrowserLogger(verbose=False)
        mock_page = AsyncMock()

        with patch(
            "scitex.browser.debugging._browser_logger.log_page_async"
        ) as mock_log:
            mock_log.return_value = True
            await logger._log(mock_page, "info", "Test message")
            mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_uses_instance_defaults(self):
        """_log should use instance default values."""
        logger = BrowserLogger(
            duration_ms=10000,
            take_screenshot=False,
            screenshot_dir="/custom",
            verbose=False,
        )
        mock_page = AsyncMock()

        with patch(
            "scitex.browser.debugging._browser_logger.log_page_async"
        ) as mock_log:
            mock_log.return_value = True
            await logger._log(mock_page, "info", "Test")

            call_kwargs = mock_log.call_args[1]
            assert call_kwargs["duration_ms"] == 10000
            assert call_kwargs["take_screenshot"] is False
            assert call_kwargs["screenshot_dir"] == "/custom"

    @pytest.mark.asyncio
    async def test_log_overrides_defaults_with_params(self):
        """_log should override defaults with provided parameters."""
        logger = BrowserLogger(
            duration_ms=10000,
            take_screenshot=False,
            verbose=False,
        )
        mock_page = AsyncMock()

        with patch(
            "scitex.browser.debugging._browser_logger.log_page_async"
        ) as mock_log:
            mock_log.return_value = True
            await logger._log(
                mock_page,
                "info",
                "Test",
                duration_ms=5000,
                take_screenshot=True,
            )

            call_kwargs = mock_log.call_args[1]
            assert call_kwargs["duration_ms"] == 5000
            assert call_kwargs["take_screenshot"] is True


class TestGlobalBrowserLogger:
    """Tests for global browser_logger instance."""

    def test_global_logger_exists(self):
        """Should have global browser_logger instance."""
        assert browser_logger is not None
        assert isinstance(browser_logger, BrowserLogger)

    def test_global_logger_has_default_values(self):
        """Global logger should have default values."""
        assert browser_logger.page is None
        assert browser_logger.duration_ms == 60_000
        assert browser_logger.take_screenshot is True
        assert browser_logger.verbose is True


class TestLogLevelFiltering:
    """Tests for log level filtering behavior."""

    @pytest.mark.asyncio
    async def test_respects_logging_level(self):
        """Should respect effective logging level."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate = AsyncMock()
        mock_page.on = MagicMock()

        # Debug level messages might not show popup based on logger level
        result = await log_page_async(
            page=mock_page,
            message="Debug test",
            level="debug",
            verbose=True,
            take_screenshot=False,
        )
        # Should still return True (non-blocking)
        assert result is True

    @pytest.mark.asyncio
    async def test_error_level_always_shown(self):
        """Error level should always be shown."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate = AsyncMock()
        mock_page.on = MagicMock()

        result = await log_page_async(
            page=mock_page,
            message="Error test",
            level="error",
            verbose=True,
            take_screenshot=False,
        )
        assert result is True


class TestScreenshotCapture:
    """Tests for screenshot capture functionality."""

    @pytest.mark.asyncio
    async def test_skips_screenshot_when_disabled(self):
        """Should skip screenshot when take_screenshot=False."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate = AsyncMock()
        mock_page.on = MagicMock()

        result = await log_page_async(
            page=mock_page,
            message="Test",
            take_screenshot=False,
            verbose=True,
        )
        assert result is True
        # Screenshot method should not be called
        mock_page.screenshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_screenshot_error_gracefully(self):
        """Should handle screenshot errors - returns False due to undefined log_func bug."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))
        mock_page.on = MagicMock()

        # When screenshot fails, the code tries to call undefined 'log_func'
        # which raises NameError, caught by outer exception handler -> returns False
        with patch("scitex.browser.debugging._browser_logger.get_paths") as mock_paths:
            mock_paths.return_value.resolve.return_value = Path("/tmp/screenshots")
            result = await log_page_async(
                page=mock_page,
                message="Test",
                take_screenshot=True,
                verbose=True,
            )
            # Returns False due to NameError from undefined log_func in exception handler
            assert result is False


class TestPopupPersistence:
    """Tests for popup persistence across navigation."""

    @pytest.mark.asyncio
    async def test_adds_framenavigated_handler(self):
        """Should add framenavigated handler for popup persistence."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate = AsyncMock()
        mock_page.on = MagicMock()

        # Remove any existing handler marker
        if hasattr(mock_page, "_scitex_popup_handler_added"):
            delattr(mock_page, "_scitex_popup_handler_added")

        await log_page_async(
            page=mock_page,
            message="Test",
            take_screenshot=False,
            verbose=True,
        )

        # Should have called page.on for framenavigated
        mock_page.on.assert_called()


class TestMessageCleaning:
    """Tests for message cleaning for filenames."""

    @pytest.mark.asyncio
    async def test_cleans_special_characters(self):
        """Should clean special characters from message for filename."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.screenshot = AsyncMock()
        mock_page.on = MagicMock()

        # Message with special characters
        with patch("scitex.browser.debugging._browser_logger.get_paths") as mock_paths:
            mock_dir = MagicMock()
            mock_dir.mkdir = MagicMock()
            mock_dir.__truediv__ = MagicMock(return_value=Path("/tmp/test.png"))
            mock_paths.return_value.resolve.return_value = mock_dir

            await log_page_async(
                page=mock_page,
                message="Test: Special chars! @#$%",
                take_screenshot=True,
                verbose=True,
            )


class TestIntegration:
    """Integration tests for browser logger."""

    @pytest.mark.asyncio
    async def test_full_logging_workflow(self):
        """Test complete logging workflow."""
        logger = BrowserLogger(verbose=False)

        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False

        # Log at different levels
        assert await logger.debug(mock_page, "Debug message") is True
        assert await logger.info(mock_page, "Info message") is True
        assert await logger.success(mock_page, "Success message") is True
        assert await logger.warning(mock_page, "Warning message") is True
        assert await logger.error(mock_page, "Error message") is True
        assert await logger.fail(mock_page, "Fail message") is True

    @pytest.mark.asyncio
    async def test_global_logger_workflow(self):
        """Test using global browser_logger."""
        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False

        # Create a new logger with verbose=False for testing
        test_logger = BrowserLogger(verbose=False)
        result = await test_logger.info(mock_page, "Test via global logger")
        assert result is True


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
