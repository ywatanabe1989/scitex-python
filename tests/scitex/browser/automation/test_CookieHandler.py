#!/usr/bin/env python3
"""Tests for CookieAutoAcceptor class."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scitex.browser.automation.CookieHandler import CookieAutoAcceptor


class TestCookieAutoAcceptorInit:
    """Tests for CookieAutoAcceptor initialization."""

    def test_init_creates_instance(self):
        """CookieAutoAcceptor should initialize without errors."""
        acceptor = CookieAutoAcceptor()
        assert acceptor is not None
        assert acceptor.name == "CookieAutoAcceptor"

    def test_init_sets_cookie_texts(self):
        """Should set cookie_texts list."""
        acceptor = CookieAutoAcceptor()
        assert isinstance(acceptor.cookie_texts, list)
        assert len(acceptor.cookie_texts) > 0

    def test_init_sets_selectors(self):
        """Should set selectors list."""
        acceptor = CookieAutoAcceptor()
        assert isinstance(acceptor.selectors, list)
        assert len(acceptor.selectors) > 0


class TestCookieAutoAcceptorCookieTexts:
    """Tests for cookie_texts list content."""

    def test_contains_accept_all(self):
        """Should contain 'Accept All' text."""
        acceptor = CookieAutoAcceptor()
        assert "Accept All" in acceptor.cookie_texts

    def test_contains_accept(self):
        """Should contain 'Accept' text."""
        acceptor = CookieAutoAcceptor()
        assert "Accept" in acceptor.cookie_texts

    def test_contains_ok(self):
        """Should contain 'OK' text."""
        acceptor = CookieAutoAcceptor()
        assert "OK" in acceptor.cookie_texts

    def test_contains_agree(self):
        """Should contain 'Agree' text."""
        acceptor = CookieAutoAcceptor()
        assert "Agree" in acceptor.cookie_texts

    def test_contains_continue(self):
        """Should contain 'Continue' text."""
        acceptor = CookieAutoAcceptor()
        assert "Continue" in acceptor.cookie_texts

    def test_contains_i_accept(self):
        """Should contain 'I Accept' text."""
        acceptor = CookieAutoAcceptor()
        assert "I Accept" in acceptor.cookie_texts


class TestCookieAutoAcceptorSelectors:
    """Tests for selectors list content."""

    def test_contains_testid_accept_selector(self):
        """Should contain testid accept selector."""
        acceptor = CookieAutoAcceptor()
        assert "[data-testid*='accept']" in acceptor.selectors

    def test_contains_id_accept_selector(self):
        """Should contain id accept selector."""
        acceptor = CookieAutoAcceptor()
        assert "[id*='accept']" in acceptor.selectors

    def test_contains_class_accept_selector(self):
        """Should contain class accept selector."""
        acceptor = CookieAutoAcceptor()
        assert "[class*='accept']" in acceptor.selectors

    def test_contains_aria_label_selector(self):
        """Should contain aria-label selector."""
        acceptor = CookieAutoAcceptor()
        assert "button[aria-label*='Accept']" in acceptor.selectors

    def test_contains_cookie_banner_selector(self):
        """Should contain cookie-banner selector."""
        acceptor = CookieAutoAcceptor()
        assert ".cookie-banner button:first-of-type" in acceptor.selectors


class TestGetAutoAcceptorScript:
    """Tests for get_auto_acceptor_script method."""

    def test_returns_string(self):
        """Should return a string."""
        acceptor = CookieAutoAcceptor()
        script = acceptor.get_auto_acceptor_script()
        assert isinstance(script, str)

    def test_returns_non_empty_script(self):
        """Should return non-empty script."""
        acceptor = CookieAutoAcceptor()
        script = acceptor.get_auto_acceptor_script()
        assert len(script) > 0

    def test_contains_cookie_texts_json(self):
        """Should contain JSON-encoded cookie texts."""
        acceptor = CookieAutoAcceptor()
        script = acceptor.get_auto_acceptor_script()
        # The script should contain the cookie texts as JSON
        for text in acceptor.cookie_texts[:3]:  # Check first few
            assert text in script

    def test_contains_selectors_json(self):
        """Should contain JSON-encoded selectors."""
        acceptor = CookieAutoAcceptor()
        script = acceptor.get_auto_acceptor_script()
        # The script should contain the selectors
        for selector in acceptor.selectors[:2]:  # Check first few
            assert selector in script

    def test_contains_accept_function(self):
        """Should contain acceptCookies function."""
        acceptor = CookieAutoAcceptor()
        script = acceptor.get_auto_acceptor_script()
        assert "function acceptCookies()" in script

    def test_contains_interval_setup(self):
        """Should contain setInterval for periodic checking."""
        acceptor = CookieAutoAcceptor()
        script = acceptor.get_auto_acceptor_script()
        assert "setInterval" in script

    def test_contains_timeout_cleanup(self):
        """Should contain setTimeout for cleanup."""
        acceptor = CookieAutoAcceptor()
        script = acceptor.get_auto_acceptor_script()
        assert "setTimeout" in script
        assert "30000" in script  # 30 second timeout

    def test_skips_scitex_buttons(self):
        """Should skip SciTeX buttons to avoid interfering."""
        acceptor = CookieAutoAcceptor()
        script = acceptor.get_auto_acceptor_script()
        assert "data-scitex-no-auto-click" in script
        assert "scitex" in script.lower()

    def test_contains_queryselector(self):
        """Should use querySelector to find elements."""
        acceptor = CookieAutoAcceptor()
        script = acceptor.get_auto_acceptor_script()
        assert "querySelectorAll" in script

    def test_checks_element_visibility(self):
        """Should check element visibility with offsetParent."""
        acceptor = CookieAutoAcceptor()
        script = acceptor.get_auto_acceptor_script()
        assert "offsetParent" in script


class TestInjectAutoAcceptorAsync:
    """Tests for inject_auto_acceptor_async method."""

    @pytest.mark.asyncio
    async def test_calls_add_init_script(self):
        """Should call context.add_init_script with script."""
        acceptor = CookieAutoAcceptor()
        mock_context = MagicMock()
        mock_context.add_init_script = AsyncMock()

        await acceptor.inject_auto_acceptor_async(mock_context)

        mock_context.add_init_script.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_script_to_context(self):
        """Should pass auto-acceptor script to context."""
        acceptor = CookieAutoAcceptor()
        mock_context = MagicMock()
        mock_context.add_init_script = AsyncMock()

        await acceptor.inject_auto_acceptor_async(mock_context)

        call_args = mock_context.add_init_script.call_args
        script = call_args[0][0]
        assert "acceptCookies" in script

    @pytest.mark.asyncio
    async def test_logs_warning_about_deprecated(self):
        """Should log warning about using get_auto_acceptor_script instead."""
        acceptor = CookieAutoAcceptor()
        mock_context = MagicMock()
        mock_context.add_init_script = AsyncMock()

        with patch("scitex.browser.automation.CookieHandler.logger") as mock_logger:
            await acceptor.inject_auto_acceptor_async(mock_context)
            mock_logger.warning.assert_called()


class TestCheckCookieBannerExistsAsync:
    """Tests for check_cookie_banner_exists_async method."""

    @pytest.mark.asyncio
    async def test_returns_true_when_visible(self):
        """Should return True when cookie banner is visible."""
        acceptor = CookieAutoAcceptor()
        mock_page = MagicMock()
        mock_locator = MagicMock()
        mock_locator.first.is_visible = AsyncMock(return_value=True)
        mock_page.locator.return_value = mock_locator

        result = await acceptor.check_cookie_banner_exists_async(mock_page)

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_visible(self):
        """Should return False when cookie banner is not visible."""
        acceptor = CookieAutoAcceptor()
        mock_page = MagicMock()
        mock_locator = MagicMock()
        mock_locator.first.is_visible = AsyncMock(return_value=False)
        mock_page.locator.return_value = mock_locator

        result = await acceptor.check_cookie_banner_exists_async(mock_page)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self):
        """Should return False when exception occurs."""
        acceptor = CookieAutoAcceptor()
        mock_page = MagicMock()
        mock_page.locator.side_effect = Exception("Locator failed")

        result = await acceptor.check_cookie_banner_exists_async(mock_page)

        assert result is False

    @pytest.mark.asyncio
    async def test_uses_correct_selectors(self):
        """Should use cookie-banner and class*='cookie' selectors."""
        acceptor = CookieAutoAcceptor()
        mock_page = MagicMock()
        mock_locator = MagicMock()
        mock_locator.first.is_visible = AsyncMock(return_value=True)
        mock_page.locator.return_value = mock_locator

        await acceptor.check_cookie_banner_exists_async(mock_page)

        call_args = mock_page.locator.call_args
        selector = call_args[0][0]
        assert ".cookie-banner" in selector
        assert "[class*='cookie']" in selector


class TestCookieAutoAcceptorIntegration:
    """Integration tests for CookieAutoAcceptor."""

    def test_multiple_instances_independent(self):
        """Multiple instances should be independent."""
        acceptor1 = CookieAutoAcceptor()
        acceptor2 = CookieAutoAcceptor()

        # Add custom text to one
        acceptor1.cookie_texts.append("Custom Text")

        # Other should not have it
        assert "Custom Text" not in acceptor2.cookie_texts

    def test_script_is_valid_javascript_iife(self):
        """Script should be a valid IIFE pattern."""
        acceptor = CookieAutoAcceptor()
        script = acceptor.get_auto_acceptor_script()

        # Should start with IIFE pattern
        assert "(() => {" in script
        # Should end with IIFE invocation
        assert "})();" in script

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete cookie auto-acceptor workflow."""
        acceptor = CookieAutoAcceptor()

        # Verify initialization
        assert acceptor.name == "CookieAutoAcceptor"
        assert len(acceptor.cookie_texts) > 0
        assert len(acceptor.selectors) > 0

        # Get script
        script = acceptor.get_auto_acceptor_script()
        assert "acceptCookies" in script

        # Mock context injection
        mock_context = MagicMock()
        mock_context.add_init_script = AsyncMock()
        await acceptor.inject_auto_acceptor_async(mock_context)
        mock_context.add_init_script.assert_called_once()

        # Mock page banner check
        mock_page = MagicMock()
        mock_locator = MagicMock()
        mock_locator.first.is_visible = AsyncMock(return_value=False)
        mock_page.locator.return_value = mock_locator

        banner_exists = await acceptor.check_cookie_banner_exists_async(mock_page)
        assert banner_exists is False


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
