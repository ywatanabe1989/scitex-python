#!/usr/bin/env python3
"""Tests for GoogleAuthHelper class."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scitex.browser.auth.google import GoogleAuthHelper, google_login


class TestGoogleAuthHelperInit:
    """Tests for GoogleAuthHelper initialization."""

    def test_init_creates_instance(self):
        """GoogleAuthHelper should initialize without errors."""
        auth = GoogleAuthHelper()
        assert auth is not None

    def test_init_stores_email(self):
        """Should store provided email."""
        auth = GoogleAuthHelper(email="test@gmail.com")
        assert auth.email == "test@gmail.com"

    def test_init_stores_password(self):
        """Should store provided password."""
        auth = GoogleAuthHelper(password="secret123")
        assert auth.password == "secret123"

    def test_init_stores_debug_flag(self):
        """Should store debug flag."""
        auth = GoogleAuthHelper(debug=True)
        assert auth.debug is True

    def test_init_uses_env_email_when_not_provided(self):
        """Should use GOOGLE_EMAIL env var when email not provided."""
        with patch.dict(os.environ, {"GOOGLE_EMAIL": "env@gmail.com"}):
            auth = GoogleAuthHelper()
            assert auth.email == "env@gmail.com"

    def test_init_uses_env_password_when_not_provided(self):
        """Should use GOOGLE_PASSWORD env var when password not provided."""
        with patch.dict(os.environ, {"GOOGLE_PASSWORD": "envpass"}):
            auth = GoogleAuthHelper()
            assert auth.password == "envpass"

    def test_init_uses_env_debug_when_not_provided(self):
        """Should use GOOGLE_AUTH_DEBUG env var when debug not provided."""
        with patch.dict(os.environ, {"GOOGLE_AUTH_DEBUG": "1"}):
            auth = GoogleAuthHelper()
            assert auth.debug is True

    def test_init_prefers_param_over_env(self):
        """Provided params should override env vars."""
        with patch.dict(
            os.environ, {"GOOGLE_EMAIL": "env@gmail.com", "GOOGLE_PASSWORD": "envpass"}
        ):
            auth = GoogleAuthHelper(email="param@gmail.com", password="parampass")
            assert auth.email == "param@gmail.com"
            assert auth.password == "parampass"

    def test_init_defaults_to_empty_strings(self):
        """Should default to empty strings when nothing provided."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_EMAIL", None)
            os.environ.pop("GOOGLE_PASSWORD", None)
            os.environ.pop("GOOGLE_AUTH_DEBUG", None)
            auth = GoogleAuthHelper()
            assert auth.email == ""
            assert auth.password == ""
            assert auth.debug is False


class TestGoogleAuthHelperLog:
    """Tests for _log method."""

    def test_log_prints_when_debug_enabled(self, capsys):
        """_log should print when debug is True."""
        auth = GoogleAuthHelper(debug=True)
        auth._log("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.err
        assert "[GoogleAuth]" in captured.err

    def test_log_silent_when_debug_disabled(self, capsys):
        """_log should not print when debug is False."""
        auth = GoogleAuthHelper(debug=False)
        auth._log("Test message")
        captured = capsys.readouterr()
        assert captured.err == ""


class TestLoginViaGoogleButton:
    """Tests for login_via_google_button method."""

    @pytest.mark.asyncio
    async def test_returns_false_when_button_not_found(self):
        """Should return False when Google button not found."""
        auth = GoogleAuthHelper()
        mock_page = MagicMock()
        mock_page.query_selector = AsyncMock(return_value=None)

        result = await auth.login_via_google_button(mock_page)

        assert result is False

    @pytest.mark.asyncio
    async def test_tries_alternative_selectors(self):
        """Should try alternative selectors when primary fails."""
        auth = GoogleAuthHelper()
        mock_page = MagicMock()
        # First call fails, second succeeds
        call_count = 0

        async def mock_query_selector(selector):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                return MagicMock()
            return None

        mock_page.query_selector = mock_query_selector
        mock_page.context.expect_page = MagicMock()

        # This will fail due to other issues but tests selector logic
        result = await auth.login_via_google_button(mock_page)

        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        """Should return False on exception."""
        auth = GoogleAuthHelper()
        mock_page = MagicMock()
        mock_page.query_selector = AsyncMock(side_effect=Exception("Test error"))

        result = await auth.login_via_google_button(mock_page)

        assert result is False

    @pytest.mark.asyncio
    async def test_clicks_google_button(self):
        """Should click the Google button when found."""
        auth = GoogleAuthHelper()
        mock_page = MagicMock()
        mock_button = MagicMock()
        mock_button.click = AsyncMock()
        mock_page.query_selector = AsyncMock(return_value=mock_button)

        # Setup popup context manager
        mock_popup = MagicMock()
        mock_popup.url = "https://accounts.google.com"
        mock_popup_info = MagicMock()
        mock_popup_info.value = mock_popup

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_popup_info)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_page.context.expect_page = MagicMock(return_value=mock_cm)

        # Mock popup handler
        with patch.object(auth, "_handle_google_popup", AsyncMock(return_value=False)):
            await auth.login_via_google_button(mock_page)
            mock_button.click.assert_called_once()


class TestHandleGooglePopup:
    """Tests for _handle_google_popup method."""

    @pytest.mark.asyncio
    async def test_returns_false_on_email_failure(self):
        """Should return False when email fill fails."""
        auth = GoogleAuthHelper()
        mock_popup = MagicMock()
        mock_popup.wait_for_load_state = AsyncMock()
        mock_popup.wait_for_timeout = AsyncMock()

        with patch.object(auth, "_fill_email", AsyncMock(return_value=False)):
            result = await auth._handle_google_popup(mock_popup)
            assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_password_failure(self):
        """Should return False when password fill fails."""
        auth = GoogleAuthHelper()
        mock_popup = MagicMock()
        mock_popup.wait_for_load_state = AsyncMock()
        mock_popup.wait_for_timeout = AsyncMock()

        with patch.object(auth, "_fill_email", AsyncMock(return_value=True)):
            with patch.object(auth, "_fill_password", AsyncMock(return_value=False)):
                result = await auth._handle_google_popup(mock_popup)
                assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_when_popup_closes(self):
        """Should return True when popup closes (indicates success)."""
        auth = GoogleAuthHelper()
        mock_popup = MagicMock()
        mock_popup.wait_for_load_state = AsyncMock()
        mock_popup.wait_for_timeout = AsyncMock()
        mock_popup.wait_for_event = AsyncMock(return_value=None)

        with patch.object(auth, "_fill_email", AsyncMock(return_value=True)):
            with patch.object(auth, "_fill_password", AsyncMock(return_value=True)):
                result = await auth._handle_google_popup(mock_popup)
                assert result is True

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        """Should return False on exception."""
        auth = GoogleAuthHelper()
        mock_popup = MagicMock()
        mock_popup.wait_for_load_state = AsyncMock(side_effect=Exception("Load error"))

        result = await auth._handle_google_popup(mock_popup)
        assert result is False


class TestFillEmail:
    """Tests for _fill_email method."""

    @pytest.mark.asyncio
    async def test_fills_email_input(self):
        """Should fill email in input field."""
        auth = GoogleAuthHelper(email="test@gmail.com")
        mock_popup = MagicMock()
        mock_popup.wait_for_selector = AsyncMock()
        mock_popup.fill = AsyncMock()
        mock_popup.wait_for_timeout = AsyncMock()

        mock_next_btn = MagicMock()
        mock_next_btn.click = AsyncMock()
        mock_popup.query_selector = AsyncMock(return_value=mock_next_btn)

        result = await auth._fill_email(mock_popup)

        mock_popup.fill.assert_called_with('input[type="email"]', "test@gmail.com")
        assert result is True

    @pytest.mark.asyncio
    async def test_clicks_next_button(self):
        """Should click Next button after filling email."""
        auth = GoogleAuthHelper(email="test@gmail.com")
        mock_popup = MagicMock()
        mock_popup.wait_for_selector = AsyncMock()
        mock_popup.fill = AsyncMock()
        mock_popup.wait_for_timeout = AsyncMock()

        mock_next_btn = MagicMock()
        mock_next_btn.click = AsyncMock()
        mock_popup.query_selector = AsyncMock(return_value=mock_next_btn)

        await auth._fill_email(mock_popup)

        mock_next_btn.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_next_button_not_found(self):
        """Should return False when Next button not found."""
        auth = GoogleAuthHelper(email="test@gmail.com")
        mock_popup = MagicMock()
        mock_popup.wait_for_selector = AsyncMock()
        mock_popup.fill = AsyncMock()
        mock_popup.wait_for_timeout = AsyncMock()
        mock_popup.query_selector = AsyncMock(return_value=None)

        result = await auth._fill_email(mock_popup)

        assert result is False

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        """Should return False on exception."""
        auth = GoogleAuthHelper(email="test@gmail.com")
        mock_popup = MagicMock()
        mock_popup.wait_for_selector = AsyncMock(
            side_effect=Exception("Selector error")
        )

        result = await auth._fill_email(mock_popup)

        assert result is False


class TestFillPassword:
    """Tests for _fill_password method."""

    @pytest.mark.asyncio
    async def test_fills_password_input(self):
        """Should fill password in input field."""
        auth = GoogleAuthHelper(password="secret123")
        mock_popup = MagicMock()
        mock_popup.wait_for_selector = AsyncMock()
        mock_popup.fill = AsyncMock()
        mock_popup.wait_for_timeout = AsyncMock()

        mock_next_btn = MagicMock()
        mock_next_btn.click = AsyncMock()
        mock_popup.query_selector = AsyncMock(return_value=mock_next_btn)

        with patch.object(auth, "_wait_for_2fa", AsyncMock(return_value=True)):
            with patch.object(auth, "_handle_consent_screens", AsyncMock()):
                result = await auth._fill_password(mock_popup)

        mock_popup.fill.assert_called_with('input[type="password"]', "secret123")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_next_button_not_found(self):
        """Should return False when Next button not found."""
        auth = GoogleAuthHelper(password="secret123")
        mock_popup = MagicMock()
        mock_popup.wait_for_selector = AsyncMock()
        mock_popup.fill = AsyncMock()
        mock_popup.wait_for_timeout = AsyncMock()
        mock_popup.query_selector = AsyncMock(return_value=None)

        result = await auth._fill_password(mock_popup)

        assert result is False

    @pytest.mark.asyncio
    async def test_calls_2fa_handler(self):
        """Should call 2FA handler after password."""
        auth = GoogleAuthHelper(password="secret123")
        mock_popup = MagicMock()
        mock_popup.wait_for_selector = AsyncMock()
        mock_popup.fill = AsyncMock()
        mock_popup.wait_for_timeout = AsyncMock()

        mock_next_btn = MagicMock()
        mock_next_btn.click = AsyncMock()
        mock_popup.query_selector = AsyncMock(return_value=mock_next_btn)

        mock_2fa = AsyncMock(return_value=True)
        with patch.object(auth, "_wait_for_2fa", mock_2fa):
            with patch.object(auth, "_handle_consent_screens", AsyncMock()):
                await auth._fill_password(mock_popup)

        mock_2fa.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_2fa_fails(self):
        """Should return False when 2FA fails."""
        auth = GoogleAuthHelper(password="secret123")
        mock_popup = MagicMock()
        mock_popup.wait_for_selector = AsyncMock()
        mock_popup.fill = AsyncMock()
        mock_popup.wait_for_timeout = AsyncMock()

        mock_next_btn = MagicMock()
        mock_next_btn.click = AsyncMock()
        mock_popup.query_selector = AsyncMock(return_value=mock_next_btn)

        with patch.object(auth, "_wait_for_2fa", AsyncMock(return_value=False)):
            result = await auth._fill_password(mock_popup)

        assert result is False


class TestHandleConsentScreens:
    """Tests for _handle_consent_screens method."""

    @pytest.mark.asyncio
    async def test_clicks_continue_button(self):
        """Should click Continue button when found."""
        auth = GoogleAuthHelper()
        mock_popup = MagicMock()

        mock_btn = MagicMock()
        mock_btn.is_visible = AsyncMock(return_value=True)
        mock_btn.click = AsyncMock()
        mock_popup.query_selector = AsyncMock(return_value=mock_btn)
        mock_popup.wait_for_timeout = AsyncMock()

        await auth._handle_consent_screens(mock_popup)

        mock_btn.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_no_consent_screen(self):
        """Should handle case when no consent screen present."""
        auth = GoogleAuthHelper()
        mock_popup = MagicMock()
        mock_popup.query_selector = AsyncMock(return_value=None)

        # Should not raise
        await auth._handle_consent_screens(mock_popup)

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        """Should handle exceptions gracefully."""
        auth = GoogleAuthHelper()
        mock_popup = MagicMock()
        mock_popup.query_selector = AsyncMock(side_effect=Exception("Error"))

        # Should not raise
        await auth._handle_consent_screens(mock_popup)


class TestWaitFor2FA:
    """Tests for _wait_for_2fa method."""

    @pytest.mark.asyncio
    async def test_returns_true_when_not_2fa_page(self):
        """Should return True when not on 2FA page."""
        auth = GoogleAuthHelper()
        mock_popup = MagicMock()
        mock_popup.inner_text = AsyncMock(return_value="Welcome to Google")

        result = await auth._wait_for_2fa(mock_popup)

        assert result is True

    @pytest.mark.asyncio
    async def test_detects_2fa_indicators(self):
        """Should detect 2FA indicators in page text."""
        auth = GoogleAuthHelper()
        mock_popup = MagicMock()
        mock_popup.inner_text = AsyncMock(return_value="2-Step Verification required")
        mock_popup.url = "https://accounts.google.com/2fa"
        mock_popup.wait_for_timeout = AsyncMock()

        # Simulate popup closing
        call_count = 0

        @property
        def url_getter():
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                raise Exception("Popup closed")
            return "https://accounts.google.com/2fa"

        type(mock_popup).url = url_getter

        result = await auth._wait_for_2fa(mock_popup, timeout=5000)

        # Should have detected 2FA
        assert mock_popup.inner_text.called

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        """Should return False on exception."""
        auth = GoogleAuthHelper()
        mock_popup = MagicMock()
        mock_popup.inner_text = AsyncMock(side_effect=Exception("Error"))

        result = await auth._wait_for_2fa(mock_popup)

        assert result is False


class TestIsLoggedIn:
    """Tests for is_logged_in method."""

    @pytest.mark.asyncio
    async def test_returns_false_for_login_url(self):
        """Should return False when URL contains login."""
        auth = GoogleAuthHelper()
        mock_page = MagicMock()
        mock_page.url = "https://example.com/login"

        result = await auth.is_logged_in(mock_page)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_for_signin_url(self):
        """Should return False when URL contains signin."""
        auth = GoogleAuthHelper()
        mock_page = MagicMock()
        mock_page.url = "https://example.com/signin"

        result = await auth.is_logged_in(mock_page)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_for_oauth_url(self):
        """Should return False when URL contains oauth."""
        auth = GoogleAuthHelper()
        mock_page = MagicMock()
        mock_page.url = "https://example.com/oauth/authorize"

        result = await auth.is_logged_in(mock_page)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_for_google_accounts_url(self):
        """Should return False when URL is Google accounts."""
        auth = GoogleAuthHelper()
        mock_page = MagicMock()
        mock_page.url = "https://accounts.google.com/signin"

        result = await auth.is_logged_in(mock_page)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_for_dashboard_url(self):
        """Should return True when URL suggests logged in."""
        auth = GoogleAuthHelper()
        mock_page = MagicMock()
        mock_page.url = "https://example.com/dashboard"

        result = await auth.is_logged_in(mock_page)

        assert result is True

    @pytest.mark.asyncio
    async def test_accepts_custom_indicators(self):
        """Should use custom login indicators."""
        auth = GoogleAuthHelper()
        mock_page = MagicMock()
        mock_page.url = "https://example.com/auth"

        result = await auth.is_logged_in(mock_page, login_indicators=["auth"])

        assert result is False


class TestGoogleLoginConvenienceFunction:
    """Tests for google_login convenience function."""

    @pytest.mark.asyncio
    async def test_creates_auth_helper(self):
        """Should create GoogleAuthHelper with correct params."""
        mock_page = MagicMock()
        mock_page.query_selector = AsyncMock(return_value=None)

        with patch("scitex.browser.auth.google.GoogleAuthHelper") as mock_class:
            mock_instance = MagicMock()
            mock_instance.login_via_google_button = AsyncMock(return_value=True)
            mock_class.return_value = mock_instance

            await google_login(mock_page, "test@gmail.com", "password", debug=True)

            mock_class.assert_called_with(
                email="test@gmail.com", password="password", debug=True
            )

    @pytest.mark.asyncio
    async def test_calls_login_method(self):
        """Should call login_via_google_button method."""
        mock_page = MagicMock()
        mock_page.query_selector = AsyncMock(return_value=None)

        with patch("scitex.browser.auth.google.GoogleAuthHelper") as mock_class:
            mock_instance = MagicMock()
            mock_instance.login_via_google_button = AsyncMock(return_value=True)
            mock_class.return_value = mock_instance

            result = await google_login(
                mock_page, "test@gmail.com", "password", button_selector="custom"
            )

            mock_instance.login_via_google_button.assert_called_with(
                mock_page, "custom"
            )
            assert result is True


class TestGoogleAuthHelperIntegration:
    """Integration tests for GoogleAuthHelper."""

    def test_multiple_instances_independent(self):
        """Multiple instances should be independent."""
        auth1 = GoogleAuthHelper(email="user1@gmail.com")
        auth2 = GoogleAuthHelper(email="user2@gmail.com")

        assert auth1.email != auth2.email
        assert auth1.email == "user1@gmail.com"
        assert auth2.email == "user2@gmail.com"

    def test_full_config_from_env(self):
        """Should configure fully from environment."""
        with patch.dict(
            os.environ,
            {
                "GOOGLE_EMAIL": "env@gmail.com",
                "GOOGLE_PASSWORD": "envpass",
                "GOOGLE_AUTH_DEBUG": "1",
            },
        ):
            auth = GoogleAuthHelper()
            assert auth.email == "env@gmail.com"
            assert auth.password == "envpass"
            assert auth.debug is True


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
