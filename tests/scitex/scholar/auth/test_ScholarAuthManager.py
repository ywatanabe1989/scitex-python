#!/usr/bin/env python3
"""Tests for ScholarAuthManager class."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scitex.scholar.auth.ScholarAuthManager import ScholarAuthManager


class TestScholarAuthManagerInit:
    """Tests for ScholarAuthManager initialization."""

    def test_init_creates_instance_without_emails(self):
        """ScholarAuthManager should initialize without emails (with warning)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                # Pass explicit None to override any env var defaults
                manager = ScholarAuthManager(
                    email_openathens=None,
                    email_ezproxy=None,
                    email_shibboleth=None,
                )
                assert manager is not None
                assert manager.name == "ScholarAuthManager"
                assert manager.providers == {}
                assert manager.active_provider is None

    def test_init_sets_name(self):
        """Should set name to class name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarAuthManager(
                    email_openathens=None,
                    email_ezproxy=None,
                    email_shibboleth=None,
                )
                assert manager.name == "ScholarAuthManager"

    def test_init_creates_empty_providers_dict(self):
        """Should initialize with empty providers dict when no emails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarAuthManager(
                    email_openathens=None,
                    email_ezproxy=None,
                    email_shibboleth=None,
                )
                assert isinstance(manager.providers, dict)
                assert len(manager.providers) == 0

    def test_init_sets_auth_session_none(self):
        """Should initialize auth_session as None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarAuthManager(
                    email_openathens=None,
                    email_ezproxy=None,
                    email_shibboleth=None,
                )
                assert manager.auth_session is None

    @patch(
        "scitex.scholar.auth.ScholarAuthManager.OpenAthensAuthenticator", autospec=True
    )
    def test_init_registers_openathens_provider(self, mock_openathens):
        """Should register OpenAthens provider when email provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarAuthManager(email_openathens="test@uni.edu")
                assert "openathens" in manager.providers
                assert manager.active_provider == "openathens"

    @patch("scitex.scholar.auth.ScholarAuthManager.EZProxyAuthenticator", autospec=True)
    def test_init_registers_ezproxy_provider(self, mock_ezproxy):
        """Should register EZProxy provider when email provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarAuthManager(email_ezproxy="test@uni.edu")
                assert "ezproxy" in manager.providers
                assert manager.active_provider == "ezproxy"

    @patch(
        "scitex.scholar.auth.ScholarAuthManager.ShibbolethAuthenticator", autospec=True
    )
    def test_init_registers_shibboleth_provider(self, mock_shibboleth):
        """Should register Shibboleth provider when email provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarAuthManager(email_shibboleth="test@uni.edu")
                assert "shibboleth" in manager.providers
                assert manager.active_provider == "shibboleth"

    @patch(
        "scitex.scholar.auth.ScholarAuthManager.OpenAthensAuthenticator", autospec=True
    )
    @patch("scitex.scholar.auth.ScholarAuthManager.EZProxyAuthenticator", autospec=True)
    def test_init_registers_multiple_providers(self, mock_ezproxy, mock_openathens):
        """Should register multiple providers when multiple emails provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarAuthManager(
                    email_openathens="test1@uni.edu",
                    email_ezproxy="test2@uni.edu",
                )
                assert "openathens" in manager.providers
                assert "ezproxy" in manager.providers
                assert len(manager.providers) == 2
                # First registered provider becomes active
                assert manager.active_provider == "openathens"


class TestScholarAuthManagerListProviders:
    """Tests for list_providers method."""

    def test_list_providers_returns_empty_list(self):
        """list_providers should return empty list when no providers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarAuthManager(
                    email_openathens=None,
                    email_ezproxy=None,
                    email_shibboleth=None,
                )
                providers = manager.list_providers()
                assert isinstance(providers, list)
                assert len(providers) == 0

    @patch(
        "scitex.scholar.auth.ScholarAuthManager.OpenAthensAuthenticator", autospec=True
    )
    @patch("scitex.scholar.auth.ScholarAuthManager.EZProxyAuthenticator", autospec=True)
    def test_list_providers_returns_registered_providers(
        self, mock_ezproxy, mock_openathens
    ):
        """list_providers should return list of registered provider names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarAuthManager(
                    email_openathens="test1@uni.edu",
                    email_ezproxy="test2@uni.edu",
                )
                providers = manager.list_providers()
                assert "openathens" in providers
                assert "ezproxy" in providers


class TestScholarAuthManagerSetActiveProvider:
    """Tests for set_active_provider method."""

    @patch(
        "scitex.scholar.auth.ScholarAuthManager.OpenAthensAuthenticator", autospec=True
    )
    @patch("scitex.scholar.auth.ScholarAuthManager.EZProxyAuthenticator", autospec=True)
    def test_set_active_provider_changes_provider(self, mock_ezproxy, mock_openathens):
        """set_active_provider should change active provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarAuthManager(
                    email_openathens="test1@uni.edu",
                    email_ezproxy="test2@uni.edu",
                )
                assert manager.active_provider == "openathens"

                manager.set_active_provider("ezproxy")
                assert manager.active_provider == "ezproxy"

    def test_set_active_provider_raises_for_unknown(self):
        """set_active_provider should raise for unknown provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                os.environ.pop("SCITEX_SCHOLAR_OPENATHENS_EMAIL", None)
                os.environ.pop("SCITEX_SCHOLAR_EZPROXY_EMAIL", None)
                os.environ.pop("SCITEX_SCHOLAR_SHIBBOLETH_EMAIL", None)

                manager = ScholarAuthManager()
                with pytest.raises(ValueError) as excinfo:
                    manager.set_active_provider("unknown")
                assert "not found" in str(excinfo.value)


class TestScholarAuthManagerGetActiveProvider:
    """Tests for get_active_provider method."""

    @patch(
        "scitex.scholar.auth.ScholarAuthManager.OpenAthensAuthenticator", autospec=True
    )
    def test_get_active_provider_returns_provider(self, mock_openathens):
        """get_active_provider should return active provider instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarAuthManager(email_openathens="test@uni.edu")
                provider = manager.get_active_provider()
                assert provider is not None

    def test_get_active_provider_raises_when_none(self):
        """get_active_provider should raise when no active provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                os.environ.pop("SCITEX_SCHOLAR_OPENATHENS_EMAIL", None)
                os.environ.pop("SCITEX_SCHOLAR_EZPROXY_EMAIL", None)
                os.environ.pop("SCITEX_SCHOLAR_SHIBBOLETH_EMAIL", None)

                manager = ScholarAuthManager()
                with pytest.raises(ValueError) as excinfo:
                    manager.get_active_provider()
                assert "Active provider not found" in str(excinfo.value)


class TestScholarAuthManagerRegisterProvider:
    """Tests for _register_provider method."""

    @patch(
        "scitex.scholar.auth.ScholarAuthManager.OpenAthensAuthenticator", autospec=True
    )
    def test_register_provider_adds_to_dict(self, mock_openathens):
        """_register_provider should add provider to dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarAuthManager(email_openathens="test@uni.edu")
                assert "openathens" in manager.providers

    @patch(
        "scitex.scholar.auth.ScholarAuthManager.OpenAthensAuthenticator", autospec=True
    )
    def test_register_provider_sets_first_as_active(self, mock_openathens):
        """First registered provider should become active."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarAuthManager(email_openathens="test@uni.edu")
                assert manager.active_provider == "openathens"


class TestScholarAuthManagerAsync:
    """Tests for async methods with mocking."""

    @pytest.fixture
    def manager_with_mock_provider(self):
        """Create manager with mocked provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                os.environ.pop("SCITEX_SCHOLAR_OPENATHENS_EMAIL", None)
                os.environ.pop("SCITEX_SCHOLAR_EZPROXY_EMAIL", None)
                os.environ.pop("SCITEX_SCHOLAR_SHIBBOLETH_EMAIL", None)

                manager = ScholarAuthManager()

                # Create mock provider
                mock_provider = MagicMock()
                mock_provider.is_authenticate_async = AsyncMock(return_value=True)
                mock_provider.authenticate_async = AsyncMock(
                    return_value={"authenticated": True, "cookies": []}
                )
                mock_provider.get_auth_headers_async = AsyncMock(
                    return_value={"Authorization": "Bearer token"}
                )
                mock_provider.get_auth_cookies_async = AsyncMock(
                    return_value=[{"name": "session", "value": "abc123"}]
                )
                mock_provider.logout_async = AsyncMock()

                # Register mock provider
                manager.providers["mock"] = mock_provider
                manager.active_provider = "mock"

                yield manager

    @pytest.mark.asyncio
    async def test_is_authenticate_async_returns_bool(self, manager_with_mock_provider):
        """is_authenticate_async should return boolean."""
        result = await manager_with_mock_provider.is_authenticate_async()
        assert isinstance(result, bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_authenticate_async_checks_active_provider(
        self, manager_with_mock_provider
    ):
        """is_authenticate_async should check active provider first."""
        await manager_with_mock_provider.is_authenticate_async()
        manager_with_mock_provider.providers[
            "mock"
        ].is_authenticate_async.assert_called()

    @pytest.mark.asyncio
    async def test_authenticate_async_returns_session(self, manager_with_mock_provider):
        """authenticate_async should return auth session dict."""
        result = await manager_with_mock_provider.authenticate_async()
        assert isinstance(result, dict)
        assert "authenticated" in result

    @pytest.mark.asyncio
    async def test_authenticate_async_raises_for_unknown_provider(
        self, manager_with_mock_provider
    ):
        """authenticate_async should raise for unknown provider name."""
        from scitex.logging import AuthenticationError

        with pytest.raises(AuthenticationError) as excinfo:
            await manager_with_mock_provider.authenticate_async(
                provider_name="unknown_provider"
            )
        assert "not found" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_get_auth_headers_async_returns_dict(
        self, manager_with_mock_provider
    ):
        """get_auth_headers_async should return headers dict."""
        result = await manager_with_mock_provider.get_auth_headers_async()
        assert isinstance(result, dict)
        assert "Authorization" in result

    @pytest.mark.asyncio
    async def test_get_auth_cookies_async_returns_list(
        self, manager_with_mock_provider
    ):
        """get_auth_cookies_async should return list of cookies."""
        result = await manager_with_mock_provider.get_auth_cookies_async(
            essential_only=False
        )
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_auth_options_returns_dict(self, manager_with_mock_provider):
        """get_auth_options should return options dict."""
        # Set auth_session with cookies
        manager_with_mock_provider.auth_session = {"cookies": [{"name": "test"}]}
        result = await manager_with_mock_provider.get_auth_options()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_logout_async_clears_state(self, manager_with_mock_provider):
        """logout_async should clear active provider and session."""
        manager_with_mock_provider.auth_session = {"authenticated": True}
        await manager_with_mock_provider.logout_async()

        assert manager_with_mock_provider.active_provider is None
        assert manager_with_mock_provider.auth_session is None


class TestScholarAuthManagerCookieFiltering:
    """Tests for cookie filtering in get_auth_cookies_async."""

    @pytest.fixture
    def manager_with_cookies(self):
        """Create manager with mock provider that returns various cookies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                os.environ.pop("SCITEX_SCHOLAR_OPENATHENS_EMAIL", None)
                os.environ.pop("SCITEX_SCHOLAR_EZPROXY_EMAIL", None)
                os.environ.pop("SCITEX_SCHOLAR_SHIBBOLETH_EMAIL", None)

                manager = ScholarAuthManager()

                mock_provider = MagicMock()
                mock_provider.is_authenticate_async = AsyncMock(return_value=True)
                mock_provider.get_auth_cookies_async = AsyncMock(
                    return_value=[
                        {"name": "oa-session", "value": "abc"},
                        {"name": "oa-xsrf-token", "value": "xyz"},
                        {"name": "other-cookie", "value": "123"},
                    ]
                )

                manager.providers["openathens"] = mock_provider
                manager.active_provider = "openathens"
                manager.auth_session = {"authenticated": True}

                yield manager

    @pytest.mark.asyncio
    async def test_filters_openathens_cookies(self, manager_with_cookies):
        """Should filter to essential OpenAthens cookies."""
        result = await manager_with_cookies.get_auth_cookies_async(essential_only=True)

        # Should only include essential OpenAthens cookies
        cookie_names = [c["name"] for c in result]
        assert "oa-session" in cookie_names
        assert "oa-xsrf-token" in cookie_names
        # Non-essential cookie should be filtered
        assert "other-cookie" not in cookie_names

    @pytest.mark.asyncio
    async def test_returns_all_cookies_when_not_essential_only(
        self, manager_with_cookies
    ):
        """Should return all cookies when essential_only=False."""
        result = await manager_with_cookies.get_auth_cookies_async(essential_only=False)

        cookie_names = [c["name"] for c in result]
        assert len(cookie_names) == 3
        assert "other-cookie" in cookie_names


class TestScholarAuthManagerIntegration:
    """Integration tests for ScholarAuthManager."""

    @pytest.fixture
    def fully_mocked_manager(self):
        """Create fully mocked manager for integration testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                os.environ.pop("SCITEX_SCHOLAR_OPENATHENS_EMAIL", None)
                os.environ.pop("SCITEX_SCHOLAR_EZPROXY_EMAIL", None)
                os.environ.pop("SCITEX_SCHOLAR_SHIBBOLETH_EMAIL", None)

                manager = ScholarAuthManager()

                # Create mock providers
                mock_openathens = MagicMock()
                mock_openathens.is_authenticate_async = AsyncMock(return_value=False)
                mock_openathens.authenticate_async = AsyncMock(
                    return_value={"provider": "openathens", "cookies": []}
                )

                mock_ezproxy = MagicMock()
                mock_ezproxy.is_authenticate_async = AsyncMock(return_value=True)
                mock_ezproxy.authenticate_async = AsyncMock(
                    return_value={"provider": "ezproxy", "cookies": []}
                )

                manager.providers["openathens"] = mock_openathens
                manager.providers["ezproxy"] = mock_ezproxy
                manager.active_provider = "openathens"

                yield manager

    @pytest.mark.asyncio
    async def test_is_authenticate_fallback_to_other_provider(
        self, fully_mocked_manager
    ):
        """is_authenticate_async should try other providers if active fails."""
        result = await fully_mocked_manager.is_authenticate_async()

        # Should succeed with ezproxy (which returns True)
        assert result is True
        # Should switch active provider to the one that succeeded
        assert fully_mocked_manager.active_provider == "ezproxy"

    @pytest.mark.asyncio
    async def test_provider_switching_workflow(self, fully_mocked_manager):
        """Test complete provider switching workflow."""
        # Start with openathens
        assert fully_mocked_manager.active_provider == "openathens"

        # Check auth status - should switch to ezproxy
        await fully_mocked_manager.is_authenticate_async()
        assert fully_mocked_manager.active_provider == "ezproxy"

        # Switch back manually
        fully_mocked_manager.set_active_provider("openathens")
        assert fully_mocked_manager.active_provider == "openathens"


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
