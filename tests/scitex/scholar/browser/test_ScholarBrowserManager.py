#!/usr/bin/env python3
"""Tests for ScholarBrowserManager class."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scitex.scholar.browser.ScholarBrowserManager import ScholarBrowserManager


class TestScholarBrowserManagerInit:
    """Tests for ScholarBrowserManager initialization."""

    def test_init_creates_instance(self):
        """ScholarBrowserManager should initialize without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                # Must pass browser_mode explicitly due to source code bug
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )
                assert manager is not None
                assert manager.name == "ScholarBrowserManager"

    def test_init_stores_browser_mode(self):
        """Should store browser mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )
                assert manager.browser_mode == "interactive"

    def test_init_stores_auth_manager(self):
        """Should store auth manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                mock_auth = MagicMock()
                manager = ScholarBrowserManager(
                    browser_mode="interactive",
                    auth_manager=mock_auth,
                    chrome_profile_name="test",
                )
                assert manager.auth_manager is mock_auth

    def test_init_creates_chrome_profile_manager(self):
        """Should create ChromeProfileManager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test_profile"
                )
                assert manager.chrome_profile_manager is not None

    def test_init_creates_stealth_manager(self):
        """Should create StealthManager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )
                assert manager.stealth_manager is not None

    def test_init_creates_cookie_acceptor(self):
        """Should create CookieAutoAcceptor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )
                assert manager.cookie_acceptor is not None

    def test_init_sets_persistent_attributes_to_none(self):
        """Should initialize persistent attributes to None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )
                assert manager._persistent_browser is None
                assert manager._persistent_context is None
                assert manager._persistent_playwright is None

    def test_init_uses_config(self):
        """Should use provided config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                from scitex.scholar.config import ScholarConfig

                config = ScholarConfig()
                manager = ScholarBrowserManager(
                    browser_mode="interactive",
                    config=config,
                    chrome_profile_name="test",
                )
                assert manager.config is config


class TestSetInteractiveOrStealth:
    """Tests for _set_interactive_or_stealth method."""

    def test_interactive_mode_settings(self):
        """Interactive mode should set correct settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )
                assert manager.headless is False
                assert manager.spoof_dimension is False
                assert manager.viewport_size == (1920, 1080)
                assert manager.display == 0

    def test_stealth_mode_settings(self):
        """Stealth mode should set correct settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="stealth", chrome_profile_name="test"
                )
                assert manager.headless is False
                assert manager.spoof_dimension is True
                assert manager.viewport_size == (1920, 1080)
                assert manager.display == 99

    def test_invalid_mode_raises_error(self):
        """Invalid browser mode should raise AssertionError from BrowserMixin."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                # BrowserMixin.__init__ raises AssertionError before
                # _set_interactive_or_stealth can raise ValueError
                with pytest.raises(AssertionError):
                    ScholarBrowserManager(
                        browser_mode="invalid_mode", chrome_profile_name="test"
                    )


class TestGetAuthenticatedBrowserAndContextAsync:
    """Tests for get_authenticated_browser_and_context_async method."""

    @pytest.mark.asyncio
    async def test_raises_without_auth_manager(self):
        """Should raise ValueError when auth_manager is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive",
                    auth_manager=None,
                    chrome_profile_name="test",
                )
                with pytest.raises(ValueError) as excinfo:
                    await manager.get_authenticated_browser_and_context_async()
                assert "Authentication manager is not set" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_calls_ensure_authenticate(self):
        """Should call auth_manager.ensure_authenticate_async."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                mock_auth = MagicMock()
                mock_auth.ensure_authenticate_async = AsyncMock()
                manager = ScholarBrowserManager(
                    browser_mode="interactive",
                    auth_manager=mock_auth,
                    chrome_profile_name="test",
                )

                with patch.object(
                    manager,
                    "_get_persistent_browser_with_profile_but_not_with_auth_async",
                    AsyncMock(return_value=MagicMock()),
                ):
                    manager._persistent_context = MagicMock()
                    await manager.get_authenticated_browser_and_context_async()

                mock_auth.ensure_authenticate_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_persistent_context_when_available(self):
        """Should return persistent context when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                mock_auth = MagicMock()
                mock_auth.ensure_authenticate_async = AsyncMock()
                manager = ScholarBrowserManager(
                    browser_mode="interactive",
                    auth_manager=mock_auth,
                    chrome_profile_name="test",
                )

                mock_browser = MagicMock()
                mock_context = MagicMock()
                manager._persistent_context = mock_context

                with patch.object(
                    manager,
                    "_get_persistent_browser_with_profile_but_not_with_auth_async",
                    AsyncMock(return_value=mock_browser),
                ):
                    (
                        browser,
                        context,
                    ) = await manager.get_authenticated_browser_and_context_async()

                assert browser is mock_browser
                assert context is mock_context


class TestNewContextAsync:
    """Tests for _new_context_async method."""

    @pytest.mark.asyncio
    async def test_creates_context_with_stealth_options(self):
        """Should create context with stealth options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                mock_browser = MagicMock()
                mock_context = MagicMock()
                mock_context.add_init_script = AsyncMock()
                mock_browser.new_context = AsyncMock(return_value=mock_context)

                result = await manager._new_context_async(mock_browser)

                mock_browser.new_context.assert_called_once()
                assert result is mock_context

    @pytest.mark.asyncio
    async def test_applies_stealth_scripts(self):
        """Should apply stealth scripts to context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                mock_browser = MagicMock()
                mock_context = MagicMock()
                mock_context.add_init_script = AsyncMock()
                mock_browser.new_context = AsyncMock(return_value=mock_context)

                await manager._new_context_async(mock_browser)

                # Should call add_init_script 3 times (stealth, dimension, cookie)
                assert mock_context.add_init_script.call_count == 3


class TestBuildPersistentContextLaunchOptions:
    """Tests for _build_persistent_context_launch_options method."""

    def test_returns_dict_with_required_keys(self):
        """Should return dict with all required launch options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )
                options = manager._build_persistent_context_launch_options()

                assert "user_data_dir" in options
                assert "headless" in options
                assert "args" in options
                assert "accept_downloads" in options
                assert "downloads_path" in options
                assert "viewport" in options
                assert "screen" in options

    def test_includes_extension_args(self):
        """Should include extension args in launch options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                with patch.object(
                    manager.chrome_profile_manager,
                    "get_extension_args",
                    return_value=["--load-extension=test"],
                ):
                    options = manager._build_persistent_context_launch_options()
                    assert "--load-extension=test" in options["args"]

    def test_includes_stealth_args(self):
        """Should include stealth args in launch options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )
                options = manager._build_persistent_context_launch_options()

                # Should have display arg
                display_args = [arg for arg in options["args"] if "--display=" in arg]
                assert len(display_args) > 0

    def test_sets_accept_downloads_true(self):
        """Should set accept_downloads to True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )
                options = manager._build_persistent_context_launch_options()
                assert options["accept_downloads"] is True

    def test_viewport_matches_viewport_size(self):
        """Viewport should match configured viewport size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )
                options = manager._build_persistent_context_launch_options()

                assert options["viewport"]["width"] == manager.viewport_size[0]
                assert options["viewport"]["height"] == manager.viewport_size[1]


class TestTakeScreenshotAsync:
    """Tests for take_screenshot_async method."""

    @pytest.mark.asyncio
    async def test_calls_page_screenshot(self):
        """Should call page.screenshot with correct parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                mock_page = MagicMock()
                mock_page.screenshot = AsyncMock()

                screenshot_path = Path(tmpdir) / "test.png"
                await manager.take_screenshot_async(mock_page, screenshot_path)

                mock_page.screenshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_correct_timeout(self):
        """Should use correct timeout in milliseconds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                mock_page = MagicMock()
                mock_page.screenshot = AsyncMock()

                await manager.take_screenshot_async(
                    mock_page, "/tmp/test.png", timeout_sec=10.0
                )

                call_kwargs = mock_page.screenshot.call_args[1]
                assert call_kwargs["timeout"] == 10000  # 10 seconds in ms

    @pytest.mark.asyncio
    async def test_handles_full_page_option(self):
        """Should pass full_page option to screenshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                mock_page = MagicMock()
                mock_page.screenshot = AsyncMock()

                await manager.take_screenshot_async(
                    mock_page, "/tmp/test.png", full_page=True
                )

                call_kwargs = mock_page.screenshot.call_args[1]
                assert call_kwargs["full_page"] is True

    @pytest.mark.asyncio
    async def test_handles_screenshot_error_gracefully(self):
        """Should handle screenshot errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                mock_page = MagicMock()
                mock_page.screenshot = AsyncMock(
                    side_effect=Exception("Screenshot failed")
                )

                # Should not raise
                await manager.take_screenshot_async(mock_page, "/tmp/test.png")


class TestPeriodicScreenshots:
    """Tests for periodic screenshot functionality."""

    @pytest.mark.asyncio
    async def test_start_returns_task(self):
        """start_periodic_screenshots_async should return an asyncio Task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                mock_page = MagicMock()
                mock_page.screenshot = AsyncMock()

                task = await manager.start_periodic_screenshots_async(
                    mock_page, tmpdir, duration_seconds=0
                )

                assert isinstance(task, asyncio.Task)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        """stop_periodic_screenshots_async should cancel the task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                mock_page = MagicMock()
                mock_page.screenshot = AsyncMock()

                task = await manager.start_periodic_screenshots_async(
                    mock_page, tmpdir, duration_seconds=0
                )

                await manager.stop_periodic_screenshots_async(task)

                assert task.cancelled() or task.done()


class TestVerifyXvfbRunning:
    """Tests for _verify_xvfb_running method."""

    def test_returns_true_when_display_running(self):
        """Should return True when Xvfb display is running."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0)
                    result = manager._verify_xvfb_running()
                    assert result is True

    def test_handles_xvfb_not_found(self):
        """Should handle case when Xvfb not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                with patch("subprocess.run") as mock_run:
                    mock_run.side_effect = Exception("xdpyinfo not found")
                    result = manager._verify_xvfb_running()
                    assert result is False


class TestClose:
    """Tests for close method."""

    @pytest.mark.asyncio
    async def test_closes_persistent_context(self):
        """Should close persistent context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                mock_context = MagicMock()
                mock_context.close = AsyncMock()
                mock_browser = MagicMock()
                mock_browser.is_connected.return_value = True
                mock_context.browser = mock_browser

                manager._persistent_context = mock_context

                await manager.close()

                mock_context.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_closes_persistent_browser(self):
        """Should close persistent browser."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                mock_browser = MagicMock()
                mock_browser.is_connected.return_value = True
                mock_browser.close = AsyncMock()

                mock_context = MagicMock()
                mock_context.close = AsyncMock()
                mock_context.browser.is_connected.return_value = True

                manager._persistent_context = mock_context
                manager._persistent_browser = mock_browser

                await manager.close()

                mock_browser.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stops_playwright(self):
        """Should stop playwright instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                mock_playwright = MagicMock()
                mock_playwright.stop = AsyncMock()

                mock_context = MagicMock()
                mock_context.close = AsyncMock()
                mock_context.browser.is_connected.return_value = True

                manager._persistent_context = mock_context
                manager._persistent_playwright = mock_playwright

                await manager.close()

                mock_playwright.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_resets_persistent_attributes(self):
        """Should reset persistent attributes to None after close."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                mock_context = MagicMock()
                mock_context.close = AsyncMock()
                mock_context.browser.is_connected.return_value = True

                manager._persistent_context = mock_context
                manager._persistent_browser = MagicMock()
                manager._persistent_playwright = MagicMock()
                manager._persistent_playwright.stop = AsyncMock()

                await manager.close()

                assert manager._persistent_context is None
                assert manager._persistent_browser is None
                assert manager._persistent_playwright is None

    @pytest.mark.asyncio
    async def test_handles_already_closed_browser(self):
        """Should handle case when browser already closed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="test"
                )

                mock_context = MagicMock()
                mock_context.browser.is_connected.return_value = False
                manager._persistent_context = mock_context

                # Should not raise
                await manager.close()


class TestZenRowsProxy:
    """Tests for ZenRows proxy functionality."""

    def test_zenrows_import_fails_when_enabled(self):
        """Should raise ModuleNotFoundError when use_zenrows_proxy=True.

        The remote module with ZenRowsProxyManager doesn't exist in the
        scholar browser path, so enabling zenrows_proxy causes an import error.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                with pytest.raises(ModuleNotFoundError):
                    ScholarBrowserManager(
                        browser_mode="interactive",
                        use_zenrows_proxy=True,
                        chrome_profile_name="test",
                    )

    def test_no_zenrows_manager_when_disabled(self):
        """Should not create ZenRowsProxyManager when use_zenrows_proxy=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager = ScholarBrowserManager(
                    browser_mode="interactive",
                    use_zenrows_proxy=False,
                    chrome_profile_name="test",
                )
                assert manager.use_zenrows_proxy is False
                assert not hasattr(manager, "zenrows_proxy_manager")


class TestScholarBrowserManagerIntegration:
    """Integration tests for ScholarBrowserManager."""

    def test_full_initialization(self):
        """Test complete initialization with all components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                mock_auth = MagicMock()
                manager = ScholarBrowserManager(
                    browser_mode="interactive",
                    auth_manager=mock_auth,
                    chrome_profile_name="test",
                )

                assert manager.name == "ScholarBrowserManager"
                assert manager.browser_mode == "interactive"
                assert manager.auth_manager is mock_auth
                assert manager.chrome_profile_manager is not None
                assert manager.stealth_manager is not None
                assert manager.cookie_acceptor is not None

    def test_multiple_instances_independent(self):
        """Multiple instances should be independent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                manager1 = ScholarBrowserManager(
                    browser_mode="interactive", chrome_profile_name="profile1"
                )
                manager2 = ScholarBrowserManager(
                    browser_mode="stealth", chrome_profile_name="profile2"
                )

                assert manager1.browser_mode != manager2.browser_mode
                assert manager1.display != manager2.display


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
