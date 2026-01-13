#!/usr/bin/env python3
# Timestamp: "2025-10-11 01:05:03 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/scholar/auth/core/BrowserAuthenticator.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = "./src/scitex/scholar/auth/core/BrowserAuthenticator.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""Browser-based authentication operations.

This module handles browser interactions for authentication,
including login detection, navigation, and session verification.
"""

import asyncio
from typing import Any, Dict, List, Optional

from playwright.async_api import Page, async_playwright

from scitex import logging
from scitex.browser.core import BrowserMixin
from scitex.browser.interaction import (
    click_with_fallbacks_async,
    fill_with_fallbacks_async,
)

logger = logging.getLogger(__name__)


class BrowserAuthenticator(BrowserMixin):
    """Handles browser-based authentication operations."""

    def __init__(
        self, mode: str = "interactive", timeout: int = 300, sso_automator=None
    ):
        """Initialize browser authenticator.

        Args:
            mode: Browser mode - 'interactive' for authentication, 'stealth' for scraping
            timeout: Timeout for browser operations in seconds
            sso_automator: Optional SSO automator instance for institution-specific handling
        """
        super().__init__(mode=mode)
        self.name = self.__class__.__name__
        self.timeout = timeout
        self.sso_automator = sso_automator

    async def navigate_to_login_async(self, url: str) -> Page:
        """Navigate to login URL and return page.

        Args:
            url: Login URL to navigate to

        Returns
        -------
            Page object for further operations
        """
        # Use the BrowserMixin's get_browser_async method to get properly configured browser
        browser = await self.get_browser_async()

        # Create context with proper interactive viewport settings
        context_options = {}
        if self.mode == "interactive":
            context_options["viewport"] = {"width": 1280, "height": 720}
        else:
            context_options["viewport"] = {"width": 1, "height": 1}

        context = await browser.new_context(**context_options)
        await context.add_init_script(self.cookie_acceptor.get_auto_acceptor_script())
        # await self.cookie_acceptor.inject_auto_acceptor_async(context)
        page = await context.new_page()

        logger.info(f"{self.name}: Navigating to: {url}")
        await page.goto(url, wait_until="domcontentloaded")

        # Check for cookie banner and warn if present
        if await self.cookie_acceptor.check_cookie_banner_exists_async(page):
            logger.warning(
                "{self.name}: Cookie banner detected - may need manual acceptance"
            )

        return page

    async def wait_for_login_completion_async(
        self,
        page: Page,
        success_indicators: List[str],
        on_screenshot=None,
        on_intervention_needed=None,
    ) -> bool:
        """Wait for login completion by monitoring URL changes and handling SSO automation.

        Args:
            page: Browser page to monitor
            success_indicators: List of URL patterns indicating successful login
            on_screenshot: Optional async callback(screenshot_data) for notifications
            on_intervention_needed: Optional async callback() when user intervention needed

        Returns
        -------
            True if login successful, False if timed out
        """
        max_wait_time = self.timeout
        check_interval = 2
        elapsed_time = 0
        seen_sso_page = False
        openathens_automated = False
        institution_sso_attempted = False
        generic_sso_attempted = False
        intervention_notified = False
        screenshot_sent = False  # Only send ONE screenshot after intervention

        while elapsed_time < max_wait_time:
            current_url = page.url

            # Track SSO navigation
            if self._is_sso_page(current_url):
                seen_sso_page = True
                logger.debug(f"{self.name}: Detected SSO/login page: {current_url}")

            # Handle different authentication flows based on current URL
            automation_attempted = False

            # Priority 1: OpenAthens page automation (if on OpenAthens and not yet automated)
            if not openathens_automated and self._is_openathens_page(current_url):
                logger.debug(
                    f"{self.name}: OpenAthens page detected - attempting automation"
                )
                automation_attempted = True

                try:
                    from ..sso.OpenAthensSSOAutomator import (
                        OpenAthensSSOAutomator,
                    )

                    openathens_automator = OpenAthensSSOAutomator()

                    if openathens_automator.is_sso_page(current_url):
                        oa_success = (
                            await openathens_automator.handle_sso_redirect_async(page)
                        )
                        if oa_success:
                            logger.info(
                                f"{self.name}: OpenAthens page automation completed"
                            )
                            openathens_automated = True
                        else:
                            logger.warning(
                                f"{self.name}: OpenAthens page automation failed"
                            )
                except Exception as e:
                    logger.warning(f"{self.name}: OpenAthens automation failed: {e}")

            # Priority 2: Institution-specific SSO automation (if available and on SSO page)
            # Only attempt ONCE - don't retry after failure
            elif (
                self.sso_automator
                and self.sso_automator.is_sso_page(current_url)
                and not institution_sso_attempted
            ):
                institution_name = self.sso_automator.get_institution_name()
                logger.info(
                    f"{self.name}: {institution_name} SSO detected - attempting automation"
                )
                automation_attempted = True
                institution_sso_attempted = True

                sso_success = await self.sso_automator.handle_sso_redirect_async(page)
                if sso_success:
                    logger.info(
                        f"{self.name}: {institution_name} SSO automation completed"
                    )
                else:
                    logger.warning(
                        f"{self.name}: {institution_name} SSO automation failed"
                    )

                # Notify user AFTER automation completes - intervention may be needed
                if not intervention_notified and on_intervention_needed:
                    intervention_notified = True
                    # Capture screenshot for 2FA page and pass to callback
                    screenshot_data = await self.capture_screenshot_async(
                        page, description="2fa_page"
                    )
                    try:
                        await on_intervention_needed(screenshot_data)
                    except Exception as e:
                        logger.debug(f"{self.name}: Intervention callback failed: {e}")

            # Priority 3: Generic automation attempt for unknown SSO pages (only once)
            elif (
                self._is_sso_page(current_url)
                and not automation_attempted
                and not generic_sso_attempted
            ):
                logger.info(
                    f"{self.name}: Generic SSO page detected - attempting basic automation"
                )
                automation_attempted = True
                generic_sso_attempted = True

                try:
                    generic_success = await self._attempt_generic_sso_automation(page)
                    if generic_success:
                        logger.info(f"{self.name}: Generic SSO automation completed")
                    else:
                        logger.info(
                            f"{self.name}: Generic SSO automation not applicable"
                        )
                except Exception as e:
                    logger.debug(f"{self.name}: Generic SSO automation failed: {e}")

            # If automation was attempted but we're still on the same page,
            # it likely failed and requires manual intervention
            if automation_attempted and elapsed_time > 30:
                if elapsed_time % 30 == 0:  # Show message every 30 seconds
                    logger.info(
                        f"{self.name}: Automation completed - waiting for manual completion if needed"
                    )

            # Check for success
            if self._check_success_indicators(current_url, success_indicators):
                if await self._verify_login_success_async(
                    page, seen_sso_page, elapsed_time
                ):
                    logger.info(
                        f"{self.name}: Login successful detected at URL: {current_url}"
                    )
                    logger.info(f"{self.name}: Login detected! Capturing session...")
                    return True

            # Show progress
            if elapsed_time % 10 == 0 and elapsed_time > 0:
                logger.info(
                    f"{self.name}: Waiting for login... ({elapsed_time}s elapsed)"
                )

            # Send ONE screenshot immediately after intervention notification
            # This captures the 2FA page for user to see the code
            if intervention_notified and not screenshot_sent and on_screenshot:
                screenshot_sent = True
                screenshot_data = await self.capture_screenshot_async(
                    page, description="2fa_page"
                )
                if screenshot_data.get("success"):
                    try:
                        await on_screenshot(screenshot_data)
                    except Exception as e:
                        logger.debug(f"{self.name}: Screenshot callback failed: {e}")

            await asyncio.sleep(check_interval)
            elapsed_time += check_interval

        logger.error(f"{self.name}: Login timeout - please try again")
        return False

    async def verify_authentication_async(
        self, verification_url: str, cookies: List[Dict[str, Any]]
    ) -> bool:
        """Verify authentication by checking access to protected page.

        Args:
            verification_url: URL to test access to
            cookies: Authentication cookies to use

        Returns
        -------
            True if authentication verified, False otherwise
        """
        try:
            # Use stealth mode for verification (minimal viewport)
            original_mode = self.mode
            self.mode = "stealth"

            async with async_playwright() as p:
                browser, context = await self.create_browser_context_async(p)

                # Add cookies
                if cookies:
                    await context.add_cookies(cookies)

                page = await context.new_page()

                # Navigate to verification URL
                response = await page.goto(
                    verification_url,
                    wait_until="domcontentloaded",
                    timeout=15000,
                )

                current_url = page.url
                is_authenticate_async = self._check_authenticate_async_page(current_url)

                await browser.close()

                if is_authenticate_async:
                    logger.info(
                        f"{self.name}: Verified live authentication at {current_url}"
                    )
                else:
                    logger.debug(
                        f"{self.name}: Authentication verification failed at {current_url}"
                    )

                return is_authenticate_async

        except Exception as e:
            logger.error(f"{self.name}: Authentication verification failed: {e}")
            return False
        finally:
            # Restore original mode setting
            self.mode = original_mode

    async def extract_session_cookies_async(
        self, page: Page
    ) -> tuple[Dict[str, str], List[Dict[str, Any]]]:
        """Extract session cookies from browser page.

        Args:
            page: Browser page to extract cookies from

        Returns
        -------
            Tuple of (simple_cookies_dict, full_cookies_list)
        """
        cookies = await page.context.cookies()
        simple_cookies = {c["name"]: c["value"] for c in cookies}
        return simple_cookies, cookies

    async def reliable_click_async(self, page: Page, selector: str) -> bool:
        """Perform reliable click using shared utility."""
        return await click_with_fallbacks_async(page, selector)

    async def reliable_fill_async(self, page: Page, selector: str, value: str) -> bool:
        """Perform reliable form fill using shared utility."""
        return await fill_with_fallbacks_async(page, selector, value)

    def display_login_instructions(self, email: Optional[str], timeout: int) -> None:
        """Display login instructions to user using proper logging.

        Args:
            email: User email to display
            timeout: Timeout to display
        """
        logger.info(f"{self.name}: OpenAthens Authentication Required")
        logger.info(f"{self.name}: MyAthens login page is opening...")
        if email:
            logger.info(f"{self.name}: Account: {email}")
        logger.info(f"{self.name}: Please complete the login process:")
        logger.info(f"{self.name}: 1. Enter your institutional email")
        logger.info(f"{self.name}: 2. Click your institution when it appears")
        logger.info(f"{self.name}: 3. Complete login on your institution's page")
        logger.info(
            f"{self.name}: 4. You'll be redirected back to OpenAthens when done"
        )
        logger.info(f"{self.name}: 5. Timeout is {timeout} seconds")
        logger.info(f"{self.name}: 6. Close the window after successful login")

        # Show environment variables
        logger.debug(f"{self.name}: OpenAthens Environment Variables:")
        for key, value in os.environ.items():
            if "SCITEX_SCHOLAR_OPENATHENS" in key:
                logger.debug(f"{self.name}:   {key}: {value}")

    async def capture_screenshot_async(
        self,
        page: Page,
        description: str = "auth",
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Capture screenshot of current page state.

        Useful for monitoring authentication progress, especially for 2FA pages.

        Args:
            page: Browser page to capture
            description: Short description for filename
            save_dir: Directory to save screenshot (uses temp if None)

        Returns
        -------
            Dict with 'path', 'base64', 'url', 'timestamp'
        """
        import base64
        from datetime import datetime
        from pathlib import Path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"auth_{description}_{timestamp}.png"

        # Determine save directory
        if save_dir:
            save_path = Path(save_dir)
        else:
            save_path = Path(os.environ.get("SCITEX_DIR", "~/.scitex")).expanduser()
            save_path = save_path / "scholar" / "screenshots"
        save_path.mkdir(parents=True, exist_ok=True)

        filepath = save_path / filename

        try:
            # Capture screenshot
            screenshot_bytes = await page.screenshot(full_page=False)

            # Save to file
            with open(filepath, "wb") as f:
                f.write(screenshot_bytes)

            # Convert to base64 for email embedding
            b64_data = base64.b64encode(screenshot_bytes).decode("utf-8")

            logger.info(f"{self.name}: Screenshot saved: {filepath}")

            return {
                "path": str(filepath),
                "base64": b64_data,
                "url": page.url,
                "timestamp": timestamp,
                "success": True,
            }
        except Exception as e:
            logger.warning(f"{self.name}: Screenshot capture failed: {e}")
            return {
                "path": None,
                "base64": None,
                "url": page.url,
                "timestamp": timestamp,
                "success": False,
                "error": str(e),
            }

    def _is_sso_page(self, url: str) -> bool:
        """Check if URL indicates SSO/login page."""
        return "sso" in url.lower() or "login" in url.lower()

    def _is_openathens_page(self, url: str) -> bool:
        """Check if URL is OpenAthens page."""
        return "my.openathens.net" in url.lower() or "openathens.net" in url.lower()

    def _check_success_indicators(self, url: str, indicators: List[str]) -> bool:
        """Check if URL matches any success indicators."""
        return any(indicator in url for indicator in indicators)

    async def _verify_login_success_async(
        self, page: Page, seen_sso_page: bool, elapsed_time: int
    ) -> bool:
        """Verify login success with additional checks."""
        # Additional verification logic can be added here
        return seen_sso_page or elapsed_time > 30

    def _check_authenticate_async_page(self, url: str) -> bool:
        """Check if URL indicates authenticate_async page."""
        authenticate_async_patterns = ["/account", "/app", "/library"]
        unauthenticate_async_patterns = ["login", "signin"]

        # Check for unauthenticate_async patterns first
        if any(pattern in url for pattern in unauthenticate_async_patterns):
            return False

        # Check for authenticate_async patterns
        return any(pattern in url for pattern in authenticate_async_patterns)


# EOF
