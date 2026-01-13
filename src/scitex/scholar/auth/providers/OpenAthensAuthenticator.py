#!/usr/bin/env python3
# Timestamp: "2026-01-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/auth/providers/OpenAthensAuthenticator.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = "./src/scitex/scholar/auth/providers/OpenAthensAuthenticator.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""OpenAthens authentication for institutional access to academic papers.

This module provides authentication through OpenAthens single sign-on
to enable legal PDF downloads via institutional subscriptions.

This refactored version uses smaller, focused helper classes:
- SessionManager: Handles session state and validation
- AuthCacheManager: Handles session caching operations
- AuthLockManager: Handles concurrent authentication prevention
- BrowserAuthenticator: Handles browser-based authentication
- ScholarNotifier: Handles user notifications with configurable backends
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright

from scitex import logging
from scitex.logging import ScholarError
from scitex.scholar.config import ScholarConfig

from ..core.BrowserAuthenticator import BrowserAuthenticator
from ..session import AuthCacheManager, SessionManager
from ._notifications import ScholarNotifier
from .BaseAuthenticator import BaseAuthenticator

logger = logging.getLogger(__name__)


class OpenAthensError(ScholarError):
    """Raised when OpenAthens authentication fails."""

    pass


class OpenAthensAuthenticator(BaseAuthenticator):
    """Handles OpenAthens authentication for institutional access.

    OpenAthens is a single sign-on system used by many universities
    and institutions to provide seamless access to academic resources.

    This refactored authenticator:
    1. Uses SessionManager for session state management
    2. Uses AuthCacheManager for session persistence
    3. Uses AuthLockManager for concurrent authentication prevention
    4. Uses BrowserAuthenticator for browser-based operations
    5. Uses ScholarNotifier for configurable user notifications
    6. Maintains backward compatibility with original interface
    """

    MYATHENS_URL = "https://my.openathens.net/?passiveLogin=false"
    VERIFICATION_URL = "https://my.openathens.net/account"
    SUCCESS_INDICATORS = [
        "my.openathens.net/account",
        "my.openathens.net/app",
    ]

    def __init__(
        self,
        email: Optional[str] = None,
        timeout: int = 300,
        debug_mode: Optional[bool] = None,
        config: Optional[ScholarConfig] = None,
    ):
        """Initialize OpenAthens authenticator.

        Args:
            email: Institutional email for identification (uses config if None)
            timeout: Authentication timeout in seconds
            debug_mode: Enable debug logging (uses config if None)
            config: ScholarConfig instance (creates new if None)
        """
        # Initialize config first
        if config is None:
            config = ScholarConfig()
        self.scholar_config = (
            config  # Store ScholarConfig separately from BaseAuthenticator's config
        )

        # Use config resolution for email and debug_mode
        self.email = self.scholar_config.resolve("openathens_email", email, None, str)
        self.debug_mode = self.scholar_config.resolve(
            "debug_mode", debug_mode, False, bool
        )
        self.timeout = timeout  # Keep timeout as passed parameter

        BaseAuthenticator.__init__(
            self, config={"email": self.email, "debug_mode": self.debug_mode}
        )

        # Initialize helper components
        self.session_manager = SessionManager(default_expiry_hours=8)
        self.cache_manager = AuthCacheManager(
            "openathens", self.scholar_config, self.email
        )
        self.notifier = ScholarNotifier(self.scholar_config)

        # Create SSO automator for institution-specific automation
        sso_automator = self._create_sso_automator(
            openathens_email=self.email, config=self.scholar_config
        )

        self.browser_authenticator = BrowserAuthenticator(
            mode="interactive", timeout=timeout, sso_automator=sso_automator
        )

    def _create_sso_automator(self, openathens_email=None, config=None):
        """Create appropriate SSO automator based on email domain."""
        try:
            from ..sso import SSOAutomator

            openathens_email = config.resolve(
                "openathens_email", openathens_email, default=None
            )
            return SSOAutomator(email=openathens_email, config=config)
        except Exception as e:
            logger.warning(f"SSO automators not available\n{str(e)}")
            return None

    async def _ensure_session_loaded_async(self) -> None:
        """Ensure session data is loaded from cache if available."""
        if not self.session_manager.has_valid_session_data():
            await self.cache_manager.load_session_async(self.session_manager)

    async def authenticate_async(self, force: bool = False, **kwargs) -> dict:
        """Authenticate with OpenAthens and return session data.

        Args:
            force: Force re-authentication even if session exists
            **kwargs: Additional parameters

        Returns
        -------
            Dictionary containing session cookies
        """
        # Check if we have a valid session
        if not force and await self.is_authenticate_async():
            logger.info(
                f"Using existing OpenAthens session"
                f"{self.session_manager.format_expiry_info()}"
            )
            return self.session_manager.create_auth_response()

        # No lock needed - workers read cached auth concurrently
        # Only one worker should authenticate at a time, but if cache is valid,
        # all workers skip this section
        try:
            # Double-check session after acquiring lock
            await self._ensure_session_loaded_async()
            if not force and await self.is_authenticate_async():
                logger.info(
                    f"Using session authenticate_async by another process"
                    f"{self.session_manager.format_expiry_info()}"
                )
                return self.session_manager.create_auth_response()

            # Perform browser-based authentication
            return await self._perform_browser_authentication_async()

        except Exception:
            # Check if another process authenticated while we were waiting
            await self._ensure_session_loaded_async()
            if await self.is_authenticated_async():
                logger.info(f"{self.name}: Another process authenticated successfully")
                return self.session_manager.create_auth_response()
            raise

    async def _perform_browser_authentication_async(self) -> dict:
        """Perform OpenAthens authentication with automatic SSO automation."""
        logger.info(f"{self.name}: Starting OpenAthens authentication")
        if self.email:
            logger.info(f"Account: {self.email}")

        async with async_playwright() as p:
            # Always start with OpenAthens page
            page = await self.browser_authenticator.navigate_to_login_async(
                self.MYATHENS_URL
            )

            # Display simple instructions
            self._display_login_instructions()

            # Callback for when automation is done and user intervention needed (2FA)
            async def on_intervention_needed(screenshot_data=None):
                await self.notifier.notify_intervention_needed(
                    email=self.email,
                    timeout=self.timeout,
                    login_url=self.MYATHENS_URL,
                    screenshot_data=screenshot_data,
                )

            # Wait for login completion (includes all automation)
            success = await self.browser_authenticator.wait_for_login_completion_async(
                page,
                self.SUCCESS_INDICATORS,
                on_intervention_needed=on_intervention_needed,
            )

            if success:
                # Send success notification via configurable backends
                await self.notifier.notify_success(
                    email=self.email,
                    expiry_info=self.session_manager.format_expiry_info(),
                    verification_url=self.VERIFICATION_URL,
                )
                return await self._handle_successful_authentication_async(page)
            else:
                # Authentication failed - notify user
                await self.notifier.notify_failed(
                    email=self.email,
                    error_details="Authentication timed out",
                    timeout=self.timeout,
                    login_url=self.MYATHENS_URL,
                )
                await page.context.browser.close()
                raise OpenAthensError("Authentication failed or timed out")

    async def _handle_successful_authentication_async(self, page) -> dict:
        """Handle successful authentication by extracting and saving session."""
        # Extract session cookies
        (
            simple_cookies,
            full_cookies,
        ) = await self.browser_authenticator.extract_session_cookies_async(page)

        # Update session manager
        expiry = datetime.now() + timedelta(hours=8)
        self.session_manager.set_session_data(simple_cookies, full_cookies, expiry)

        # Save to cache
        await self.cache_manager.save_session_async(self.session_manager)

        logger.info(
            f"OpenAthens authentication successful"
            f"{self.session_manager.format_expiry_info()}"
        )

        await page.context.browser.close()
        return self.session_manager.create_auth_response()

    async def is_authenticate_async(self, verify_live: bool = True) -> bool:
        """Check if we have a valid authenticate_async session.

        Args:
            verify_live: If True, performs a live check against OpenAthens

        Returns
        -------
            True if authenticate_async, False otherwise
        """
        # Ensure session data is loaded
        await self._ensure_session_loaded_async()

        # Quick local checks
        if not self.session_manager.has_valid_session_data():
            logger.debug(f"{self.name}: No cookies or session expiry found")
            return False

        if self.session_manager.is_session_expired():
            logger.warning("OpenAthens session expired")
            return False

        expiry = self.session_manager.get_session_async_expiry()
        logger.debug(f"Session valid until {expiry}")

        # Perform live verification if requested
        if verify_live:
            return await self._verify_live_authentication_async()

        return True

    async def _verify_live_authentication_async(self) -> bool:
        """Verify authentication with live check if needed."""
        if not self.session_manager.needs_live_verification():
            return True

        cookies = self.session_manager.get_full_cookies()
        is_authenticate_async = (
            await self.browser_authenticator.verify_authentication_async(
                self.VERIFICATION_URL, cookies
            )
        )

        if is_authenticate_async:
            self.session_manager.mark_live_verification()

        return is_authenticate_async

    async def get_auth_headers_async(self) -> Dict[str, str]:
        """Get authentication headers (OpenAthens uses cookies, not headers)."""
        return {}

    async def get_auth_cookies_async(self) -> List[Dict[str, Any]]:
        """Get authentication cookies."""
        if not await self.is_authenticate_async():
            raise OpenAthensError("Not authenticate_async")
        return self.session_manager.get_full_cookies()

    async def logout_async(self, clear_cache=False) -> None:
        """Log out and clear authentication state."""
        self.session_manager.reset_session()

        # Clear cache if requested
        if clear_cache:
            self.cache_manager.clear_cache()

        logger.info(f"{self.name}: Logged out from OpenAthens")

    def _display_login_instructions(self) -> None:
        """Display simple login instructions to user."""
        logger.info(f"{self.name}: OpenAthens Authentication")
        logger.info(f"{self.name}: This will automatically:")
        logger.info(f"{self.name}: 1. Fill in your institutional email")
        logger.info(f"{self.name}: 2. Select your institution")
        logger.info(f"{self.name}: 3. Handle institution SSO if needed")
        logger.info(f"{self.name}: 4. Manual completion if automation fails")

        if self.email:
            logger.info(f"Account: {self.email}")
        logger.info(f"Timeout: {self.timeout} seconds")

    async def get_session_info_async(self) -> Dict[str, Any]:
        """Get information about current session."""
        if not await self.is_authenticate_async():
            return {"authenticate_async": False}

        session_info = self.session_manager.get_session_info_async()
        session_info.update(
            {
                "authenticate_async": True,
                "email": self.email,
            }
        )
        return session_info


if __name__ == "__main__":
    import asyncio

    async def main():
        import argparse

        parser = argparse.ArgumentParser(
            description="OpenAthens authenticator with automatic SSO automation"
        )
        parser.add_argument(
            "--force", action="store_true", help="Force re-authentication"
        )
        parser.add_argument(
            "--manual",
            action="store_true",
            help="Skip automation and require manual intervention",
        )
        parser.add_argument("--email", type=str, help="Override institutional email")
        args = parser.parse_args()

        # Create authenticator
        auth = OpenAthensAuthenticator(email=args.email)

        # Check if already authenticated (unless forced)
        if not args.force:
            try:
                is_authenticated = await auth.is_authenticated_async()
                if is_authenticated:
                    logger.info("Already authenticated! Using cached session.")
                    session_info = await auth.get_session_info_async()
                    logger.info("Current session details:")
                    for key, value in session_info.items():
                        if key != "cookies":  # Don't log sensitive data
                            logger.info(f"  {key}: {value}")
                    return 0
                else:
                    logger.info(
                        "No valid session found, proceeding with authentication..."
                    )
            except Exception as e:
                logger.debug(f"Session check failed: {e}")
                logger.info("Proceeding with authentication...")

        if args.manual:
            logger.info("Manual mode requested - automation will be skipped")
            # Temporarily disable automation by removing SSO automator
            auth.browser_authenticator.sso_automator = None

        try:
            # Always attempt full authentication flow with SSO automation
            result = await auth.authenticate_async(force=args.force)

            if result:
                logger.info("Authentication completed successfully!")
                logger.info("Session details:")
                session_info = await auth.get_session_info_async()
                for key, value in session_info.items():
                    if key != "cookies":  # Don't log sensitive cookie data
                        logger.info(f"  {key}: {value}")

                # Show available cookies count
                if "cookies" in result:
                    logger.info(f"  cookies_count: {len(result['cookies'])}")
            else:
                logger.error("Authentication failed")

        except Exception as e:
            logger.error(f"Authentication error: {e}")

            # Show helpful information for debugging
            logger.info("Troubleshooting tips:")
            logger.info("1. Check your institutional email configuration")
            logger.info("2. Verify your institution has OpenAthens access")
            logger.info("3. Try with --manual flag for manual authentication")
            logger.info("4. Try with --force flag to ignore cache")

            return 1

        return 0

    exit_code = asyncio.run(main())


# python -m scitex.scholar.auth.providers.OpenAthensAuthenticator

# EOF
