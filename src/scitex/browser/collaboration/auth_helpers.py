#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-19 05:40:00 (ywatanabe)"
# File: ./src/scitex/browser/collaboration/auth_helpers.py
# ----------------------------------------
"""
Simple authentication helpers for SharedBrowserSession.

Start small - just Django login for now.
"""

import os
from typing import Optional
from playwright.async_api import Page


class DjangoAuthHelper:
    """
    Simple Django authentication helper.

    Example:
        auth = DjangoAuthHelper(
            login_url="http://127.0.0.1:8000/auth/login/",
            username="your_user",
            password="your_pass",
        )

        success = await auth.login(page)
    """

    def __init__(
        self,
        login_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        username_field: str = "#id_username",
        password_field: str = "#id_password",
        submit_button: str = "button[type='submit']",
        success_indicator: str = "/core/",  # URL contains this after login
    ):
        self.login_url = login_url
        self.username = username or os.getenv("SCITEX_CLOUD_USERNAME", "")
        self.password = password or os.getenv("SCITEX_CLOUD_PASSWORD", "")
        self.username_field = username_field
        self.password_field = password_field
        self.submit_button = submit_button
        self.success_indicator = success_indicator

    async def login(self, page: Page) -> bool:
        """
        Perform Django login.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Navigate to login page
            await page.goto(self.login_url, wait_until="load")

            # Fill username
            await page.fill(self.username_field, self.username)

            # Fill password
            await page.fill(self.password_field, self.password)

            # Submit
            await page.click(self.submit_button)

            # Wait for redirect (success indicator in URL)
            await page.wait_for_url(f"**{self.success_indicator}**", timeout=5000)

            print(f"✅ Logged in as: {self.username}")
            return True

        except Exception as e:
            print(f"❌ Login failed: {e}")
            return False

    async def is_logged_in(self, page: Page) -> bool:
        """
        Check if currently logged in.

        Simple check: not on login page.
        """
        current_url = page.url
        is_logged_in = "login" not in current_url.lower()
        return is_logged_in

    async def logout(
        self, page: Page, logout_url: str = "http://127.0.0.1:8000/auth/logout/"
    ):
        """Logout (navigate to logout URL)."""
        await page.goto(logout_url)


# EOF
