# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/auth/google.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-04 (ywatanabe)"
# # File: ./src/scitex/browser/auth/google.py
# # ----------------------------------------
# """
# Google OAuth authentication helper for Playwright.
# 
# Handles Google OAuth popup flow for services that use "Continue with Google".
# 
# Example:
#     from scitex.browser.auth import GoogleAuthHelper
# 
#     auth = GoogleAuthHelper(
#         email="user@gmail.com",
#         password="your_password",
#     )
# 
#     # Login to a service that uses Google OAuth
#     success = await auth.login_via_google_button(page, 'button:has-text("Continue with Google")')
# """
# 
# import os
# import sys
# from typing import Optional
# from playwright.async_api import Page
# 
# 
# class GoogleAuthHelper:
#     """
#     Google OAuth authentication helper.
# 
#     Handles the popup-based Google OAuth flow used by many services.
# 
#     Environment Variables:
#         GOOGLE_EMAIL: Default email if not provided
#         GOOGLE_PASSWORD: Default password if not provided
#     """
# 
#     def __init__(
#         self,
#         email: Optional[str] = None,
#         password: Optional[str] = None,
#         debug: bool = False,
#     ):
#         """
#         Initialize GoogleAuthHelper.
# 
#         Args:
#             email: Google account email
#             password: Google account password
#             debug: Print debug messages to stderr
#         """
#         self.email = email or os.getenv("GOOGLE_EMAIL", "")
#         self.password = password or os.getenv("GOOGLE_PASSWORD", "")
#         self.debug = debug or bool(os.getenv("GOOGLE_AUTH_DEBUG"))
# 
#     def _log(self, msg: str):
#         """Print debug message if debug mode is enabled."""
#         if self.debug:
#             print(f"[GoogleAuth] {msg}", file=sys.stderr)
# 
#     async def login_via_google_button(
#         self,
#         page: Page,
#         google_button_selector: str = 'button:has-text("Continue with Google")',
#         timeout: int = 60000,
#     ) -> bool:
#         """
#         Perform Google OAuth login via a "Continue with Google" button.
# 
#         This handles the popup-based OAuth flow:
#         1. Click the Google button on the main page
#         2. Handle the Google popup for email/password entry
#         3. Wait for redirect back to the original service
# 
#         Args:
#             page: Playwright Page object (the main page with the Google button)
#             google_button_selector: CSS selector for the Google login button
#             timeout: Maximum time to wait for login (ms)
# 
#         Returns:
#             True if login successful, False otherwise
#         """
#         try:
#             # Find the Google button
#             google_btn = await page.query_selector(google_button_selector)
#             if not google_btn:
#                 # Try alternative selectors
#                 alternatives = [
#                     'button:has-text("Google")',
#                     '[data-testid="google-login"]',
#                     "button >> text=Continue with Google",
#                     "button >> text=Sign in with Google",
#                 ]
#                 for selector in alternatives:
#                     try:
#                         google_btn = await page.query_selector(selector)
#                         if google_btn:
#                             break
#                     except:
#                         continue
# 
#             if not google_btn:
#                 self._log("Google button not found")
#                 return False
# 
#             self._log("Found Google button, clicking...")
# 
#             # Google OAuth opens in a popup - listen for it
#             async with page.context.expect_page(timeout=timeout) as popup_info:
#                 await google_btn.click()
# 
#             popup = await popup_info.value
#             self._log(f"Popup opened: {popup.url[:100]}...")
# 
#             # Handle Google OAuth in popup
#             success = await self._handle_google_popup(popup, timeout)
# 
#             if success:
#                 # Wait for main page to update after OAuth completes
#                 await page.wait_for_timeout(3000)
#                 self._log(f"Login complete, main page URL: {page.url}")
# 
#             return success
# 
#         except Exception as e:
#             self._log(f"Login error: {e}")
#             return False
# 
#     async def _handle_google_popup(self, popup: Page, timeout: int = 60000) -> bool:
#         """
#         Handle the Google OAuth popup flow.
# 
#         Args:
#             popup: The Google OAuth popup page
#             timeout: Maximum time to wait (ms)
# 
#         Returns:
#             True if authentication successful, False otherwise
#         """
#         try:
#             # Wait for Google login page to load
#             await popup.wait_for_load_state("domcontentloaded")
#             await popup.wait_for_timeout(2000)
# 
#             # Step 1: Enter email
#             email_filled = await self._fill_email(popup)
#             if not email_filled:
#                 self._log("Failed to fill email")
#                 return False
# 
#             # Step 2: Wait for password page and enter password
#             password_filled = await self._fill_password(popup)
#             if not password_filled:
#                 self._log("Failed to fill password")
#                 return False
# 
#             # Step 3: Wait for popup to close (indicates success)
#             try:
#                 await popup.wait_for_event("close", timeout=20000)
#                 self._log("Popup closed - login successful")
#                 return True
#             except:
#                 # Check if we're still on Google or redirected
#                 current_url = popup.url
#                 if "accounts.google.com" not in current_url:
#                     self._log("Redirected away from Google - login successful")
#                     return True
#                 self._log("Popup didn't close - possible error")
#                 return False
# 
#         except Exception as e:
#             self._log(f"Popup handling error: {e}")
#             return False
# 
#     async def _fill_email(self, popup: Page) -> bool:
#         """Fill email on Google login page."""
#         try:
#             # Wait for email input
#             await popup.wait_for_selector(
#                 'input[type="email"]', state="visible", timeout=10000
#             )
# 
#             self._log(f"Filling email: {self.email}")
#             await popup.fill('input[type="email"]', self.email)
#             await popup.wait_for_timeout(500)
# 
#             # Click Next button
#             next_btn = await popup.query_selector("#identifierNext")
#             if not next_btn:
#                 next_btn = await popup.query_selector('button:has-text("Next")')
# 
#             if next_btn:
#                 self._log("Clicking Next after email")
#                 await next_btn.click()
#                 await popup.wait_for_timeout(3000)
#                 return True
#             else:
#                 self._log("Next button not found after email")
#                 return False
# 
#         except Exception as e:
#             self._log(f"Email fill error: {e}")
#             return False
# 
#     async def _fill_password(self, popup: Page) -> bool:
#         """Fill password on Google login page."""
#         try:
#             # Wait for password page to load (Google transitions between pages)
#             self._log("Waiting for password page...")
# 
#             # Wait for password input to become visible
#             await popup.wait_for_selector(
#                 'input[type="password"]', state="visible", timeout=15000
#             )
# 
#             self._log("Filling password")
#             await popup.fill('input[type="password"]', self.password)
#             await popup.wait_for_timeout(500)
# 
#             # Click Next button
#             next_btn = await popup.query_selector("#passwordNext")
#             if not next_btn:
#                 next_btn = await popup.query_selector('button:has-text("Next")')
# 
#             if next_btn:
#                 self._log("Clicking Next after password")
#                 await next_btn.click()
#                 await popup.wait_for_timeout(5000)
# 
#                 # Handle 2FA if present
#                 twofa_ok = await self._wait_for_2fa(popup, timeout=60000)
#                 if not twofa_ok:
#                     return False
# 
#                 # Handle potential consent/continue screens
#                 await self._handle_consent_screens(popup)
# 
#                 return True
#             else:
#                 self._log("Next button not found after password")
#                 return False
# 
#         except Exception as e:
#             self._log(f"Password fill error: {e}")
#             return False
# 
#     async def _handle_consent_screens(self, popup: Page) -> None:
#         """Handle OAuth consent or 'Continue' screens that may appear."""
#         try:
#             # Check for Continue button (consent screen)
#             continue_selectors = [
#                 'button:has-text("Continue")',
#                 'button:has-text("Allow")',
#                 "#submit_approve_access",
#                 'button[data-idom-class*="continue"]',
#             ]
# 
#             for selector in continue_selectors:
#                 try:
#                     btn = await popup.query_selector(selector)
#                     if btn and await btn.is_visible():
#                         self._log(f"Found consent button: {selector}")
#                         await btn.click()
#                         await popup.wait_for_timeout(3000)
#                         break
#                 except:
#                     continue
# 
#         except Exception as e:
#             self._log(f"Consent handling: {e}")
# 
#     async def _wait_for_2fa(self, popup: Page, timeout: int = 60000) -> bool:
#         """
#         Wait for 2FA verification to complete.
# 
#         Detects 2FA screens and waits for user to approve on their device.
# 
#         Args:
#             popup: The Google OAuth popup page
#             timeout: Maximum time to wait for 2FA (ms)
# 
#         Returns:
#             True if 2FA completed, False if timed out
#         """
#         try:
#             # Check if we're on a 2FA page
#             page_text = await popup.inner_text("body")
#             twofa_indicators = [
#                 "2-Step Verification",
#                 "Verify it's you",
#                 "confirm it's you",
#                 "Open the Gmail app",
#                 "Check your phone",
#             ]
# 
#             is_2fa = any(
#                 indicator.lower() in page_text.lower() for indicator in twofa_indicators
#             )
# 
#             if is_2fa:
#                 self._log("2FA detected - waiting for user approval...")
#                 # Wait for popup to close or URL to change (indicating 2FA success)
#                 start_url = popup.url
#                 check_interval = 2000  # Check every 2 seconds
#                 elapsed = 0
# 
#                 while elapsed < timeout:
#                     await popup.wait_for_timeout(check_interval)
#                     elapsed += check_interval
# 
#                     # Check if popup closed
#                     try:
#                         current_url = popup.url
#                         if (
#                             current_url != start_url
#                             and "accounts.google.com" not in current_url
#                         ):
#                             self._log("2FA completed - redirected")
#                             return True
#                     except:
#                         # Popup closed
#                         self._log("2FA completed - popup closed")
#                         return True
# 
#                 self._log("2FA timeout")
#                 return False
# 
#             return True  # Not a 2FA page
# 
#         except Exception as e:
#             self._log(f"2FA check error: {e}")
#             return False
# 
#     async def is_logged_in(self, page: Page, login_indicators: list = None) -> bool:
#         """
#         Check if user appears to be logged in.
# 
#         Args:
#             page: Page to check
#             login_indicators: List of URL substrings that indicate NOT logged in
#                              (default: ["login", "signin", "oauth"])
# 
#         Returns:
#             True if appears logged in, False otherwise
#         """
#         if login_indicators is None:
#             login_indicators = ["login", "signin", "oauth", "accounts.google.com"]
# 
#         current_url = page.url.lower()
#         for indicator in login_indicators:
#             if indicator.lower() in current_url:
#                 return False
#         return True
# 
# 
# # Convenience function for quick usage
# async def google_login(
#     page: Page,
#     email: str,
#     password: str,
#     button_selector: str = 'button:has-text("Continue with Google")',
#     debug: bool = False,
# ) -> bool:
#     """
#     Quick Google OAuth login.
# 
#     Args:
#         page: Playwright Page with Google login button
#         email: Google account email
#         password: Google account password
#         button_selector: CSS selector for Google button
#         debug: Print debug messages
# 
#     Returns:
#         True if login successful, False otherwise
# 
#     Example:
#         success = await google_login(page, "user@gmail.com", "password")
#     """
#     auth = GoogleAuthHelper(email=email, password=password, debug=debug)
#     return await auth.login_via_google_button(page, button_selector)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/auth/google.py
# --------------------------------------------------------------------------------
