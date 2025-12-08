#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-04 (ywatanabe)"
# File: ./src/scitex/browser/auth/__init__.py
# ----------------------------------------
"""
scitex.browser.auth - Authentication helpers for browser automation.

Provides reusable authentication handlers for various OAuth providers
and login flows used in browser automation tasks.

Features:
- Google OAuth (popup-based flow)
- Django session auth
- Generic credential management

Example:
    from scitex.browser.auth import GoogleAuthHelper, google_login

    # Quick login
    success = await google_login(page, "user@gmail.com", "password")

    # Or with helper class
    auth = GoogleAuthHelper(email="user@gmail.com", password="password")
    success = await auth.login_via_google_button(page)
"""

from .google import GoogleAuthHelper, google_login

__all__ = [
    "GoogleAuthHelper",
    "google_login",
]

# EOF
