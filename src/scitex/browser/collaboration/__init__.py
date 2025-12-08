#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-19 05:20:00 (ywatanabe)"
# File: ./src/scitex/browser/collaboration/__init__.py
# ----------------------------------------
"""
scitex.browser.collaboration - Interactive browser automation for AI-human teams.

This is a NEW module that does NOT affect existing scitex.browser functionality.

Features:
- Persistent shared browser sessions
- Multi-agent coordination
- AI-human collaboration
- Authentication handling
- Content extraction

Version: 0.1.0-alpha (experimental)
"""

__version__ = "0.1.0-alpha"
__experimental__ = True

# Import components
from .shared_session import SharedBrowserSession, SessionConfig
from .visual_feedback import VisualFeedback
from .credential_manager import CredentialManager

# Re-export auth helpers for convenience
from scitex.browser.auth import GoogleAuthHelper, google_login

# Exports
__all__ = [
    "SharedBrowserSession",
    "SessionConfig",
    "VisualFeedback",
    "CredentialManager",
    "GoogleAuthHelper",
    "google_login",
]


# Compatibility check - ensure we don't break existing code
def _check_compatibility():
    """Verify existing scitex.browser still works."""
    try:
        from scitex.browser import browser_logger
        from scitex.browser.automation import CookieAutoAcceptor
        from scitex.browser.interaction import click_center_async

        return True
    except ImportError as e:
        raise RuntimeError(
            f"❌ Collaboration module broke existing imports: {e}\n"
            "This should never happen! Please report this bug."
        )


_check_compatibility()

print("✅ scitex.browser.collaboration loaded (experimental)")

# EOF
