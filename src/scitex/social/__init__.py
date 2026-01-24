#!/usr/bin/env python3
# Timestamp: 2026-01-22
# File: /home/ywatanabe/proj/scitex-code/src/scitex/social/__init__.py

"""SciTeX Social - Unified social media management.

This module provides a thin wrapper around socialia, the core social media
integration package. It uses scitex branding and environment variable prefixes.

Features
--------
- Twitter/X posting and analytics
- LinkedIn posting
- Reddit posting
- YouTube analytics
- Google Analytics integration

Environment Variables
---------------------
Credentials use SCITEX_SOCIAL_ prefix (falls back to SOCIALIA_):
- SCITEX_SOCIAL_X_CONSUMER_KEY, SCITEX_SOCIAL_X_CONSUMER_KEY_SECRET
- SCITEX_SOCIAL_X_ACCESS_TOKEN, SCITEX_SOCIAL_X_ACCESS_TOKEN_SECRET
- SCITEX_SOCIAL_X_BEARER_TOKEN
- SCITEX_SOCIAL_LINKEDIN_CLIENT_ID, SCITEX_SOCIAL_LINKEDIN_CLIENT_SECRET
- SCITEX_SOCIAL_LINKEDIN_ACCESS_TOKEN
- SCITEX_SOCIAL_REDDIT_CLIENT_ID, SCITEX_SOCIAL_REDDIT_CLIENT_SECRET
- SCITEX_SOCIAL_YOUTUBE_API_KEY
- SCITEX_SOCIAL_GOOGLE_ANALYTICS_PROPERTY_ID

Usage
-----
    import scitex as stx

    # Twitter/X
    x = stx.social.Twitter()
    x.post("Hello from SciTeX!")

    # LinkedIn
    linkedin = stx.social.LinkedIn()
    linkedin.post("Research update", visibility="public")

    # YouTube analytics
    yt = stx.social.YouTube()
    stats = yt.get_channel_stats()

    # Google Analytics
    ga = stx.social.GoogleAnalytics()
    report = ga.get_report(start_date="7daysAgo")

See Also
--------
- socialia: https://github.com/ywatanabe1989/socialia
- scitex: https://scitex.ai
"""

import os as _os

# Set branding BEFORE importing socialia
_os.environ.setdefault("SOCIALIA_BRAND", "scitex.social")
_os.environ.setdefault("SOCIALIA_ENV_PREFIX", "SCITEX_SOCIAL")

# Check socialia availability
try:
    import socialia as _socialia

    # Re-export platform clients
    from socialia import (
        # Content strategies for MCP
        PLATFORM_STRATEGIES,
        # Base class
        BasePoster,
        GoogleAnalytics,
        LinkedIn,
        LinkedInPoster,
        Reddit,
        RedditPoster,
        # Platform clients (preferred names)
        Twitter,
        # Backward compatibility aliases
        TwitterPoster,
        YouTube,
        YouTubePoster,
    )
    from socialia import __version__ as _socialia_version

    SOCIALIA_AVAILABLE = True
    __socialia_version__ = _socialia_version

except ImportError:
    SOCIALIA_AVAILABLE = False
    __socialia_version__ = None

    # Provide helpful error on access
    class _SocialiaNotAvailable:
        """Placeholder when socialia is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "socialia is required for scitex.social. "
                "Install with: pip install socialia"
            )

        def __getattr__(self, name):
            raise ImportError(
                "socialia is required for scitex.social. "
                "Install with: pip install socialia"
            )

    BasePoster = _SocialiaNotAvailable
    Twitter = _SocialiaNotAvailable
    LinkedIn = _SocialiaNotAvailable
    Reddit = _SocialiaNotAvailable
    YouTube = _SocialiaNotAvailable
    GoogleAnalytics = _SocialiaNotAvailable
    TwitterPoster = _SocialiaNotAvailable
    LinkedInPoster = _SocialiaNotAvailable
    RedditPoster = _SocialiaNotAvailable
    YouTubePoster = _SocialiaNotAvailable
    PLATFORM_STRATEGIES = ""


def has_socialia() -> bool:
    """Check if socialia is available.

    Returns
    -------
    bool
        True if socialia is installed and importable.
    """
    return SOCIALIA_AVAILABLE


__all__ = [
    # Availability check
    "SOCIALIA_AVAILABLE",
    "has_socialia",
    "__socialia_version__",
    # Base class
    "BasePoster",
    # Platform clients (preferred names)
    "Twitter",
    "LinkedIn",
    "Reddit",
    "YouTube",
    "GoogleAnalytics",
    # Backward compatibility aliases
    "TwitterPoster",
    "LinkedInPoster",
    "RedditPoster",
    "YouTubePoster",
    # Content strategies
    "PLATFORM_STRATEGIES",
]

# EOF
