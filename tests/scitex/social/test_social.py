#!/usr/bin/env python3
# Timestamp: 2026-01-22
# File: tests/scitex/social/test_social.py

"""Tests for scitex.social module."""

import pytest


class TestSocialImport:
    """Test basic import functionality."""

    def test_import_social_module(self):
        """Test that scitex.social can be imported."""
        from scitex import social

        assert social is not None

    def test_socialia_available_flag(self):
        """Test SOCIALIA_AVAILABLE flag exists and is boolean."""
        from scitex.social import SOCIALIA_AVAILABLE

        assert isinstance(SOCIALIA_AVAILABLE, bool)

    def test_has_socialia_function(self):
        """Test has_socialia() helper function."""
        from scitex.social import has_socialia

        result = has_socialia()
        assert isinstance(result, bool)

    def test_socialia_version(self):
        """Test __socialia_version__ is accessible."""
        from scitex.social import SOCIALIA_AVAILABLE, __socialia_version__

        if SOCIALIA_AVAILABLE:
            assert __socialia_version__ is not None
            assert isinstance(__socialia_version__, str)
        else:
            assert __socialia_version__ is None


class TestSocialClasses:
    """Test platform client class exports."""

    def test_twitter_class_exists(self):
        """Test Twitter class is exported."""
        from scitex.social import Twitter

        assert Twitter is not None

    def test_linkedin_class_exists(self):
        """Test LinkedIn class is exported."""
        from scitex.social import LinkedIn

        assert LinkedIn is not None

    def test_reddit_class_exists(self):
        """Test Reddit class is exported."""
        from scitex.social import Reddit

        assert Reddit is not None

    def test_youtube_class_exists(self):
        """Test YouTube class is exported."""
        from scitex.social import YouTube

        assert YouTube is not None

    def test_google_analytics_class_exists(self):
        """Test GoogleAnalytics class is exported."""
        from scitex.social import GoogleAnalytics

        assert GoogleAnalytics is not None

    def test_base_poster_class_exists(self):
        """Test BasePoster class is exported."""
        from scitex.social import BasePoster

        assert BasePoster is not None


class TestBackwardCompatibility:
    """Test backward compatibility aliases."""

    def test_twitter_poster_alias(self):
        """Test TwitterPoster alias exists."""
        from scitex.social import TwitterPoster

        assert TwitterPoster is not None

    def test_linkedin_poster_alias(self):
        """Test LinkedInPoster alias exists."""
        from scitex.social import LinkedInPoster

        assert LinkedInPoster is not None

    def test_reddit_poster_alias(self):
        """Test RedditPoster alias exists."""
        from scitex.social import RedditPoster

        assert RedditPoster is not None

    def test_youtube_poster_alias(self):
        """Test YouTubePoster alias exists."""
        from scitex.social import YouTubePoster

        assert YouTubePoster is not None


class TestAllExports:
    """Test __all__ exports."""

    def test_all_defined(self):
        """Test __all__ is defined."""
        from scitex import social

        assert hasattr(social, "__all__")
        assert isinstance(social.__all__, list)

    def test_all_exports_accessible(self):
        """Test all items in __all__ are accessible."""
        from scitex import social

        for name in social.__all__:
            assert hasattr(social, name), f"Missing export: {name}"


@pytest.mark.skipif(
    not pytest.importorskip("socialia", reason="socialia not installed"),
    reason="socialia not installed",
)
class TestSocialiaIntegration:
    """Integration tests requiring socialia to be installed."""

    def test_twitter_is_socialia_class(self):
        """Test Twitter is the actual socialia class."""
        from socialia import Twitter as SocialiaTwitter

        from scitex.social import Twitter

        assert Twitter is SocialiaTwitter

    def test_branding_env_prefix(self):
        """Test SCITEX_SOCIAL_ environment prefix is set."""
        import os

        from socialia._branding import ENV_PREFIX

        # Import to trigger branding setup
        from scitex import social  # noqa: F401

        assert ENV_PREFIX == "SCITEX_SOCIAL"


# EOF
