#!/usr/bin/env python3
# Timestamp: "2026-01-14"
# File: tests/scitex/scholar/url_finder/test_ScholarURLFinder.py
"""
Comprehensive tests for ScholarURLFinder.

Tests cover:
- Initialization
- DOI extraction from strings
- PDF URL dict conversion
- Find PDF URLs interface routing
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scitex.scholar.url_finder import ScholarURLFinder


class TestScholarURLFinderInit:
    """Tests for ScholarURLFinder initialization."""

    def test_init_with_context(self):
        """Finder should initialize with browser context."""
        mock_context = MagicMock()

        with patch("scitex.scholar.url_finder.ScholarURLFinder.ScholarConfig"):
            with patch("scitex.scholar.url_finder.ScholarURLFinder.OpenURLResolver"):
                finder = ScholarURLFinder(context=mock_context)

        assert finder.name == "ScholarURLFinder"
        assert finder.context == mock_context

    def test_init_with_custom_config(self):
        """Finder should accept custom config."""
        mock_context = MagicMock()
        mock_config = MagicMock()

        with patch("scitex.scholar.url_finder.ScholarURLFinder.OpenURLResolver"):
            finder = ScholarURLFinder(context=mock_context, config=mock_config)

        assert finder.config == mock_config


class TestScholarURLFinderExtractDOI:
    """Tests for _extract_doi method."""

    @pytest.fixture
    def finder(self):
        """Create finder with mocked context."""
        mock_context = MagicMock()
        with patch("scitex.scholar.url_finder.ScholarURLFinder.ScholarConfig"):
            with patch("scitex.scholar.url_finder.ScholarURLFinder.OpenURLResolver"):
                return ScholarURLFinder(context=mock_context)

    def test_extract_doi_plain(self, finder):
        """Should extract plain DOI string."""
        result = finder._extract_doi("10.1038/nature12373")
        assert result == "10.1038/nature12373"

    def test_extract_doi_with_prefix(self, finder):
        """Should extract DOI with 'doi:' prefix."""
        result = finder._extract_doi("doi:10.1038/nature12373")
        assert result == "10.1038/nature12373"

    def test_extract_doi_with_uppercase_prefix(self, finder):
        """Should extract DOI with 'DOI:' prefix."""
        result = finder._extract_doi("DOI:10.1038/nature12373")
        assert result == "10.1038/nature12373"

    def test_extract_doi_with_whitespace(self, finder):
        """Should extract DOI with whitespace."""
        result = finder._extract_doi("  10.1038/nature12373  ")
        assert result == "10.1038/nature12373"

    def test_extract_doi_from_url_returns_none(self, finder):
        """Should return None for HTTP URLs."""
        result = finder._extract_doi("https://example.com/paper")
        assert result is None

    def test_extract_doi_from_http_url_returns_none(self, finder):
        """Should return None for http URLs."""
        result = finder._extract_doi("http://example.com/paper")
        assert result is None

    def test_extract_doi_non_doi_string_returns_none(self, finder):
        """Should return None for non-DOI strings."""
        result = finder._extract_doi("just some text")
        assert result is None


class TestScholarURLFinderAsPdfDicts:
    """Tests for _as_pdf_dicts utility method."""

    @pytest.fixture
    def finder(self):
        """Create finder with mocked context."""
        mock_context = MagicMock()
        with patch("scitex.scholar.url_finder.ScholarURLFinder.ScholarConfig"):
            with patch("scitex.scholar.url_finder.ScholarURLFinder.OpenURLResolver"):
                return ScholarURLFinder(context=mock_context)

    def test_as_pdf_dicts_single_url(self, finder):
        """Should convert single URL to dict."""
        urls = ["https://example.com/paper.pdf"]

        result = finder._as_pdf_dicts(urls, "test_source")

        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/paper.pdf"
        assert result[0]["source"] == "test_source"

    def test_as_pdf_dicts_multiple_urls(self, finder):
        """Should convert multiple URLs to dicts."""
        urls = [
            "https://example.com/paper1.pdf",
            "https://example.com/paper2.pdf",
        ]

        result = finder._as_pdf_dicts(urls, "zotero")

        assert len(result) == 2
        assert all(d["source"] == "zotero" for d in result)

    def test_as_pdf_dicts_empty_list(self, finder):
        """Should return empty list for empty input."""
        result = finder._as_pdf_dicts([], "any_source")

        assert result == []


class TestScholarURLFinderFindPdfUrls:
    """Tests for find_pdf_urls main interface."""

    @pytest.fixture
    def finder(self):
        """Create finder with mocked context."""
        mock_context = MagicMock()
        with patch("scitex.scholar.url_finder.ScholarURLFinder.ScholarConfig"):
            with patch("scitex.scholar.url_finder.ScholarURLFinder.OpenURLResolver"):
                return ScholarURLFinder(context=mock_context)

    @pytest.mark.asyncio
    async def test_find_pdf_urls_routes_string_to_url_method(self, finder):
        """Should route string input to _find_from_url_string."""
        finder._find_from_url_string = AsyncMock(
            return_value=[{"url": "test", "source": "test"}]
        )

        result = await finder.find_pdf_urls("https://example.com/paper")

        finder._find_from_url_string.assert_called_once_with(
            "https://example.com/paper"
        )

    @pytest.mark.asyncio
    async def test_find_pdf_urls_routes_page_to_page_method(self, finder):
        """Should route Page input to _find_from_page."""
        mock_page = MagicMock()
        # Make isinstance check fail for str
        finder._find_from_page = AsyncMock(
            return_value=[{"url": "test", "source": "test"}]
        )

        result = await finder.find_pdf_urls(mock_page, base_url="https://example.com")

        finder._find_from_page.assert_called_once()


class TestScholarURLFinderFindFromUrlString:
    """Tests for _find_from_url_string method."""

    @pytest.fixture
    def finder(self):
        """Create finder with mocked context."""
        mock_context = MagicMock()
        mock_context.new_page = AsyncMock()
        with patch("scitex.scholar.url_finder.ScholarURLFinder.ScholarConfig"):
            with patch("scitex.scholar.url_finder.ScholarURLFinder.OpenURLResolver"):
                return ScholarURLFinder(context=mock_context)

    @pytest.mark.asyncio
    async def test_find_from_url_string_no_context(self, finder):
        """Should return empty list when no context."""
        finder.context = None

        result = await finder._find_from_url_string("https://example.com")

        assert result == []


class TestScholarURLFinderStrategies:
    """Tests for strategy-based PDF finding."""

    @pytest.fixture
    def finder(self):
        """Create finder with mocked context."""
        mock_context = MagicMock()
        with patch("scitex.scholar.url_finder.ScholarURLFinder.ScholarConfig"):
            with patch("scitex.scholar.url_finder.ScholarURLFinder.OpenURLResolver"):
                return ScholarURLFinder(context=mock_context)

    def test_page_load_timeout_constant(self, finder):
        """Should have reasonable page load timeout."""
        assert finder.PAGE_LOAD_TIMEOUT == 30_000


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
