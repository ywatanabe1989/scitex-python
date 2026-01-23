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

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/ScholarURLFinder.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-26 17:04:54 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/ScholarURLFinder.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/scholar/url_finder/ScholarURLFinder.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# ScholarURLFinder - Find PDF URLs from web pages.
# 
# Simple, focused responsibility: Given a page or URL, find PDF URLs.
# """
# 
# from contextlib import asynccontextmanager
# from typing import Dict
# from typing import List, Optional, Union
# 
# from playwright.async_api import BrowserContext
# from playwright.async_api import Page
# 
# from scitex import logging
# from scitex.browser.debugging import browser_logger
# from scitex.scholar.auth.gateway import OpenURLResolver
# from scitex.scholar.config import PublisherRules
# from scitex.scholar.config import ScholarConfig
# 
# # Import strategies
# from .strategies import find_pdf_urls_by_direct_links
# from .strategies import find_pdf_urls_by_navigation
# from .strategies import find_pdf_urls_by_publisher_patterns
# from .strategies import find_pdf_urls_by_zotero_translators
# 
# # Strategy execution order (priority order)
# STRATEGIES = [
#     # 1. Zotero translators (most reliable)
#     {
#         "name": "Python Zotero Translators",
#         "function": find_pdf_urls_by_zotero_translators,
#         "source_label": "zotero_translator",
#     },
#     # 2. Direct links (combined: href + dropdown)
#     {
#         "name": "Direct Links",
#         "function": find_pdf_urls_by_direct_links,
#         "source_label": "direct_link",
#     },
#     # 3. Navigation (Elsevier only)
#     {
#         "name": "Navigation",
#         "function": find_pdf_urls_by_navigation,
#         "source_label": "navigation",
#         "publisher_filter": "elsevier",  # Only for Elsevier domains
#     },
#     # 4. Publisher patterns (fallback)
#     {
#         "name": "Publisher Patterns",
#         "function": find_pdf_urls_by_publisher_patterns,
#         "source_label": "publisher_pattern",
#     },
# ]
# 
# logger = logging.getLogger(__name__)
# 
# 
# class ScholarURLFinder:
#     """Find PDF URLs from web pages.
# 
#     Simple, focused responsibility:
#     - Input: Page or URL string
#     - Output: List of PDF URLs
# 
#     Authentication/DOI resolution should be handled BEFORE calling this.
#     """
# 
#     PAGE_LOAD_TIMEOUT = 30_000
# 
#     def __init__(
#         self,
#         context: BrowserContext,
#         config: Optional[ScholarConfig] = None,
#     ):
#         self.name = self.__class__.__name__
#         self.config = config or ScholarConfig()
#         self.context = context
#         self.openurl_resolver = OpenURLResolver(config=self.config)
# 
#     # ==========================================================================
#     # Public API
#     # ==========================================================================
# 
#     async def find_pdf_urls(
#         self, page_or_url: Union[Page, str], base_url: Optional[str] = None
#     ) -> List[Dict]:
#         """Find PDF URLs from page or URL string.
# 
#         Args:
#             page_or_url: Playwright Page object or URL string
#             base_url: Optional base URL for the page
# 
#         Returns:
#             List of PDF URL dicts: [{"url": "...", "source": "zotero_translator"}]
#         """
#         if isinstance(page_or_url, str):
#             return await self._find_from_url_string(page_or_url)
#         else:
#             return await self._find_from_page(page_or_url, base_url)
# 
#     # ==========================================================================
#     # PDF Finding Implementation
#     # ==========================================================================
# 
#     async def _find_pdf_urls_with_strategies(
#         self, page: Page, base_url: Optional[str] = None
#     ) -> List[Dict]:
#         """Try strategies in priority order."""
#         base_url = base_url or page.url
#         n_strategies = len(STRATEGIES)
# 
#         for i_strategy, strategy in enumerate(STRATEGIES, 1):
#             # Check if strategy should run for this URL
#             publisher_filter = strategy.get("publisher_filter")
#             if publisher_filter and publisher_filter.lower() not in base_url.lower():
#                 # logger.debug(
#                 #     f"{self.name}: Skipping {strategy['name']} (filtered)"
#                 # )
#                 continue
# 
#             # Log progress
#             await browser_logger.info(
#                 page,
#                 f"{self.name}: {i_strategy}/{n_strategies} Trying {strategy['name']}",
#             )
# 
#             try:
#                 # Execute strategy
#                 urls = await strategy["function"](page, base_url, self.config)
# 
#                 if urls:
#                     result = self._as_pdf_dicts(urls, strategy["source_label"])
#                     await browser_logger.info(
#                         page,
#                         f"{self.name}: ✓ {strategy['name']} found {len(result)} URLs",
#                     )
#                     return result
# 
#             except Exception as e:
#                 await browser_logger.debug(
#                     page, f"{self.name}: {strategy['name']} failed: {e}"
#                 )
#                 continue
# 
#         return []
# 
#     def _extract_doi(self, url: str) -> Optional[str]:
#         """Extract DOI from string if present.
# 
#         Args:
#             url: URL string or DOI
# 
#         Returns:
#             DOI string if found, None otherwise
# 
#         Examples:
#             >>> _extract_doi("10.1038/nature12345")
#             "10.1038/nature12345"
#             >>> _extract_doi("doi:10.1038/nature12345")
#             "10.1038/nature12345"
#             >>> _extract_doi("https://example.com")
#             None
#         """
#         import re
# 
#         url = url.strip()
# 
#         # Already a valid URL - not a DOI
#         if url.startswith(("http://", "https://")):
#             return None
# 
#         # Remove common DOI prefixes
#         doi_pattern = r"^(?:doi:\s*|DOI:\s*)?(.+)$"
#         match = re.match(doi_pattern, url, re.IGNORECASE)
#         if match:
#             potential_doi = match.group(1).strip()
# 
#             # Check if it looks like a DOI (starts with 10.)
#             if potential_doi.startswith("10."):
#                 return potential_doi
# 
#         # If it starts with 10., assume it's a DOI
#         if url.startswith("10."):
#             return url
# 
#         return None
# 
#     async def _find_from_url_string(self, url: str) -> List[Dict]:
#         """Find PDFs from URL string or DOI."""
#         if not self.context:
#             logger.error(f"{self.name}: Browser context required")
#             return []
# 
#         # Check if input is a DOI and resolve it to publisher URL
#         doi = self._extract_doi(url)
#         if doi:
#             logger.info(f"{self.name}: Detected DOI: {doi}")
#             async with self._managed_page() as page:
#                 resolved_url = await self.openurl_resolver.resolve_doi(doi, page)
#                 if resolved_url:
#                     logger.info(f"{self.name}: Resolved DOI to: {resolved_url}")
#                     url = resolved_url
#                 else:
#                     # Fallback to direct DOI URL (works for open access papers like arXiv)
#                     url = f"https://doi.org/{doi}"
#                     logger.info(
#                         f"{self.name}: OpenURL failed, using direct DOI URL: {url}"
#                     )
# 
#         logger.info(f"{self.name}: Finding PDFs from URL: {url}")
# 
#         async with self._managed_page() as page:
#             try:
#                 await page.goto(
#                     url,
#                     wait_until="domcontentloaded",
#                     timeout=self.PAGE_LOAD_TIMEOUT,
#                 )
#                 pdfs = await self._find_pdf_urls_with_strategies(page)
# 
#                 if not pdfs:
#                     await browser_logger.warning(
#                         page, f"{self.name}: No PDFs from URL: {url[:50]}"
#                     )
# 
#                 return pdfs
#             except Exception as e:
#                 # logger.warning(f"{self.name}: Failed to load page: {e}")
#                 await browser_logger.error(page, f"{self.name}: Navigation Error {e}")
#                 return []
# 
#     async def _find_from_page(
#         self, page: Page, base_url: Optional[str] = None
#     ) -> List[Dict]:
#         """Find PDFs from existing page."""
#         try:
#             pdfs = await self._find_pdf_urls_with_strategies(page, base_url)
# 
#             if not pdfs:
#                 await browser_logger.warning(page, f"{self.name}: No PDFs on page")
# 
#             return pdfs
#         except Exception as e:
#             # logger.error(f"{self.name}: Error finding PDFs: {e}")
#             await browser_logger.error(page, f"{self.name}: PDF Search Error {e}")
#             return []
# 
#     # ==========================================================================
#     # Utilities
#     # ==========================================================================
# 
#     @asynccontextmanager
#     async def _managed_page(self):
#         """Context manager for page lifecycle."""
#         page = await self.context.new_page()
#         try:
#             yield page
#         finally:
#             try:
#                 await page.close()
#             except:
#                 pass
# 
#     def _as_pdf_dicts(self, urls: List[str], source: str) -> List[Dict]:
#         """Convert URL strings to dict format with source."""
#         return [{"url": url, "source": source} for url in urls]
# 
# 
# # ==============================================================================
# # CLI
# # ==============================================================================
# 
# 
# def parse_args():
#     """Parse CLI arguments."""
#     import argparse
# 
#     parser = argparse.ArgumentParser(
#         description="Find PDF URLs from a publisher page or DOI"
#     )
#     parser.add_argument(
#         "--url",
#         required=True,
#         help="URL or DOI to search for PDFs (e.g., 'https://...' or '10.1038/...' or 'doi:10.1038/...')",
#     )
#     parser.add_argument(
#         "--browser-mode",
#         choices=["interactive", "stealth"],
#         default="interactive",
#     )
#     parser.add_argument("--chrome-profile", default="system_worker_0")
#     return parser.parse_args()
# 
# 
# if __name__ == "__main__":
#     import asyncio
# 
#     async def main_async():
#         from pprint import pprint
# 
#         from scitex.scholar import ScholarAuthManager, ScholarBrowserManager
# 
#         args = parse_args()
# 
#         # Setup auth manager and browser
#         auth_manager = ScholarAuthManager()
#         browser_manager = ScholarBrowserManager(
#             auth_manager=auth_manager,
#             browser_mode=args.browser_mode,
#             chrome_profile_name=args.chrome_profile,
#         )
#         _, context = await browser_manager.get_authenticated_browser_and_context_async()
#         # time.sleep(1)
# 
#         # Find PDFs
#         url_finder = ScholarURLFinder(context)
#         pdfs = await url_finder.find_pdf_urls(args.url)
# 
#         print(f"\nFound {len(pdfs)} PDF URLs:")
#         pprint(pdfs)
# 
#         await browser_manager.close()
# 
#     asyncio.run(main_async())
# 
# """
# # With URL
# python -m scitex.scholar.url_finder.ScholarURLFinder \
# 	--url "https://arxiv.org/abs/2308.09312" \
# 	--browser-mode stealth
# 
# # With DOI
# python -m scitex.scholar.url_finder.ScholarURLFinder \
# 	--url "10.1038/s41598-017-02626-y" \
# 	--browser-mode stealth
# 
# # With doi: prefix
# python -m scitex.scholar.url_finder.ScholarURLFinder \
# 	--url "doi:10.3389/fnins.2024.1472747" \
# 	--browser-mode stealth
# 
# # No doi prefix
# python -m scitex.scholar.url_finder.ScholarURLFinder \
# 	--url "10.1016/j.clinph.2024.09.017" \
# 	--browser-mode stealth
# 
# # Found 1 PDF URLs:
# # [{'source': 'zotero_translator',
# #   'url': 'https://www.sciencedirect.com/science/article/pii/S1388245724002761/pdfft?download=true'}]
# """
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/ScholarURLFinder.py
# --------------------------------------------------------------------------------
