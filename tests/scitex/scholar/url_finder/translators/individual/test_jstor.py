# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/jstor.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """JSTOR translator.
# 
# Translates JSTOR URLs to extract PDF download links and metadata.
# Based on the official Zotero JavaScript translator.
# 
# Key features:
# - Extracts JSTOR ID (JID) from URLs
# - Converts JID to DOI format (10.2307/xxx)
# - Constructs PDF URLs for accessible articles
# - Handles multiple item types (journalArticle, book, bookSection)
# - Detects authentication and access restrictions
# """
# 
# import re
# import urllib.parse
# from typing import List, Optional
# from playwright.async_api import Page
# from ..core.base import BaseTranslator
# 
# 
# class JSTORTranslator(BaseTranslator):
#     """JSTOR translator.
# 
#     Handles JSTOR article, book, and search pages.
#     Extracts PDF URLs and metadata from JSTOR stable URLs.
#     """
# 
#     LABEL = "JSTOR"
#     URL_TARGET_PATTERN = r"^https?://([^/]+\.)?jstor\.org/(discover/|action/(showArticle|doBasicSearch|doAdvancedSearch|doLocatorSearch|doAdvancedResults|doBasicResults)|stable/|pss/|openurl\?|sici\?)"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         """Check if URL is a JSTOR page.
# 
#         Args:
#             url: URL to check
# 
#         Returns:
#             True if URL matches JSTOR pattern
#         """
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     def _extract_jid(cls, url: str) -> Optional[str]:
#         """Extract JSTOR ID (JID) from URL and convert to DOI format.
# 
#         JSTOR uses internal IDs and DOIs. This function:
#         1. Extracts the ID from stable/pss/discover URLs
#         2. Converts numeric IDs to 10.2307/xxx DOI format
#         3. Returns existing DOIs as-is
# 
#         Args:
#             url: JSTOR URL
# 
#         Returns:
#             JID in DOI format (10.xxxx/yyyy) or None if not found
# 
#         Examples:
#             'stable/1593514' -> '10.2307/1593514'
#             'stable/10.1086/245591' -> '10.1086/245591'
#             'stable/131548.pdf' -> '10.2307/131548'
#         """
#         if not url:
#             return None
# 
#         # Pattern matches stable/pss/discover URLs with IDs or DOIs
#         # Matches:
#         # - DOIs: stable/10.1086/245591 (must check DOI pattern first!)
#         # - URL-encoded DOIs: stable/10.1086%2F245591
#         # - numeric IDs: stable/1593514
#         # - With .pdf: stable/131548.pdf
#         # Group 1: the ID or DOI (may be URL-encoded)
#         # Note: DOI pattern must come BEFORE simple numeric pattern to match correctly
#         pattern = r"(?:discover|pss|stable(?:/info|/pdf)?)/(10\.\d+(?:%2F|/)[^?#\s]+|[a-z0-9.]+)"
#         match = re.search(pattern, url, re.IGNORECASE)
# 
#         if not match:
#             return None
# 
#         # Decode URL encoding (e.g., %2F -> /)
#         jid = urllib.parse.unquote(match.group(1))
# 
#         # Remove .pdf extension if present
#         if jid.lower().endswith(".pdf"):
#             jid = jid[:-4]
# 
#         # If not already a DOI (starting with 10.), convert to JSTOR DOI
#         if not jid.startswith("10."):
#             jid = f"10.2307/{jid}"
# 
#         return jid
# 
#     @classmethod
#     async def _detect_item_type(cls, page: Page) -> str:
#         """Detect the type of JSTOR item.
# 
#         Args:
#             page: Playwright page object
# 
#         Returns:
#             Item type: 'book', 'bookSection', or 'journalArticle'
#         """
#         try:
#             # Check for book indicators
#             book_button = await page.query_selector(".book_info_button")
#             if book_button:
#                 return "book"
# 
#             # Check for book chapter indicators
#             analytics = await page.query_selector("script[data-analytics-provider]")
#             if analytics:
#                 content = await analytics.inner_text()
#                 if "chapter view" in content:
#                     return "bookSection"
#         except Exception:
#             pass
# 
#         # Default to journal article
#         return "journalArticle"
# 
#     @classmethod
#     async def _check_pdf_access(cls, page: Page) -> bool:
#         """Check if PDF download is available.
# 
#         JSTOR may restrict PDF access based on:
#         - Institutional access
#         - Individual subscriptions
#         - Article type (books don't have PDFs)
# 
#         Args:
#             page: Playwright page object
# 
#         Returns:
#             True if PDF appears to be accessible
#         """
#         try:
#             # Look for PDF download link/button
#             pdf_selectors = [
#                 'a[data-qa="download-pdf"]',
#                 'a[href*="pdfplus"]',
#                 'a[href*=".pdf"]',
#                 'button:has-text("Download PDF")',
#                 ".download-pdf",
#             ]
# 
#             for selector in pdf_selectors:
#                 element = await page.query_selector(selector)
#                 if element:
#                     # Check if element is visible and not disabled
#                     is_visible = await element.is_visible()
#                     if is_visible:
#                         return True
#         except Exception:
#             pass
# 
#         return False
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         """Extract PDF URLs from JSTOR page.
# 
#         JSTOR PDF URLs follow the pattern:
#         https://www.jstor.org/stable/pdfplus/{jid}.pdf?acceptTC=true
# 
#         Books typically don't have PDFs, only articles and chapters.
# 
#         Args:
#             page: Playwright page object on JSTOR page
# 
#         Returns:
#             List of PDF URLs (empty if no access or wrong item type)
#         """
#         pdf_urls = []
# 
#         try:
#             # Get current page URL
#             url = page.url
# 
#             # Extract JSTOR ID
#             jid = cls._extract_jid(url)
#             if not jid:
#                 return pdf_urls
# 
#             # Detect item type
#             item_type = await cls._detect_item_type(page)
# 
#             # Books don't have PDFs
#             if item_type == "book":
#                 return pdf_urls
# 
#             # Check if PDF access is available
#             has_access = await cls._check_pdf_access(page)
# 
#             if has_access:
#                 # Construct PDF URL
#                 # JSTOR uses pdfplus for enhanced PDFs
#                 pdf_url = (
#                     f"https://www.jstor.org/stable/pdfplus/{jid}.pdf?acceptTC=true"
#                 )
#                 pdf_urls.append(pdf_url)
# 
#         except Exception as e:
#             # Log error but don't fail completely
#             print(f"Error extracting JSTOR PDF URLs: {e}")
# 
#         return pdf_urls
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/jstor.py
# --------------------------------------------------------------------------------
