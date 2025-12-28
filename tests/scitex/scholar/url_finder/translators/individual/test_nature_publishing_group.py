# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/nature_publishing_group.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """Nature Publishing Group translator.
# 
# Based on Nature Publishing Group.js translator from Zotero.
# Original JavaScript implementation by Aurimas Vinckevicius.
# """
# 
# import re
# from typing import List
# from playwright.async_api import Page
# from ..core.base import BaseTranslator
# 
# 
# class NaturePublishingGroupTranslator(BaseTranslator):
#     """Nature Publishing Group translator.
# 
#     Supports:
#     - Nature journals (nature.com)
#     - Journal articles (full text, abstracts)
#     - Search results and table of contents
#     - Multiple publishers under Nature Publishing Group umbrella
# 
#     Based on JavaScript translator detectWeb() (lines 476-501) and
#     scrape() function (lines 537-759).
#     """
# 
#     LABEL = "Nature Publishing Group"
#     URL_TARGET_PATTERN = r"^https?://(www\.)?nature\.com/"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         """Check if URL matches Nature Publishing Group pattern.
# 
#         Based on JavaScript detectWeb() (lines 476-501).
#         Excludes PDF URLs and includes various article and issue pages.
# 
#         The JavaScript translator uses complex logic to determine article vs
#         multiple items, including:
#         - Article pages: /full/, /abs/, /articles/
#         - Issue pages: /volumes/\d+/issues/\d+, /journal/v\d+/n\d+/
#         - Search and archive pages
# 
#         Args:
#             url: URL to check
# 
#         Returns:
#             True if URL matches Nature Publishing Group pattern
#         """
#         # Exclude PDF files (JS line 477)
#         if url.endswith(".pdf"):
#             return False
# 
#         # Basic nature.com check
#         if not re.match(cls.URL_TARGET_PATTERN, url):
#             return False
# 
#         # More specific checks based on JavaScript detectWeb (lines 478-500)
#         # Single article patterns (JS line 478)
#         if re.search(
#             r"/(full|abs)/[^/]+($|\?|#)|/fp/.+?[?&]lang=ja(?:&|$)|/articles/", url
#         ):
#             return True
# 
#         # Multiple items patterns (JS lines 481-489)
#         if (
#             "/research/" in url
#             or "/topten/" in url
#             or "/most.htm" in url
#             or "/search?" in url
#             or "sp-q=" in url
#             or "/archive/" in url
#         ):
#             return True
# 
#         # Issue table of contents patterns
#         if re.search(r"journal/v\d+/n\d+/", url) or re.search(
#             r"volumes/\d+/issues/\d+", url
#         ):
#             return True
# 
#         return False
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         """Extract PDF URLs from Nature Publishing Group page.
# 
#         Based on JavaScript getPdfUrl() function (lines 269-275) and
#         scrape() function (lines 699-706).
# 
#         The JavaScript implementation:
#         1. Tries to construct PDF URL from article URL pattern
#         2. Falls back to finding download link with data-track-action="download pdf"
#         3. URL pattern: /(full|abs)/{article_id} -> /pdf/{article_id}.pdf
# 
#         Python implementation:
#         - Method 1: Transform URL pattern (full/abs -> pdf)
#         - Method 2: Look for PDF download link on page
#         - Method 3: Extract from meta tags
# 
#         Args:
#             page: Playwright page object on Nature Publishing Group
# 
#         Returns:
#             List containing PDF URL if found, empty list otherwise
#         """
#         try:
#             # Method 1: Transform URL pattern (JS lines 270-271)
#             # JS: var m = url.match(/(^[^#?]+\/)(?:full|abs)(\/[^#?]+?\.)[a-zA-Z]+(?=$|\?|#)/);
#             # JS: if (m && m.length) return m[1] + 'pdf' + m[2] + 'pdf';
#             current_url = page.url
# 
#             # Pattern: https://www.nature.com/.../full/article_id.html -> .../pdf/article_id.pdf
#             url_match = re.match(
#                 r"(^[^#?]+/)(?:full|abs)(\/[^#?]+?\.)[a-zA-Z]+(?=$|\?|#)", current_url
#             )
# 
#             if url_match:
#                 base_path = url_match.group(1)
#                 article_path = url_match.group(2)
#                 pdf_url = f"{base_path}pdf{article_path}pdf"
#                 return [pdf_url]
# 
#         except Exception:
#             pass
# 
#         # Method 2: Look for download PDF link (JS line 273)
#         # JS: return attr(doc, 'a[data-track-action="download pdf"]', 'href');
#         try:
#             pdf_link = await page.locator(
#                 'a[data-track-action="download pdf"]'
#             ).first.get_attribute("href", timeout=2000)
#             if pdf_link:
#                 # Make absolute URL if needed
#                 if pdf_link.startswith("/"):
#                     base_url = await page.evaluate("window.location.origin")
#                     pdf_link = f"{base_url}{pdf_link}"
#                 elif not pdf_link.startswith("http"):
#                     current_url = page.url
#                     # Remove query params and fragments
#                     base_url = re.sub(r"[?#].*$", "", current_url)
#                     # Remove trailing path
#                     base_url = re.sub(r"/[^/]*$", "", base_url)
#                     pdf_link = f"{base_url}/{pdf_link}"
#                 return [pdf_link]
#         except Exception:
#             pass
# 
#         # Method 3: Try newer URL pattern for articles/* pages
#         # e.g., https://www.nature.com/articles/onc2011282
#         try:
#             current_url = page.url
#             if "/articles/" in current_url:
#                 # Extract article ID
#                 article_match = re.search(r"/articles/([^/?#]+)", current_url)
#                 if article_match:
#                     article_id = article_match.group(1)
#                     # Try different PDF URL patterns
#                     base_url = await page.evaluate("window.location.origin")
# 
#                     # Pattern 1: /articles/article_id.pdf
#                     pdf_url = f"{base_url}/articles/{article_id}.pdf"
# 
#                     # Check if PDF link exists by looking for it in the page
#                     pdf_links = await page.locator('a[href*=".pdf"]').all()
#                     if pdf_links:
#                         for link in pdf_links[:3]:  # Check first 3 PDF links
#                             try:
#                                 href = await link.get_attribute("href", timeout=1000)
#                                 if href and article_id in href:
#                                     if href.startswith("/"):
#                                         href = f"{base_url}{href}"
#                                     return [href]
#                             except Exception:
#                                 continue
# 
#                     # Fallback: construct standard PDF URL
#                     return [pdf_url]
#         except Exception:
#             pass
# 
#         # Method 4: Meta tag fallback
#         try:
#             # Some Nature pages have PDF URL in meta tags
#             pdf_url = await page.locator(
#                 'meta[name="citation_pdf_url"]'
#             ).first.get_attribute("content", timeout=2000)
#             if pdf_url:
#                 return [pdf_url]
#         except Exception:
#             pass
# 
#         return []

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/nature_publishing_group.py
# --------------------------------------------------------------------------------
