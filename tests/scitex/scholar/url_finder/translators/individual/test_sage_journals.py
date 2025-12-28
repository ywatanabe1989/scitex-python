# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/sage_journals.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """SAGE Journals translator.
# 
# Based on SAGE Journals.js translator from Zotero.
# Original JavaScript implementation by Sebastian Karcher and Philipp Zumstein.
# 
# Supports:
# - Journal articles (abs, full, pdf, doi views)
# - Search results pages
# - Table of contents pages
# - Automatic PDF URL construction from DOI
# 
# SAGE uses Atypon platform but has distinct features that warrant a separate translator.
# """
# 
# import re
# from typing import List
# from urllib.parse import quote
# from playwright.async_api import Page
# from ..core.base import BaseTranslator
# 
# 
# class SAGEJournalsTranslator(BaseTranslator):
#     """SAGE Journals translator.
# 
#     Handles journals.sagepub.com content including:
#     - Journal articles in various views (abs, full, pdf)
#     - DOI-based URLs without view prefix
#     - Search results pages
#     - Journal issue/volume table of contents
# 
#     Based on JavaScript translator (SAGE Journals.js, lines 1-427).
#     """
# 
#     LABEL = "SAGE Journals"
#     URL_TARGET_PATTERN = r"^https?://journals\.sagepub\.com(/doi/((abs|full|pdf)/)?10\.|/action/doSearch\?|/toc/)"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         """Check if URL matches SAGE Journals pattern.
# 
#         Based on JavaScript detectWeb() function (lines 40-49).
# 
#         The JavaScript implementation checks for:
#         1. Article URLs matching /(abs|full|pdf|doi)\/10\./
#         2. Search results with getSearchResults()
# 
#         Args:
#             url: URL to check
# 
#         Returns:
#             True if URL matches SAGE Journals pattern
#         """
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         """Extract PDF URLs from SAGE Journals page.
# 
#         Based on JavaScript scrape() function (lines 81-147).
# 
#         The JavaScript implementation:
#         1. Extracts DOI from meta tag dc.Identifier or URL (lines 83-86)
#         2. Constructs PDF URL as /doi/pdf/{DOI} (line 88)
#         3. Downloads RIS citation for metadata (lines 82-92)
#         4. Parses RIS and complements with page metadata (lines 103-145)
#         5. Extracts title, subtitle, abstract from DOM (lines 110-123)
#         6. Adds PDF as attachment (lines 139-143)
# 
#         Python implementation:
#         - Extracts DOI from meta tags or URL
#         - Constructs direct PDF URL pattern: /doi/pdf/{DOI}
#         - Returns PDF URL for download
# 
#         Args:
#             page: Playwright page object on SAGE Journals
# 
#         Returns:
#             List containing PDF URL if DOI found, empty list otherwise
#         """
#         pdf_urls = []
# 
#         try:
#             # Method 1: Extract DOI from meta tag (JavaScript line 83)
#             # JavaScript: var doi = ZU.xpathText(doc, '//meta[@name="dc.Identifier" and @scheme="doi"]/@content');
#             doi = await page.locator(
#                 'meta[name="dc.Identifier"][scheme="doi"]'
#             ).first.get_attribute("content", timeout=3000)
# 
#             if doi:
#                 # Construct PDF URL (JavaScript line 88)
#                 # JavaScript: let pdfurl = "//" + doc.location.host + "/doi/pdf/" + doi;
#                 base_url = await page.evaluate("window.location.origin")
#                 pdf_url = f"{base_url}/doi/pdf/{doi}"
#                 pdf_urls.append(pdf_url)
#                 return pdf_urls
# 
#         except Exception:
#             pass
# 
#         # Method 2: Extract DOI from URL (JavaScript lines 84-86)
#         # JavaScript: doi = url.match(/10\.[^?#]+/)[0];
#         try:
#             current_url = page.url
#             doi_match = re.search(r"(10\.[^?#]+)", current_url)
# 
#             if doi_match:
#                 doi = doi_match.group(1)
#                 base_url = await page.evaluate("window.location.origin")
#                 pdf_url = f"{base_url}/doi/pdf/{doi}"
#                 pdf_urls.append(pdf_url)
#                 return pdf_urls
# 
#         except Exception:
#             pass
# 
#         # Method 3: Look for PDF link on page
#         # SAGE pages typically have a PDF download button
#         try:
#             # Common selectors for SAGE PDF download links
#             selectors = [
#                 "a.show-pdf",
#                 'a[href*="/doi/pdf/"]',
#                 "a.pdf-download",
#                 'a[title*="PDF"]',
#                 '.epub-section__access-options a[href*="pdf"]',
#                 "a.download-pdf-link",
#             ]
# 
#             for selector in selectors:
#                 try:
#                     pdf_link = await page.locator(selector).first.get_attribute(
#                         "href", timeout=1000
#                     )
#                     if pdf_link:
#                         # Make absolute URL if needed
#                         if pdf_link.startswith("/"):
#                             base_url = await page.evaluate("window.location.origin")
#                             pdf_link = f"{base_url}{pdf_link}"
#                         elif not pdf_link.startswith("http"):
#                             base_url = await page.evaluate("window.location.href")
#                             base_url = re.sub(r"/[^/]*$", "", base_url)
#                             pdf_link = f"{base_url}/{pdf_link}"
# 
#                         pdf_urls.append(pdf_link)
#                         return pdf_urls
#                 except Exception:
#                     continue
# 
#         except Exception:
#             pass
# 
#         # Method 4: Check if we're on a PDF page already
#         # If URL contains /doi/pdf/, it's already a PDF URL
#         try:
#             current_url = page.url
#             if "/doi/pdf/" in current_url:
#                 pdf_urls.append(current_url)
#                 return pdf_urls
#         except Exception:
#             pass
# 
#         # Method 5: Try to construct from current page URL by changing view type
#         # Convert /doi/abs/ or /doi/full/ to /doi/pdf/
#         try:
#             current_url = page.url
#             if (
#                 "/doi/abs/" in current_url
#                 or "/doi/full/" in current_url
#                 or re.search(r"/doi/10\.", current_url)
#             ):
#                 # Replace view type with pdf
#                 pdf_url = re.sub(r"/doi/(abs|full)/", "/doi/pdf/", current_url)
#                 # If URL is /doi/10.xxx without view prefix, insert pdf
#                 pdf_url = re.sub(r"/doi/(10\.)", r"/doi/pdf/\1", pdf_url)
# 
#                 pdf_urls.append(pdf_url)
#                 return pdf_urls
#         except Exception:
#             pass
# 
#         return pdf_urls

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/sage_journals.py
# --------------------------------------------------------------------------------
