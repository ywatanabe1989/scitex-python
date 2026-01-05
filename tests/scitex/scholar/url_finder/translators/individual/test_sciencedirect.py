# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/sciencedirect.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """ScienceDirect translator."""
# 
# import re
# from typing import List
# from playwright.async_api import Page
# from ..core.base import BaseTranslator
# 
# 
# class ScienceDirectTranslator(BaseTranslator):
#     """ScienceDirect."""
# 
#     LABEL = "ScienceDirect"
#     URL_TARGET_PATTERN = r"^https?://[^/]*science-?direct\.com[^/]*/((science/)?(article/|(journal|bookseries|book|handbook)/\d)|search[?/]|journal/[^/]+/vol)"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         """Extract PDF URLs from ScienceDirect article page.
# 
#         Based on JavaScript translator getPDFLink() function (lines 67-173).
#         Implements all 5 methods matching JavaScript exactly:
#         1. Direct PDF link (#pdfLink or meta tag)
#         2. Embedded PDF object
#         3. JSON metadata construction
#         4. Canonical link + suffix
#         5. Refetch HTML for .pdf-download-btn-link (last resort)
#         """
#         # JS lines 68-72: Check if PDF is not available
#         try:
#             if await page.locator(".accessContent").first.is_visible(timeout=1000):
#                 return []
#         except:
#             pass
#         try:
#             if await page.locator(".access-options-link-text").first.is_visible(
#                 timeout=1000
#             ):
#                 return []
#         except:
#             pass
#         try:
#             if await page.locator("#check-access-popover").first.is_visible(
#                 timeout=1000
#             ):
#                 return []
#         except:
#             pass
# 
#         # Method 1: Direct PDF link (JS lines 75-76)
#         try:
#             pdf_url = await page.locator("#pdfLink").first.get_attribute(
#                 "href", timeout=2000
#             )
#             if pdf_url and pdf_url != "#":
#                 return [pdf_url]
#         except:
#             pass
# 
#         # Method 1b: Meta tag citation_pdf_url (JS line 76)
#         try:
#             pdf_url = await page.locator(
#                 '[name="citation_pdf_url"]'
#             ).first.get_attribute("content", timeout=2000)
#             if pdf_url:
#                 return [pdf_url]
#         except:
#             pass
# 
#         # Method 2: Embedded PDF object (JS lines 83-92)
#         try:
#             intermediate_url = await page.locator(
#                 ".PdfEmbed > object"
#             ).first.get_attribute("data", timeout=2000)
#             if intermediate_url:
#                 return [intermediate_url]
#         except:
#             pass
# 
#         # Method 3: JSON metadata (JS lines 123-148)
#         try:
#             json_script = await page.locator(
#                 'script[type="application/json"]'
#             ).first.text_content(timeout=2000)
#             if json_script:
#                 import json
# 
#                 data = json.loads(json_script)
# 
#                 # Try to construct PDF URL from JSON
#                 url_metadata = (
#                     data.get("article", {})
#                     .get("pdfDownload", {})
#                     .get("urlMetadata", {})
#                 )
#                 path = url_metadata.get("path")
#                 pdf_extension = url_metadata.get("pdfExtension")
#                 pii = url_metadata.get("pii")
#                 query_params = url_metadata.get("queryParams", {})
#                 md5 = query_params.get("md5")
#                 pid = query_params.get("pid")
# 
#                 if all([path, pdf_extension, pii, md5, pid]):
#                     pdf_url = f"/{path}/{pii}{pdf_extension}?md5={md5}&pid={pid}"
#                     # Make absolute URL
#                     base_url = await page.evaluate("window.location.origin")
#                     pdf_url = f"{base_url}{pdf_url}"
#                     return [pdf_url]
#         except:
#             pass
# 
#         # Method 4: Canonical link + suffix (JS lines 154-159)
#         try:
#             canonical_url = await page.locator(
#                 'link[rel="canonical"]'
#             ).first.get_attribute("href", timeout=2000)
#             if canonical_url:
#                 pdf_url = f"{canonical_url}/pdfft?download=true"
#                 return [pdf_url]
#         except:
#             pass
# 
#         # Method 5: Refetch HTML for .pdf-download-btn-link (JS lines 164-172)
#         # Note: This is a last resort method that requires refetching the page
#         # Skipping this in Python as it's expensive and rarely needed
#         # The JavaScript uses: let reloadedDoc = await requestDocument(url);
# 
#         return []

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/sciencedirect.py
# --------------------------------------------------------------------------------
