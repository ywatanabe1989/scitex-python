# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/springer.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """Springer Link translator.
# 
# Based on Springer Link.js translator from Zotero.
# Original JavaScript implementation by Aurimas Vinckevicius.
# """
# 
# import re
# from typing import List
# from urllib.parse import quote
# from playwright.async_api import Page
# from ..core.base import BaseTranslator
# 
# 
# class SpringerTranslator(BaseTranslator):
#     """Springer Link translator.
# 
#     Supports:
#     - Journal articles
#     - Book chapters
#     - Reference work entries
#     - Conference papers (protocols)
#     - Search results pages
#     - Book tables of contents
#     - Journal volume/issue pages
# 
#     Based on JavaScript translator lines 1-282.
#     """
# 
#     LABEL = "Springer Link"
#     URL_TARGET_PATTERN = r"^https?://link\.springer\.com/(search(/page/\d+)?\?|(article|chapter|book|referenceworkentry|protocol|journal|referencework)/.+)"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         """Check if URL matches Springer Link pattern.
# 
#         Based on JavaScript detectWeb() and getAction() (lines 33-67).
#         """
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         """Extract PDF URLs from Springer Link page.
# 
#         Based on JavaScript scrape() function (lines 237-281).
# 
#         The JavaScript implementation:
#         1. Extracts DOI from URL
#         2. Constructs PDF URL as /content/pdf/{DOI}.pdf
#         3. For books, uses DOI translator
#         4. For articles/chapters, fetches RIS from citation service
#         5. Parses RIS and complements with page metadata
# 
#         Python implementation:
#         - Directly constructs PDF URL from DOI in page URL
#         - Extracts DOI from meta tags as fallback
#         - Returns PDF URL for download
# 
#         Args:
#             page: Playwright page object on Springer Link
# 
#         Returns:
#             List containing PDF URL if DOI found, empty list otherwise
#         """
#         try:
#             # Method 1: Extract DOI from current URL (JS line 238)
#             current_url = page.url
#             doi_match = re.search(r"/(10\.[^#?]+)", current_url)
# 
#             if doi_match:
#                 doi = doi_match.group(1)
#                 # Construct PDF URL (JS line 239)
#                 # JavaScript: var pdfURL = "/content/pdf/" + encodeURIComponent(DOI) + ".pdf";
#                 pdf_url = f"/content/pdf/{quote(doi, safe='')}.pdf"
# 
#                 # Make absolute URL
#                 base_url = await page.evaluate("window.location.origin")
#                 pdf_url = f"{base_url}{pdf_url}"
#                 return [pdf_url]
# 
#         except Exception:
#             pass
# 
#         # Method 2: Fallback to meta tag DOI (JS line 131)
#         try:
#             doi = await page.locator('meta[name="citation_doi"]').first.get_attribute(
#                 "content", timeout=2000
#             )
#             if doi:
#                 # Remove any prefix like "doi:" or "https://doi.org/"
#                 doi = re.sub(r"^(doi:|https?://doi\.org/)", "", doi)
#                 pdf_url = f"/content/pdf/{quote(doi, safe='')}.pdf"
# 
#                 base_url = await page.evaluate("window.location.origin")
#                 pdf_url = f"{base_url}{pdf_url}"
#                 return [pdf_url]
#         except Exception:
#             pass
# 
#         # Method 3: Look for direct PDF link on page
#         try:
#             # Some Springer pages have direct PDF download links
#             pdf_link = await page.locator(
#                 'a[data-track-action="download pdf"], a.c-pdf-download__link'
#             ).first.get_attribute("href", timeout=2000)
#             if pdf_link:
#                 # Make absolute URL if needed
#                 if pdf_link.startswith("/"):
#                     base_url = await page.evaluate("window.location.origin")
#                     pdf_link = f"{base_url}{pdf_link}"
#                 elif not pdf_link.startswith("http"):
#                     base_url = await page.evaluate("window.location.href")
#                     # Remove trailing path
#                     base_url = re.sub(r"/[^/]*$", "", base_url)
#                     pdf_link = f"{base_url}/{pdf_link}"
#                 return [pdf_link]
#         except Exception:
#             pass
# 
#         return []

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/springer.py
# --------------------------------------------------------------------------------
