# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/arxiv_org.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """arXiv.org translator.
# 
# Translates arXiv.org URLs to extract PDF links and metadata.
# Based on the JavaScript translator by Sean Takats and Michael Berkowitz.
# 
# Supports:
# - Abstract pages: https://arxiv.org/abs/XXXX.XXXXX
# - PDF pages: https://arxiv.org/pdf/XXXX.XXXXX
# - Search results: https://arxiv.org/search/...
# - Legacy search: https://arxiv.org/find/...
# - Listings: https://arxiv.org/list/...
# """
# 
# import re
# from typing import List
# from playwright.async_api import Page
# from ..core.base import BaseTranslator
# 
# 
# class ArXivOrgTranslator(BaseTranslator):
#     """arXiv.org translator for extracting PDF URLs."""
# 
#     LABEL = "arXiv.org"
#     URL_TARGET_PATTERN = r"^https?://([^.]+\.)?(arxiv\.org|xxx\.lanl\.gov)/(search|find|catchup|list/\w|abs/|pdf/)"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         """Check if URL is an arXiv.org URL.
# 
#         Args:
#             url: URL to check
# 
#         Returns:
#             True if URL matches arXiv.org pattern
#         """
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         """Extract PDF URLs from arXiv.org page.
# 
#         For abstract pages (abs/), extracts the PDF link.
#         For PDF pages (pdf/), returns the current URL.
#         For search/list pages, extracts all paper PDF links.
# 
#         Args:
#             page: Playwright page object
# 
#         Returns:
#             List of PDF URLs found on the page
#         """
#         url = page.url
#         pdf_urls = []
# 
#         # Handle direct PDF pages
#         if "/pdf/" in url:
#             # Already on a PDF page - ensure it has .pdf extension
#             if not url.endswith(".pdf"):
#                 url = url + ".pdf"
#             pdf_urls.append(url)
#             return pdf_urls
# 
#         # Handle abstract pages - simple and reliable
#         if "/abs/" in url:
#             # Extract arXiv ID from URL (everything after /abs/)
#             match = re.search(r"/abs/([^?#/]+)", url)
#             if match:
#                 arxiv_id = match.group(1).strip()
#                 # Remove any version suffix if present (e.g., v1, v2)
#                 # but keep it in the URL as arXiv supports it
#                 # Construct PDF URL - arXiv.org pattern is always predictable
#                 pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
#                 pdf_urls.append(pdf_url)
#                 return pdf_urls
#             else:
#                 # Fallback: if regex fails, log error
#                 print(f"arXiv translator: Could not extract arXiv ID from URL: {url}")
#                 return []
# 
#         # Handle search, find, list, and catchup pages - multiple items
#         try:
#             # New search results format
#             if "/search/" in url:
#                 results = await page.query_selector_all(".arxiv-result")
#                 for result in results:
#                     link_elem = await result.query_selector(".list-title a")
#                     if link_elem:
#                         href = await link_elem.get_attribute("href")
#                         if href:
#                             # Extract arXiv ID
#                             arxiv_id = (
#                                 href.strip().replace("arXiv:", "").replace("/abs/", "")
#                             )
#                             pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
#                             pdf_urls.append(pdf_url)
# 
#             # Legacy search, listings, and catchup pages
#             elif any(pattern in url for pattern in ["/find/", "/list/", "/catchup"]):
#                 # Find all abstract links in <dt> elements
#                 dt_elements = await page.query_selector_all(
#                     "#dlpage dt a[title='Abstract']"
#                 )
#                 for link_elem in dt_elements:
#                     href = await link_elem.get_attribute("href")
#                     if href:
#                         # Extract arXiv ID
#                         match = re.search(r"/abs/([^?#]+)", href)
#                         if match:
#                             arxiv_id = match.group(1)
#                             pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
#                             pdf_urls.append(pdf_url)
# 
#         except Exception as e:
#             # Log error but don't fail completely
#             print(f"Error extracting PDF URLs from arXiv: {e}")
# 
#         return pdf_urls
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/arxiv_org.py
# --------------------------------------------------------------------------------
