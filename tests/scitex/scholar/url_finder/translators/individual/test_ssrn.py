# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/ssrn.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-09 23:39:08 (ywatanabe)"
# # File: /home/ywatanabe/proj/zotero-translators-python/src/zotero_translators_python/individual/ssrn.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/zotero_translators_python/individual/ssrn.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# """SSRN translator."""
# 
# import re
# from typing import List
# 
# from playwright.async_api import Page
# 
# from ..core.base import BaseTranslator
# 
# 
# class SSRNTranslator(BaseTranslator):
#     """SSRN."""
# 
#     LABEL = "SSRN"
#     URL_TARGET_PATTERN = r"^https?://(www|papers|hq)\.ssrn\.com/"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         """Extract PDF URLs from SSRN paper page.
# 
#         Based on JavaScript translator scrape() function, line 124:
#         var pdfURL = attr(doc, 'a.primary[data-abstract-id]', 'href');
# 
#         Matches JavaScript implementation exactly - no fallbacks.
#         """
#         pdf_urls = []
# 
#         try:
#             # Extract the correct PDF URL from the download button
#             # JavaScript line 124: var pdfURL = attr(doc, 'a.primary[data-abstract-id]', 'href');
#             pdf_url = await page.locator(
#                 "a.primary[data-abstract-id]"
#             ).first.get_attribute("href", timeout=5000)
# 
#             if pdf_url:
#                 # Make absolute URL if relative
#                 if pdf_url.startswith("/"):
#                     base_url = await page.evaluate("window.location.origin")
#                     pdf_url = f"{base_url}{pdf_url}"
#                 pdf_urls.append(pdf_url)
#         except Exception:
#             # No fallback - matches JavaScript behavior
#             pass
# 
#         return pdf_urls
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/ssrn.py
# --------------------------------------------------------------------------------
