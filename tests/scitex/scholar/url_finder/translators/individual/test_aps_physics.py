# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/aps_physics.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """Python implementation of APS-Physics translator.
# 
# Original JavaScript: APS-Physics.js
# Translator ID: f318ab1e-71c6-4f67-8ac3-4b1144e5bf4e
# 
# APS-Physics (American Physical Society Physics magazine) translator.
# Key features:
# - Line 40-42: detectWeb checks for citation_title meta tag
# - Line 82-96: PDF URL pattern: https://physics.aps.org/articles/pdf/{DOI}
# - Line 52: Multiple results from h3.feed-item-title > a[href*="/articles/"]
# """
# 
# import re
# from typing import List
# from playwright.async_api import Page
# 
# 
# class APSPhysicsTranslator:
#     """APS-Physics (American Physical Society Physics) translator."""
# 
#     LABEL = "APS-Physics"
#     URL_TARGET_PATTERN = r"^https?://(www\.)?(physics)\.aps\.org/"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         """Check if URL matches APS-Physics pattern."""
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         """Extract PDF URLs from APS-Physics page.
# 
#         APS-Physics uses a standard pattern: https://physics.aps.org/articles/pdf/{DOI}
#         """
#         pdf_urls = []
# 
#         try:
#             await page.wait_for_load_state("domcontentloaded", timeout=5000)
#         except:
#             pass
# 
#         # Extract DOI and construct PDF URL (line 82-96)
#         urls = await page.evaluate("""
#             () => {
#                 const pdfUrls = [];
# 
#                 // Look for DOI in citation meta tag or DOI link
#                 let doi = null;
# 
#                 // Try citation_doi meta tag
#                 const doiMeta = document.querySelector('meta[name="citation_doi"]');
#                 if (doiMeta && doiMeta.content) {
#                     doi = doiMeta.content;
#                 }
# 
#                 // Try DOI link (line 83)
#                 if (!doi) {
#                     const doiLink = document.querySelector('a[href*="link.aps.org/doi"]');
#                     if (doiLink && doiLink.href) {
#                         const match = doiLink.href.match(/doi\/(10\.\d+\/[^\s]+)/);
#                         if (match) {
#                             doi = match[1];
#                         }
#                     }
#                 }
# 
#                 // Construct PDF URL (line 95)
#                 if (doi) {
#                     pdfUrls.push(`https://physics.aps.org/articles/pdf/${doi}`);
#                 }
# 
#                 return pdfUrls;
#             }
#         """)
# 
#         pdf_urls.extend(urls)
#         return pdf_urls
# 
# 
# if __name__ == "__main__":
#     import asyncio
#     from playwright.async_api import async_playwright
# 
#     async def main():
#         """Demonstration of APSPhysicsTranslator usage."""
#         test_url = "https://physics.aps.org/articles/v5/100"
# 
#         print(f"Testing APSPhysicsTranslator with URL: {test_url}")
#         print(f"URL matches pattern: {APSPhysicsTranslator.matches_url(test_url)}\n")
# 
#         async with async_playwright() as p:
#             browser = await p.chromium.launch(headless=True)
#             context = await browser.new_context()
#             page = await context.new_page()
# 
#             print("Navigating to APS-Physics page...")
#             await page.goto(test_url, timeout=60000)
#             await page.wait_for_load_state("domcontentloaded")
# 
#             print("Extracting PDF URLs...")
#             pdf_urls = await APSPhysicsTranslator.extract_pdf_urls_async(page)
# 
#             print(f"\nResults:")
#             print(f"  Found {len(pdf_urls)} PDF URL(s)")
#             for url in pdf_urls:
#                 print(f"  - {url}")
# 
#             await browser.close()
# 
#     asyncio.run(main())
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/aps_physics.py
# --------------------------------------------------------------------------------
