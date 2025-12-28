# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/scinapse.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """Python implementation of scinapse translator.
# 
# Original JavaScript: scinapse.js
# Translator ID: 42680c5e-1ae8-4171-ab53-afe1d8e840d4
# 
# Scinapse (academic search engine) translator.
# Key features:
# - Line 42-44: detectWeb checks for /papers/ in URL
# - Line 99: PDF URL from a[href*=".pdf"][target="_blank"]
# - Line 103-106: BibTeX API at /api/citations/export
# """
# 
# import re
# from typing import List
# from playwright.async_api import Page
# 
# 
# class ScinapseTranslator:
#     """Scinapse academic search engine translator."""
# 
#     LABEL = "scinapse"
#     URL_TARGET_PATTERN = r"^https?://(www\.)?scinapse\.io/"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         """Check if URL matches Scinapse pattern."""
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         """Extract PDF URLs from Scinapse page.
# 
#         Scinapse PDFs are found via a[href*=".pdf"][target="_blank"] (line 99).
#         """
#         pdf_urls = []
# 
#         try:
#             await page.wait_for_load_state("domcontentloaded", timeout=5000)
#         except:
#             pass
# 
#         # Extract PDF URL (line 99)
#         urls = await page.evaluate("""
#             () => {
#                 const pdfUrls = [];
# 
#                 // Standard PDF link pattern (line 99)
#                 const pdfLink = document.querySelector('a[href*=".pdf"][target="_blank"]');
#                 if (pdfLink && pdfLink.href) {
#                     pdfUrls.push(pdfLink.href);
#                 }
# 
#                 // Alternative: any PDF links
#                 const allPdfLinks = document.querySelectorAll('a[href*=".pdf"]');
#                 for (const link of allPdfLinks) {
#                     if (link.href && !pdfUrls.includes(link.href)) {
#                         pdfUrls.push(link.href);
#                     }
#                 }
# 
#                 // Check citation_pdf_url meta tag
#                 const pdfMeta = document.querySelector('meta[name="citation_pdf_url"]');
#                 if (pdfMeta && pdfMeta.content && !pdfUrls.includes(pdfMeta.content)) {
#                     pdfUrls.push(pdfMeta.content);
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
#         """Demonstration of ScinapseTranslator usage."""
#         test_url = "https://www.scinapse.io/papers/2981511200"
# 
#         print(f"Testing ScinapseTranslator with URL: {test_url}")
#         print(f"URL matches pattern: {ScinapseTranslator.matches_url(test_url)}\n")
# 
#         async with async_playwright() as p:
#             browser = await p.chromium.launch(headless=True)
#             context = await browser.new_context()
#             page = await context.new_page()
# 
#             print("Navigating to Scinapse page...")
#             await page.goto(test_url, timeout=60000)
#             await page.wait_for_load_state("domcontentloaded")
# 
#             print("Extracting PDF URLs...")
#             pdf_urls = await ScinapseTranslator.extract_pdf_urls_async(page)
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/scinapse.py
# --------------------------------------------------------------------------------
