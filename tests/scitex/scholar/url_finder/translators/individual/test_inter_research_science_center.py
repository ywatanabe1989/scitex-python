# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/inter_research_science_center.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """Python implementation of Inter-Research Science Center translator.
# 
# Original JavaScript: Inter-Research Science Center.js
# Translator ID: 0eeb2ac0-fbaf-4994-b98f-203d273eb9fa
# 
# Inter-Research Science Center translator.
# Key features:
# - Line 44: Uses citation_title meta tag for detection
# - Line 86-106: Uses Embedded Metadata translator
# - Line 154-160: PDF and snapshot attachments
# """
# 
# import re
# from typing import List
# from playwright.async_api import Page
# 
# 
# class InterResearchScienceCenterTranslator:
#     """Inter-Research Science Center translator."""
# 
#     LABEL = "Inter-Research Science Center"
#     URL_TARGET_PATTERN = r"^https?://www\.int-res\.com/"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         """Check if URL matches Inter-Research Science Center pattern."""
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         """Extract PDF URLs from Inter-Research Science Center page.
# 
#         Inter-Research provides PDF URLs via citation_pdf_url meta tag.
#         """
#         pdf_urls = []
# 
#         try:
#             await page.wait_for_load_state("domcontentloaded", timeout=5000)
#         except:
#             pass
# 
#         # Extract PDF URL from meta tags and direct links
#         urls = await page.evaluate("""
#             () => {
#                 const pdfUrls = [];
# 
#                 // Check for citation_pdf_url meta tag (standard for embedded metadata)
#                 const pdfMeta = document.querySelector('meta[name="citation_pdf_url"]');
#                 if (pdfMeta && pdfMeta.content) {
#                     pdfUrls.push(pdfMeta.content);
#                 }
# 
#                 // Look for direct PDF links
#                 const pdfLinks = document.querySelectorAll('a[href*=".pdf"]');
#                 for (const link of pdfLinks) {
#                     if (link.href && !pdfUrls.includes(link.href)) {
#                         pdfUrls.push(link.href);
#                     }
#                 }
# 
#                 // Look for download/PDF buttons
#                 const downloadLinks = document.querySelectorAll('a[href*="pdf"], a[href*="download"]');
#                 for (const link of downloadLinks) {
#                     if (link.href && link.href.includes('pdf') && !pdfUrls.includes(link.href)) {
#                         pdfUrls.push(link.href);
#                     }
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
#         """Demonstration of InterResearchScienceCenterTranslator usage."""
#         test_url = "http://www.int-res.com/abstracts/meps/v403/p13-27/"
# 
#         print(f"Testing InterResearchScienceCenterTranslator with URL: {test_url}")
#         print(
#             f"URL matches pattern: {InterResearchScienceCenterTranslator.matches_url(test_url)}\n"
#         )
# 
#         async with async_playwright() as p:
#             browser = await p.chromium.launch(headless=True)
#             context = await browser.new_context()
#             page = await context.new_page()
# 
#             print("Navigating to Inter-Research Science Center page...")
#             await page.goto(test_url, timeout=60000)
#             await page.wait_for_load_state("domcontentloaded")
# 
#             print("Extracting PDF URLs...")
#             pdf_urls = (
#                 await InterResearchScienceCenterTranslator.extract_pdf_urls_async(page)
#             )
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/inter_research_science_center.py
# --------------------------------------------------------------------------------
