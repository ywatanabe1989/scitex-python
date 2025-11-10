#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python implementation of zbMATH translator.

Original JavaScript: zbMATH.js
Translator ID: 9302e5aa-f424-4ece-849a-c376cffd84d9

zbMATH is a mathematics bibliography database.
Key features:
- Line 97-105: BibTeX export via /bibtex/ links
- Line 48-66: detectWeb based on URL patterns
- Line 123-135: PDF URLs from download links
"""

import re
from typing import List
from playwright.async_api import Page


class ZbMATHTranslator:
    """zbMATH mathematics bibliography translator."""

    LABEL = "zbMATH"
    URL_TARGET_PATTERN = r"^https?://(www\.)?zbmath\.org/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL matches zbMATH pattern."""
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from zbMATH page."""
        pdf_urls = []

        try:
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except:
            pass

        # Extract PDF URLs from download links
        urls = await page.evaluate("""
            () => {
                const pdfUrls = [];

                // Look for PDF download links
                const links = document.querySelectorAll('a[href*=".pdf"], a[href*="/pdf/"]');
                for (const link of links) {
                    if (link.href && !pdfUrls.includes(link.href)) {
                        pdfUrls.push(link.href);
                    }
                }

                // Check for DOI links that might lead to PDFs
                const doiLinks = document.querySelectorAll('a[href*="doi.org"]');
                for (const link of doiLinks) {
                    // Note: DOI links typically require resolver
                    // This is just marking their presence
                }

                return pdfUrls;
            }
        """)

        pdf_urls.extend(urls)
        return pdf_urls


if __name__ == "__main__":
    import asyncio
    from playwright.async_api import async_playwright

    async def main():
        """Demonstration of ZbMATHTranslator usage."""
        test_url = "https://zbmath.org/?q=euler"

        print(f"Testing ZbMATHTranslator with URL: {test_url}")
        print(f"URL matches pattern: {ZbMATHTranslator.matches_url(test_url)}\n")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            print("Navigating to zbMATH page...")
            await page.goto(test_url, timeout=60000)
            await page.wait_for_load_state("domcontentloaded")

            print("Extracting PDF URLs...")
            pdf_urls = await ZbMATHTranslator.extract_pdf_urls_async(page)

            print(f"\nResults:")
            print(f"  Found {len(pdf_urls)} PDF URL(s)")
            for url in pdf_urls:
                print(f"  - {url}")

            await browser.close()

    asyncio.run(main())


# EOF
