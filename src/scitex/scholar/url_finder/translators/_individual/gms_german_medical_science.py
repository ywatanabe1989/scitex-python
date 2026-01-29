#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python implementation of GMS German Medical Science translator.

Original JavaScript: GMS German Medical Science.js
Translator ID: 8d5984e8-3ba9-4faa-8b84-a58adae56439

GMS German Medical Science translator.
Key features:
- Line 40-42: detectWeb checks for /journals/ or /meetings/ in URL
- Line 86: XML export via format_xml link
- Line 136-139: PDF and snapshot attachments
- Pattern: www.egms.de/static/(de|en)/
"""

import re
from typing import List
from playwright.async_api import Page


class GMSGermanMedicalScienceTranslator:
    """GMS German Medical Science translator."""

    LABEL = "GMS German Medical Science"
    URL_TARGET_PATTERN = r"^https?://www\.egms\.de/static/(de|en)/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL matches GMS German Medical Science pattern."""
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from GMS German Medical Science page.

        GMS provides PDF links directly on the page.
        """
        pdf_urls = []

        try:
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except:
            pass

        # Extract PDF URLs from download links
        urls = await page.evaluate("""
            () => {
                const pdfUrls = [];

                // Look for PDF download links (standard pattern)
                const pdfLinks = document.querySelectorAll('a[href*=".pdf"], a.format_pdf');
                for (const link of pdfLinks) {
                    if (link.href && !pdfUrls.includes(link.href)) {
                        pdfUrls.push(link.href);
                    }
                }

                // Look for download section links
                const downloadLinks = document.querySelectorAll('a[class*="format"]');
                for (const link of downloadLinks) {
                    if (link.href && link.href.includes('.pdf') && !pdfUrls.includes(link.href)) {
                        pdfUrls.push(link.href);
                    }
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
        """Demonstration of GMSGermanMedicalScienceTranslator usage."""
        test_url = "https://www.egms.de/static/de/journals/gms/2017-15/000242.shtml"

        print(f"Testing GMSGermanMedicalScienceTranslator with URL: {test_url}")
        print(
            f"URL matches pattern: {GMSGermanMedicalScienceTranslator.matches_url(test_url)}\n"
        )

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            print("Navigating to GMS German Medical Science page...")
            await page.goto(test_url, timeout=60000)
            await page.wait_for_load_state("domcontentloaded")

            print("Extracting PDF URLs...")
            pdf_urls = await GMSGermanMedicalScienceTranslator.extract_pdf_urls_async(
                page
            )

            print(f"\nResults:")
            print(f"  Found {len(pdf_urls)} PDF URL(s)")
            for url in pdf_urls:
                print(f"  - {url}")

            await browser.close()

    asyncio.run(main())


# EOF
