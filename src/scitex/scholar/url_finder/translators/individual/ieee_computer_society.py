#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python implementation of IEEE Computer Society translator.

Original JavaScript: IEEE Computer Society.js
Translator ID: 8d72adbc-376c-4a33-b6be-730bc235190f

IEEE Computer Society translator.
Key features:
- Line 114-126: BibTeX import from bibtex link
- Line 194-200: PDF URL from panel-body PDF link
- Complex target pattern for computer.org domains
"""

import re
from typing import List
from playwright.async_api import Page


class IEEEComputerSocietyTranslator:
    """IEEE Computer Society translator."""

    LABEL = "IEEE Computer Society"
    URL_TARGET_PATTERN = r"^https?://(www[0-9]?|search[0-9]?)\.computer\.org/(csdl/(mags/[0-9a-z/]+|trans/[0-9a-z/]+|letters/[0-9a-z]+|proceedings/[0-9a-z/]+|doi|abs/proceedings)|search/results|portal/web/computingnow/.*content\?)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL matches IEEE Computer Society pattern."""
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from IEEE Computer Society page.

        IEEE Computer Society PDFs are found in panel-body sections (line 194).
        """
        pdf_urls = []

        try:
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except:
            pass

        # Extract PDF URL from panel-body (line 194)
        urls = await page.evaluate("""
            () => {
                const pdfUrls = [];

                // Standard PDF link in panel-body (line 194)
                const pdfLink = document.querySelector('div.panel-body a[href*="PDF"]');
                if (pdfLink && pdfLink.href) {
                    pdfUrls.push(pdfLink.href);
                }

                // Alternative: direct PDF links
                const directPdfLinks = document.querySelectorAll('a[href*=".pdf"]');
                for (const link of directPdfLinks) {
                    if (link.href && !pdfUrls.includes(link.href)) {
                        pdfUrls.push(link.href);
                    }
                }

                // Look for download section
                const downloadLinks = document.querySelectorAll('a[href*="download"]');
                for (const link of downloadLinks) {
                    if (link.href && link.textContent &&
                        link.textContent.toLowerCase().includes('pdf') &&
                        !pdfUrls.includes(link.href)) {
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
        """Demonstration of IEEEComputerSocietyTranslator usage."""
        test_url = (
            "https://www.computer.org/csdl/trans/ta/2012/01/tta2012010003-abs.html"
        )

        print(f"Testing IEEEComputerSocietyTranslator with URL: {test_url}")
        print(
            f"URL matches pattern: {IEEEComputerSocietyTranslator.matches_url(test_url)}\n"
        )

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            print("Navigating to IEEE Computer Society page...")
            await page.goto(test_url, timeout=60000)
            await page.wait_for_load_state("domcontentloaded")

            print("Extracting PDF URLs...")
            pdf_urls = await IEEEComputerSocietyTranslator.extract_pdf_urls_async(page)

            print(f"\nResults:")
            print(f"  Found {len(pdf_urls)} PDF URL(s)")
            for url in pdf_urls:
                print(f"  - {url}")

            await browser.close()

    asyncio.run(main())


# EOF
