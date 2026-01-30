#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python implementation of LingBuzz translator.

Original JavaScript: LingBuzz.js
Translator ID: e048e70e-8fea-43e9-ac8e-940bc3d71b0b

LingBuzz linguistics preprint repository translator.
Key features:
- Line 5: Target pattern for ling.auf.net/lingbuzz/ and lingbuzz.net/lingbuzz/
- Line 103: PDF URL from a[href*='.pdf'] in tbody tr (Format field)
- Line 142: SemanticsArchive PDF from a:first-child in center block
"""

import re
import logging
from typing import List
from playwright.async_api import Page, Error as PlaywrightError

# Explicit public API - only export the translator class
__all__ = ["LingBuzzTranslator"]

# Private module-level logger - not exported
_logger = logging.getLogger(__name__)


class LingBuzzTranslator:
    """LingBuzz linguistics preprint repository translator."""

    LABEL = "LingBuzz"
    URL_TARGET_PATTERN = r"^https://(ling\.auf|lingbuzz)\.net/lingbuzz/(repo/semanticsArchive/article/)?(\d+|_search)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL matches LingBuzz pattern."""
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from LingBuzz page.

        LingBuzz has two formats:
        1. Standard: PDF in tbody tr containing "format" (line 103)
        2. SemanticsArchive: PDF from first link in center block (line 142)

        Args:
            page: Playwright page object

        Returns:
            List of PDF URLs found (may be empty)

        Note:
            Gracefully handles errors - returns empty list on failure
        """
        pdf_urls = []

        try:
            # Wait for page to be ready (with specific timeout)
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except PlaywrightError as e:
            _logger.warning(f"LingBuzz: Page load timeout - {e}")
            # Continue anyway - page may still be partially loaded
        except Exception as e:
            _logger.error(f"LingBuzz: Unexpected error during page load - {e}")
            return []

        try:
            # Extract PDF URLs via JavaScript
            urls = await page.evaluate("""
                () => {
                    const pdfUrls = [];

                    try {
                        // Check if this is a SemanticsArchive page (line 79-82)
                        const url = window.location.href;
                        const isSemanticsArchive = url.includes('semanticsArchive');

                        if (isSemanticsArchive) {
                            // SemanticsArchive: PDF from first link in center block (line 142)
                            const idBlock = document.querySelector('center');
                            if (idBlock) {
                                const firstLink = idBlock.querySelector('a:first-child');
                                if (firstLink && firstLink.href && firstLink.href.includes('.pdf')) {
                                    pdfUrls.push(firstLink.href);
                                }
                            }
                        } else {
                            // Standard LingBuzz: PDF in table row with "format" (line 102-105)
                            const tableRows = document.querySelectorAll('tbody tr');
                            for (const row of tableRows) {
                                const cells = row.querySelectorAll('td');
                                if (cells.length >= 2) {
                                    const leftCell = cells[0];
                                    const rightCell = cells[1];
                                    const fieldName = leftCell.innerText.toLowerCase();

                                    if (fieldName.includes('format')) {
                                        const pdfLink = rightCell.querySelector('a[href*=".pdf"]');
                                        if (pdfLink && pdfLink.href) {
                                            pdfUrls.push(pdfLink.href);
                                        }
                                    }
                                }
                            }
                        }

                        // Fallback: any PDF link in the page
                        if (pdfUrls.length === 0) {
                            const allPdfLinks = document.querySelectorAll('a[href*=".pdf"]');
                            for (const link of allPdfLinks) {
                                if (link.href && !pdfUrls.includes(link.href)) {
                                    pdfUrls.push(link.href);
                                }
                            }
                        }
                    } catch (error) {
                        console.error('LingBuzz PDF extraction error:', error);
                        return [];
                    }

                    return pdfUrls;
                }
            """)

            # Validate and filter URLs
            if urls and isinstance(urls, list):
                for url in urls:
                    if url and isinstance(url, str) and url.startswith("http"):
                        pdf_urls.append(url)
                    else:
                        _logger.warning(f"LingBuzz: Invalid URL format: {url}")

        except PlaywrightError as e:
            _logger.error(f"LingBuzz: Failed to evaluate JavaScript - {e}")
        except Exception as e:
            _logger.error(f"LingBuzz: Unexpected error during PDF extraction - {e}")

        return pdf_urls


if __name__ == "__main__":
    # Private demo - imports only needed when running as script
    import asyncio
    from playwright.async_api import async_playwright

    async def _demo():
        """Private demonstration of LingBuzzTranslator usage."""
        test_urls = [
            "https://ling.auf.net/lingbuzz/005988",  # Standard
            "https://lingbuzz.net/lingbuzz/006559",  # Alternative domain
            "https://ling.auf.net/lingbuzz/repo/semanticsArchive/article/001471",  # SemanticsArchive
        ]

        for test_url in test_urls:
            print(f"\nTesting LingBuzzTranslator with URL: {test_url}")
            print(f"URL matches pattern: {LingBuzzTranslator.matches_url(test_url)}")

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()

                print("Navigating to LingBuzz page...")
                await page.goto(test_url, timeout=60000)
                await page.wait_for_load_state("domcontentloaded")

                print("Extracting PDF URLs...")
                pdf_urls = await LingBuzzTranslator.extract_pdf_urls_async(page)

                print(f"\nResults:")
                print(f"  Found {len(pdf_urls)} PDF URL(s)")
                for url in pdf_urls:
                    print(f"  - {url}")

                await browser.close()

    asyncio.run(_demo())


# EOF
