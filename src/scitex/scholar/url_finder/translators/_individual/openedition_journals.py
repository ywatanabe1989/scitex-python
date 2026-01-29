#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python implementation of OpenEdition Journals translator.

Original JavaScript: OpenEdition Journals.js
Translator ID: 87766765-919e-4d3b-9071-3dd7efe984c8

OpenEdition Journals academic platform translator.
Key features:
- Line 5: Target pattern for journals.openedition.org
- Uses embedded metadata (citation_pdf_url meta tag)
- Test cases show "Full Text PDF" attachments
"""

import re
import logging
from typing import List
from playwright.async_api import Page, Error as PlaywrightError

# Explicit public API - only export the translator class
__all__ = ["OpenEditionJournalsTranslator"]

# Private module-level logger - not exported
_logger = logging.getLogger(__name__)


class OpenEditionJournalsTranslator:
    """OpenEdition Journals academic platform translator."""

    LABEL = "OpenEdition Journals"
    URL_TARGET_PATTERN = r"^https?://journals\.openedition\.org/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL matches OpenEdition Journals pattern."""
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from OpenEdition Journals page.

        OpenEdition uses embedded metadata with citation_pdf_url meta tags.

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
            _logger.warning(f"OpenEdition: Page load timeout - {e}")
            # Continue anyway - page may still be partially loaded
        except Exception as e:
            _logger.error(f"OpenEdition: Unexpected error during page load - {e}")
            return []

        try:
            # Extract PDF URLs via JavaScript
            urls = await page.evaluate("""
                () => {
                    const pdfUrls = [];

                    try {
                        // citation_pdf_url meta tag (standard embedded metadata)
                        const pdfMeta = document.querySelector('meta[name="citation_pdf_url"]');
                        if (pdfMeta && pdfMeta.content) {
                            pdfUrls.push(pdfMeta.content);
                        }

                        // DC.identifier meta tags
                        const dcIdentifiers = document.querySelectorAll('meta[name="DC.identifier"]');
                        for (const meta of dcIdentifiers) {
                            if (meta.content && meta.content.includes('.pdf')) {
                                pdfUrls.push(meta.content);
                            }
                        }

                        // Fallback: direct PDF links
                        const pdfLinks = document.querySelectorAll('a[href*=".pdf"]');
                        for (const link of pdfLinks) {
                            if (link.href && !pdfUrls.includes(link.href)) {
                                pdfUrls.push(link.href);
                            }
                        }
                    } catch (error) {
                        console.error('OpenEdition PDF extraction error:', error);
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
                        _logger.warning(f"OpenEdition: Invalid URL format: {url}")

        except PlaywrightError as e:
            _logger.error(f"OpenEdition: Failed to evaluate JavaScript - {e}")
        except Exception as e:
            _logger.error(f"OpenEdition: Unexpected error during PDF extraction - {e}")

        return pdf_urls


if __name__ == "__main__":
    # Private demo - imports only needed when running as script
    import asyncio
    from playwright.async_api import async_playwright

    async def _demo():
        """Private demonstration of OpenEditionJournalsTranslator usage."""
        test_urls = [
            "https://journals.openedition.org/e-spania/12303",
            "https://journals.openedition.org/chs/142",
            "https://journals.openedition.org/remi/2495",
        ]

        for test_url in test_urls:
            print(f"\nTesting OpenEditionJournalsTranslator with URL: {test_url}")
            print(
                f"URL matches pattern: {OpenEditionJournalsTranslator.matches_url(test_url)}"
            )

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()

                print("Navigating to OpenEdition Journals page...")
                await page.goto(test_url, timeout=60000)
                await page.wait_for_load_state("domcontentloaded")

                print("Extracting PDF URLs...")
                pdf_urls = await OpenEditionJournalsTranslator.extract_pdf_urls_async(
                    page
                )

                print(f"\nResults:")
                print(f"  Found {len(pdf_urls)} PDF URL(s)")
                for url in pdf_urls:
                    print(f"  - {url}")

                await browser.close()

    asyncio.run(_demo())


# EOF
