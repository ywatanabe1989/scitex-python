#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atypon Journals translator.

Translates the Zotero JavaScript translator "Atypon Journals" to Python.
Original JavaScript translator by Sebastian Karcher and Abe Jellinek.

Copyright (c) 2011-2022 Sebastian Karcher and Abe Jellinek
Copyright (c) 2025 Python translation

This file is part of Zotero.

Zotero is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Zotero is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with Zotero. If not, see <http://www.gnu.org/licenses/>.

Original metadata:
- Translator ID: 5af42734-7cd5-4c69-97fc-bc406999bdba
- Label: Atypon Journals
- Creator: Sebastian Karcher and Abe Jellinek
- Target: ^https?://[^?#]+(/doi/((abs|abstract|full|figure|ref|citedby|book)/)?10\.|/action/doSearch\?)|^https?://[^/]+/toc/
- Priority: 270
- Translator Type: 4 (web)
- Last Updated: 2022-10-25

Atypon is a major publishing platform serving 100+ publishers including:
- Radiological Society of North America (RSNA)
- Society for Industrial and Applied Mathematics (SIAM)
- Mary Ann Liebert
- Japan Physical Society
- World Scientific
- Annual Reviews
- American Society for Microbiology (ASM)
- RMIT Publishing (Informit)
- Science/AAAS
- PNAS
- EMBO Press
"""

import re
from typing import List, Optional
from playwright.async_api import Page
from ..core.base import BaseTranslator


class AtyponJournalsTranslator(BaseTranslator):
    """Atypon Journals platform translator.

    Handles PDF extraction from the Atypon publishing platform which serves
    over 100 publishers worldwide.
    """

    LABEL = "Atypon Journals"
    URL_TARGET_PATTERN = (
        r"^https?://[^?#]+(/doi/((abs|abstract|full|figure|ref|citedby|book)/)?10\.|"
        r"/action/doSearch\?)|^https?://[^/]+/toc/"
    )

    # Regex matching sites that load PDFs in an embedded reader (needs bypass)
    NEED_BYPASS_EMBEDDED_READER = re.compile(r"^https?://www\.embopress\.org/")

    # URL pattern for replacing article view paths with PDF paths
    REPL_URL_REGEXP = re.compile(
        r"/doi/((?:abs|abstract|full|figure|ref|citedby|book)/)?"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL matches Atypon pattern.

        Args:
            url: URL to check

        Returns:
            True if URL matches Atypon Journals sites
        """
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    def _build_pdf_url(
        cls, url: str, has_pdf_link: bool, pdf_path: str
    ) -> Optional[str]:
        """Build PDF URL from article URL.

        Based on JavaScript buildPdfUrl() function (lines 194-223).

        Args:
            url: Article URL
            has_pdf_link: Whether a PDF link exists on the page
            pdf_path: PDF path to use (/doi/pdf/, /doi/epdf/, or /doi/pdfplus/)

        Returns:
            PDF URL or None if construction failed
        """
        if not cls.REPL_URL_REGEXP.search(url):
            return None

        if not has_pdf_link:
            return None

        pdf_url = cls.REPL_URL_REGEXP.sub(pdf_path, url)

        # Special case: EMBO Press needs embedded reader bypass
        if cls.NEED_BYPASS_EMBEDDED_READER.match(url):
            pdf_url = re.sub(r"/e?pdf/", "/pdfdirect/", pdf_url)
            separator = "&" if "?" in pdf_url else "?"
            pdf_url = f"{pdf_url}{separator}download=true"

        return pdf_url

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from Atypon page.

        Implements the buildPdfUrl logic from JavaScript (lines 194-223).
        Atypon uses standardized PDF URL patterns:
        - /doi/pdf/ - Standard PDF
        - /doi/epdf/ - Enhanced PDF
        - /doi/pdfplus/ - PDF with extras
        - /doi/pdfdirect/ - Direct PDF (used by some publishers like Neurology)

        Args:
            page: Playwright page object on Atypon site

        Returns:
            List of PDF URLs found
        """
        pdf_urls = []
        url = page.url

        try:
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass

        # PDF paths to try (line 197 + pdfdirect for Neurology and similar)
        pdf_paths = ["/doi/pdfdirect/", "/doi/pdf/", "/doi/epdf/", "/doi/pdfplus/"]

        # Try each PDF path pattern
        for pdf_path in pdf_paths:
            try:
                # Check if PDF link exists on page
                pdf_link = await page.query_selector(f'a[href*="{pdf_path}"]')

                if pdf_link:
                    # Try to get href directly from link
                    href = await pdf_link.get_attribute("href")
                    if href:
                        # Make absolute URL if needed
                        if href.startswith("http"):
                            pdf_urls.append(href)
                        else:
                            # Construct absolute URL
                            base_url = re.match(r"^https?://[^/]+", url)
                            if base_url:
                                abs_url = base_url.group(0) + href
                                pdf_urls.append(abs_url)
                        break

                    # If href extraction failed, try building URL
                    pdf_url = cls._build_pdf_url(url, True, pdf_path)
                    if pdf_url:
                        pdf_urls.append(pdf_url)
                        break

            except Exception:
                continue

        # If no PDF links found, try meta tags
        if not pdf_urls:
            try:
                # Check for citation_pdf_url meta tag
                pdf_meta = await page.query_selector('meta[name="citation_pdf_url"]')
                if pdf_meta:
                    content = await pdf_meta.get_attribute("content")
                    if content:
                        pdf_urls.append(content)
            except Exception:
                pass

        # If still no PDF links, try URL construction
        if not pdf_urls:
            for pdf_path in pdf_paths:
                pdf_url = cls._build_pdf_url(url, False, pdf_path)
                if pdf_url:
                    # Only add if it follows the expected pattern
                    if cls.REPL_URL_REGEXP.search(url):
                        pdf_urls.append(pdf_url)
                        break

        return pdf_urls


if __name__ == "__main__":
    import asyncio
    from playwright.async_api import async_playwright

    async def main():
        """Demonstration of AtyponJournalsTranslator usage."""
        test_urls = [
            "https://pubs.rsna.org/doi/full/10.1148/rg.337125073",
            "https://www.neurology.org/doi/10.1212/WNL.0000000000200348",
            "https://www.science.org/doi/10.1126/science.aag1582",
        ]

        for test_url in test_urls:
            print(f"\nTesting AtyponJournalsTranslator with URL: {test_url}")
            print(
                f"URL matches pattern: {AtyponJournalsTranslator.matches_url(test_url)}"
            )

            if not AtyponJournalsTranslator.matches_url(test_url):
                print("  URL doesn't match Atypon pattern, skipping...")
                continue

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()

                try:
                    print("  Navigating to Atypon page...")
                    await page.goto(test_url, timeout=30000)
                    await page.wait_for_load_state("domcontentloaded")

                    print("  Extracting PDF URLs...")
                    pdf_urls = await AtyponJournalsTranslator.extract_pdf_urls_async(
                        page
                    )

                    print(f"  Found {len(pdf_urls)} PDF URL(s)")
                    for url in pdf_urls:
                        print(f"    - {url}")

                except Exception as e:
                    print(f"  Error: {e}")
                finally:
                    await browser.close()

    asyncio.run(main())


# EOF
