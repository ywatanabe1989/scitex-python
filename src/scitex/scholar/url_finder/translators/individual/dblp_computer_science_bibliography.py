#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python implementation of DBLP Computer Science Bibliography translator.

Original JavaScript: DBLP Computer Science Bibliography.js
Translator ID: 625c6435-e235-4402-a48f-3095a9c1a09c

DBLP extracts BibTeX from embedded <pre> tags.
Key features:
- Line 67-68: BibTeX data in #bibtex-section > pre
- Line 39-63: detectWeb based on URL patterns
- Line 96-130: Handles both single entry and crossref (conferences/books)
"""

import re
from typing import List, Optional
from playwright.async_api import Page


class DBLPTranslator:
    """DBLP Computer Science Bibliography translator."""

    LABEL = "DBLP Computer Science Bibliography"
    URL_TARGET_PATTERN = (
        r"^https?://(www\.)?(dblp\d?(\.org|\.uni-trier\.de/|\.dagstuhl\.de/))"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL matches DBLP pattern."""
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from DBLP page.

        DBLP typically doesn't host PDFs but links to them via 'ee' field
        in BibTeX which becomes 'url' in the parsed data.
        """
        pdf_urls = []

        try:
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except:
            pass

        # Extract URLs from BibTeX data
        bibtex_data = await page.evaluate("""
            () => {
                const pre = document.querySelector('#bibtex-section > pre');
                if (!pre) return null;
                return pre.textContent;
            }
        """)

        if bibtex_data:
            # Look for ee (electronic edition) or url fields
            # ee = {http://...} or url = {http://...}
            url_match = re.search(
                r"(?:ee|url)\s*=\s*\{([^}]+)\}", bibtex_data, re.IGNORECASE
            )
            if url_match:
                url = url_match.group(1).strip()
                # Check if it's a PDF
                if ".pdf" in url.lower():
                    pdf_urls.append(url)

        return pdf_urls


if __name__ == "__main__":
    import asyncio
    from playwright.async_api import async_playwright

    async def main():
        """Demonstration of DBLPTranslator usage."""
        test_url = "https://dblp.org/rec/journals/cssc/XuY12.html?view=bibtex"

        print(f"Testing DBLPTranslator with URL: {test_url}")
        print(f"URL matches pattern: {DBLPTranslator.matches_url(test_url)}\n")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            print("Navigating to DBLP page...")
            await page.goto(test_url, timeout=60000)
            await page.wait_for_load_state("domcontentloaded")

            print("Extracting PDF URLs...")
            pdf_urls = await DBLPTranslator.extract_pdf_urls_async(page)

            print(f"\nResults:")
            print(f"  Found {len(pdf_urls)} PDF URL(s)")
            for url in pdf_urls:
                print(f"  - {url}")

            await browser.close()

    asyncio.run(main())


# EOF
