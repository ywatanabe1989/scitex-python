#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cambridge Core translator.

Based on Cambridge Core.js translator from Zotero.
Original JavaScript implementation by Sebastian Karcher.
"""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class CambridgeCoreTranslator(BaseTranslator):
    """Cambridge Core translator.

    Supports:
    - Journal articles (/article/)
    - Books (/books/)
    - Book chapters (/books/ with chapter-wrapper)
    - Search results and listing pages
    - Journal issues

    Based on JavaScript translator Cambridge Core.js (lines 1-580).
    """

    LABEL = "Cambridge Core"
    URL_TARGET_PATTERN = (
        r"^https?://www\.cambridge\.org/core/(search\?|journals/|books/|.+/listing?)"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL matches Cambridge Core pattern.

        Based on JavaScript detectWeb() function (lines 38-65).

        The JavaScript implementation checks:
        1. If URL contains /search? or /listing? or /issue/ -> multiple results
        2. If URL contains /article/ -> journal article
        3. If URL contains /books/:
           - If has chapter-wrapper -> book chapter/section
           - Otherwise -> book
        """
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from Cambridge Core page.

        Based on JavaScript scrape() function (lines 99-178).

        The JavaScript implementation:
        1. For books/book sections:
           - Extracts product ID from URL
           - Fetches RIS citation from /core/services/aop-easybib/export
           - Gets PDF from citation_pdf_url meta tag or .actions a[target="_blank"]
           - Uses RIS translator to parse metadata
           - Adds PDF as attachment

        2. For journal articles:
           - Uses Embedded Metadata translator
           - Extracts abstract from div.abstract
           - Gets citation_online_date if date is undefined
           - Removes asterisk/number from title if needed
           - Sets library catalog to "Cambridge University Press"

        Python implementation:
        - Focuses on PDF URL extraction
        - Tries multiple methods to find PDF link
        - Returns list of PDF URLs for download

        Args:
            page: Playwright page object on Cambridge Core

        Returns:
            List containing PDF URL if found, empty list otherwise
        """
        pdf_urls = []

        try:
            # Method 1: Check for citation_pdf_url meta tag (JS line 108-109)
            # This is the most reliable method for both books and articles
            citation_pdf = await page.locator(
                'meta[name="citation_pdf_url"], meta[name="citation_ pdf_url"]'
            ).first.get_attribute("content", timeout=2000)

            if citation_pdf:
                # Make absolute URL if needed
                if not citation_pdf.startswith("http"):
                    if citation_pdf.startswith("/"):
                        base_url = await page.evaluate("window.location.origin")
                        citation_pdf = f"{base_url}{citation_pdf}"
                    else:
                        base_url = await page.evaluate("window.location.href")
                        # Remove trailing path
                        base_url = re.sub(r"/[^/]*$", "", base_url)
                        citation_pdf = f"{base_url}/{citation_pdf}"

                pdf_urls.append(citation_pdf)
                return pdf_urls

        except Exception:
            pass

        try:
            # Method 2: Look for PDF link in actions section (JS line 111-112)
            # Pattern: .actions a[target="_blank"][href*=".pdf"]
            pdf_link = await page.locator(
                '.actions a[target="_blank"][href*=".pdf"]'
            ).first.get_attribute("href", timeout=2000)

            if pdf_link:
                # Make absolute URL if needed
                if not pdf_link.startswith("http"):
                    if pdf_link.startswith("/"):
                        base_url = await page.evaluate("window.location.origin")
                        pdf_link = f"{base_url}{pdf_link}"
                    else:
                        base_url = await page.evaluate("window.location.href")
                        base_url = re.sub(r"/[^/]*$", "", base_url)
                        pdf_link = f"{base_url}/{pdf_link}"

                pdf_urls.append(pdf_link)
                return pdf_urls

        except Exception:
            pass

        try:
            # Method 3: Look for standard PDF download button/link
            # Cambridge Core often has a "PDF" button or link
            pdf_button = await page.locator(
                'a[href*=".pdf"], a[data-track-label="pdf"], a:has-text("PDF")'
            ).first.get_attribute("href", timeout=2000)

            if pdf_button and ".pdf" in pdf_button.lower():
                # Make absolute URL if needed
                if not pdf_button.startswith("http"):
                    if pdf_button.startswith("/"):
                        base_url = await page.evaluate("window.location.origin")
                        pdf_button = f"{base_url}{pdf_button}"
                    else:
                        base_url = await page.evaluate("window.location.href")
                        base_url = re.sub(r"/[^/]*$", "", base_url)
                        pdf_button = f"{base_url}/{pdf_button}"

                pdf_urls.append(pdf_button)
                return pdf_urls

        except Exception:
            pass

        try:
            # Method 4: Extract DOI and construct PDF URL
            # Cambridge Core PDF URLs often follow pattern: /core/services/aop-cambridge-core/content/view/{DOI}
            doi_meta = await page.locator(
                'meta[name="citation_doi"]'
            ).first.get_attribute("content", timeout=2000)

            if doi_meta:
                # Remove any prefix like "doi:" or "https://doi.org/"
                doi = re.sub(r"^(doi:|https?://doi\.org/)", "", doi_meta)

                # Try constructing PDF URL from DOI
                base_url = await page.evaluate("window.location.origin")
                pdf_url = (
                    f"{base_url}/core/services/aop-cambridge-core/content/view/{doi}"
                )

                pdf_urls.append(pdf_url)
                return pdf_urls

        except Exception:
            pass

        return []
