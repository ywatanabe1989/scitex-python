#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BioMed Central translator.

Based on BioMed Central.js translator from Zotero.
Original JavaScript implementation by Philipp Zumstein.

This translator covers BioMedCentral and SpringerOpen journals.
"""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class BioMedCentralTranslator(BaseTranslator):
    """BioMed Central and SpringerOpen translator.

    Supports:
    - Journal articles from biomedcentral.com domains
    - Journal articles from springeropen.com domains
    - Search results pages
    - Open-access PDF downloads

    Based on JavaScript translator (BioMed Central.js lines 1-152).
    """

    LABEL = "BioMed Central"
    URL_TARGET_PATTERN = (
        r"^https?://[^\.]+\.(biomedcentral|springeropen)\.com/(articles|search)"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL matches BioMed Central pattern.

        Based on JavaScript detectWeb() function (lines 45-53).
        Matches:
        - Article pages with DOI (10.1186/...)
        - Search results pages

        Args:
            url: URL to check

        Returns:
            True if URL matches BioMed Central pattern
        """
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from BioMed Central article page.

        Based on JavaScript scrape() function (lines 89-125).

        The JavaScript implementation:
        1. Extracts DOI from URL
        2. Fetches RIS metadata from Springer citation service
        3. Extracts PDF URL from citation_pdf_url meta tag
        4. Returns PDF attachment

        Python implementation:
        - Extracts PDF URL from citation_pdf_url meta tag (primary method)
        - Falls back to constructing PDF URL from DOI
        - Returns PDF URL for download

        Args:
            page: Playwright page object on BioMed Central site

        Returns:
            List containing PDF URL if found, empty list otherwise
        """
        pdf_urls = []

        # Method 1: citation_pdf_url meta tag (JS line 92)
        # This is the most reliable method for BioMed Central
        try:
            pdf_url = await page.locator(
                'meta[name="citation_pdf_url"]'
            ).first.get_attribute("content", timeout=2000)
            if pdf_url:
                pdf_urls.append(pdf_url)
                return pdf_urls
        except Exception:
            pass

        # Method 2: Direct PDF download link
        # BioMed Central has download links in various formats
        try:
            # Try common BMC PDF link patterns
            selectors = [
                'a[data-track-action="download pdf"]',
                "a.c-pdf-download__link",
                'a[href*="/track/pdf"]',
                "a.download-article-pdf-link",
            ]

            for selector in selectors:
                try:
                    pdf_link = await page.locator(selector).first.get_attribute(
                        "href", timeout=1000
                    )
                    if pdf_link:
                        # Make absolute URL if needed
                        if pdf_link.startswith("/"):
                            base_url = await page.evaluate("window.location.origin")
                            pdf_link = f"{base_url}{pdf_link}"
                        elif not pdf_link.startswith("http"):
                            current_url = await page.evaluate("window.location.href")
                            base_url = re.sub(r"/[^/]*$", "", current_url)
                            pdf_link = f"{base_url}/{pdf_link}"
                        pdf_urls.append(pdf_link)
                        return pdf_urls
                except Exception:
                    continue
        except Exception:
            pass

        # Method 3: Construct PDF URL from DOI (JS line 90)
        # Extract DOI pattern: /10.1186/...
        try:
            current_url = page.url
            doi_match = re.search(r"/(10\.1186/[^#?]+)", current_url)

            if doi_match:
                doi = doi_match.group(1)
                # BMC PDFs are typically available at the article URL with /pdf suffix
                # or via direct DOI-based URL
                base_url = await page.evaluate("window.location.origin")
                pdf_url = f"{base_url}/track/pdf/{doi}"
                pdf_urls.append(pdf_url)
                return pdf_urls
        except Exception:
            pass

        # Method 4: Look for PDF in link elements
        try:
            pdf_link_elem = await page.locator(
                'link[type="application/pdf"]'
            ).first.get_attribute("href", timeout=1000)
            if pdf_link_elem:
                if pdf_link_elem.startswith("/"):
                    base_url = await page.evaluate("window.location.origin")
                    pdf_link_elem = f"{base_url}{pdf_link_elem}"
                pdf_urls.append(pdf_link_elem)
                return pdf_urls
        except Exception:
            pass

        return pdf_urls
