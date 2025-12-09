#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PLoS Journals translator.

Based on PLoS Journals.js
Authors: Michael Berkowitz, Rintze Zelle, and Sebastian Karcher
"""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class PLoSTranslator(BaseTranslator):
    """PLoS Journals translator for PLOS open-access publications.

    Supports:
    - journals.plos.org (all PLOS journals)
    - Legacy domains: plosone.org, plosntds.org, ploscompbiol.org, etc.
    - Article pages and search results
    """

    LABEL = "PLoS Journals"
    # Based on JavaScript target pattern (line 5)
    URL_TARGET_PATTERN = r"^https?://(www\.)?(plos(one|ntds|compbiol|pathogens|genetics|medicine|biology)\.org|journals\.plos\.org(/\w+)?)/(?:search|(?:\w+/)?article)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL is a PLOS article or search page.

        Args:
            url: URL to check

        Returns:
            True if URL matches PLOS pattern
        """
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from PLOS article page.

        PLOS uses embedded metadata (citation_pdf_url) for PDF links.
        Based on JavaScript translator which delegates to Embedded Metadata.

        PDF URL format:
        https://journals.plos.org/{journal}/article/file?id={DOI}&type=printable

        Args:
            page: Playwright page object on PLOS article

        Returns:
            List containing PDF URL if found, empty list otherwise
        """
        pdf_urls = []

        # Method 1: citation_pdf_url meta tag (primary method for PLOS)
        # PLOS embeds PDF URL in standard meta tag
        try:
            pdf_url = await page.locator(
                'meta[name="citation_pdf_url"]'
            ).first.get_attribute("content", timeout=3000)
            if pdf_url:
                pdf_urls.append(pdf_url)
                return pdf_urls
        except Exception:
            pass

        # Method 2: Look for PDF download link
        # PLOS may have direct download links on article pages
        try:
            pdf_link = await page.locator(
                'a[href*="/article/file"][href*="type=printable"]'
            ).first.get_attribute("href", timeout=2000)
            if pdf_link:
                # Make absolute URL if needed
                if pdf_link.startswith("/"):
                    base_url = await page.evaluate("window.location.origin")
                    pdf_link = f"{base_url}{pdf_link}"
                pdf_urls.append(pdf_link)
                return pdf_urls
        except Exception:
            pass

        # Method 3: Construct PDF URL from DOI if present
        # Pattern: https://journals.plos.org/journal/article/file?id=DOI&type=printable
        try:
            doi = await page.locator('meta[name="citation_doi"]').first.get_attribute(
                "content", timeout=2000
            )
            current_url = await page.evaluate("window.location.href")

            if doi:
                # Extract journal name from URL
                # Pattern: journals.plos.org/{journal}/article?id=...
                journal_match = re.search(r"journals\.plos\.org/([^/]+)/", current_url)
                if journal_match:
                    journal = journal_match.group(1)
                    pdf_url = f"https://journals.plos.org/{journal}/article/file?id={doi}&type=printable"
                    pdf_urls.append(pdf_url)
                    return pdf_urls
        except Exception:
            pass

        return pdf_urls


# Demo usage
# Note: Due to relative imports, run the demo script instead:
# python scripts/demo_plos.py
