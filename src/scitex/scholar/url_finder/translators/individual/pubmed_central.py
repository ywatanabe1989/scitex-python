#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PubMed Central translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class PubMedCentralTranslator(BaseTranslator):
    """PubMed Central."""

    LABEL = "PubMed Central"
    URL_TARGET_PATTERN = (
        r"^https://(www\.)?(pmc\.ncbi\.nlm\.nih\.gov/|ncbi\.nlm\.nih\.gov/pmc)"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from PubMed Central article page.

        Based on PubMed Central.js translator logic:
        - Try multiple XPath selectors for PDF links
        - Check if already on PDF page
        - Construct full URL if relative path found

        Args:
            page: Playwright page object with loaded PubMed Central content

        Returns:
            List of PDF URLs found on the page
        """
        pdf_urls = []
        current_url = page.url

        # Check if already viewing a PDF
        if re.search(r"/pdf/.+\.pdf", current_url):
            return [current_url]

        # Try multiple selectors (from JavaScript translator lines 70-74)
        selectors = [
            'td.format-menu a[href*=".pdf"]',
            'div.format-menu a[href*=".pdf"]',
            'aside#jr-alt-p div a[href*=".pdf"]',
            'li[class*="pdf-link"] a',
            'a[data-ga-label*="pdf_download_"]',
        ]

        for selector in selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    href = await element.get_attribute("href")
                    if href and ".pdf" in href.lower():
                        # Make absolute URL if relative
                        if href.startswith("/"):
                            href = f"https://www.ncbi.nlm.nih.gov{href}"
                        elif not href.startswith("http"):
                            href = f"https://www.ncbi.nlm.nih.gov/pmc/{href}"

                        if href not in pdf_urls:
                            pdf_urls.append(href)
            except Exception:
                continue

        return pdf_urls
