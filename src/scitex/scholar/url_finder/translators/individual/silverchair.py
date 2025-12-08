#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Silverchair translator.

Based on Silverchair.js translator from Zotero.
Handles academic journals hosted on Silverchair platform, including:
- Oxford University Press journals (academic.oup.com)
- JAMA Network journals (jamanetwork.com)
- Rockefeller University Press journals (rupress.org)
- American Society of Hematology (ashpublications.org)
- GeoScienceWorld (pubs.geoscienceworld.org)
- ARVO Journals (arvojournals.org)
- American Institute of Physics (pubs.aip.org)

Original JavaScript implementation by Sebastian Karcher.
"""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class SilverchairTranslator(BaseTranslator):
    """Silverchair platform translator.

    Supports:
    - Journal articles (article, fullarticle, advance-article)
    - Article abstracts (article-abstract, advance-article-abstract)
    - Book chapters (chapter, chapter-abstract)
    - Books and edited volumes
    - Search results pages
    - Journal issue pages

    Based on JavaScript translator (Silverchair.js) lines 1-886.
    """

    LABEL = "Silverchair"
    URL_TARGET_PATTERN = r"/(article|fullarticle|advance-article|advance-article-abstract|article-abstract|book|edited-volume|chapter|chapter-abstract)(/|\.aspx)|search-results\?|/issue(/|s\.aspx|$)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL matches Silverchair platform pattern.

        Based on JavaScript detectWeb() function (lines 39-55).
        Matches URLs containing article, book, or issue patterns.

        Args:
            url: URL to check

        Returns:
            True if URL matches Silverchair pattern
        """
        return bool(re.search(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from Silverchair platform pages.

        Based on JavaScript scrape() function (lines 105-157).

        The JavaScript implementation:
        1. Extracts article ID from the page
        2. Downloads RIS citation format for metadata
        3. Extracts PDF URL from <a class="article-pdfLink"> or <a#pdf-link>
        4. For JAMA journals, uses data-article-url attribute

        Python implementation:
        - Extracts PDF URL using selectors from JS implementation
        - Handles multiple publisher-specific variations
        - Returns direct download URL

        Args:
            page: Playwright page object on Silverchair platform

        Returns:
            List containing PDF URL if found, empty list otherwise
        """
        pdf_urls = []

        def is_supplementary_material(url: str) -> bool:
            """Check if URL points to supplementary material."""
            supp_patterns = [
                r"/mmc\d+\.pdf",  # Multimedia component (Lancet, Elsevier)
                r"/attachment/",  # Generic attachment
                r"/supplement",  # Supplementary material
                r"/supp",  # Supplementary material (short)
                r"supplementary",  # Supplementary in filename
                r"appendix",  # Appendix
                r"SI\.pdf",  # Supporting Information
            ]
            return any(
                re.search(pattern, url, re.IGNORECASE) for pattern in supp_patterns
            )

        try:
            # Method 1: Extract from standard Silverchair PDF link (JS line 121)
            # JavaScript: let pdfURL = attr(doc, 'a.article-pdfLink', 'href');
            # Get ALL article-pdfLink elements to filter out supplementary materials
            all_links = await page.locator("a.article-pdfLink").all()

            for link in all_links:
                try:
                    pdf_url = await link.get_attribute("href", timeout=1000)
                    if not pdf_url:
                        continue

                    # Make absolute URL if needed
                    if pdf_url.startswith("/"):
                        base_url = await page.evaluate("window.location.origin")
                        pdf_url = f"{base_url}{pdf_url}"
                    elif not pdf_url.startswith("http"):
                        base_url = await page.evaluate("window.location.href")
                        base_url = re.sub(r"/[^/]*$", "", base_url)
                        pdf_url = f"{base_url}/{pdf_url}"

                    # Skip supplementary materials - we want the main article PDF
                    if is_supplementary_material(pdf_url):
                        continue

                    # Found main article PDF
                    pdf_urls.append(pdf_url)
                    return pdf_urls

                except Exception:
                    continue

        except Exception:
            pass

        try:
            # Method 2: JAMA Network journals (JS line 123)
            # JavaScript: if (!pdfURL) pdfURL = attr(doc, 'a#pdf-link', 'data-article-url');
            pdf_url = await page.locator("a#pdf-link").first.get_attribute(
                "data-article-url", timeout=2000
            )

            if pdf_url:
                # Make absolute URL if needed
                if pdf_url.startswith("/"):
                    base_url = await page.evaluate("window.location.origin")
                    pdf_url = f"{base_url}{pdf_url}"
                elif not pdf_url.startswith("http"):
                    base_url = await page.evaluate("window.location.href")
                    base_url = re.sub(r"/[^/]*$", "", base_url)
                    pdf_url = f"{base_url}/{pdf_url}"

                pdf_urls.append(pdf_url)
                return pdf_urls

        except Exception:
            pass

        try:
            # Method 3: Alternative PDF link selector
            # Some Silverchair sites use different selectors
            alternative_selectors = [
                "a.pdf-download",
                'a[href*="/pdf/"]',
                'a[href*=".pdf"]',
                '.article-tools a[href*="pdf"]',
                'a:has-text("Download PDF")',
                'a[data-article-url*="pdf"]',
            ]

            for selector in alternative_selectors:
                try:
                    # Get all matching links to filter supplementary materials
                    all_links = await page.locator(selector).all()

                    for link in all_links:
                        try:
                            pdf_link = await link.get_attribute("href", timeout=500)
                            if not pdf_link:
                                # Try data-article-url attribute
                                pdf_link = await link.get_attribute(
                                    "data-article-url", timeout=500
                                )

                            if pdf_link:
                                # Make absolute URL
                                if pdf_link.startswith("/"):
                                    base_url = await page.evaluate(
                                        "window.location.origin"
                                    )
                                    pdf_link = f"{base_url}{pdf_link}"
                                elif not pdf_link.startswith("http"):
                                    base_url = await page.evaluate(
                                        "window.location.href"
                                    )
                                    base_url = re.sub(r"/[^/]*$", "", base_url)
                                    pdf_link = f"{base_url}/{pdf_link}"

                                # Skip supplementary materials
                                if is_supplementary_material(pdf_link):
                                    continue

                                pdf_urls.append(pdf_link)
                                return pdf_urls
                        except Exception:
                            continue
                except Exception:
                    continue

        except Exception:
            pass

        # Method 3.5: Lancet-specific PDF URL construction
        try:
            # Lancet uses /action/showPdf?pii=<PII> pattern
            current_url = await page.evaluate("window.location.href")
            if "thelancet.com" in current_url:
                # Extract PII from URL like /article/PIIS1474-4422(13)70075-9/fulltext
                pii_match = re.search(
                    r"/article/(PII[A-Z0-9\-()]+)", current_url, re.IGNORECASE
                )
                if pii_match:
                    pii = pii_match.group(1)
                    base_url = await page.evaluate("window.location.origin")
                    pdf_url = f"{base_url}/action/showPdf?pii={pii}"
                    pdf_urls.append(pdf_url)
                    return pdf_urls
        except Exception:
            pass

        # Method 4: Extract from meta tags (common fallback)
        try:
            # Many Silverchair sites use citation_pdf_url meta tag
            pdf_url = await page.locator(
                'meta[name="citation_pdf_url"]'
            ).first.get_attribute("content", timeout=2000)

            if pdf_url:
                # Make absolute URL if needed
                if pdf_url.startswith("/"):
                    base_url = await page.evaluate("window.location.origin")
                    pdf_url = f"{base_url}{pdf_url}"
                elif not pdf_url.startswith("http"):
                    base_url = await page.evaluate("window.location.href")
                    base_url = re.sub(r"/[^/]*$", "", base_url)
                    pdf_url = f"{base_url}/{pdf_url}"

                pdf_urls.append(pdf_url)
                return pdf_urls

        except Exception:
            pass

        # Method 5: Construct PDF URL from DOI (for Oxford and similar)
        try:
            # Get DOI from meta tag
            doi = await page.locator('meta[name="citation_doi"]').first.get_attribute(
                "content", timeout=2000
            )

            if doi:
                # Clean DOI (remove prefix if present)
                doi = re.sub(r"^(doi:|https?://doi\.org/)", "", doi)

                # Try common Silverchair PDF URL patterns
                base_url = await page.evaluate("window.location.origin")

                # Pattern 1: /article-pdf/DOI
                pdf_url = f"{base_url}/article-pdf/{doi}"
                pdf_urls.append(pdf_url)
                return pdf_urls

        except Exception:
            pass

        # Method 6: Look for download center links
        try:
            # Some publishers have a download center
            download_selectors = [
                "a.downloadsLink",
                'a:has-text("Download")',
                '.article-navigation a[href*="download"]',
            ]

            for selector in download_selectors:
                try:
                    download_link = await page.locator(selector).first.get_attribute(
                        "href", timeout=1000
                    )
                    if download_link and "pdf" in download_link.lower():
                        if download_link.startswith("/"):
                            base_url = await page.evaluate("window.location.origin")
                            download_link = f"{base_url}{download_link}"
                        elif not download_link.startswith("http"):
                            base_url = await page.evaluate("window.location.href")
                            base_url = re.sub(r"/[^/]*$", "", base_url)
                            download_link = f"{base_url}/{download_link}"

                        pdf_urls.append(download_link)
                        return pdf_urls
                except Exception:
                    continue
        except Exception:
            pass

        return pdf_urls
