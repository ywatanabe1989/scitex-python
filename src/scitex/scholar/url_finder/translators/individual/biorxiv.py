#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bioRxiv/medRxiv translator.

Translates bioRxiv and medRxiv URLs to extract PDF links and metadata.
bioRxiv is the preprint server for biology, medRxiv for medical sciences.

Supports:
- Article pages: https://www.biorxiv.org/content/10.1101/YYYY.MM.DD.NNNNNNvN
- Search results: https://www.biorxiv.org/search/...
- Collections: https://www.biorxiv.org/collection/...
- Both bioRxiv and medRxiv domains
"""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class BioRxivTranslator(BaseTranslator):
    """bioRxiv/medRxiv translator for extracting PDF URLs.

    bioRxiv and medRxiv are open-access preprint servers.
    All articles have freely available PDFs.
    """

    LABEL = "bioRxiv/medRxiv"
    URL_TARGET_PATTERN = (
        r"^https?://(www\.)?(biorxiv|medrxiv)\.org/(content|search|collection)/"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL is a bioRxiv or medRxiv URL.

        Args:
            url: URL to check

        Returns:
            True if URL matches bioRxiv/medRxiv pattern
        """
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from bioRxiv/medRxiv page.

        For article pages, extracts the PDF download link.
        For search/collection pages, extracts all article PDF links.

        bioRxiv/medRxiv PDF URL patterns:
        - Article: https://www.biorxiv.org/content/10.1101/YYYY.MM.DD.NNNNNNvN.full.pdf
        - Early: https://www.biorxiv.org/content/biorxiv/early/YYYY/MM/DD/DOI.full.pdf

        Args:
            page: Playwright page object

        Returns:
            List of PDF URLs found on the page
        """
        url = page.url
        pdf_urls = []

        # Method 1: citation_pdf_url meta tag (primary method)
        # bioRxiv embeds PDF URL in standard citation meta tag
        try:
            pdf_url = await page.locator(
                'meta[name="citation_pdf_url"]'
            ).first.get_attribute("content", timeout=3000)
            if pdf_url:
                pdf_urls.append(pdf_url)
                return pdf_urls
        except Exception:
            pass

        # Method 2: Direct PDF download link
        # Look for the "Download PDF" button
        try:
            pdf_link = await page.locator(
                'a.article-dl-pdf-link, a[href*=".full.pdf"]'
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

        # Method 3: Construct PDF URL from DOI if on article page
        # bioRxiv pattern: /content/{DOI}v{version} -> {DOI}v{version}.full.pdf
        if "/content/" in url and not any(x in url for x in ["/search", "/collection"]):
            try:
                # Extract DOI from URL
                # Pattern: /content/10.1101/YYYY.MM.DD.NNNNNNvN
                match = re.search(r"/content/([^?#]+)", url)
                if match:
                    doi_version = match.group(1)
                    # Determine base URL (biorxiv or medrxiv)
                    base_domain = "biorxiv.org" if "biorxiv" in url else "medrxiv.org"
                    pdf_url = (
                        f"https://www.{base_domain}/content/{doi_version}.full.pdf"
                    )
                    pdf_urls.append(pdf_url)
                    return pdf_urls
            except Exception:
                pass

        # Method 4: Search/collection pages - extract DOIs from results
        if any(x in url for x in ["/search", "/collection"]):
            try:
                # Find all DOI metadata elements in search results
                doi_elements = await page.query_selector_all(
                    ".highwire-cite-metadata-doi"
                )

                for doi_elem in doi_elements:
                    doi_text = await doi_elem.inner_text()
                    # Extract DOI, removing "doi:" prefix and "https://doi.org/" if present
                    doi = (
                        doi_text.strip()
                        .replace("doi:", "")
                        .replace("https://doi.org/", "")
                        .strip()
                    )

                    if doi.startswith("10.1101/"):
                        # Determine base URL (biorxiv or medrxiv)
                        base_domain = (
                            "biorxiv.org" if "biorxiv" in url else "medrxiv.org"
                        )
                        # bioRxiv search results don't include version, so use bare DOI
                        pdf_url = f"https://www.{base_domain}/content/{doi}.full.pdf"
                        pdf_urls.append(pdf_url)

                if pdf_urls:
                    return pdf_urls

            except Exception as e:
                # Log error but don't fail completely
                print(f"Error extracting PDF URLs from bioRxiv search: {e}")

        return pdf_urls
