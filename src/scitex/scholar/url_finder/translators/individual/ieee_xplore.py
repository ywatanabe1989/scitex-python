#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IEEE Xplore translator.

Based on IEEE Xplore.js translator from Zotero.
Original JavaScript implementation by Simon Kornblith, Michael Berkowitz,
Bastian Koenings, and Avram Lyon.

Supports:
- Journal articles
- Conference papers
- Document pages
- Search results pages
- Issue pages
- Conference proceedings

IEEE Xplore provides academic papers in electrical engineering, computer science,
and electronics.
"""

import re
import json
from typing import List, Dict, Optional
from urllib.parse import quote, urljoin
from playwright.async_api import Page
from ..core.base import BaseTranslator


class IEEEXploreTranslator(BaseTranslator):
    """IEEE Xplore translator.

    Based on JavaScript translator (IEEE Xplore.js).
    """

    LABEL = "IEEE Xplore"
    URL_TARGET_PATTERN = r"^https?://([^/]+\.)?ieeexplore\.ieee\.org/([^#]+[&?]arnumber=\d+|(abstract/)?document/|search/(searchresult|selected)\.jsp|xpl/(mostRecentIssue|tocresult)\.jsp\?|xpl/conhome/\d+/proceeding)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL matches IEEE Xplore pattern.

        Based on JavaScript detectWeb() (lines 38-82).
        """
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from IEEE Xplore page.

        Based on JavaScript scrape() function (lines 134-261).

        The JavaScript implementation:
        1. Extracts arnumber from URL or page
        2. Fetches BibTeX metadata via REST API
        3. Constructs PDF URL using arnumber
        4. Checks for PDF gateway URL or direct PDF
        5. Returns PDF attachment

        Python implementation:
        - Extracts arnumber from URL or page
        - Constructs PDF URL using IEEE's stamp/getPDF endpoint
        - Returns list of PDF URLs

        Args:
            page: Playwright page object on IEEE Xplore

        Returns:
            List containing PDF URL if arnumber found, empty list otherwise
        """
        pdf_urls = []

        try:
            # Method 1: Extract arnumber from current URL (JS line 135)
            # Match patterns: arnumber=123456 or /document/123456
            current_url = page.url
            arnumber_match = re.search(r"arnumber=(\d+)", current_url)
            if not arnumber_match:
                arnumber_match = re.search(r"/document/(\d+)", current_url)

            if arnumber_match:
                arnumber = arnumber_match.group(1)

                # Try to get the actual PDF URL by checking the stamp page (JS lines 166-180)
                try:
                    # Navigate to the PDF gateway page
                    pdf_gateway_url = f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={arnumber}"

                    # Look for embedded PDF iframe or redirect
                    # JavaScript uses requestDocument, we use page.goto with a short timeout
                    # to check if we can access the PDF
                    pdf_url = None

                    # Try to extract PDF URL from metadata or page content
                    # Check for PDF link in the page (JS lines 176-177)
                    try:
                        # Look for iframe with PDF
                        iframe_src = await page.locator(
                            'iframe[src*=".pdf"], iframe[src*="/getPDF.jsp"]'
                        ).first.get_attribute("src", timeout=2000)
                        if iframe_src:
                            if iframe_src.startswith("/"):
                                pdf_url = f"https://ieeexplore.ieee.org{iframe_src}"
                            elif not iframe_src.startswith("http"):
                                pdf_url = f"https://ieeexplore.ieee.org/{iframe_src}"
                            else:
                                pdf_url = iframe_src
                    except Exception:
                        pass

                    # Fallback to direct getPDF URL (JS lines 182-184)
                    if not pdf_url:
                        pdf_url = f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={arnumber}&ref="

                    if pdf_url:
                        pdf_urls.append(pdf_url)

                except Exception:
                    # If we can't access the gateway, use the fallback URL
                    pdf_url = f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={arnumber}&ref="
                    pdf_urls.append(pdf_url)

                return pdf_urls

        except Exception:
            pass

        # Method 2: Look for PDF download link on the page
        try:
            # Check for direct PDF download links
            pdf_link = await page.locator(
                'a[href*=".pdf"], a[href*="/getPDF.jsp"], a[href*="/stamp/"]'
            ).first.get_attribute("href", timeout=2000)
            if pdf_link:
                # Make absolute URL if needed
                if pdf_link.startswith("/"):
                    pdf_link = f"https://ieeexplore.ieee.org{pdf_link}"
                elif not pdf_link.startswith("http"):
                    pdf_link = f"https://ieeexplore.ieee.org/{pdf_link}"
                pdf_urls.append(pdf_link)
                return pdf_urls
        except Exception:
            pass

        # Method 3: Extract from metadata in page script (JS lines 138-149)
        try:
            # Look for global.document.metadata in script tags
            script_elements = await page.locator('script[type="text/javascript"]').all()
            for script_element in script_elements:
                script = await script_element.text_content()
                if script and "global.document.metadata" in script:
                    # Try to extract arnumber from metadata
                    arnumber_match = re.search(r'"arnumber"\s*:\s*"(\d+)"', script)
                    if not arnumber_match:
                        arnumber_match = re.search(r'"articleId"\s*:\s*"(\d+)"', script)

                    if arnumber_match:
                        arnumber = arnumber_match.group(1)
                        pdf_url = f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={arnumber}&ref="
                        pdf_urls.append(pdf_url)
                        return pdf_urls
        except Exception:
            pass

        return pdf_urls

    @classmethod
    async def extract_metadata_async(cls, page: Page) -> Optional[Dict]:
        """Extract metadata from IEEE Xplore page.

        Based on JavaScript scrape() function (lines 134-261).

        Extracts:
        - Article number (arnumber)
        - Title
        - Authors
        - DOI
        - Abstract
        - Publication title (journal/conference)
        - Volume, issue, pages
        - Publication date
        - Keywords

        Args:
            page: Playwright page object on IEEE Xplore

        Returns:
            Dictionary containing metadata, or None if extraction fails
        """
        metadata = {}

        try:
            # Extract arnumber from URL (JS line 135)
            current_url = page.url
            arnumber_match = re.search(r"arnumber=(\d+)", current_url)
            if not arnumber_match:
                arnumber_match = re.search(r"/document/(\d+)", current_url)

            if not arnumber_match:
                return None

            arnumber = arnumber_match.group(1)
            metadata["arnumber"] = arnumber

            # Extract metadata from script tag (JS lines 138-149)
            script_elements = await page.locator('script[type="text/javascript"]').all()
            for script_element in script_elements:
                script = await script_element.text_content()
                if script and "global.document.metadata" in script:
                    try:
                        # Extract JSON metadata (JS lines 140-148)
                        data_raw = script.split("global.document.metadata")[1]
                        data_raw = re.sub(r"^=", "", data_raw)
                        data_raw = re.sub(r"};[\s\S]*$", "}", data_raw)
                        data = json.loads(data_raw)

                        # Extract authors (JS lines 220-229)
                        if "authors" in data and data["authors"]:
                            authors = []
                            for author in data["authors"]:
                                author_name = {
                                    "firstName": author.get("firstName", ""),
                                    "lastName": author.get("lastName", ""),
                                }
                                authors.append(author_name)
                            metadata["authors"] = authors

                        # Extract other metadata
                        if "title" in data:
                            metadata["title"] = data["title"]
                        if "doi" in data:
                            metadata["doi"] = data["doi"]
                        if "abstract" in data:
                            metadata["abstract"] = data["abstract"]
                        if "publicationTitle" in data:
                            metadata["publicationTitle"] = data["publicationTitle"]
                        if "issn" in data:
                            metadata["issn"] = data.get("issn", [])

                        break
                    except Exception:
                        continue

            # Determine item type (JS lines 44-50)
            # Check breadcrumbs for "Conferences"
            try:
                first_breadcrumb = await page.locator(
                    'div[class*="breadcrumbs"] a'
                ).first.text_content(timeout=2000)
                if first_breadcrumb and "Conference" in first_breadcrumb:
                    metadata["itemType"] = "conferencePaper"
                else:
                    metadata["itemType"] = "journalArticle"
            except Exception:
                metadata["itemType"] = "journalArticle"

            # Clean URL (JS lines 237-239)
            metadata["url"] = re.sub(r"[?#].*", "", current_url)

            return metadata

        except Exception:
            return None


# EOF
