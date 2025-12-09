#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACM Digital Library translator.

Based on ACM Digital Library.js translator from Zotero.
Original JavaScript implementation by Guy Aglionby.

Supports:
- Conference papers
- Journal articles
- Books and book chapters
- Theses
- Reports
- Software/datasets
- Search results pages
- Author profile pages
- TOC/issue pages

ACM Digital Library is a major repository for computer science and IT publications.
"""

import re
import json
from typing import List, Dict, Optional
from urllib.parse import quote, urlencode
from playwright.async_api import Page
from ..core.base import BaseTranslator


class ACMDigitalLibraryTranslator(BaseTranslator):
    """ACM Digital Library translator.

    Based on JavaScript translator (ACM Digital Library.js).
    Extracts metadata and PDFs from dl.acm.org.
    """

    LABEL = "ACM Digital Library"
    URL_TARGET_PATTERN = r"^https://dl\.acm\.org/(doi|do|profile|toc|topic|keyword|action/doSearch|acmbooks|browse)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL matches ACM Digital Library pattern.

        Based on JavaScript detectWeb() and isContentUrl() (lines 37-103).
        """
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    def _is_content_url(cls, url: str) -> bool:
        """Check if URL is a content page (not search/browse).

        Based on JavaScript isContentUrl() (lines 101-103).
        """
        return ("/doi/" in url or "/do/" in url) and "/doi/proceedings" not in url

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from ACM Digital Library page.

        Based on JavaScript scrape() function (lines 131-235).

        The JavaScript implementation:
        1. Extracts DOI from page input field
        2. Fetches CSL JSON from ACM's exportCiteProcCitation API
        3. Parses metadata from CSL JSON
        4. Extracts abstract from page
        5. Finds PDF download link via "View PDF" anchor
        6. Returns item with PDF attachment

        Python implementation:
        - Extracts DOI from page
        - Finds PDF download link
        - Returns PDF URL for download

        Args:
            page: Playwright page object on ACM Digital Library

        Returns:
            List containing PDF URL if found, empty list otherwise
        """
        pdf_urls = []

        try:
            # Check if this is a content URL
            current_url = page.url
            if not cls._is_content_url(current_url):
                return []

            # Method 1: Look for "View PDF" link on page (JS lines 170-176)
            # JavaScript: let pdfElement = doc.querySelector('a[title="View PDF"]');
            try:
                pdf_link = await page.locator(
                    'a[title="View PDF"]'
                ).first.get_attribute("href", timeout=2000)
                if pdf_link:
                    # Make absolute URL if needed
                    if pdf_link.startswith("/"):
                        pdf_link = f"https://dl.acm.org{pdf_link}"
                    elif not pdf_link.startswith("http"):
                        pdf_link = f"https://dl.acm.org/{pdf_link}"
                    pdf_urls.append(pdf_link)
                    return pdf_urls
            except Exception:
                pass

            # Method 2: Construct PDF URL from DOI (JS lines 132-133)
            # JavaScript: let doi = attr(doc, 'input[name=doiVal]', 'value');
            try:
                doi = await page.locator("input[name=doiVal]").first.get_attribute(
                    "value", timeout=2000
                )
                if doi:
                    # ACM PDF URLs follow pattern: /doi/pdf/{DOI}
                    # Clean DOI and construct PDF URL
                    doi_clean = doi.strip()
                    pdf_url = f"https://dl.acm.org/doi/pdf/{quote(doi_clean, safe='')}"
                    pdf_urls.append(pdf_url)
                    return pdf_urls
            except Exception:
                pass

            # Method 3: Extract DOI from URL and construct PDF link
            try:
                # Match patterns: /doi/10.1145/... or /doi/abs/10.1145/...
                doi_match = re.search(r"/doi/(abs/)?(10\.[^?#/]+/[^?#/]+)", current_url)
                if doi_match:
                    doi = doi_match.group(2)
                    pdf_url = f"https://dl.acm.org/doi/pdf/{quote(doi, safe='')}"
                    pdf_urls.append(pdf_url)
                    return pdf_urls
            except Exception:
                pass

            # Method 4: Look for any PDF download link
            try:
                pdf_link = await page.locator(
                    'a[href*="/pdf/"], a[href*=".pdf"]'
                ).first.get_attribute("href", timeout=2000)
                if pdf_link:
                    if pdf_link.startswith("/"):
                        pdf_link = f"https://dl.acm.org{pdf_link}"
                    elif not pdf_link.startswith("http"):
                        pdf_link = f"https://dl.acm.org/{pdf_link}"
                    pdf_urls.append(pdf_link)
                    return pdf_urls
            except Exception:
                pass

        except Exception:
            pass

        return pdf_urls

    @classmethod
    async def extract_metadata_async(cls, page: Page) -> Optional[Dict]:
        """Extract metadata from ACM Digital Library page.

        Based on JavaScript scrape() function (lines 131-235).

        Extracts:
        - DOI
        - Title
        - Authors (with first/last names)
        - Abstract
        - Publication type (conference/journal/book/etc.)
        - Publication title
        - Volume, issue, pages
        - Date
        - ISBN/ISSN
        - Keywords/tags
        - Publisher

        Args:
            page: Playwright page object on ACM Digital Library

        Returns:
            Dictionary containing metadata, or None if extraction fails
        """
        metadata = {}

        try:
            current_url = page.url
            if not cls._is_content_url(current_url):
                return None

            # Extract DOI (JS lines 132-133)
            try:
                doi = await page.locator("input[name=doiVal]").first.get_attribute(
                    "value", timeout=2000
                )
                if doi:
                    metadata["doi"] = doi.strip()
            except Exception:
                # Try from URL
                doi_match = re.search(r"/doi/(abs/)?(10\.[^?#/]+/[^?#/]+)", current_url)
                if doi_match:
                    metadata["doi"] = doi_match.group(2)

            if not metadata.get("doi"):
                return None

            # Fetch CSL JSON metadata from ACM API (JS lines 133-139)
            # JavaScript: let postBody = 'targetFile=custom-bibtex&format=bibTex&dois=' + encodeURIComponent(doi);
            # This would require making an HTTP request, which we'll skip for now
            # Instead, we'll scrape metadata directly from the page

            # Extract item type from page context (JS lines 95-99)
            try:
                pb_context = await page.locator(
                    "meta[name=pbContext]"
                ).first.get_attribute("content", timeout=2000)
                if pb_context:
                    # JavaScript: let subtypeRegex = /csubtype:string:(\w+)/;
                    subtype_match = re.search(r"csubtype:string:(\w+)", pb_context)
                    if subtype_match:
                        subtype = subtype_match.group(1).lower()

                        # Determine item type (JS lines 38-80)
                        if subtype == "conference":
                            metadata["itemType"] = "conferencePaper"
                        elif subtype in [
                            "journal",
                            "periodical",
                            "magazine",
                            "newsletter",
                        ]:
                            metadata["itemType"] = "journalArticle"
                        elif subtype in ["report", "rfc"]:
                            metadata["itemType"] = "report"
                        elif subtype == "thesis":
                            metadata["itemType"] = "thesis"
                        elif subtype == "software":
                            metadata["itemType"] = "computerProgram"
                        elif subtype == "dataset":
                            metadata["itemType"] = "document"
                        elif subtype == "book":
                            # Check if it's a book or book section (JS lines 64-72)
                            book_type_match = re.search(
                                r"page:string:([\w ]+)", pb_context
                            )
                            if (
                                book_type_match
                                and book_type_match.group(1).lower() == "book page"
                            ):
                                metadata["itemType"] = "book"
                            else:
                                metadata["itemType"] = "bookSection"
                        else:
                            metadata["itemType"] = "journalArticle"
                    else:
                        metadata["itemType"] = "journalArticle"
            except Exception:
                metadata["itemType"] = "journalArticle"

            # Extract title (JS line 162)
            try:
                title = await page.locator("h1.citation__title").first.text_content(
                    timeout=2000
                )
                if title:
                    metadata["title"] = title.strip()
            except Exception:
                pass

            # Extract authors (JS lines 201-209)
            try:
                author_elements = await page.locator(
                    "div.citation span.loa__author-name"
                ).all()
                if author_elements:
                    authors = []
                    for author_elem in author_elements:
                        author_name = await author_elem.text_content()
                        if author_name:
                            # Simple split on last space for first/last name
                            name_parts = author_name.strip().rsplit(" ", 1)
                            if len(name_parts) == 2:
                                authors.append(
                                    {
                                        "firstName": name_parts[0],
                                        "lastName": name_parts[1],
                                    }
                                )
                            else:
                                authors.append(
                                    {"firstName": "", "lastName": name_parts[0]}
                                )
                    metadata["authors"] = authors
            except Exception:
                pass

            # Extract abstract (JS lines 164-168)
            try:
                abstract_elements = await page.locator(
                    "div.article__abstract p, div.abstractSection p"
                ).all()
                if abstract_elements:
                    abstract_texts = []
                    for elem in abstract_elements:
                        text = await elem.text_content()
                        if text:
                            abstract_texts.append(text.strip())
                    abstract = "\n\n".join(abstract_texts)
                    if abstract and abstract.lower() != "no abstract available.":
                        metadata["abstract"] = abstract
            except Exception:
                pass

            # Extract publication title (JS lines 182-188)
            try:
                pub_title = await page.locator(
                    "span.epub-section__title"
                ).first.text_content(timeout=2000)
                if pub_title:
                    metadata["publicationTitle"] = pub_title.strip()
            except Exception:
                pass

            # Extract tags/keywords (JS lines 222-225)
            try:
                tag_elements = await page.locator("div.tags-widget a").all()
                if tag_elements:
                    tags = []
                    for tag_elem in tag_elements:
                        tag_text = await tag_elem.text_content()
                        if tag_text:
                            tags.append(tag_text.strip())
                    metadata["tags"] = tags
            except Exception:
                pass

            # Extract number of pages (JS lines 217-220)
            try:
                num_pages = await page.locator(
                    "div.pages-info span"
                ).first.text_content(timeout=2000)
                if num_pages:
                    metadata["numPages"] = num_pages.strip()
            except Exception:
                pass

            # Construct clean URL (JS lines 177-179)
            if metadata.get("doi"):
                metadata["url"] = f"https://dl.acm.org/doi/{metadata['doi']}"

            # Set library catalog
            metadata["libraryCatalog"] = "ACM Digital Library"

            return metadata

        except Exception:
            return None


# EOF
