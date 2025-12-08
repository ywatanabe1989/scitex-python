#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python implementation of ORCID Zotero translator.

Original JavaScript: ORCID.js
Translator ID: e83248bb-caa4-4dd2-a470-11f4cd164083
Creator: Philipp Zumstein

Key logic:
- detectWeb (lines 39-50): Checks for ORCID ID and work list
- doWeb (lines 65-94): Fetches works via ORCID API and allows user selection
- lookupWork (lines 53-62): Retrieves individual work metadata as CSL JSON

Note: This translator extracts bibliographic metadata from ORCID profiles,
not PDF URLs. It uses the ORCID public API to fetch work information.
"""

import re
from typing import List, Dict, Optional
from xml.etree import ElementTree as ET

import httpx
from playwright.async_api import Page


class ORCIDTranslator:
    """ORCID metadata extractor - Python implementation.

    This translator extracts bibliographic information from ORCID profiles
    via the ORCID public API. Unlike other translators that extract PDF URLs,
    this one retrieves publication metadata in CSL JSON format.
    """

    LABEL = "ORCID"
    URL_TARGET_PATTERN = r"^https?://orcid\.org/"

    # ORCID API endpoints
    API_BASE = "https://pub.orcid.org/v2.0"

    # XML namespaces used in ORCID API responses
    NAMESPACES = {
        "work": "http://www.orcid.org/ns/work",
        "activities": "http://www.orcid.org/ns/activities",
    }

    @classmethod
    def matches_url(cls, url: str) -> bool:
        """Check if URL matches ORCID pattern.

        Args:
            url: URL to check

        Returns:
            True if URL matches ORCID profile page pattern
        """
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def detect_web(cls, page: Page) -> Optional[str]:
        """Detect if page is an ORCID profile with works.

        Args:
            page: Playwright page on potential ORCID page

        Returns:
            "multiple" if works are found, None otherwise
        """
        # Check for ORCID ID element
        orcid_id = await page.query_selector("#orcid-id")
        if not orcid_id:
            return None

        # Check for work list
        works = await page.query_selector_all("ul#body-work-list > li")
        if works:
            return "multiple"

        return None

    @classmethod
    async def extract_orcid_from_page(cls, page: Page) -> Optional[str]:
        """Extract ORCID identifier from page.

        Args:
            page: Playwright page on ORCID profile

        Returns:
            ORCID identifier (e.g., "0000-0003-0902-4386") or None
        """
        orcid_element = await page.query_selector("#orcid-id")
        if not orcid_element:
            return None

        orcid_text = await orcid_element.text_content()
        if not orcid_text:
            return None

        # Remove URL prefix if present and clean extra UI text
        orcid = orcid_text.replace("https://orcid.org/", "").strip()

        # Extract just the ORCID ID pattern (0000-0000-0000-0000)
        # Remove any extra text like "content_copyprint" that might be from UI elements
        import re

        match = re.search(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", orcid)
        if match:
            return match.group(0)

        return orcid

    @classmethod
    async def fetch_works_list(cls, orcid: str) -> Dict[str, str]:
        """Fetch list of works from ORCID API.

        Args:
            orcid: ORCID identifier (e.g., "0000-0003-0902-4386")

        Returns:
            Dictionary mapping put-codes to work titles
        """
        url = f"{cls.API_BASE}/{orcid}/works"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.content)

        put_codes = {}

        # Find all work groups
        for group in root.findall(".//activities:group", cls.NAMESPACES):
            # Get first work summary in the group
            work_summary = group.find("./work:work-summary", cls.NAMESPACES)
            if work_summary is not None:
                code = work_summary.get("put-code")
                title_elem = work_summary.find(".//work:title", cls.NAMESPACES)
                if code and title_elem is not None and title_elem.text:
                    put_codes[code] = title_elem.text.strip()

        return put_codes

    @classmethod
    async def fetch_work_metadata(cls, orcid: str, put_code: str) -> Optional[Dict]:
        """Fetch individual work metadata as CSL JSON.

        Args:
            orcid: ORCID identifier
            put_code: Work put-code from ORCID API

        Returns:
            CSL JSON metadata dictionary or None if fetch fails
        """
        url = f"{cls.API_BASE}/{orcid}/work/{put_code}"

        headers = {"Accept": "application/vnd.citationstyles.csl+json"}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=30.0)
                response.raise_for_status()

            # Response should be CSL JSON
            return response.json()
        except Exception:
            return None

    @classmethod
    async def extract_metadata_async(cls, page: Page) -> List[Dict]:
        """Extract bibliographic metadata from ORCID profile.

        This is the main entry point that mimics the JavaScript doWeb function.
        It fetches all works from the ORCID profile and returns their metadata.

        Args:
            page: Playwright page on ORCID profile

        Returns:
            List of CSL JSON metadata dictionaries
        """
        # Extract ORCID from page
        orcid = await cls.extract_orcid_from_page(page)
        if not orcid:
            return []

        # Fetch list of works
        try:
            works = await cls.fetch_works_list(orcid)
        except Exception:
            return []

        if not works:
            return []

        # Fetch metadata for all works
        # Note: In the original JS, user selects which works to fetch
        # Here we fetch all works for simplicity
        metadata_list = []

        for put_code in works.keys():
            metadata = await cls.fetch_work_metadata(orcid, put_code)
            if metadata:
                metadata_list.append(metadata)

        return metadata_list

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        """Extract PDF URLs from ORCID profile.

        Note: ORCID profiles don't contain direct PDF URLs. This method
        is provided for API compatibility but returns an empty list.
        Use extract_metadata_async() instead to get bibliographic data.

        Args:
            page: Playwright page on ORCID profile

        Returns:
            Empty list (ORCID doesn't provide PDF URLs)
        """
        # ORCID profiles don't contain PDF URLs
        # They provide bibliographic metadata instead
        return []


if __name__ == "__main__":
    import asyncio
    from playwright.async_api import async_playwright

    async def main():
        """Demonstration of ORCIDTranslator usage."""
        # Example ORCID profile URL
        test_url = "https://orcid.org/0000-0003-0902-4386"

        print(f"Testing ORCIDTranslator with URL: {test_url}")
        print(f"URL matches pattern: {ORCIDTranslator.matches_url(test_url)}\n")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()

            print("Navigating to ORCID profile...")
            await page.goto(test_url, timeout=60000)
            await page.wait_for_load_state("domcontentloaded")

            # Detect if page has works
            result = await ORCIDTranslator.detect_web(page)
            print(f"Detection result: {result}")

            # Extract ORCID
            orcid = await ORCIDTranslator.extract_orcid_from_page(page)
            print(f"ORCID: {orcid}\n")

            if orcid:
                # Fetch works list
                print("Fetching works list...")
                works = await ORCIDTranslator.fetch_works_list(orcid)
                print(f"Found {len(works)} works:")
                for put_code, title in list(works.items())[:5]:
                    print(f"  [{put_code}] {title}")

                if works:
                    # Fetch metadata for first work
                    first_code = list(works.keys())[0]
                    print(f"\nFetching metadata for work {first_code}...")
                    metadata = await ORCIDTranslator.fetch_work_metadata(
                        orcid, first_code
                    )

                    if metadata:
                        print(f"Metadata keys: {list(metadata.keys())}")
                        if "title" in metadata:
                            print(f"Title: {metadata['title']}")
                        if "author" in metadata:
                            authors = metadata["author"]
                            if isinstance(authors, list) and authors:
                                print(f"First author: {authors[0]}")

            await browser.close()

    asyncio.run(main())


# EOF
