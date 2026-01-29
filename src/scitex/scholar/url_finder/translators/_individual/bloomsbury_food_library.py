"""
Bloomsbury Food Library Translator

Translates content from Bloomsbury Food Library.

Metadata:
    translatorID: 0524c89b-2a96-4d81-bb05-ed91ed8b2b47
    label: Bloomsbury Food Library
    creator: Abe Jellinek
    target: ^https?://(www\.)?bloomsburyfoodlibrary\.com/
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2021-08-03 01:17:12
"""

from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import re


class BloomsburyFoodLibraryTranslator:
    """Translator for Bloomsbury Food Library content."""

    METADATA = {
        "translatorID": "0524c89b-2a96-4d81-bb05-ed91ed8b2b47",
        "label": "Bloomsbury Food Library",
        "creator": "Abe Jellinek",
        "target": r"^https?://(www\.)?bloomsburyfoodlibrary\.com/",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2021-08-03 01:17:12",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect page type."""
        ris_link = doc.select_one('a[href*="/getris"]')
        if not ris_link:
            if self._get_search_results(doc, check_only=True):
                return "multiple"
            return ""

        # Determine type based on URL and facets
        if "bloomsburyfoodlibrary.com/encyclopedia-chapter" in url:
            subfacet = doc.select_one(".subfacet")
            if subfacet and "Book chapter" in subfacet.get_text():
                return "bookSection"
            else:
                return "encyclopediaArticle"
        elif "bloomsburyfoodlibrary.com/audio" in url:
            # Not well supported
            return ""
        elif "bloomsburyfoodlibrary.com/museum" in url:
            return "artwork"
        else:
            return "book"

    def _get_search_results(
        self, doc: BeautifulSoup, check_only: bool = False
    ) -> Dict[str, str]:
        """Get search results."""
        items = {}
        rows = doc.select("a#search-result-link")

        for row in rows:
            href = row.get("href")
            title = row.get_text(strip=True)
            if not href or not title:
                continue
            if check_only:
                return {"found": True}
            items[href] = title

        return items

    def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Extract data from the page."""
        page_type = self.detect_web(doc, url)

        if page_type == "multiple":
            items = self._get_search_results(doc, check_only=False)
            return [{"url": u} for u in items.keys()]
        else:
            return [self.scrape(doc, url, page_type)]

    def scrape(self, doc: BeautifulSoup, url: str, item_type: str) -> Dict[str, Any]:
        """
        Scrape item data using RIS download.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page
            item_type: Detected item type

        Returns:
            Dictionary containing metadata
        """
        item = {
            "itemType": item_type if item_type else "book",
            "libraryCatalog": "Bloomsbury Food Library",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Get RIS URL
        ris_link = doc.select_one('a[href*="/getris"]')
        if ris_link:
            item["_ris_url"] = ris_link.get("href")

        # Extract basic metadata from page (fallback)
        title_tag = doc.find("meta", {"property": "og:title"})
        if title_tag:
            title = title_tag.get("content", "")
            # Clean title
            title = title.replace(" : ", ": ")
            item["title"] = title

        # Extract ISBN
        isbn_tag = doc.find("meta", {"property": "og:isbn"})
        if isbn_tag:
            item["ISBN"] = isbn_tag.get("content", "")

        # Add snapshot
        item["attachments"].append(
            {"title": "Snapshot", "mimeType": "text/html", "url": url}
        )

        return item
