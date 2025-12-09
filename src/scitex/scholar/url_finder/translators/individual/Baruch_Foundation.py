"""
Baruch Foundation Translator

Translates Baruch Foundation artwork pages to Zotero format.

Metadata:
    translatorID: 283d6b78-d3d7-48d4-8fc0-0bdabef7c4ee
    label: Baruch Foundation
    creator: Abe Jellinek
    target: ^https?://baruchfoundation\\.org/
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2021-08-20 18:55:13
"""

from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import re


class BaruchFoundationTranslator:
    """Translator for Baruch Foundation artwork pages."""

    METADATA = {
        "translatorID": "283d6b78-d3d7-48d4-8fc0-0bdabef7c4ee",
        "label": "Baruch Foundation",
        "creator": "Abe Jellinek",
        "target": r"^https?://baruchfoundation\.org/",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2021-08-20 18:55:13",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """
        Detect if the page is a single artwork or multiple artworks.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            'artwork' for single item, 'multiple' for list, empty string otherwise
        """
        if doc.find(id="img-artist"):
            return "artwork"
        elif self._get_search_results(doc, check_only=True):
            return "multiple"
        return ""

    def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """
        Extract artwork data from Baruch Foundation page.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            List of dictionaries containing artwork metadata
        """
        if self.detect_web(doc, url) == "multiple":
            results = self._get_search_results(doc, check_only=False)
            # In a real implementation, this would process each URL
            # For now, return placeholder indicating multiple items
            return [{"itemType": "multiple", "urls": results}]
        else:
            return [self.scrape(doc, url)]

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Scrape artwork data from the document.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            Dictionary containing artwork metadata
        """
        item = {
            "itemType": "artwork",
            "archive": "Baruch Foundation",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Extract title
        title_elem = doc.select_one("h1#title b")
        if title_elem:
            item["title"] = title_elem.get_text().strip()

        # Extract abstract/description
        desc_elem = doc.select_one(".description")
        if desc_elem:
            item["abstractNote"] = desc_elem.get_text().strip()

        # Extract artwork medium from tags
        tag_elems = doc.select(".taglist a")
        if tag_elems:
            media = [tag.get_text().strip() for tag in tag_elems]
            item["artworkMedium"] = ", ".join(media)

        # Extract artwork size (stored oddly in zp_uneditable_image_location class)
        size_elem = doc.select_one(".zp_uneditable_image_location")
        if size_elem:
            item["artworkSize"] = size_elem.get_text().strip()

        # Extract date (stored in zp_uneditable_image_city class)
        date_elem = doc.select_one(".zp_uneditable_image_city")
        if date_elem:
            date_text = date_elem.get_text().strip()
            if date_text.lower() != "no date":
                item["date"] = date_text

        # Extract rights/credit
        credit_elem = doc.select_one(".credit")
        if credit_elem:
            item["rights"] = credit_elem.get_text().strip()

        # Extract artist
        artist_elem = doc.select_one("#img-artist em")
        if artist_elem:
            artist_name = artist_elem.get_text().strip()
            # Remove "Dr." prefix
            artist_name = re.sub(r"^Dr\.?\b\s*", "", artist_name)
            item["creators"].append(self._clean_author(artist_name, "artist"))

        # Extract image attachment
        img_link = doc.select_one("#img-full")
        if img_link and img_link.get("href"):
            item["attachments"].append(
                {"title": "Image", "mimeType": "image/jpeg", "url": img_link["href"]}
            )

        return item

    def _get_search_results(
        self, doc: BeautifulSoup, check_only: bool = False
    ) -> Optional[Dict[str, str]]:
        """
        Get search results from a page.

        Args:
            doc: BeautifulSoup parsed document
            check_only: If True, just check if results exist

        Returns:
            Dictionary mapping URLs to titles, or True/None if check_only
        """
        items = {}
        rows = doc.select('h4 > a[href*=".jpg.php"]')

        if not rows:
            return None

        if check_only:
            return True

        for row in rows:
            href = row.get("href")
            title = row.get_text().strip()
            if href and title:
                items[href] = title

        return items if items else None

    def _clean_author(self, name: str, creator_type: str = "artist") -> Dict[str, Any]:
        """
        Parse author name into first and last name.

        Args:
            name: Full author name
            creator_type: Type of creator (default: 'artist')

        Returns:
            Dictionary with firstName and lastName
        """
        name = name.strip()
        parts = name.split()

        if len(parts) >= 2:
            return {
                "firstName": " ".join(parts[:-1]),
                "lastName": parts[-1],
                "creatorType": creator_type,
            }
        else:
            return {"lastName": name, "creatorType": creator_type, "fieldMode": True}
