"""
DPLA Translator

Translates metadata from Digital Public Library of America.

Metadata:
    translatorID: 117feb72-21bb-4424-a47b-c9ca6b71f979
    label: DPLA
    creator: Sebastian Karcher
    target: ^https?://dp\\.la/
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    lastUpdated: 2016-09-12 06:17:30
"""

from typing import Dict, Any, List, Optional
import json


class DPLATranslator:
    """Translator for DPLA."""

    METADATA = {
        "translatorID": "117feb72-21bb-4424-a47b-c9ca6b71f979",
        "label": "DPLA",
        "creator": "Sebastian Karcher",
        "target": r"^https?://dp\.la/",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,  # Web
        "lastUpdated": "2016-09-12 06:17:30",
    }

    # Type mapping
    TYPE_MAP = {
        "image": "artwork",
        "physical object": "artwork",
        "text": "book",
        "collection": "book",
        "moving image": "film",
        "interactive resource": "webpage",
        "dataset": "webpage",
        "software": "computerProgram",
        "sound": "audioRecording",
    }

    API_KEY = "910de961922b85c6e95ee1311938ece6"

    def detect_web(self, doc, url: str) -> Optional[str]:
        """
        Detect if URL is DPLA and what type of content.

        Args:
            doc: HTML document
            url: URL of the page

        Returns:
            Item type or None
        """
        if "/search?" in url:
            return "multiple"
        elif "/item/" in url:
            # Get type from page
            type_text = self._get_type_text(doc)
            if type_text:
                if re.search(r"^(image|physical)", type_text):
                    return "artwork"
                elif re.search(r"^sound", type_text):
                    return "audioRecording"
                elif re.search(r"^moving", type_text):
                    return "film"
                elif re.search(r"^software", type_text):
                    return "computerProgram"
                elif re.search(r"^(dataset|interactive)", type_text):
                    return "webpage"
                else:
                    return "book"
            else:
                return "book"
        return None

    def do_web(self, doc, url: str) -> List[Dict[str, Any]]:
        """
        Scrape DPLA pages.

        Args:
            doc: HTML document
            url: URL of the page

        Returns:
            List of scraped items
        """
        item_type = self.detect_web(doc, url)

        if item_type == "multiple":
            return []  # Handled by selectItems
        else:
            # Extract ID from URL
            import re

            match = re.search(r"item/([^\?]+)", url)
            if match:
                item_id = match.group(1)
                return self.fetch_from_api([item_id])
        return []

    def fetch_from_api(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch items from DPLA API.

        Args:
            ids: List of DPLA item IDs

        Returns:
            List of items
        """
        # Construct API URL
        api_url = f"http://api.dp.la/v2/items/{','.join(ids)}?api_key={self.API_KEY}"

        # In real implementation, would fetch from API
        # For now, return empty list
        return []

    def parse_api_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse DPLA API response.

        Args:
            response_text: JSON response text

        Returns:
            List of items
        """
        items = []

        try:
            data = json.loads(response_text)
            docs = data.get("docs", [])

            for doc in docs:
                item = self._parse_doc(doc)
                items.append(item)

        except json.JSONDecodeError:
            pass

        return items

    def _parse_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single DPLA document."""
        source = doc.get("sourceResource", {})

        item = {
            "itemType": "book",  # default
            "creators": [],
            "tags": [],
            "attachments": [],
            "libraryCatalog": "DPLA",
        }

        # Extract title
        title = source.get("title")
        if isinstance(title, list):
            item["title"] = title[0]
        else:
            item["title"] = title

        # Extract date
        date_info = source.get("date")
        if date_info:
            item["date"] = date_info.get("displayDate")

        # Extract creators
        creators = source.get("creator", [])
        if isinstance(creators, str):
            creators = [creators]

        for creator in creators:
            item["creators"].append(
                {"lastName": creator, "creatorType": "author", "fieldMode": True}
            )

        # Extract type
        item_type = source.get("type")
        if item_type:
            if isinstance(item_type, list):
                item_type = item_type[0]
            mapped_type = self.TYPE_MAP.get(item_type)
            if mapped_type:
                item["itemType"] = mapped_type

        return item

    def _get_type_text(self, doc) -> Optional[str]:
        """Extract type text from document."""
        # Placeholder
        return None
