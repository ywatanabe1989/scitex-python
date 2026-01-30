"""
DAI-Zenon Translator

Translates DAI-Zenon library catalog records to Zotero format.

Metadata:
    translatorID: 16199bf0-a365-4aad-baeb-225019ae32dc
    label: DAI-Zenon
    creator: Philipp Zumstein, Sebastian Karcher
    target: ^https?://zenon\\.dainst\\.org/(Record/|Search/)
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2020-10-13 15:24:32
"""

from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
import re


class DAIZenonTranslator:
    """Translator for DAI-Zenon library catalog."""

    METADATA = {
        "translatorID": "16199bf0-a365-4aad-baeb-225019ae32dc",
        "label": "DAI-Zenon",
        "creator": "Philipp Zumstein, Sebastian Karcher",
        "target": r"^https?://zenon\.dainst\.org/(Record/|Search/)",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2020-10-13 15:24:32",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if the page is a record or search results."""
        if "/Record" in url:
            return "book"  # Could be book, journalArticle, or bookSection
        elif self._get_search_results(doc, True):
            return "multiple"
        return ""

    def _get_search_results(
        self, doc: BeautifulSoup, check_only: bool = False
    ) -> Dict[str, str]:
        """Get search results from the page."""
        items = {}
        found = False
        rows = doc.select("div.row a.title")

        for row in rows:
            href = row.get("href")
            title = row.get_text(strip=True)
            if not href or not title:
                continue
            if check_only:
                return True
            found = True
            items[href] = title

        return items if found else False

    def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Extract record data."""
        if self.detect_web(doc, url) in ["book", "journalArticle", "bookSection"]:
            return [self.scrape(doc, url)]
        return []

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Scrape record data from the document.

        Note: This is a simplified version. The full translator requires
        MARC processing which is complex in Python.
        """
        item = {"itemType": "book", "creators": [], "tags": [], "attachments": []}

        # Extract title
        title_elem = doc.select_one("h1.title, .record-title")
        if title_elem:
            item["title"] = title_elem.get_text(strip=True)

        # Extract metadata from table rows
        for row in doc.select("table tr"):
            label_elem = row.find("th")
            value_elem = row.find("td")

            if not label_elem or not value_elem:
                continue

            label = label_elem.get_text(strip=True)
            value = value_elem.get_text(strip=True)

            if label == "Author" or label == "Creator":
                parts = value.split(",")
                if len(parts) >= 2:
                    item["creators"].append(
                        {
                            "firstName": parts[1].strip(),
                            "lastName": parts[0].strip(),
                            "creatorType": "author",
                        }
                    )
                else:
                    item["creators"].append(
                        {"lastName": value, "creatorType": "author", "fieldMode": True}
                    )

            elif label == "Publication Date" or label == "Year":
                item["date"] = value

            elif label == "Publisher":
                item["publisher"] = value

            elif label == "Place of Publication" or label == "Place":
                item["place"] = value

            elif label == "Series":
                item["series"] = value

            elif label == "Pages" or label == "Extent":
                item["numPages"] = re.sub(r"\s*p\.?$", "", value)

            elif label == "ISBN":
                item["ISBN"] = value

            elif label == "Language":
                item["language"] = value

            elif label == "Call Number":
                item["callNumber"] = value

        # Extract tags/subjects
        for tag_elem in doc.select(".subjects a, .tags a"):
            tag_text = tag_elem.get_text(strip=True)
            if tag_text:
                item["tags"].append({"tag": tag_text})

        # Extract abstract/description
        abstract_elem = doc.select_one(".description, .abstract")
        if abstract_elem:
            item["abstractNote"] = abstract_elem.get_text(strip=True)

        # Add attachment
        clean_url = re.sub(r"#.*", "", url)
        item["attachments"].append(
            {
                "url": clean_url,
                "title": "DAI Zenon Entry",
                "mimeType": "text/html",
                "snapshot": False,
            }
        )

        return item
