"""
Cornell University Press Translator

Translates Cornell University Press book pages to Zotero format.

Metadata:
    translatorID: 4363275e-5cc5-4627-9a7f-951fb58a02c3
    label: Cornell University Press
    creator: Sebastian Karcer
    target: ^https?://www\\.cornellpress\\.cornell\\.edu/
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2016-09-10 11:32:31
"""

from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
import re


class CornellUniversityPressTranslator:
    """Translator for Cornell University Press books."""

    METADATA = {
        "translatorID": "4363275e-5cc5-4627-9a7f-951fb58a02c3",
        "label": "Cornell University Press",
        "creator": "Sebastian Karcer",
        "target": r"^https?://www\.cornellpress\.cornell\.edu/",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2016-09-10 11:32:31",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if the page is a book or search page."""
        if "/book/" in url:
            return "book"
        elif "/search/?" in url or "/catalog/?" in url:
            return "multiple"
        return ""

    def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Extract book data."""
        if self.detect_web(doc, url) == "book":
            return [self.scrape(doc, url)]
        return []

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Scrape book data from the document."""
        item = {"itemType": "book", "creators": [], "tags": [], "attachments": []}

        # Extract fields from span.detailbox elements
        fields = doc.select("span.detailbox")

        for field_elem in fields:
            field_name = field_elem.get_text(strip=True)
            value_elem = field_elem.find_next_sibling("span", class_="DetailLabelText")
            if not value_elem:
                continue

            value = value_elem.get_text(strip=True)

            if field_name == "Title":
                item["title"] = value

            elif field_name == "Subtitle":
                if "title" in item:
                    item["title"] = f"{item['title']}: {value}"

            elif field_name == "Author":
                item["creators"].append(self._clean_author(value, "author"))

            elif field_name == "Authors":
                authors = value.split(",")
                for author in authors:
                    author = author.strip()
                    if author:
                        item["creators"].append(self._clean_author(author, "author"))

            elif field_name == "Edited by":
                editors = value.split(",")
                for editor in editors:
                    editor = editor.strip()
                    if editor:
                        item["creators"].append(self._clean_author(editor, "editor"))

            elif field_name == "Translated by":
                translators = value.split(",")
                for translator in translators:
                    translator = translator.strip()
                    if translator:
                        item["creators"].append(
                            self._clean_author(translator, "translator")
                        )

            elif field_name == "Publisher":
                item["publisher"] = value

            elif field_name == "ISBN-13":
                item["ISBN"] = value

            elif field_name in ["Publication Date", "Title First Published"]:
                item["date"] = value

            elif field_name == "Collection":
                item["series"] = value

            elif field_name in ["Language", "Languages"]:
                item["language"] = value

            elif field_name in ["Nb of pages", "Main content page count"]:
                item["numPages"] = value

            elif field_name == "BISAC Subject Heading":
                tags = value.split("\n")
                for tag in tags:
                    tag = re.sub(r".+/", "", tag).strip()
                    if tag:
                        item["tags"].append({"tag": tag})

        # Add default publisher and place if not specified
        if "publisher" not in item:
            item["publisher"] = "Cornell University Press"
            item["place"] = "Ithaca, NY"
        elif "place" not in item:
            if "Leuven" in item.get("publisher", ""):
                item["place"] = "Leuven"
            else:
                item["place"] = "Ithaca, NY"

        # Extract abstract
        abstract_elem = doc.select_one("div#bookpagedescription")
        if abstract_elem:
            item["abstractNote"] = abstract_elem.get_text(strip=True)

        return item

    def _clean_author(self, name: str, creator_type: str) -> Dict[str, Any]:
        """Parse author name into first and last name."""
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
