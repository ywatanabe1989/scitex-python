"""
Canadiana.ca Translator

Translates historical Canadian books from Canadiana to Zotero format.

Metadata:
    translatorID: 2d174277-7651-458f-86dd-20e168d2f1f3
    label: Canadiana.ca
    creator: Adam Crymble, Sebastian Karcher
    target: ^https?://eco\.canadiana\.ca
    minVersion: 1.0.0b4.r5
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2012-07-03 16:44:04
"""

from typing import Dict, Any
from bs4 import BeautifulSoup
import re


class CanadianaCATranslator:
    """Translator for Canadiana.ca historical books."""

    METADATA = {
        "translatorID": "2d174277-7651-458f-86dd-20e168d2f1f3",
        "label": "Canadiana.ca",
        "creator": "Adam Crymble, Sebastian Karcher",
        "target": r"^https?://eco\.canadiana\.ca",
        "minVersion": "1.0.0b4.r5",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2012-07-03 16:44:04",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if page is a book or search results."""
        if "/view/" in url:
            return "book"
        elif "/search?" in url:
            return "multiple"
        return ""

    def do_web(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract book data."""
        return self.scrape(doc, url)

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Scrape book data from the document.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            Dictionary containing book metadata
        """
        item = {
            "itemType": "book",
            "creators": [],
            "tags": [],
            "attachments": [],
            "url": url,
        }

        data_tags = {}

        # Extract metadata from table
        table = doc.select_one("#documentRecord table tbody")
        if table:
            rows = table.find_all("tr")
            for row in rows:
                header = row.find("th")
                value = row.find("td")
                if header and value:
                    field_title = header.get_text().strip().replace("\s+", "")
                    field_value = value.get_text().strip()

                    # Normalize French to English field names
                    field_title = self._normalize_field_name(field_title)
                    data_tags[field_title] = field_value

        # Process Creator/Author
        if "Creator" in data_tags or "PrincipalAuthor" in data_tags:
            author_name = data_tags.get("Creator") or data_tags.get("PrincipalAuthor")
            if author_name:
                item["creators"].append(self._clean_author(author_name))

        # Process Published/Imprint information
        if "Published" in data_tags or "Imprint" in data_tags:
            imprint = data_tags.get("Published") or data_tags.get("Imprint")
            if imprint:
                # Extract date
                date_match = re.search(r"\d+[-?\s\d]*", imprint)
                if date_match:
                    item["date"] = date_match.group(0).strip()

                # Extract place
                place_match = re.search(r".+?:", imprint)
                if place_match:
                    item["place"] = (
                        place_match.group(0)
                        .replace(":", "")
                        .replace("[", "")
                        .replace("]", "")
                        .strip()
                    )

                # Extract publisher
                publisher_match = re.search(r":[^,\d]+", imprint)
                if publisher_match:
                    item["publisher"] = (
                        publisher_match.group(0)
                        .replace(":", "")
                        .replace("[", "")
                        .replace("]", "")
                        .replace("?", "")
                        .strip()
                    )

        # Process subjects/tags
        if "Subjects" in data_tags:
            subjects = data_tags["Subjects"].replace("\n", "||").split("||")
            for subject in subjects:
                subject = subject.strip()
                if subject:
                    item["tags"].append({"tag": subject})

        # Process identifier
        if "Identifier" in data_tags:
            item["extra"] = "CIHM Number: " + data_tags["Identifier"].strip()

        # Other fields
        self._associate_data(item, data_tags, "Title", "title")
        self._associate_data(item, data_tags, "Language", "language")
        self._associate_data(item, data_tags, "Pages", "pages")
        self._associate_data(item, data_tags, "DocumentSource", "rights")
        self._associate_data(item, data_tags, "PermanentLink", "url")

        # Clean title
        if item.get("title"):
            item["title"] = " ".join(item["title"].split())

        # Fix language encoding
        if item.get("language"):
            if "English" in item["language"] or "Anglais" in item["language"]:
                item["language"] = "en-CA"

        return item

    def _normalize_field_name(self, field_name: str) -> str:
        """Normalize French field names to English."""
        mapping = {
            "Créateur": "Creator",
            "Titre": "Title",
            "Adressebibliographique": "Published",
            "Éditeur": "Published",
            "Langue": "Language",
            "Nombredepages": "Pages",
            "Description": "Extent",
            "Sujet": "Subjects",
            "Identificateur": "Identifier",
            "ICMHno": "Identifier",
            "Documentoriginal": "DocumentSource",
            "Lienpermanent": "PermanentLink",
        }
        return mapping.get(field_name, field_name)

    def _clean_author(self, name: str) -> Dict[str, Any]:
        """Parse author name."""
        name = name.strip()

        if "," in name:
            parts = name.split(",", 1)
            first_name = parts[1].strip() if len(parts) > 1 else ""
            last_name = parts[0].strip()
            return {
                "firstName": first_name,
                "lastName": last_name,
                "creatorType": "author",
            }
        else:
            return {"lastName": name, "creatorType": "author", "fieldMode": True}

    def _associate_data(
        self, item: Dict, data_tags: Dict, field: str, zotero_field: str
    ):
        """Associate data from tags to item."""
        if field in data_tags:
            item[zotero_field] = data_tags[field]
