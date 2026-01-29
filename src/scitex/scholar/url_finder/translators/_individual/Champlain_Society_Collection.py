"""
Champlain Society - Collection Translator

Translates historical books from Champlain Society Collection to Zotero format.

Metadata:
    translatorID: 50d3ca81-3c4c-406b-afb2-0fe8105b9b38
    label: Champlain Society - Collection
    creator: Adam Crymble
    target: ^https?://link\.library\.utoronto\.ca
    minVersion: 1.0.0b4.r5
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2021-12-28 04:41:28
"""

from typing import Dict, Any
from bs4 import BeautifulSoup
import re


class ChamplainSocietyCollectionTranslator:
    """Translator for Champlain Society Collection."""

    METADATA = {
        "translatorID": "50d3ca81-3c4c-406b-afb2-0fe8105b9b38",
        "label": "Champlain Society - Collection",
        "creator": "Adam Crymble",
        "target": r"^https?://link\.library\.utoronto\.ca",
        "minVersion": "1.0.0b4.r5",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2021-12-28 04:41:28",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if page is a book or search results."""
        if "search_results" in url:
            return "multiple"
        elif "item_record" in url:
            return "book"
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
        table = doc.find("table")
        if table:
            rows = table.find_all("tr")
            for row in rows:
                header = row.find("b")
                value_cell = row.find_all("td")

                if header and len(value_cell) >= 2:
                    field_title = header.get_text().strip().replace("\s+", "")
                    field_value = value_cell[1].get_text().strip()

                    # Normalize French to English
                    field_title = self._normalize_field_name(field_title)
                    data_tags[field_title] = field_value

        # Process authors
        if "Author:" in data_tags:
            author_text = data_tags["Author:"]
            if "; " in author_text:
                authors = author_text.split("; ")
            else:
                authors = [author_text]

            for author in authors:
                author = author.strip()
                if ", " in author:
                    parts = author.split(", ", 1)
                    first_name = parts[1] if len(parts) > 1 else ""
                    last_name = parts[0]
                    item["creators"].append(
                        {
                            "firstName": first_name,
                            "lastName": last_name,
                            "creatorType": "author",
                        }
                    )
                else:
                    item["creators"].append(
                        {"lastName": author, "creatorType": "author", "fieldMode": True}
                    )

        # Process published information
        if "Published:" in data_tags:
            published = data_tags["Published:"]

            # Extract place
            if ": " in published:
                place_end = published.index(": ")
                item["place"] = published[:place_end]
                remainder = published[place_end + 2 :]

                # Extract date and publisher
                if ", " in remainder:
                    date_start = remainder.rfind(", ")
                    item["date"] = remainder[date_start + 2 :]
                    item["publisher"] = remainder[:date_start]
                else:
                    item["publisher"] = remainder

        # Process subjects/tags
        if "Subjects:" in data_tags:
            subjects = data_tags["Subjects:"].split("\n")
            for subject in subjects:
                subject = subject.strip()
                if subject and re.search(r"\w", subject):
                    item["tags"].append({"tag": subject})

        # Other fields
        self._associate_data(item, data_tags, "Extent:", "pages")
        self._associate_data(item, data_tags, "ID:", "callNumber")
        self._associate_data(item, data_tags, "Notes:", "abstractNote")

        # Title
        if "Title:" in data_tags:
            item["title"] = data_tags["Title:"]
        elif doc.title:
            item["title"] = doc.title.string
        else:
            item["title"] = "No Title Found: Champlain Collection"

        return item

    def _normalize_field_name(self, field_name: str) -> str:
        """Normalize French field names to English."""
        mapping = {
            "Auteur:": "Author:",
            "Titre:": "Title:",
            "Description:": "Extent:",
            "Éditeur:": "Published:",
            "Sujet:": "Subjects:",
        }
        return mapping.get(field_name, field_name)

    def _associate_data(
        self, item: Dict, data_tags: Dict, field: str, zotero_field: str
    ):
        """Associate data from tags to item."""
        if field in data_tags:
            item[zotero_field] = data_tags[field]
