"""
Canada.com Translator

Translates newspaper articles from Canada.com to Zotero format.

Metadata:
    translatorID: 4da40f07-904b-4472-93b6-9bea1fe7d4df
    label: Canada.com
    creator: Adam Crymble
    target: ^https?://www\.canada\.com
    minVersion: 1.0.0b4.r5
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2021-12-28 04:34:11
"""

from typing import Dict, Any
from bs4 import BeautifulSoup
import re


class CanadaComTranslator:
    """Translator for Canada.com newspaper articles."""

    METADATA = {
        "translatorID": "4da40f07-904b-4472-93b6-9bea1fe7d4df",
        "label": "Canada.com",
        "creator": "Adam Crymble",
        "target": r"^https?://www\.canada\.com",
        "minVersion": "1.0.0b4.r5",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2021-12-28 04:34:11",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if page is an article or search results."""
        if "story" in url:
            return "newspaperArticle"
        elif "search" in url:
            return "multiple"
        return ""

    def do_web(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract article data."""
        return self.scrape(doc, url)

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Scrape article data from the document.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            Dictionary containing article metadata
        """
        item = {
            "itemType": "newspaperArticle",
            "creators": [],
            "tags": [],
            "attachments": [],
            "url": url,
        }

        # Title
        item["title"] = doc.title.string if doc.title else ""

        # Abstract
        abstract_elem = doc.select_one(".storyheader h4")
        if abstract_elem:
            item["abstractNote"] = abstract_elem.get_text().strip()
        elif doc.select_one(".storyheader h2"):
            item["abstractNote"] = doc.select_one(".storyheader h2").get_text().strip()

        # Author
        author_meta = doc.find("meta", attrs={"name": "Author"})
        if author_meta and author_meta.get("content"):
            author_text = author_meta["content"]

            # Handle multiple authors separated by newlines or "and"
            if "\n" in author_text:
                author_parts = author_text.split("\n")
                author_text = author_parts[0]

            if " and " in author_text:
                authors = author_text.split(" and ")
                for author in authors:
                    item["creators"].append(self._clean_author(author.strip()))
            else:
                item["creators"].append(self._clean_author(author_text.strip()))

        # Date
        date_meta = doc.find("meta", attrs={"name": "PubDate"})
        if date_meta and date_meta.get("content"):
            item["date"] = date_meta["content"].strip()

        # Publication title
        pub_title_elem = doc.select_one("ul.home li a span")
        if pub_title_elem:
            pub_title = pub_title_elem.get_text().strip()
            if pub_title.endswith("Home"):
                pub_title = pub_title[:-4].strip()
            item["publicationTitle"] = pub_title
        else:
            item["publicationTitle"] = "Canada.com"

        return item

    def _clean_author(self, name: str) -> Dict[str, Any]:
        """Parse author name into first and last name."""
        name = name.strip()
        parts = name.split()

        if len(parts) >= 2:
            return {
                "firstName": " ".join(parts[:-1]),
                "lastName": parts[-1],
                "creatorType": "author",
            }
        else:
            return {"lastName": name, "creatorType": "author", "fieldMode": True}
