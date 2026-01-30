"""
Christian Science Monitor Translator

Translates magazine articles from Christian Science Monitor to Zotero format.

Metadata:
    translatorID: 04c0db88-a7fc-4d1a-9cf7-471d0990acb1
    label: Christian Science Monitor
    creator: Sebastian Karcher
    target: ^https?://(features\.csmonitor|www\.csmonitor)\.com
    minVersion: 2.1.9
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2016-11-01 18:26:01
"""

from typing import Dict, Any
from bs4 import BeautifulSoup


class ChristianScienceMonitorTranslator:
    """Translator for Christian Science Monitor magazine articles."""

    METADATA = {
        "translatorID": "04c0db88-a7fc-4d1a-9cf7-471d0990acb1",
        "label": "Christian Science Monitor",
        "creator": "Sebastian Karcher",
        "target": r"^https?://(features\.csmonitor|www\.csmonitor)\.com",
        "minVersion": "2.1.9",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2016-11-01 18:26:01",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if page is an article or search results."""
        if doc.select_one("h1#headline"):
            return "magazineArticle"

        if "/content/search?" in url:
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
            "itemType": "magazineArticle",
            "creators": [],
            "tags": [],
            "attachments": [],
            "url": url,
            "ISSN": "0882-7729",
            "publicationTitle": "Christian Science Monitor",
        }

        # Title
        headline = doc.select_one("h1#headline")
        if headline:
            item["title"] = headline.get_text().strip()

        # Authors
        author_meta = doc.find("meta", attrs={"name": "sailthru.author"})
        if author_meta and author_meta.get("content"):
            author_text = author_meta["content"]
            # Handle multiple authors
            authors = [a.strip() for a in author_text.split(",")]
            for author in authors:
                if author:
                    item["creators"].append(self._clean_author(author))

        # Date
        date_elem = doc.select_one("time#date-published")
        if date_elem and date_elem.get("datetime"):
            item["date"] = date_elem["datetime"]

        # Abstract
        summary = doc.select_one("h2#summary")
        if summary:
            item["abstractNote"] = summary.get_text().strip()

        # Snapshot attachment
        item["attachments"].append(
            {"title": "CS Monitor Snapshot", "mimeType": "text/html", "url": url}
        )

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
