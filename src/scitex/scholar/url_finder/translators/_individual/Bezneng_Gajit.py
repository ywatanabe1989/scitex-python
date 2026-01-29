"""
Bezneng Gajit Translator

Translates Bezneng Gajit (Tatar newspaper) articles to Zotero format.

Metadata:
    translatorID: 7500180d-ca99-4ef7-a9a9-3e58bba91d28
    label: Bezneng Gajit
    creator: Avram Lyon
    target: ^https?://(www\\.)?beznen\\.ru
    minVersion: 1.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsbv
    lastUpdated: 2016-11-01 18:22:20
"""

from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import re


class BeznengGajitTranslator:
    """Translator for Bezneng Gajit (Безнең гәҗит) Tatar newspaper."""

    METADATA = {
        "translatorID": "7500180d-ca99-4ef7-a9a9-3e58bba91d28",
        "label": "Bezneng Gajit",
        "creator": "Avram Lyon",
        "target": r"^https?://(www\.)?beznen\.ru",
        "minVersion": "1.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsbv",
        "lastUpdated": "2016-11-01 18:22:20",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """
        Detect if the page is a single article or multiple articles.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            'newspaperArticle' for single item, 'multiple' for list, empty string otherwise
        """
        # Check if URL matches article pattern
        if re.search(r"(/basma)?/[0-9-]+/\w+|/[0-9-]+\w+/\w+", url):
            return "newspaperArticle"
        # Check for search results
        elif re.search(r"/rubric/|\?s=", url):
            if self._get_search_results(doc, check_only=True):
                return "multiple"
        return ""

    def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """
        Extract article data from Bezneng Gajit page.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            List of dictionaries containing article metadata
        """
        if self.detect_web(doc, url) == "multiple":
            results = self._get_search_results(doc, check_only=False)
            return [{"itemType": "multiple", "urls": results}]
        else:
            return [self.scrape(doc, url)]

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
            "publicationTitle": "Безнең гәҗит",
            "language": "татарча",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Extract title
        title_elem = doc.select_one("h1.entry-title")
        if title_elem:
            item["title"] = title_elem.get_text().strip()

        # Extract creators from authors paragraph
        # Skip entries that say "Килгән хатлардан" (from letters)
        author_elems = doc.select("div.entry-meta p.authors a")
        for author_elem in author_elems:
            author_text = author_elem.get_text().strip()
            # Skip if it's "Килгән хатлардан" pattern
            if not re.search(r"Килгән\sхатлардан", author_text):
                # Capitalize names that are in all caps
                if author_text == author_text.upper():
                    author_text = self._capitalize_name(author_text)
                item["creators"].append(self._clean_author(author_text))

        # Extract date from post-category paragraph
        # Format: "2011, № 41 (19 октябрь 2011)"
        date_elem = doc.select_one("div.entry-meta p.post-category")
        if date_elem:
            date_text = date_elem.get_text().strip()
            # Extract date from format: YYYY, № N (DD месяц YYYY)
            date_match = re.search(r"(\d{4}),\s*№\s*\d+\s*\((.*)\)", date_text)
            if date_match:
                year, date_full = date_match.groups()
                item["date"] = date_full

            # Extract issue number
            issue_match = re.search(r"№\s*(\d+)", date_text)
            if issue_match:
                item["issue"] = issue_match.group(1)

        # Add snapshot attachment
        item["attachments"].append(
            {"title": "Безнең гәҗит Snapshot", "mimeType": "text/html", "url": url}
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
        # Look for article links in h2.entry-title that contain /basma/
        rows = doc.select('h2.entry-title a[href*="/basma/"]')

        if not rows:
            return None

        if check_only:
            return True

        items = {}
        for row in rows:
            href = row.get("href")
            title = row.get_text().strip()
            if href and title:
                items[href] = title

        return items if items else None

    def _clean_author(self, name: str) -> Dict[str, Any]:
        """
        Parse author name into first and last name.

        Args:
            name: Full author name

        Returns:
            Dictionary with firstName and lastName
        """
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

    def _capitalize_name(self, name: str) -> str:
        """
        Capitalize a name that is in all uppercase.

        Args:
            name: Name in uppercase

        Returns:
            Name with proper capitalization
        """
        # Simple title case for Cyrillic text
        words = name.lower().split()
        return " ".join(word.capitalize() for word in words)
