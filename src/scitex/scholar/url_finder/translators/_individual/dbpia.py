"""
DBpia Translator

Translates metadata from DBpia (Korean database).

Metadata:
    translatorID: 0c31f371-e012-4b1c-b793-f89ab1ae2610
    label: DBpia
    creator: Yunwoo Song, Philipp Zumstein
    target: ^https?://[^/]+\\.dbpia\\.co\\.kr/
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    lastUpdated: 2021-09-14 00:23:40
"""

from typing import Dict, Any, List, Optional


class DBpiaTranslator:
    """Translator for DBpia."""

    METADATA = {
        "translatorID": "0c31f371-e012-4b1c-b793-f89ab1ae2610",
        "label": "DBpia",
        "creator": "Yunwoo Song, Philipp Zumstein",
        "target": r"^https?://[^/]+\.dbpia\.co\.kr/",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,  # Web
        "lastUpdated": "2021-09-14 00:23:40",
    }

    def detect_web(self, doc, url: str) -> Optional[str]:
        """
        Detect if URL is DBpia and what type of content.

        Args:
            doc: HTML document
            url: URL of the page

        Returns:
            Item type or None
        """
        if "/journal/articleDetail" in url:
            return "journalArticle"
        elif (
            "/search/" in url or "/journal/articleList/" in url
        ) and self._has_search_results(doc):
            return "multiple"
        return None

    def do_web(self, doc, url: str) -> List[Dict[str, Any]]:
        """
        Scrape DBpia pages.

        Args:
            doc: HTML document
            url: URL of the page

        Returns:
            List of scraped items
        """
        item_type = self.detect_web(doc, url)

        if item_type == "multiple":
            return []  # Handled by selectItems
        elif item_type == "journalArticle":
            return [self.scrape(doc, url)]
        return []

    def scrape(self, doc, url: str) -> Dict[str, Any]:
        """
        Scrape a single DBpia article.

        Args:
            doc: HTML document
            url: URL of the page

        Returns:
            Scraped item data
        """
        item = {
            "itemType": "journalArticle",
            "creators": [],
            "tags": [],
            "attachments": [],
            "libraryCatalog": "www.dbpia.co.kr",
            "url": url,
        }

        # Uses embedded metadata translator in JavaScript version
        # Would delegate to Embedded Metadata translator

        return item

    def _has_search_results(self, doc) -> bool:
        """Check if document has search results."""
        # Placeholder - would check for h5 > a[href*="/journal/articleDetail"]
        return False
