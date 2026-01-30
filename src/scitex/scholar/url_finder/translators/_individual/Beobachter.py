"""
Beobachter Translator

Translates Beobachter (Swiss magazine) articles to Zotero format.

Metadata:
    translatorID: a571680e-6338-46c2-a740-3cd9eb80fc7f
    label: Beobachter
    creator: Sebastian Karcher
    target: ^https?://((www\\.)?beobachter\\.ch/.)
    minVersion: 2.1.9
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2022-02-05 20:11:36
"""

from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup


class BeobachterTranslator:
    """Translator for Beobachter Swiss magazine."""

    METADATA = {
        "translatorID": "a571680e-6338-46c2-a740-3cd9eb80fc7f",
        "label": "Beobachter",
        "creator": "Sebastian Karcher",
        "target": r"^https?://((www\.)?beobachter\.ch/.)",
        "minVersion": "2.1.9",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2022-02-05 20:11:36",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """
        Detect if the page is a single article or multiple articles.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            'magazineArticle' for single item, 'multiple' for list, empty string otherwise
        """
        if doc.select(".article-header"):
            return "magazineArticle"
        elif self._get_search_results(doc, check_only=True):
            return "multiple"
        return ""

    def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """
        Extract article data from Beobachter page.

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
            "itemType": "magazineArticle",
            "ISSN": "1661-7444",
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Extract title from meta tags
        title_tag = doc.find("meta", {"property": "og:title"})
        if title_tag and title_tag.get("content"):
            # Remove " | Beobachter" suffix
            title = title_tag["content"]
            title = title.replace(" | Beobachter", "").strip()
            item["title"] = title

        # Extract authors from parsely-author meta tags
        author_tags = doc.find_all("meta", {"name": "parsely-author"})
        for author_tag in author_tags:
            if author_tag.get("content"):
                author_name = author_tag["content"]
                item["creators"].append(self._clean_author(author_name))

        # Extract date from published_at meta tag
        date_tag = doc.find("meta", {"name": "published_at"})
        if date_tag and date_tag.get("content"):
            item["date"] = date_tag["content"]

        # Extract abstract
        abstract_tag = doc.find("meta", {"property": "og:description"})
        if abstract_tag and abstract_tag.get("content"):
            item["abstractNote"] = abstract_tag["content"]

        # Extract language
        lang_tag = doc.find("meta", {"property": "og:locale"})
        if lang_tag and lang_tag.get("content"):
            item["language"] = lang_tag["content"]

        # Extract URL
        url_tag = doc.find("meta", {"property": "og:url"})
        if url_tag and url_tag.get("content"):
            item["url"] = url_tag["content"]
        else:
            item["url"] = url

        # Add snapshot attachment
        item["attachments"].append(
            {"title": "Snapshot", "mimeType": "text/html", "url": url}
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
        # Look for teaser links
        rows = doc.select('a[class*="teaser"]')

        if not rows:
            return None

        if check_only:
            return True

        items = {}
        for row in rows:
            href = row.get("href")
            # Get title from span within the link
            title_elem = row.select_one("span")
            if title_elem:
                title = title_elem.get_text().strip()
            else:
                continue

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
