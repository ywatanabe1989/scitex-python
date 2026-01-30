"""
Business Standard Translator

Translates Business Standard India articles to Zotero format.

Metadata:
    translatorID: e8d40f4b-c4c9-41ca-a59f-cf4deb3d3dc5
    label: Business Standard
    creator: Sebastian Karcher
    target: ^https?://www\.business-standard\.com
    minVersion: 2.1.9
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2016-11-01 18:25:24
"""

from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
import re


class BusinessStandardTranslator:
    """Translator for Business Standard India newspaper articles."""

    METADATA = {
        "translatorID": "e8d40f4b-c4c9-41ca-a59f-cf4deb3d3dc5",
        "label": "Business Standard",
        "creator": "Sebastian Karcher",
        "target": r"^https?://www\.business-standard\.com",
        "minVersion": "2.1.9",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2016-11-01 18:25:24",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """
        Detect if the page is a Business Standard article or search results.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            'multiple' for search results, 'newspaperArticle' for articles, empty string otherwise
        """
        if "/search?" in url:
            if doc.select('div[class*="main-cont-left"] h2 a'):
                return "multiple"
        if doc.select_one("div.content-main h1"):
            return "newspaperArticle"
        return ""

    def do_web(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract article data from Business Standard page.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            Dictionary containing article metadata or list of items for multiple
        """
        if self.detect_web(doc, url) == "multiple":
            return self.get_search_results(doc)
        return self.scrape(doc, url)

    def get_search_results(self, doc: BeautifulSoup) -> Dict[str, str]:
        """
        Get search results from the page.

        Args:
            doc: BeautifulSoup parsed document

        Returns:
            Dictionary mapping URLs to titles
        """
        items = {}
        rows = doc.select('div[class*="main-cont-left"] h2 a')
        for row in rows:
            href = row.get("href")
            title = row.get_text(strip=True)
            if href and title:
                items[href] = title
        return items

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
            "publicationTitle": "Business Standard India",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Extract title
        title_elem = doc.select_one("div.content-main h1")
        if title_elem:
            item["title"] = title_elem.get_text(strip=True)

        # Extract creators
        creator_elem = doc.select_one('p[class*="fL"] span:first-child')
        if creator_elem:
            creator_text = creator_elem.get_text(strip=True)
            # Remove pipe and everything after it
            creator_text = re.sub(r"\|\n?.+", "", creator_text)
            if creator_text:
                item["creators"].append(self._clean_author(creator_text))

        # Extract date
        date_meta = doc.select_one('meta[itemprop="datePublished"]')
        if date_meta and date_meta.get("content"):
            item["date"] = date_meta["content"]

        # Extract abstract
        abstract_meta = doc.select_one('meta[name="description"]')
        if abstract_meta and abstract_meta.get("content"):
            item["abstractNote"] = abstract_meta["content"]

        # Extract tags
        tags_div = doc.select('div.related-keyword div[class*="readmore_tagBG"]')
        for tag_elem in tags_div:
            tag_text = tag_elem.get_text(strip=True)
            if tag_text:
                for tag in tag_text.split(","):
                    tag = tag.strip()
                    if tag:
                        item["tags"].append({"tag": tag})

        # Add snapshot attachment
        print_url = url.replace("node", "print")
        item["attachments"].append(
            {
                "title": "Business Standard India Snapshot",
                "mimeType": "text/html",
                "url": print_url,
            }
        )

        return item

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
