"""
Bloomberg Translator

Translates articles from Bloomberg.

Metadata:
    translatorID: a509f675-cf80-4b70-8cbc-2ea8664dd38f
    label: Bloomberg
    creator: Philipp Zumstein
    target: ^https?://(www)?\.bloomberg\.com
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2016-09-08 20:56:54
"""

from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import re


class BloombergTranslator:
    """Translator for Bloomberg articles and videos."""

    METADATA = {
        "translatorID": "a509f675-cf80-4b70-8cbc-2ea8664dd38f",
        "label": "Bloomberg",
        "creator": "Philipp Zumstein",
        "target": r"^https?://(www)?\.bloomberg\.com",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2016-09-08 20:56:54",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect page type."""
        if "/articles/" in url:
            return "newspaperArticle"

        og_type_tag = doc.find("meta", {"property": "og:type"})
        if og_type_tag and og_type_tag.get("content") == "article":
            return "newspaperArticle"

        if "/videos/" in url:
            return "tvBroadcast"

        if self._get_search_results(doc, check_only=True):
            return "multiple"

        return ""

    def _get_search_results(
        self, doc: BeautifulSoup, check_only: bool = False
    ) -> Dict[str, str]:
        """Get search results."""
        items = {}
        rows = doc.select('a[data-resource-id], a[data-tracker-label="headline"]')

        for row in rows:
            href = row.get("href")
            title = row.get_text(strip=True)
            if not href or not title:
                continue
            if check_only:
                return {"found": True}
            items[href] = title

        return items

    def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Extract data from the page."""
        page_type = self.detect_web(doc, url)

        if page_type == "multiple":
            items = self._get_search_results(doc, check_only=False)
            return [{"url": u} for u in items.keys()]
        else:
            return [self.scrape(doc, url, page_type)]

    def scrape(self, doc: BeautifulSoup, url: str, item_type: str) -> Dict[str, Any]:
        """
        Scrape article/video data using embedded metadata.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page
            item_type: Type of item (newspaperArticle or tvBroadcast)

        Returns:
            Dictionary containing metadata
        """
        item = {
            "itemType": item_type if item_type else "newspaperArticle",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Extract title
        title_tag = doc.find("meta", {"property": "og:title"})
        if title_tag:
            item["title"] = title_tag.get("content", "")

        # Extract authors
        author_tags = doc.select('meta[name="author"], meta[property="article:author"]')
        for author_tag in author_tags:
            author_name = author_tag.get("content", "").strip()
            if author_name:
                item["creators"].append(self._clean_author(author_name, "author"))

        # Extract date
        date_tag = doc.find("meta", {"property": "article:published_time"})
        if date_tag:
            item["date"] = date_tag.get("content", "")
        elif not item.get("date"):
            # Try alternate source from microdata
            time_tag = doc.select_one('main time[itemprop="datePublished"]')
            if time_tag:
                item["date"] = time_tag.get("datetime", "")

        # Extract abstract
        abstract_tag = doc.find("meta", {"property": "og:description"})
        if abstract_tag:
            item["abstractNote"] = abstract_tag.get("content", "")

        # Extract publication title
        site_name_tag = doc.find("meta", {"property": "og:site_name"})
        if site_name_tag:
            pub_title = site_name_tag.get("content", "")
            if item_type == "tvBroadcast":
                item["programTitle"] = pub_title
            else:
                item["publicationTitle"] = pub_title

        # Extract tags
        keywords_tag = doc.find("meta", {"property": "keywords"})
        if keywords_tag:
            keywords = keywords_tag.get("content", "")
            if keywords:
                tag_list = [k.strip() for k in keywords.split(",") if k.strip()]
                item["tags"] = [{"tag": t} for t in tag_list]

        # Add snapshot
        item["attachments"].append(
            {"title": "Snapshot", "mimeType": "text/html", "url": url}
        )

        return item

    def _clean_author(self, name: str, creator_type: str) -> Dict[str, Any]:
        """Parse author name into structured format."""
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
