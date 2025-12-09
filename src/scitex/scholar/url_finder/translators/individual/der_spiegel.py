"""
Der Spiegel Translator

Translates Der Spiegel (German news magazine) articles to Zotero format.

Metadata:
    translatorID: eef50507-c756-4081-86fd-700ae4ebf22e
    label: Der Spiegel
    creator: Martin Meyerhoff and Abe Jellinek
    target: ^https?://www\\.spiegel\\.de/
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2021-07-05 17:55:21
"""

from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
import json


class DerSpiegelTranslator:
    """Translator for Der Spiegel German news magazine."""

    METADATA = {
        "translatorID": "eef50507-c756-4081-86fd-700ae4ebf22e",
        "label": "Der Spiegel",
        "creator": "Martin Meyerhoff and Abe Jellinek",
        "target": r"^https?://www\.spiegel\.de/",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2021-07-05 17:55:21",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if page is article or search results."""
        ld_json_tag = doc.select_one('script[type="application/ld+json"]')
        if ld_json_tag and ld_json_tag.string:
            try:
                data = json.loads(ld_json_tag.string)
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]
                if data.get("@type") == "NewsArticle":
                    return "newspaperArticle"
            except (json.JSONDecodeError, IndexError):
                pass

        if self.get_search_results(doc, check_only=True):
            return "multiple"

        return ""

    def get_search_results(
        self, doc: BeautifulSoup, check_only: bool = False
    ) -> Optional[Dict[str, str]]:
        """Extract search results."""
        items = {}
        rows = doc.select(
            '[data-search-results] article h2 > a, [data-area="article-teaser-list"] article h2 > a'
        )

        for row in rows:
            href = row.get("href")
            title = row.get_text(strip=True)

            if not href or not title:
                continue

            if check_only:
                return {"found": "true"}

            items[href] = title

        return items if items else None

    def do_web(self, doc: BeautifulSoup, url: str) -> Any:
        """Main extraction method."""
        page_type = self.detect_web(doc, url)

        if page_type == "multiple":
            return self.get_search_results(doc, check_only=False)
        else:
            return self.scrape(doc, url)

    def _clean_author_objects(self, authors: List[Dict]) -> List[Dict]:
        """Clean and parse author objects from JSON-LD."""
        creators = []

        if not isinstance(authors, list):
            authors = [authors]

        for author in authors:
            if author.get("@type") == "Organization":
                continue

            name = author.get("name", "")
            if name:
                names = name.split()
                if len(names) >= 2:
                    creators.append(
                        {
                            "firstName": " ".join(names[:-1]),
                            "lastName": names[-1],
                            "creatorType": "author",
                        }
                    )
                else:
                    creators.append(
                        {"lastName": name, "creatorType": "author", "fieldMode": True}
                    )

        return creators

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Scrape article metadata."""
        item = {
            "itemType": "newspaperArticle",
            "publicationTitle": "Der Spiegel",
            "ISSN": "2195-1349",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Parse JSON-LD
        ld_json_tag = doc.select_one('script[type="application/ld+json"]')
        if ld_json_tag and ld_json_tag.string:
            try:
                json_data = json.loads(ld_json_tag.string)
                if isinstance(json_data, list) and len(json_data) > 0:
                    json_data = json_data[0]

                # Extract title
                if json_data.get("headline"):
                    item["title"] = json_data["headline"]

                # Extract authors
                if json_data.get("author"):
                    item["creators"] = self._clean_author_objects(json_data["author"])

                # Extract URL
                if json_data.get("url"):
                    item["url"] = json_data["url"]

                # Extract section
                if json_data.get("articleSection"):
                    item["section"] = json_data["articleSection"]

                # Extract date
                date = json_data.get("dateModified") or json_data.get("dateCreated")
                if date:
                    # Convert to ISO format
                    item["date"] = date[:10] if len(date) >= 10 else date

                # Extract abstract
                if json_data.get("description"):
                    item["abstractNote"] = json_data["description"]

                # Extract language
                if json_data.get("inLanguage"):
                    item["language"] = json_data["inLanguage"]

            except (json.JSONDecodeError, IndexError, KeyError):
                pass

        # Extract keywords as tags
        keywords_meta = doc.find("meta", {"name": "news_keywords"})
        if keywords_meta and keywords_meta.get("content"):
            keywords = keywords_meta["content"].split(", ")
            for kw in keywords:
                if kw.strip():
                    item["tags"].append({"tag": kw.strip()})

        # Set library catalog from domain
        hostname = url.split("/")[2] if "/" in url else "www.spiegel.de"
        item["libraryCatalog"] = hostname

        # Add snapshot attachment
        item["attachments"].append(
            {"title": "Snapshot", "mimeType": "text/html", "url": url}
        )

        return item
