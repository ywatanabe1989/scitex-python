"""
Der Freitag Translator

Translates Der Freitag (German newspaper) articles to Zotero format.

Metadata:
    translatorID: 1ab8b9a4-72b5-4ef4-adc8-4956a50718f7
    label: Der Freitag
    creator: Sebastian Karcher
    target: ^https?://www\\.freitag\\.de
    minVersion: 2.1.9
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2022-04-15 16:29:59
"""

from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
import json


class DerFreitagTranslator:
    """Translator for Der Freitag German newspaper."""

    METADATA = {
        "translatorID": "1ab8b9a4-72b5-4ef4-adc8-4956a50718f7",
        "label": "Der Freitag",
        "creator": "Sebastian Karcher",
        "target": r"^https?://www\.freitag\.de",
        "minVersion": "2.1.9",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2022-04-15 16:29:59",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if page is article or search results."""
        ld_json_tag = doc.select_one(
            'script.qa-structured-data[type="application/ld+json"]'
        )
        if ld_json_tag and ld_json_tag.string:
            try:
                data = json.loads(ld_json_tag.string)
                item_type = data.get("@type", "")
                if item_type == "NewsArticle" or (
                    isinstance(item_type, list) and "NewsArticle" in item_type
                ):
                    return "newspaperArticle"
            except json.JSONDecodeError:
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
            ".o-search-results__container .c-article-card a.js-article-card-url"
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
            if author.get("@type") != "Person":
                continue

            name = author.get("name", "")
            if not name:
                continue

            # If name contains comma, split into multiple authors
            if "," in name:
                for one_author in name.split(","):
                    one_author = one_author.strip()
                    if one_author:
                        names = one_author.split()
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
                                {
                                    "lastName": one_author,
                                    "creatorType": "author",
                                    "fieldMode": True,
                                }
                            )
            else:
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
            "publicationTitle": "Der Freitag",
            "ISSN": "0945-2095",
            "libraryCatalog": "Der Freitag",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Parse JSON-LD
        ld_json_tag = doc.select_one(
            'script.qa-structured-data[type="application/ld+json"]'
        )
        if ld_json_tag and ld_json_tag.string:
            try:
                json_data = json.loads(ld_json_tag.string)

                # Extract title
                if json_data.get("headline"):
                    item["title"] = json_data["headline"]

                # Extract authors
                if json_data.get("author"):
                    item["creators"] = self._clean_author_objects(json_data["author"])

                # Extract date (use the latest date)
                date = json_data.get("dateModified")
                if not date or (
                    json_data.get("datePublished")
                    and json_data.get("datePublished") > date
                ):
                    date = json_data.get("datePublished")
                if date:
                    item["date"] = date

                # Extract abstract
                if json_data.get("description"):
                    item["abstractNote"] = json_data["description"]

                # Extract language
                if json_data.get("inLanguage"):
                    item["language"] = json_data["inLanguage"]

            except json.JSONDecodeError:
                pass

        # Extract section
        section_elem = doc.select_one("section ul li:nth-child(3) a span")
        if section_elem:
            item["section"] = section_elem.get_text(strip=True)

        # Extract tags
        tag_elems = doc.select(".qa-tags-container .qa-tags-item")
        for tag_elem in tag_elems:
            tag_text = tag_elem.get_text(strip=True)
            if tag_text:
                item["tags"].append({"tag": tag_text})

        # Add snapshot attachment
        item["attachments"].append(
            {"title": "Snapshot", "mimeType": "text/html", "url": url}
        )

        return item
