"""
Denik CZ Translator

Translates Denik.cz (Czech newspaper) articles to Zotero format.

Metadata:
    translatorID: 4ed446ca-b480-43ee-a8fb-5f9730915edc
    label: Denik CZ
    creator: Jiří Sedláček, Philipp Zumstein
    target: ^https?://[^/]*denik\\.cz
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2018-01-07 09:27:42
"""

from typing import Dict, Any, Optional
from bs4 import BeautifulSoup


class DenikCZTranslator:
    """Translator for Denik.cz Czech newspaper."""

    METADATA = {
        "translatorID": "4ed446ca-b480-43ee-a8fb-5f9730915edc",
        "label": "Denik CZ",
        "creator": "Jiří Sedláček, Philipp Zumstein",
        "target": r"^https?://[^/]*denik\.cz",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2018-01-07 09:27:42",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if page is article or search results."""
        og_type = doc.find("meta", {"property": "og:type"})
        if og_type and og_type.get("content") == "article":
            return "newspaperArticle"
        elif self.get_search_results(doc, check_only=True):
            return "multiple"
        return ""

    def get_search_results(
        self, doc: BeautifulSoup, check_only: bool = False
    ) -> Optional[Dict[str, str]]:
        """Extract search results."""
        items = {}
        rows = doc.select(".right h2 a")

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

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Scrape article metadata."""
        item = {
            "itemType": "newspaperArticle",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Extract title
        title_tag = doc.find("meta", {"property": "og:title"})
        if title_tag and title_tag.get("content"):
            item["title"] = title_tag["content"]

        # Extract authors from meta tag
        author_meta = doc.find("meta", {"property": "author"})
        if author_meta and author_meta.get("content"):
            author_list = author_meta["content"].split(",")
            for author in author_list:
                author = author.strip()
                # Exclude generic names like "Redakce" (editorial)
                if author and author != "Redakce":
                    names = author.split()
                    if len(names) >= 2:
                        item["creators"].append(
                            {
                                "firstName": " ".join(names[:-1]),
                                "lastName": names[-1],
                                "creatorType": "author",
                            }
                        )
                    else:
                        item["creators"].append(
                            {
                                "lastName": author,
                                "creatorType": "author",
                                "fieldMode": True,
                            }
                        )

        # Extract date
        date_tag = doc.find("meta", {"property": "article:published_time"})
        if date_tag and date_tag.get("content"):
            item["date"] = date_tag["content"]

        # Extract abstract
        abstract_tag = doc.find("meta", {"property": "og:description"})
        if abstract_tag and abstract_tag.get("content"):
            item["abstractNote"] = abstract_tag["content"]

        # Extract publication title
        site_name_tag = doc.find("meta", {"property": "og:site_name"})
        if site_name_tag and site_name_tag.get("content"):
            item["publicationTitle"] = site_name_tag["content"]

        # Extract language
        item["language"] = "cs"  # Czech

        # Set library catalog
        hostname = url.split("/")[2] if "/" in url else "denik.cz"
        item["libraryCatalog"] = hostname

        # Extract keywords as tags
        keywords_tag = doc.find("meta", {"name": "keywords"})
        if keywords_tag and keywords_tag.get("content"):
            keywords = keywords_tag["content"].split(",")
            for kw in keywords:
                kw = kw.strip()
                if kw:
                    item["tags"].append({"tag": kw})

        # Add snapshot attachment
        item["attachments"].append(
            {"title": "Snapshot", "mimeType": "text/html", "url": url}
        )

        return item
