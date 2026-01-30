"""
DABI Translator

Translates DABI (Datenbank Bibliothekswesen) journal articles to Zotero format.

Metadata:
    translatorID: 5cf8bb21-e350-444f-b9b4-f46d9fab7827
    label: DABI
    creator: Jens Mittelbach
    target: ^https?://dabi\\.ib\\.hu-berlin\\.de/
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2021-09-10 18:58:10
"""

from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
import re


class DABITranslator:
    """Translator for DABI (Datenbank Bibliothekswesen) journal articles."""

    METADATA = {
        "translatorID": "5cf8bb21-e350-444f-b9b4-f46d9fab7827",
        "label": "DABI",
        "creator": "Jens Mittelbach",
        "target": r"^https?://dabi\.ib\.hu-berlin\.de/",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2021-09-10 18:58:10",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if the page is an article or search results."""
        if "/vollanzeige.pl?" in url:
            return "journalArticle"
        elif "/suche.pl?" in url and self._get_search_results(doc, True):
            return "multiple"
        return ""

    def _get_search_results(
        self, doc: BeautifulSoup, check_only: bool = False
    ) -> Dict[str, str]:
        """Get search results from the page."""
        items = {}
        found = False
        rows = doc.find_all("tr")[1:]  # Skip header row

        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 3:
                continue

            link = tds[0].find("a")
            if not link:
                continue

            url = link.get("href")
            author = tds[1].get_text(strip=True)
            title = tds[2].get_text(strip=True).replace("<br>", ". ")

            if author:
                item_text = f"{title} ({re.sub(r';.*', ' et al.', author)})"
            else:
                item_text = title

            if not item_text or not url:
                continue

            if check_only:
                return True
            found = True
            items[url] = item_text

        return items if found else False

    def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Extract article data."""
        if self.detect_web(doc, url) == "journalArticle":
            return [self.scrape(doc, url)]
        return []

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Scrape article data from the document."""
        item = {
            "itemType": "journalArticle",
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Parse table rows for metadata
        data = {}
        rows = doc.find_all("tr")

        for row in rows:
            headers = row.find_all("th")
            contents = row.find_all("td")

            if headers and contents:
                header = headers[0].get_text(strip=True).replace(" ", "")
                content_html = str(contents[0])
                data[header] = content_html.strip()

        # Set URL to fulltext resource if present
        if "URL" in data:
            url_match = re.search(r'<a.*?href="(.*?)"', data["URL"])
            if url_match:
                url_value = url_match.group(1)

                if re.search(r"\.pdf(#.*)?$", url_value):
                    item["attachments"].append(
                        {
                            "url": url_value,
                            "title": "DABI Full Text PDF",
                            "mimeType": "application/pdf",
                        }
                    )
                else:
                    item["url"] = url_value

        # Handle title fields
        if "Titel" not in data and "Untertitel" in data:
            data["Titel"] = data["Untertitel"]
            del data["Untertitel"]

        if "Titel" in data:
            item["title"] = data["Titel"].replace("*", "")

            if "Untertitel" in data:
                subtitle = data["Untertitel"]
                if re.search(r"(\?|!|\.)\W?$", item["title"]):
                    item["title"] += " " + subtitle
                else:
                    item["title"] += ": " + subtitle

        # Parse authors
        if "Autoren" in data:
            authors = data["Autoren"].split("; ")
            for author in authors:
                item["creators"].append(self._clean_author(author, "author"))

        # Parse pages
        if "Anfangsseite" in data and int(data.get("Anfangsseite", 0)) > 0:
            start_page = data["Anfangsseite"]
            end_page = data.get("Endseite", "")
            if end_page and int(end_page) > int(start_page):
                item["pages"] = f"{start_page}-{end_page}"
            else:
                item["pages"] = start_page

        # Parse tags
        if "Schlagwörter" in data:
            tags = data["Schlagwörter"].split("; ")
            item["tags"] = [{"tag": tag} for tag in tags if tag]

        # Clean publication title
        if "Zeitschrift" in data:
            item["publicationTitle"] = data["Zeitschrift"].replace(" : ", ": ")

        # Other fields
        if "Jahr" in data:
            item["date"] = data["Jahr"]
        if "Heft" in data:
            item["issue"] = data["Heft"]
        if "Band" in data:
            item["volume"] = data["Band"]
        if "Abstract" in data:
            item["abstractNote"] = data["Abstract"]

        return item

    def _clean_author(self, name: str, creator_type: str) -> Dict[str, Any]:
        """Parse author name into first and last name."""
        name = name.strip()
        parts = name.split(",")

        if len(parts) >= 2:
            # Format: "Last, First"
            return {
                "firstName": parts[1].strip(),
                "lastName": parts[0].strip(),
                "creatorType": creator_type,
            }
        else:
            parts = name.split()
            if len(parts) >= 2:
                return {
                    "firstName": " ".join(parts[:-1]),
                    "lastName": parts[-1],
                    "creatorType": creator_type,
                }
            else:
                return {
                    "lastName": name,
                    "creatorType": creator_type,
                    "fieldMode": True,
                }
