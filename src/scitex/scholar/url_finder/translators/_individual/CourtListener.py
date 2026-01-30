"""
CourtListener Translator

Translates CourtListener case law pages to Zotero format.

Metadata:
    translatorID: 07890a30-866e-452a-ac3e-c19fcb39b597
    label: CourtListener
    creator: Sebastian Karcher
    target: ^https?://www\\.courtlistener\\.com/
    minVersion: 5.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2025-04-29 03:02:00
"""

from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
import re


class CourtListenerTranslator:
    """Translator for CourtListener case law."""

    METADATA = {
        "translatorID": "07890a30-866e-452a-ac3e-c19fcb39b597",
        "label": "CourtListener",
        "creator": "Sebastian Karcher",
        "target": r"^https?://www\.courtlistener\.com/",
        "minVersion": "5.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2025-04-29 03:02:00",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if the page contains a case or search results."""
        if "/opinion/" in url:
            return "case"
        elif self._get_search_results(doc, True):
            return "multiple"
        return ""

    def _get_search_results(
        self, doc: BeautifulSoup, check_only: bool = False
    ) -> Dict[str, str]:
        """Get search results from the page."""
        items = {}
        found = False
        rows = doc.select("article > h3 > a")

        for row in rows:
            href = row.get("href")
            title = row.get_text(strip=True)
            if not href or not title:
                continue
            if check_only:
                return True
            found = True
            items[href] = title

        return items if found else False

    def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Extract case data."""
        if self.detect_web(doc, url) == "case":
            return [self.scrape(doc, url)]
        return []

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Scrape case data from the document."""
        item = {
            "itemType": "case",
            "url": re.sub(r"/\?.*", "", url),  # Remove query parameters
            "creators": [],
            "tags": [],
            "attachments": [],
            "notes": [],
        }

        # Extract citations
        citations = []
        for elem in doc.select("li strong"):
            if "Citations:" in elem.get_text():
                for span in elem.find_next_siblings("span"):
                    citations.append(span.get_text(strip=True))

        # Get main citation from center
        citation_elem = doc.select_one("center b .citation")
        citation = citation_elem.get_text(strip=True) if citation_elem else ""

        # Extract basic fields
        caption_elem = doc.select_one("#caption")
        if caption_elem:
            item["caseName"] = caption_elem.get_text(strip=True)

        court_elem = doc.select_one(".case-court")
        if court_elem:
            item["court"] = court_elem.get_text(strip=True)

        reporter_elem = doc.select_one(".citation .reporter")
        if reporter_elem:
            item["reporter"] = reporter_elem.get_text(strip=True)

        volume_elem = doc.select_one(".citation .volume")
        if volume_elem:
            item["reporterVolume"] = volume_elem.get_text(strip=True)

        page_elem = doc.select_one(".citation .page")
        if page_elem:
            item["firstPage"] = page_elem.get_text(strip=True)

        # If reporter elements aren't tagged, parse from citation
        if not item.get("reporter") and not item.get("reporterVolume"):
            if not citation and citations:
                citation = citations[0]

            if citation:
                cite_match = re.match(
                    r"^(\d+)\s((?:[A-Z][a-z]?\.\s?)+(?:[2-3]d)?(?:Supp\.)?)\s(\d{1,4})(,|$)",
                    citation.strip(),
                )
                if cite_match:
                    item["reporterVolume"] = cite_match.group(1)
                    item["reporter"] = cite_match.group(2)
                    item["firstPage"] = cite_match.group(3)
                else:
                    item["history"] = citation

        # Set history from additional citations
        if "history" not in item and len(citations) > 1:
            item["history"] = ", ".join(citations[1:])

        # Extract date
        date_elem = doc.select_one(".case-date-new")
        if date_elem:
            item["dateDecided"] = date_elem.get_text(strip=True)

        # Extract docket number
        docket_li = doc.find("li", string=re.compile("Docket Number:"))
        if docket_li:
            docket_text = "".join(docket_li.find_all(string=True, recursive=False))
            item["docketNumber"] = docket_text.strip()

        # Extract authors (judges)
        for author_link in doc.select(".opinion-section-title > a[href*='/person/']"):
            author_name = author_link.get_text(strip=True)
            item["creators"].append(self._clean_author(author_name, "author"))

        # Add snapshot attachment
        item["attachments"].append(
            {"title": "Full Text", "mimeType": "text/html", "url": url}
        )

        return item

    def _clean_author(self, name: str, creator_type: str) -> Dict[str, Any]:
        """Parse author name into first and last name."""
        name = name.strip()
        parts = name.split()

        if len(parts) >= 2:
            return {
                "firstName": " ".join(parts[:-1]),
                "lastName": parts[-1],
                "creatorType": creator_type,
            }
        else:
            return {"lastName": name, "creatorType": creator_type, "fieldMode": False}
