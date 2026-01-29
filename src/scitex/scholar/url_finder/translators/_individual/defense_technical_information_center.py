"""
Defense Technical Information Center Translator

Translates DTIC reports to Zotero format.

Metadata:
    translatorID: 99be9976-2ff9-40df-96e8-82edfa79d9f3
    label: Defense Technical Information Center
    creator: Matt Burton
    target: ^https?://oai\\.dtic\\.mil/oai/
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2013-01-09 15:36:32
"""

from typing import Dict, Any, Optional
from bs4 import BeautifulSoup


class DefenseTechnicalInformationCenterTranslator:
    """Translator for Defense Technical Information Center (DTIC)."""

    METADATA = {
        "translatorID": "99be9976-2ff9-40df-96e8-82edfa79d9f3",
        "label": "Defense Technical Information Center",
        "creator": "Matt Burton",
        "target": r"^https?://oai\.dtic\.mil/oai/",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2013-01-09 15:36:32",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if page is DTIC report or index."""
        if "DTIC OAI Index for" in doc.title.string if doc.title else "":
            return "multiple"
        elif "verb=getRecord" in url:
            return "report"
        return ""

    def get_search_results(
        self, doc: BeautifulSoup, check_only: bool = False
    ) -> Optional[Dict[str, str]]:
        """Extract search results."""
        items = {}
        links = doc.select('a[href*="verb=getRecord"]')

        for link in links:
            href = link.get("href", "")
            # Get preceding text as title
            title_text = link.find_previous(text=True)
            if title_text:
                title = title_text.strip()
            else:
                continue

            if not href or not title:
                continue

            if check_only:
                return {"found": "true"}

            # Replace HTML metadata with Dublin Core
            href = href.replace("&metadataPrefix=html", "&metadataPrefix=oai_dc")
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
        """Scrape report metadata."""
        item = {
            "itemType": "report",
            "libraryCatalog": "Defense Technical Information Center",
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Find PDF URL
        pdf_link = doc.find("a", href=lambda x: x and "doc=GetTRDoc.pdf" in x)
        if pdf_link:
            pdf_url = pdf_link.get("href")
            item["attachments"].append(
                {
                    "url": pdf_url,
                    "title": "DTIC Full Text PDF",
                    "mimeType": "application/pdf",
                }
            )

        # Add snapshot
        item["attachments"].append(
            {"title": "DTIC Snapshot", "mimeType": "text/html", "url": url}
        )

        # Note: Full implementation would fetch OAI-DC XML and parse it
        # For now, returning basic structure

        return item
