"""
CCfr (BnF) Translator

Translates CCfr (Catalogue Collectif de France) - BnF records to Zotero format.

Metadata:
    translatorID: 899d10f5-3f35-40e6-8dfb-f8ee2dfb1849
    label: CCfr (BnF)
    creator: Sylvain Machefert, Aurimas Vinckevicius
    target: ^https?://ccfr\.bnf\.fr/portailccfr/.*\b(action=search|menu=menu_view_grappage|search\.jsp)\b
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: g
    lastUpdated: 2014-09-18 14:08:05
"""

from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
import re


class CCfrBnFTranslator:
    """Translator for CCfr (Catalogue Collectif de France) - BnF."""

    METADATA = {
        "translatorID": "899d10f5-3f35-40e6-8dfb-f8ee2dfb1849",
        "label": "CCfr (BnF)",
        "creator": "Sylvain Machefert, Aurimas Vinckevicius",
        "target": r"^https?://ccfr\.bnf\.fr/portailccfr/.*\b(action=search|menu=menu_view_grappage|search\.jsp)\b",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "g",
        "lastUpdated": "2014-09-18 14:08:05",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """
        Detect if the page is a CCfr record or search results.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            Item type detected or 'multiple' for search results
        """
        if self.get_search_results(doc, True):
            return "multiple"
        if "menu=menu_view_grappage" in url:
            return self.ccfr_type_doc(doc)
        return ""

    def get_search_results(
        self, doc: BeautifulSoup, check_only: bool = False
    ) -> Dict[str, str]:
        """
        Get search results from the page.

        Args:
            doc: BeautifulSoup parsed document
            check_only: If True, return early on first match

        Returns:
            Dictionary mapping record IDs to titles
        """
        if not doc.select_one("#sourceResultsPane"):
            return {}

        items = {}
        rows = doc.select('form[name="frmSearchResult"] table')

        for row in rows:
            checkbox = row.select_one('td.ident-check input[type="checkbox"]')
            title_elem = row.select_one('td.Ident span a[title="Voir la Notice"]')

            if checkbox and title_elem:
                record_id = checkbox.get("value")
                title = title_elem.get_text(strip=True)
                if record_id and title:
                    if check_only:
                        return {"found": "true"}
                    items[record_id] = title

        return items

    def ccfr_type_doc(self, doc: BeautifulSoup) -> str:
        """
        Determine the document type from CCfr record.

        Args:
            doc: BeautifulSoup parsed document

        Returns:
            Zotero item type
        """
        if not doc.select_one("div.notice-contenu"):
            return ""
        if not doc.select("div#vueCourante table tbody tr"):
            return ""

        rows = doc.select("div#vueCourante table tbody tr")
        for row in rows:
            label_elem = row.select_one("th.view-field-label-ccfr")
            if not label_elem:
                continue

            label = label_elem.get_text(strip=True)
            if label == "Type document":
                value_elem = row.select_one("td.view-field-value-ccfr")
                if value_elem:
                    value_text = value_elem.get_text(strip=True)

                    type_mapping = {
                        "Livre": "book",
                        "Document électronique": "book",
                        "Document sonore": "audioRecording",
                        "Images Animées": "film",
                        "Carte": "map",
                    }

                    return type_mapping.get(value_text, "book")

        return ""

    def do_web(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract data from CCfr page.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            Dictionary containing metadata or search results
        """
        if self.detect_web(doc, url) == "multiple":
            return self.get_search_results(doc)
        return self.scrape(doc, url)

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Scrape record data from the document.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            Dictionary containing record metadata
        """
        # Note: The original translator uses MARC format import
        # This is a simplified version that extracts basic metadata

        item_type = self.ccfr_type_doc(doc)

        item = {
            "itemType": item_type if item_type else "book",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Extract basic fields from the display
        rows = doc.select("div#vueCourante table tbody tr")
        for row in rows:
            label_elem = row.select_one("th.view-field-label-ccfr")
            value_elem = row.select_one("td.view-field-value-ccfr")

            if not label_elem or not value_elem:
                continue

            label = label_elem.get_text(strip=True)
            value = value_elem.get_text(strip=True)

            # Map common fields
            if label == "Titre":
                item["title"] = value
            elif label == "Auteur":
                # Parse author
                author = self._clean_author(value)
                if author:
                    item["creators"].append(author)
            elif label == "Date de publication":
                item["date"] = value
            elif label == "Editeur":
                item["publisher"] = value

        # Add snapshot attachment
        item["attachments"].append(
            {"title": "Snapshot", "mimeType": "text/html", "url": url}
        )

        return item

    def _clean_author(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Parse author name into first and last name.

        Args:
            name: Full author name

        Returns:
            Dictionary with firstName and lastName, or None
        """
        name = name.strip()
        if not name:
            return None

        parts = name.split()

        if len(parts) >= 2:
            return {
                "firstName": " ".join(parts[:-1]),
                "lastName": parts[-1],
                "creatorType": "author",
            }
        else:
            return {"lastName": name, "creatorType": "author", "fieldMode": True}
