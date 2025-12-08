"""
Biblioteca Nacional de Maestros Translator

Translates records from Biblioteca Nacional de Maestros library catalog.

Metadata:
    translatorID: b383df35-15e7-43ee-acd9-88fd62669083
    label: Biblioteca Nacional de Maestros
    creator: Sebastian Karcher
    target: ^https?://www\.bnm\.me\.gov\.ar/catalogo
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2020-06-22 00:23:44
"""

from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import re


class BibliotecaNacionalDeMaestrosTranslator:
    """Translator for Biblioteca Nacional de Maestros catalog."""

    METADATA = {
        "translatorID": "b383df35-15e7-43ee-acd9-88fd62669083",
        "label": "Biblioteca Nacional de Maestros",
        "creator": "Sebastian Karcher",
        "target": r"^https?://www\.bnm\.me\.gov\.ar/catalogo",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2020-06-22 00:23:44",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """
        Detect if the page is a single record or multiple records.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            'book' for single record, 'multiple' for search results, empty string otherwise
        """
        if re.search(r"/Record/\d+", url):
            return "book"
        elif self._get_search_results(doc, check_only=True):
            return "multiple"
        return ""

    def _get_search_results(
        self, doc: BeautifulSoup, check_only: bool = False
    ) -> Dict[str, str]:
        """
        Get search results from the page.

        Args:
            doc: BeautifulSoup parsed document
            check_only: If True, return immediately on first result

        Returns:
            Dictionary mapping URLs to titles
        """
        items = {}
        rows = doc.select(
            '.result div[class*="resultItemLine"]>a.title[href*="/Record/"]'
        )

        for row in rows:
            href = row.get("href")
            title = row.get_text(strip=True)
            if not href or not title:
                continue
            if check_only:
                return {"found": True}
            items[href] = title

        return items if items else {}

    def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """
        Extract data from the page.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            List of item dictionaries
        """
        page_type = self.detect_web(doc, url)

        if page_type == "multiple":
            # For multiple items, return list of URLs for processing
            items = self._get_search_results(doc, check_only=False)
            return [{"url": url} for url in items.keys()]
        else:
            # For single item, construct MARC URL and return
            marc_url = self._construct_marc_url(url)
            return [{"_marc_url": marc_url, "url": url}]

    def _construct_marc_url(self, url: str) -> str:
        """
        Construct MARC XML export URL from record URL.

        Args:
            url: Record URL

        Returns:
            MARC XML export URL
        """
        # Remove panels like /Details or /Holdings
        url = re.sub(r"/(Details|Holdings)([#?].*)?", "", url)
        return url + "/Export?style=MARCXML"

    def scrape(self, marc_text: str, url: str) -> Dict[str, Any]:
        """
        Scrape item data from MARC XML.

        Args:
            marc_text: MARC XML text
            url: Original URL

        Returns:
            Dictionary containing item metadata
        """
        # This would typically call the MARC XML translator
        # For now, return a minimal structure
        item = {
            "itemType": "book",
            "libraryCatalog": "Biblioteca Nacional de Maestros",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        return item
