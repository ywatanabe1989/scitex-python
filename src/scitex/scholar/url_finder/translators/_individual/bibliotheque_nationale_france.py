"""
Bibliotheque nationale de France Translator

Translates records from BnF catalog using UNIMARC.

Metadata:
    translatorID: 47533cd7-ccaa-47a7-81bb-71c45e68a74d
    label: Bibliothèque nationale de France
    creator: Florian Ziche, Sylvain Machefert
    target: ^https?://[^/]*catalogue\.bnf\.fr
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2024-01-09 03:40:58
"""

from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import re


class BibliothequeNationaleFranceTranslator:
    """Translator for BnF catalog using UNIMARC format."""

    METADATA = {
        "translatorID": "47533cd7-ccaa-47a7-81bb-71c45e68a74d",
        "label": "Bibliothèque nationale de France",
        "creator": "Florian Ziche, Sylvain Machefert",
        "target": r"^https?://[^/]*catalogue\.bnf\.fr",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2024-01-09 03:40:58",
    }

    TYPE_MAPPING = {
        "moving image": "film",
        "text": "book",
        "printed text": "book",
        "electronic resource": "book",
        "score": "book",
        "sound": "audioRecording",
        "sound recording": "audioRecording",
        "cartographic resource": "map",
        "still image": "artwork",
        "kit": "document",
        "modern manuscript or archive": "manuscript",
        "coin or medal": "document",
        "physical object": "document",
        "three dimensional object": "document",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect page type."""
        result_regexp = re.compile(r"ark:/12148/cb[0-9]+", re.IGNORECASE)

        if result_regexp.search(url):
            # Single result - check item type
            dc_type_tag = doc.find("meta", {"name": "DC.type", "lang": "eng"})
            if dc_type_tag:
                item_type = dc_type_tag.get("content", "")
                return self.TYPE_MAPPING.get(item_type, "document")
            return "document"
        elif self._get_results_table(doc):
            return "multiple"

        return ""

    def _get_results_table(self, doc: BeautifulSoup) -> bool:
        """Check if results table exists."""
        results = doc.select("div.liste-notices")
        return len(results) > 0

    def _get_selected_items(self, doc: BeautifulSoup) -> Dict[str, str]:
        """Get selected items from search results."""
        items = {}
        rows = doc.select("div.liste-notices div.notice-item div.notice-contenu")

        for row in rows:
            title_link = row.select_one("div.notice-synthese a")
            if not title_link:
                continue

            href = title_link.get("href")
            title = title_link.get_text(strip=True)

            # Add document year if available
            year_elem = row.select_one("span.notice-ordre")
            if year_elem:
                year = year_elem.get_text(strip=True)
                if len(year) == 6:
                    title += " / " + year

            if href and title:
                items[href] = title

        return items

    def _reform_url(self, url: str) -> str:
        """Convert URL to UNIMARC format."""
        url = re.sub(r"(^.*\/ark:\/12148\/cb[0-9]+[a-z]*)(.*$)", r"\1.unimarc", url)
        return url

    def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Extract data from the page."""
        page_type = self.detect_web(doc, url)

        if page_type == "multiple":
            items = self._get_selected_items(doc)
            return [{"url": self._reform_url(u)} for u in items.keys()]
        else:
            return [{"url": self._reform_url(url), "_unimarc": True}]

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Scrape UNIMARC data from the document.

        This processes UNIMARC formatted catalog data.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            Dictionary containing item metadata
        """
        item = {
            "itemType": self.detect_web(doc, url) or "document",
            "libraryCatalog": "BnF Catalogue général (http://catalogue.bnf.fr)",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Extract title from meta tags
        title_tag = doc.find("meta", {"property": "og:title"})
        if title_tag:
            item["title"] = title_tag.get("content", "")

        # Add attachment for catalog record
        ark_match = re.search(r"ark:/12148/cb[0-9]+[a-z]*", url)
        if ark_match:
            permalink = "http://catalogue.bnf.fr/" + ark_match.group()
            item["attachments"].append(
                {
                    "title": "Lien vers la notice du catalogue",
                    "url": permalink,
                    "mimeType": "text/html",
                    "snapshot": False,
                }
            )

        # Check for Gallica URL (digital version)
        gallica_link = doc.find("a", href=re.compile(r"gallica\.bnf\.fr"))
        if gallica_link:
            item["url"] = gallica_link.get("href")

        return item
