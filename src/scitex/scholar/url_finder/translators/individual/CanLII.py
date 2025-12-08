"""
CanLII Translator

Translates Canadian legal cases from CanLII to Zotero format.

Metadata:
    translatorID: 84799379-7bc5-4e55-9817-baf297d129fe
    label: CanLII
    creator: Sebastian Karcher
    target: ^https?://(www\.)?canlii\.org/(en|fr)/
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2023-03-15 05:20:22
"""

from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
import re


class CanLIITranslator:
    """Translator for CanLII legal cases."""

    METADATA = {
        "translatorID": "84799379-7bc5-4e55-9817-baf297d129fe",
        "label": "CanLII",
        "creator": "Sebastian Karcher",
        "target": r"^https?://(www\.)?canlii\.org/(en|fr)/",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2023-03-15 05:20:22",
    }

    CAN_LII_REGEXP = re.compile(
        r"https?://(?:www\.)?canlii\.org[^/]*/(?:en|fr)/[^/]+/[^/]+/doc/.+"
    )

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if page is a CanLII case or search results."""
        if self.CAN_LII_REGEXP.match(url):
            return "case"

        # Check for links to cases
        for link in doc.find_all("a", href=True):
            if self.CAN_LII_REGEXP.match(link["href"]):
                return "multiple"

        return ""

    def do_web(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract case data."""
        if self.detect_web(doc, url) == "case":
            return self.scrape(doc, url)
        return {}

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Scrape case data from the document.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            Dictionary containing case metadata
        """
        item = {
            "itemType": "case",
            "creators": [],
            "tags": [],
            "attachments": [],
            "notes": [],
        }

        # Get citation text
        citation_div = doc.select_one(".documentMeta-citation + div")
        if citation_div:
            voliss = citation_div.get_text().strip()

            # Parse citation
            # e.g. Reference re Secession of Quebec, 1998 CanLII 793 (SCC), [1998] 2 SCR 217
            citation_parts = voliss.split(",")
            if citation_parts:
                item["caseName"] = citation_parts[0].strip()

            # Extract reporter information
            reporter_regex = re.compile(r"\[(\d{4})\]\s+(\d+)\s+([A-Z]+)\s+(\d+)")
            reporter_match = reporter_regex.search(voliss)
            if reporter_match:
                item["reporterVolume"] = reporter_match.group(2)
                item["reporter"] = reporter_match.group(3)
                item["firstPage"] = reporter_match.group(4)

        # Court
        court_elem = doc.select_one('#breadcrumbs *[itemprop="name"]:nth-of-type(3)')
        if court_elem:
            item["court"] = court_elem.get_text().strip()

        # Date decided
        date_div = doc.find("div", string=re.compile(r"Date"))
        if date_div:
            date_sibling = date_div.find_next_sibling("div")
            if date_sibling:
                item["dateDecided"] = date_sibling.get_text().strip()

        # Docket number
        docket_div = doc.find(
            "div", string=re.compile(r"File number|Numéro de dossier")
        )
        if docket_div:
            docket_sibling = docket_div.find_next_sibling("div")
            if docket_sibling:
                item["docketNumber"] = docket_sibling.get_text().strip()

        # Other citations
        other_citations_div = doc.find(
            "div", string=re.compile(r"Other citations|Autres citations")
        )
        if other_citations_div:
            citations_sibling = other_citations_div.find_next_sibling("div")
            if citations_sibling:
                other_citations = citations_sibling.get_text().strip()
                item["notes"].append({"note": f"Other Citations: {other_citations}"})

        # Short URL
        short_url_elem = doc.select_one(".documentStaticUrl")
        if short_url_elem:
            item["url"] = short_url_elem.get_text().strip()

        # PDF attachment
        pdf_url = re.sub(r"\.html(?:[?#].*)?", ".pdf", url)
        item["attachments"].append(
            {
                "url": pdf_url,
                "title": "CanLII Full Text PDF",
                "mimeType": "application/pdf",
            }
        )

        # Snapshot attachment
        item["attachments"].append(
            {"title": "CanLII Snapshot", "mimeType": "text/html", "url": url}
        )

        return item
