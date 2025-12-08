"""
Demographic Research Translator

Translates Demographic Research journal articles to Zotero format.

Metadata:
    translatorID: ed317bdd-0416-4762-856d-435004a9f05c
    label: Demographic Research
    creator: Sebatian Karcher
    target: ^https?://www\\.demographic-research\\.org
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2018-05-05 11:04:17
"""

from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
import re


class DemographicResearchTranslator:
    """Translator for Demographic Research journal."""

    METADATA = {
        "translatorID": "ed317bdd-0416-4762-856d-435004a9f05c",
        "label": "Demographic Research",
        "creator": "Sebatian Karcher",
        "target": r"^https?://www\.demographic-research\.org",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2018-05-05 11:04:17",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """
        Detect if the page is an article or search results.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            'journalArticle', 'multiple', or empty string
        """
        if re.search(r"vol\d+/default\.htm|search/search\.aspx\?", url):
            return "multiple"

        # Check for refman link (indicates article page)
        refman_link = doc.find("a", href=re.compile(r"/refman\.plx\?"))
        if refman_link:
            return "journalArticle"

        return ""

    def get_search_results(
        self, doc: BeautifulSoup, check_only: bool = False
    ) -> Optional[Dict[str, str]]:
        """
        Extract search results from the page.

        Args:
            doc: BeautifulSoup parsed document
            check_only: If True, only check if results exist

        Returns:
            Dictionary mapping URLs to titles, or None if none found
        """
        items = {}

        # Search for article links
        titles = doc.select("p.articles_title>a, div.result_title>a")

        for title_elem in titles:
            href = title_elem.get("href")
            title = title_elem.get_text(strip=True)

            if not href or not title:
                continue

            if check_only:
                return {"found": "true"}

            # Fix search results that link to PDFs instead of article pages
            href = re.sub(r"\d+-\d+\.pdf.*", "", href)
            items[href] = title

        return items if items else None

    def do_web(self, doc: BeautifulSoup, url: str) -> Any:
        """
        Main method to extract data from page.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            Item data or list of items for multiple results
        """
        page_type = self.detect_web(doc, url)

        if page_type == "multiple":
            return self.get_search_results(doc, check_only=False)
        else:
            return self.scrape(doc, url)

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Scrape article metadata.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            Dictionary containing article metadata
        """
        item = {
            "itemType": "journalArticle",
            "publicationTitle": "Demographic Research",
            "journalAbbreviation": "Demographic Research",
            "ISSN": "1435-9871",
            "libraryCatalog": "Demographic Research",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Find the refman export link
        refman_link = doc.find("a", href=re.compile(r"/refman\.plx\?"))
        if refman_link:
            refman_url = refman_link.get("href")
            # Note: In full implementation, would fetch RIS data from refman_url
            # and parse it to extract metadata
            # For now, extract basic info from the page

            # Try to extract volume and issue from URL
            match = re.search(r"/vol(\d+)/(\d+)/", url)
            if match:
                item["volume"] = match.group(1)
                item["issue"] = match.group(2)

                # Construct PDF URL
                pdf_url = f"{url}{item['volume']}-{item['issue']}.pdf"
                item["attachments"].append(
                    {
                        "url": pdf_url,
                        "title": "Demographic Research Full Text PDF",
                        "mimeType": "application/pdf",
                    }
                )

        return item
