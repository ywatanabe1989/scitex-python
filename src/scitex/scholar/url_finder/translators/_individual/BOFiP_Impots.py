"""
BOFiP-Impôts Translator

Translates French tax bulletin (BOFiP-Impôts) pages to Zotero format.

Metadata:
    translatorID: 7d03e952-04ad-4d1d-845a-50b9eb545b10
    label: BOFiP-Impôts
    creator: Guillaume Adreani
    target: ^https?://bofip\\.impots\\.gouv\\.fr/bofip/
    minVersion: 2.1.9
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsib
    lastUpdated: 2017-01-01 14:53:42
"""

from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import re


class BOFiPImpotsTranslator:
    """Translator for BOFiP-Impôts French tax bulletin."""

    METADATA = {
        "translatorID": "7d03e952-04ad-4d1d-845a-50b9eb545b10",
        "label": "BOFiP-Impôts",
        "creator": "Guillaume Adreani",
        "target": r"^https?://bofip\.impots\.gouv\.fr/bofip/",
        "minVersion": "2.1.9",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsib",
        "lastUpdated": "2017-01-01 14:53:42",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """
        Detect the type of BOFiP document.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            'journalArticle', 'newspaperArticle', 'multiple', or empty string
        """
        # Check if it's a document screen (official document)
        if doc.find(id="docScreenName"):
            return "journalArticle"
        # Check if it's a news/actualités screen
        elif doc.find(id="actuScreenName"):
            return "newspaperArticle"
        # Check if it's a search/list page
        elif re.search(r"http://bofip\.impots\.gouv\.fr/bofip/(?:ext|simple)/.+", url):
            return "multiple"
        return ""

    def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """
        Extract document data from BOFiP page.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            List of dictionaries containing document metadata
        """
        item_type = self.detect_web(doc, url)

        if item_type == "multiple":
            results = self._get_search_results(doc)
            return [{"itemType": "multiple", "urls": results}]
        else:
            return [self.scrape(doc, url, item_type)]

    def scrape(self, doc: BeautifulSoup, url: str, item_type: str) -> Dict[str, Any]:
        """
        Scrape document data from the page.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page
            item_type: Type of item detected

        Returns:
            Dictionary containing document metadata
        """
        item = {
            "itemType": item_type,
            "publicationTitle": "Bulletin Officiel des Finances Publiques-Impôts",
            "rights": "public",
            "ISSN": "2262-1954",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Extract title from head/title
        title_elem = doc.find("title")
        if title_elem:
            item["title"] = title_elem.get_text().strip()

        if item_type == "journalArticle":
            # Extract creators - remove copyright symbol
            creator_elems = doc.select("td.texteGris")
            for elem in creator_elems:
                creator_text = elem.get_text().strip()
                creator_text = creator_text.replace("©", "").strip()
                if creator_text:
                    item["creators"].append(
                        {
                            "lastName": creator_text,
                            "creatorType": "author",
                            "fieldMode": True,
                        }
                    )

            # Extract call number
            callnum_elem = doc.select_one(
                '#contentResult > div > div[style="float:left"]'
            )
            if callnum_elem:
                item["callNumber"] = callnum_elem.get_text().strip()

            # Extract date and reformat from DD/MM/YYYY to YYYY-MM-DD
            date_elem = doc.select_one("#date_publication")
            if date_elem:
                date_text = date_elem.get_text().strip()
                # Convert DD/MM/YYYY to YYYY-MM-DD
                match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_text)
                if match:
                    day, month, year = match.groups()
                    item["date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

            # Extract permalink URL
            permalink_elem = doc.select_one("#permalienDoc")
            if permalink_elem and permalink_elem.get("value"):
                item["url"] = permalink_elem["value"]

        elif item_type == "newspaperArticle":
            # For actualités (news articles)
            # Extract date from title
            title_elem = doc.select_one("#titre-actualite")
            if title_elem:
                title_text = title_elem.get_text()
                # Remove newlines and tabs
                title_text = re.sub(r"\n\t\t\t", "", title_text)
                # Extract date before colon
                date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", title_text)
                if date_match:
                    date_str = date_match.group(1)
                    # Convert DD/MM/YYYY to YYYY-MM-DD
                    match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_str)
                    if match:
                        day, month, year = match.groups()
                        item["date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # Extract PDF attachment
        pdf_link = doc.select_one(
            'a img[src="/resources/skins/pergam/images/exporter.gif"]'
        )
        if pdf_link:
            parent_link = pdf_link.find_parent("a")
            if parent_link and parent_link.get("href"):
                item["attachments"].append(
                    {
                        "url": parent_link["href"],
                        "title": "BOFiP PDF",
                        "mimeType": "application/pdf",
                    }
                )

        return item

    def _get_search_results(self, doc: BeautifulSoup) -> Optional[Dict[str, str]]:
        """
        Get search results from a results page.

        Args:
            doc: BeautifulSoup parsed document

        Returns:
            Dictionary mapping URLs to titles
        """
        items = {}
        rows = doc.select("#contentResult table tbody tr td:nth-of-type(2) a")

        for row in rows:
            href = row.get("href")
            title = row.get_text().strip()
            if href and title:
                items[href] = title

        return items if items else None
