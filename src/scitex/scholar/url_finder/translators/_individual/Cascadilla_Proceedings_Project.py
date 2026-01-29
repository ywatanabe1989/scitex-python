"""
Cascadilla Proceedings Project Translator

Translates conference papers from lingref.com to Zotero format.

Metadata:
    translatorID: cbed2134-f963-43a0-a8ad-9813e94de9a7
    label: Cascadilla Proceedings Project
    creator: Abe Jellinek
    target: ^https?://(www\.)?lingref\.com/
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2021-10-29 00:35:04
"""

from typing import Dict, Any
from bs4 import BeautifulSoup


class CascadillaProceedingsProjectTranslator:
    """Translator for Cascadilla Proceedings Project."""

    METADATA = {
        "translatorID": "cbed2134-f963-43a0-a8ad-9813e94de9a7",
        "label": "Cascadilla Proceedings Project",
        "creator": "Abe Jellinek",
        "target": r"^https?://(www\.)?lingref\.com/",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2021-10-29 00:35:04",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if page is a conference paper or search results."""
        if doc.select_one(".bookinfo"):
            return "conferencePaper"

        # Check for article links
        links = doc.select(
            '.pagecontent > .contentpadding > a[href*="abstract"][href$=".html"]'
        )
        if links:
            return "multiple"

        return ""

    def do_web(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract conference paper data."""
        return self.scrape(doc, url)

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Scrape conference paper data using embedded metadata.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            Dictionary containing conference paper metadata
        """
        item = {
            "itemType": "conferencePaper",
            "creators": [],
            "tags": [],
            "attachments": [],
            "url": url,
        }

        # Extract embedded metadata
        self._extract_meta_tags(doc, item)

        # Fill book info
        self._fill_book_info(doc, item)

        # Add PDF attachment if available
        pdf_link = doc.select_one('a[href$=".pdf"]')
        if pdf_link and pdf_link.get("href"):
            item["attachments"].append(
                {
                    "url": pdf_link["href"]
                    if pdf_link["href"].startswith("http")
                    else "https://www.lingref.com" + pdf_link["href"],
                    "title": "Full Text PDF",
                    "mimeType": "application/pdf",
                }
            )

        # Add snapshot
        item["attachments"].append(
            {"title": "Snapshot", "mimeType": "text/html", "url": url}
        )

        return item

    def _extract_meta_tags(self, doc: BeautifulSoup, item: Dict[str, Any]):
        """Extract metadata from meta tags."""
        # Title
        title_meta = doc.find("meta", attrs={"name": "citation_title"})
        if title_meta:
            item["title"] = title_meta.get("content", "")

        # Authors
        author_metas = doc.find_all("meta", attrs={"name": "citation_author"})
        for author_meta in author_metas:
            author_name = author_meta.get("content", "")
            if author_name:
                item["creators"].append(self._clean_author(author_name, "author"))

        # Date
        date_meta = doc.find("meta", attrs={"name": "citation_publication_date"})
        if date_meta:
            item["date"] = date_meta.get("content", "")

        # ISBN
        isbn_meta = doc.find("meta", attrs={"name": "citation_isbn"})
        if isbn_meta:
            item["ISBN"] = isbn_meta.get("content", "")

        # Publisher
        publisher_meta = doc.find("meta", attrs={"name": "citation_publisher"})
        if publisher_meta:
            item["publisher"] = publisher_meta.get("content", "")

        # Conference name
        conf_meta = doc.find("meta", attrs={"name": "citation_conference_title"})
        if conf_meta:
            item["conferenceName"] = conf_meta.get("content", "")

        # Pages
        first_page_meta = doc.find("meta", attrs={"name": "citation_firstpage"})
        last_page_meta = doc.find("meta", attrs={"name": "citation_lastpage"})
        if first_page_meta and last_page_meta:
            first_page = first_page_meta.get("content", "")
            last_page = last_page_meta.get("content", "")
            if first_page and last_page:
                item["pages"] = f"{first_page}-{last_page}"

    def _fill_book_info(self, doc: BeautifulSoup, item: Dict[str, Any]):
        """Fill proceedings information."""
        bookinfo = doc.select_one(".bookinfo > .contentpadding")
        if not bookinfo:
            bookinfo = doc.select_one(".pagecontent > .contentpadding")

        if bookinfo:
            # Proceedings title
            proceedings_title = bookinfo.select_one(".largerfont")
            if proceedings_title:
                item["proceedingsTitle"] = proceedings_title.get_text().strip()

            # Publisher place for Cascadilla
            if item.get("publisher", "").startswith("Cascadilla"):
                item["place"] = "Somerville, MA"

            # Extract editors
            for child in bookinfo.children:
                if hasattr(child, "get_text"):
                    text = child.get_text().strip()
                    if text.startswith("edited by"):
                        editor_text = text.replace("edited by", "").strip()
                        # Split by commas and "and"
                        editors = []
                        for part in (
                            editor_text.replace(", and ", "|||")
                            .replace(" and ", "|||")
                            .split(",")
                        ):
                            for editor in part.split("|||"):
                                editor = editor.strip()
                                if editor:
                                    editors.append(editor)

                        for editor_name in editors:
                            item["creators"].append(
                                self._clean_author(editor_name, "editor")
                            )
                        break

        # Abstract
        abstract_div = doc.select_one(".abstract div")
        if abstract_div:
            item["abstractNote"] = abstract_div.get_text().strip()

    def _clean_author(self, name: str, creator_type: str) -> Dict[str, Any]:
        """Parse author/editor name."""
        name = name.strip()
        parts = name.split()

        if len(parts) >= 2:
            return {
                "firstName": " ".join(parts[:-1]),
                "lastName": parts[-1],
                "creatorType": creator_type,
            }
        else:
            return {"lastName": name, "creatorType": creator_type, "fieldMode": True}
