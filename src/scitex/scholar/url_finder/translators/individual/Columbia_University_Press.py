"""
Columbia University Press Translator

Translates Columbia University Press book pages to Zotero format.

Metadata:
    translatorID: a75e0594-a9e8-466e-9ce8-c10560ea59fd
    label: Columbia University Press
    creator: Philipp Zumstein
    target: ^https?://(www\\.)?cup\\.columbia\\.edu/
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2016-09-10 11:35:07
"""

from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
import re


class ColumbiaUniversityPressTranslator:
    """Translator for Columbia University Press books."""

    METADATA = {
        "translatorID": "a75e0594-a9e8-466e-9ce8-c10560ea59fd",
        "label": "Columbia University Press",
        "creator": "Philipp Zumstein",
        "target": r"^https?://(www\.)?cup\.columbia\.edu/",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2016-09-10 11:35:07",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """
        Detect if the page is a Columbia University Press book or search page.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            'book' if single book, 'multiple' if search results, empty string otherwise
        """
        if "/book/" in url:
            return "book"
        elif self._get_search_results(doc, True):
            return "multiple"
        return ""

    def _get_search_results(
        self, doc: BeautifulSoup, check_only: bool = False
    ) -> Dict[str, str]:
        """
        Get search results from the page.

        Args:
            doc: BeautifulSoup parsed document
            check_only: If True, return as soon as first result is found

        Returns:
            Dictionary mapping URLs to titles, or False if none found
        """
        items = {}
        found = False
        rows = doc.select("div.search-list h2 > a")

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
        """
        Extract book data from Columbia University Press page.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            List of dictionaries containing book metadata
        """
        if self.detect_web(doc, url) == "multiple":
            # In a real implementation, this would allow user selection
            # For now, we'll just return empty list
            return []
        else:
            return [self.scrape(doc, url)]

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Scrape book data from the document.

        Args:
            doc: BeautifulSoup parsed document
            url: URL of the page

        Returns:
            Dictionary containing book metadata
        """
        item = {
            "itemType": "book",
            "creators": [],
            "tags": [],
            "attachments": [],
            "notes": [],
            "seeAlso": [],
        }

        # Extract title
        title_elem = doc.select_one("div.book-header h1.title")
        if title_elem:
            item["title"] = title_elem.get_text(strip=True)

        # Extract book details
        book_details = doc.select(
            "div.book-header.pc-only p[class], div.book-details p[class]"
        )

        for detail in book_details:
            class_name = " ".join(detail.get("class", []))
            content = detail.get_text(strip=True)

            if class_name == "subtitle":
                if "title" in item:
                    item["title"] = f"{item['title']}: {content}"

            elif class_name == "author":
                # Parse creators (authors, editors, translators)
                creator_string = content

                # Find positions of different creator types
                pos_editors = creator_string.find("Edited")
                if pos_editors == -1:
                    pos_editors = len(creator_string)

                pos_translators = creator_string.find("Translated")
                if pos_translators == -1:
                    pos_translators = len(creator_string)

                # Extract authors
                authors_str = creator_string[: min(pos_editors, pos_translators)]
                authors = re.split(r"\band\b|,", authors_str)
                for author in authors:
                    author = author.strip()
                    if author:
                        item["creators"].append(self._clean_author(author, "author"))

                # Extract editors
                if pos_editors < len(creator_string):
                    editors_str = creator_string[pos_editors:pos_translators]
                    editors_str = re.sub(r"Edited\s*(by)?", "", editors_str)
                    editors = re.split(r"\band\b|,", editors_str)
                    for editor in editors:
                        editor = editor.strip()
                        if editor:
                            item["creators"].append(
                                self._clean_author(editor, "editor")
                            )

                # Extract translators
                if pos_translators < len(creator_string):
                    translators_str = creator_string[pos_translators:]
                    translators_str = re.sub(r"Translated\s*(by)?", "", translators_str)
                    translators = re.split(r"\band\b|,", translators_str)
                    for translator in translators:
                        translator = translator.strip()
                        if translator:
                            item["creators"].append(
                                self._clean_author(translator, "translator")
                            )

            elif class_name == "pubdate":
                # Convert date to ISO format
                item["date"] = content

            elif class_name == "publisher":
                item["publisher"] = content

            elif class_name == "isbn":
                item["ISBN"] = content

            elif class_name == "pages":
                item["pages"] = content

        # If there is no publisher field, assume it's published by CUP
        if "publisher" not in item:
            item["publisher"] = "Columbia University Press"

        # Extract abstract
        abstract_elem = doc.select_one("div.sp__the-description")
        if abstract_elem:
            item["abstractNote"] = abstract_elem.get_text(strip=True)

        return item

    def _clean_author(self, name: str, creator_type: str) -> Dict[str, Any]:
        """
        Parse author name into first and last name.

        Args:
            name: Full author name
            creator_type: Type of creator (author, editor, translator)

        Returns:
            Dictionary with firstName, lastName, and creatorType
        """
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
