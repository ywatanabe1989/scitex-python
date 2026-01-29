"""
Bibliotheque et Archives Nationales du Quebec Translator

Translates records from BAnQ main catalog.

Metadata:
    translatorID: 59cce211-9d77-4cdd-876d-6229ea20367f
    label: Bibliothèque et Archives Nationales du Québec
    creator: Adam Crymble
    target: ^https?://catalogue\.banq\.qc\.ca/
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsibv
    lastUpdated: 2015-06-29 17:02:02
"""

from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import re


class BibliothequeArchivesNationalesQuebecTranslator:
    """Translator for BAnQ main catalog."""

    METADATA = {
        "translatorID": "59cce211-9d77-4cdd-876d-6229ea20367f",
        "label": "Bibliothèque et Archives Nationales du Québec",
        "creator": "Adam Crymble",
        "target": r"^https?://catalogue\.banq\.qc\.ca/",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsibv",
        "lastUpdated": "2015-06-29 17:02:02",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect page type based on title and images."""
        title = doc.title.string if doc.title else ""

        if "Search" in title or "Recherche" in title:
            return "multiple"

        # Check for media type indicators
        img = doc.select_one("td:nth-of-type(2) a img")
        if img and img.get("src"):
            src = img["src"]
            if "book" in src or "mmusic" in src or "manalytic" in src:
                return "book"
            elif "msdisc" in src or "msound" in src or "mscas" in src:
                return "audioRecording"
            elif "mvdisc" in src:
                return "videoRecording"
            elif "mpaint" in src:
                return "artwork"
            elif "mserial" in src:
                return "report"
            elif "mcomponent" in src:
                return "newspaperArticle"

        return ""

    def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Extract data from the page."""
        page_type = self.detect_web(doc, url)

        if page_type == "multiple":
            return self._get_search_results(doc)
        else:
            return [self.scrape(doc, url, page_type)]

    def _get_search_results(self, doc: BeautifulSoup) -> List[Dict[str, Any]]:
        """Get search results."""
        items = []
        rows = doc.select("p table tbody tr td b")
        links = doc.select("p table tbody tr td a[href] img")

        for i, row in enumerate(rows[:10]):  # Limit to 10 results
            title = row.get_text(strip=True)
            if links and i < len(links):
                href = links[i].find_parent("a").get("href")
                if href and title:
                    items.append({"url": href, "title": title})

        return items

    def scrape(self, doc: BeautifulSoup, url: str, item_type: str) -> Dict[str, Any]:
        """Scrape item data from the document."""
        # Map item types
        description_field = "pages"
        if item_type in ["audioRecording", "videoRecording"]:
            description_field = "runningTime"
        elif item_type == "artwork":
            description_field = "artworkSize"

        item = {
            "itemType": item_type if item_type else "book",
            "url": url,
            "creators": [],
            "tags": [],
            "attachments": [],
        }

        # Determine language from images
        lang_imgs = doc.select("td:nth-of-type(2) a img")
        if len(lang_imgs) > 1:
            lang_src = lang_imgs[1].get("src", "")
            if "lfre" in lang_src:
                item["language"] = "French"
            elif "leng" in lang_src:
                item["language"] = "English"

        # Extract data fields
        headings = doc.select("td table tbody tr td:nth-of-type(2) b")
        contents = doc.select(
            "td:nth-of-type(2) table tbody tr td table tbody tr td:nth-of-type(4)"
        )

        for i, heading in enumerate(headings):
            if i >= len(contents):
                break

            field_title = heading.get_text().strip().replace(" ", "")
            content = contents[i].get_text(strip=True)

            # Remove tags marked with [*]
            if "[*]" in content and field_title not in ["Publisher", "Éditeur"]:
                content = content[: content.find("[")].strip()

            # Process authors
            if field_title in ["Author", "Auteur"]:
                if "," in content:
                    parts = content.split(",", 1)
                    first_name = parts[1].strip() if len(parts) > 1 else ""
                    last_name = parts[0].strip()
                    item["creators"].append(
                        {
                            "firstName": first_name,
                            "lastName": last_name,
                            "creatorType": "author",
                        }
                    )

            # Process publisher info
            elif field_title in ["Publisher", "Éditeur"]:
                if ":" in content:
                    parts = content.split(":", 1)
                    item["place"] = parts[0].strip().replace("[", "").replace("]", "")

                    if len(parts) > 1:
                        pub_parts = parts[1].split(",")
                        item["publisher"] = (
                            pub_parts[0].strip().replace("[", "").replace("]", "")
                        )

                        # Extract date (last 4 characters)
                        date_text = parts[1].strip().replace("[", "").replace("]", "")
                        if len(date_text) >= 4:
                            item["date"] = date_text[-4:]
                else:
                    item["date"] = content

            # Process subjects/tags
            elif field_title in ["Subjects", "Sujets"]:
                tags = content.split("\n")
                item["tags"] = [{"tag": t.strip()} for t in tags if t.strip()]

            # Process other fields
            elif field_title == "Description":
                item[description_field] = content
            elif field_title == "Title" or field_title == "Titre":
                item["title"] = content
            elif field_title == "ISBN":
                item["ISBN"] = content
            elif field_title in ["Series", "Collection"]:
                item["series"] = content
            elif field_title == "Notes":
                item["abstractNote"] = content
            elif field_title in ["Numbering", "Numérotation"]:
                item["reportNumber"] = content
            elif field_title == "Source":
                item["extra"] = "Source: " + content

        return item
