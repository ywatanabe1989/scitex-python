"""
Cornell LII Translator

Translates Cornell Legal Information Institute case law pages to Zotero format.

Metadata:
    translatorID: 930d49bc-44a1-4c22-9dde-aa6f72fb11e5
    label: Cornell LII
    creator: Bill McKinney
    target: ^https?://www\\.law\\.cornell\\.edu/supct/.+
    minVersion: 3.0
    priority: 100
    inRepository: True
    translatorType: 4
    browserSupport: gcsbv
    lastUpdated: 2013-02-09 12:09:10
"""

from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
import re


class CornellLIITranslator:
    """Translator for Cornell Legal Information Institute case law."""

    METADATA = {
        "translatorID": "930d49bc-44a1-4c22-9dde-aa6f72fb11e5",
        "label": "Cornell LII",
        "creator": "Bill McKinney",
        "target": r"^https?://www\.law\.cornell\.edu/supct/.+",
        "minVersion": "3.0",
        "priority": 100,
        "inRepository": True,
        "translatorType": 4,
        "browserSupport": "gcsbv",
        "lastUpdated": "2013-02-09 12:09:10",
    }

    def detect_web(self, doc: BeautifulSoup, url: str) -> str:
        """Detect if the page contains case law."""
        lii_regexp = re.compile(r"/supct/html/.+")
        if lii_regexp.search(url):
            return "case"
        else:
            # Check for links to cases
            for a in doc.find_all("a", href=True):
                if lii_regexp.search(a["href"]):
                    return "multiple"
        return ""

    def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Extract case data."""
        if self.detect_web(doc, url) == "case":
            return [self.scrape(doc, url)]
        return []

    def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Scrape case data from the document."""
        item = {
            "itemType": "case",
            "url": url,
            "language": "en-us",
            "court": "U.S. Supreme Court",
            "reporter": "U.S.",
            "creators": [],
            "tags": [],
            "attachments": [],
            "notes": [],
        }

        # Extract casename
        casename_meta = doc.find("meta", {"name": "CASENAME"})
        if casename_meta and casename_meta.get("content"):
            casename = casename_meta["content"]
            item["title"] = casename
            item["caseName"] = casename
            # Capitalize properly
            casename_lower = casename.lower()
            casename_capitalized = " ".join(
                [word.capitalize() for word in casename_lower.split()]
            )
            casename_capitalized = casename_capitalized.replace(" V. ", " v. ")
            item["caseName"] = casename_capitalized
            item["shortTitle"] = casename_capitalized

        # Extract history
        history_meta = doc.find("meta", {"name": "COURTBELOW"})
        if history_meta and history_meta.get("content"):
            item["history"] = history_meta["content"]

        # Extract judge/author
        author_meta = doc.find("meta", attrs={"name": re.compile("AUTHOR", re.I)})
        if author_meta:
            author = author_meta.get("content", "Author Not Provided")
            item["creators"].append(
                {
                    "lastName": author if author else "Author Not Provided",
                    "creatorType": "judge",
                    "fieldMode": True,
                }
            )

        # Extract tags from GROUP meta tags
        for tag_meta in doc.find_all("meta", attrs={"name": re.compile("GROUP", re.I)}):
            tag_value = tag_meta.get("content")
            if tag_value:
                item["tags"].append({"tag": tag_value})

        # Parse decision date
        decdate_meta = doc.find("meta", attrs={"name": re.compile("DECDATE", re.I)})
        if decdate_meta:
            decdate = decdate_meta.get("content", "")
            # Parse format like "January 15, 2003"
            date_match = re.match(r"(\w+)\s+(\d+),\s+(\d+)", decdate)
            if date_match:
                month, day, year = date_match.groups()
                item["dateDecided"] = f"{year} {month} {day}"

        # Create attachment to PDF
        pdf_match = re.match(r"^(.+)/html/(.+)(\.Z\w+)\.html$", url)
        if pdf_match:
            pdf_url = (
                f"{pdf_match.group(1)}/pdf/{pdf_match.group(2)}P{pdf_match.group(3)}"
            )
            item["attachments"].append(
                {"url": pdf_url, "title": "PDF version", "mimeType": "application/pdf"}
            )

        # Parse citation from CASENUMBER
        cite_elem = doc.find("CASENUMBER")
        if cite_elem:
            cite_text = cite_elem.get_text()
            cite_match = re.search(r"([0-9]+)\s+U\.S\.\s+([0-9]+)", cite_text)
            if cite_match:
                item["reporterVolume"] = cite_match.group(1)
                item["firstPage"] = cite_match.group(2)

        # Look for citation in span.offcite
        for span in doc.find_all("span", class_="offcite"):
            cite_match = re.search(r"([0-9]+)\s+U\.S\.\s+([0-9]+)", span.get_text())
            if cite_match:
                item["reporterVolume"] = cite_match.group(1)
                item["firstPage"] = cite_match.group(2)
                break

        # Create bluebook citation note
        if "reporterVolume" in item and "firstPage" in item:
            year = re.search(r"\d{4}", item.get("dateDecided", ""))
            year_str = year.group(0) if year else ""
            bluebook = f"Bluebook citation: {item.get('shortTitle', '')}, {item['reporterVolume']} U.S. {item['firstPage']}"
            if year_str:
                bluebook += f" ({year_str})"
            bluebook += "."
            item["notes"].append({"note": bluebook})

        # Parse disposition
        disposition_elem = doc.find("DISPOSITION")
        if disposition_elem:
            disposition = disposition_elem.get_text().strip()
            disposition = re.sub(r"\s+", " ", disposition)
            item["title"] = f"{item.get('title', '')} ({disposition})"
            item["caseName"] = f"{item.get('caseName', '')} ({disposition})"

        return item
