"""
Translator: BIBSYS
Description: BIBSYS translator for Zotero
Translator ID: ab961e61-2a8a-4be1-b8a3-044f20d52d78
"""

from typing import Any, Dict, Optional, List
import re


TRANSLATOR_INFO = {
    "translator_id": "ab961e61-2a8a-4be1-b8a3-044f20d52d78",
    "label": "BIBSYS",
    "creator": "Ramesh Srigiriraju",
    "target": r"^https?://ask\.bibsys\.no/ask/action",
    "min_version": "1.0.0b4.r1",
    "priority": 100,
    "in_repository": True,
    "translator_type": 4,
    "browser_support": "gcsibv",
    "last_updated": "2021-12-28 04:44:52",
}


def detect_web(doc: Any, url: str) -> Optional[str]:
    """Detect if the page is a book or multiple items"""
    if re.match(r"^https?://ask\.bibsys\.no/ask/action/result", url):
        return "multiple"
    elif re.match(r"^https?://ask\.bibsys\.no/ask/action/show", url):
        return "book"
    return None


def get_search_results(doc: Any, check_only: bool = False) -> Optional[Dict[str, str]]:
    """Get search results from a multiple item page"""
    items = {}

    # Get titles
    title_rows = doc.select('tr td[width="49%"][align="left"][valign="top"] a')

    # Get codes (IDs)
    code_inputs = doc.select('tr td input[type="checkbox"][name="valg"]')

    if not title_rows or not code_inputs:
        return None

    # Skip first title (it's usually a header)
    for i, (title_elem, code_input) in enumerate(zip(title_rows, code_inputs)):
        if i == 0:  # Skip first
            continue

        title = title_elem.get_text(strip=True)
        code = code_input.get("value", "")

        if title and code:
            if check_only:
                return True
            items[code] = title

    return items if items else None


def scrape_from_ris(ris_text: str) -> Dict[str, Any]:
    """Parse RIS format text and return item"""
    item = {
        "itemType": "book",
        "title": "",
        "creators": [],
        "date": "",
        "publisher": "",
        "place": "",
        "ISBN": "",
        "numPages": "",
        "libraryCatalog": "BIBSYS",
        "shortTitle": "",
        "attachments": [],
        "tags": [],
        "notes": [],
        "seeAlso": [],
    }

    lines = ris_text.split("\n")
    for line in lines:
        if line.startswith("TI  - "):
            # Clean up title - remove extra spaces and fix spacing around colons
            title = line[6:].strip()
            title = re.sub(r"\s\s+", " ", title)
            title = re.sub(r"\s:", ":", title)
            item["title"] = title
        elif line.startswith("AU  - "):
            author_name = line[6:].strip()
            parts = author_name.split(",")
            if len(parts) >= 2:
                item["creators"].append(
                    {
                        "firstName": parts[1].strip(),
                        "lastName": parts[0].strip(),
                        "creatorType": "author",
                    }
                )
            else:
                parts = author_name.split()
                if len(parts) > 1:
                    item["creators"].append(
                        {
                            "firstName": " ".join(parts[:-1]),
                            "lastName": parts[-1],
                            "creatorType": "author",
                        }
                    )
        elif line.startswith("PY  - "):
            item["date"] = line[6:].strip()
        elif line.startswith("PB  - "):
            item["publisher"] = line[6:].strip()
        elif line.startswith("CY  - "):
            item["place"] = line[6:].strip()
        elif line.startswith("SN  - "):
            item["ISBN"] = line[6:].strip()
        elif line.startswith("SP  - "):
            pages = line[6:].strip()
            item["numPages"] = pages
        elif line.startswith("N1  - "):
            note_text = line[6:].strip()
            if note_text:
                item["notes"].append({"note": note_text})

    return item


def scrape(doc: Any, url: str) -> Dict[str, Any]:
    """Scrape a single item page

    Note: This requires making an HTTP POST request to get RIS data,
    which would need to be implemented with proper HTTP handling
    """
    # In a real implementation, you would:
    # 1. Make POST request to http://ask.bibsys.no/ask/action/show
    # 2. With data: visningsformat=ris&eksportFormat=refmanager&eksportEpostAdresse=&eksportEpostFormat=fortekst&cmd=sendtil
    # 3. Parse the returned RIS text
    # 4. Return the item

    return {
        "itemType": "book",
        "title": "",
        "creators": [],
        "libraryCatalog": "BIBSYS",
        "attachments": [],
        "tags": [],
        "notes": [],
        "seeAlso": [],
    }


def do_web(doc: Any, url: str) -> List[Dict[str, Any]]:
    """Main entry point for the translator"""
    web_type = detect_web(doc, url)

    if web_type == "multiple":
        return []
    elif web_type == "book":
        return [scrape(doc, url)]
    return []
