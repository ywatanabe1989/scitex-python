"""
Translator: BBC Genome
Description: BBC Genome translator for Zotero
Translator ID: 777e5ce0-0b16-4a12-8e6c-5a1a2cb33189
"""

from typing import Any, Dict, Optional, List
import re


TRANSLATOR_INFO = {
    "translator_id": "777e5ce0-0b16-4a12-8e6c-5a1a2cb33189",
    "label": "BBC Genome",
    "creator": "Philipp Zumstein",
    "target": r"^https?://genome\.ch\.bbc\.co\.uk/",
    "min_version": "3.0",
    "priority": 100,
    "in_repository": True,
    "translator_type": 4,
    "browser_support": "gcsibv",
    "last_updated": "2017-09-04 22:38:30",
}


TV_PROGRAMS = [
    "bbcone",
    "bbctwo",
    "bbcthree",
    "bbcfour",
    "cbbc",
    "cbeebies",
    "bbcnews",
    "bbcparliament",
    "bbchd",
    "bbctv",
    "bbcchoice",
    "bbcknowledge",
]


def detect_web(doc: Any, url: str) -> Optional[str]:
    """Detect if the page is a magazine article or multiple items"""
    if "/search/" in url and get_search_results(doc, check_only=True):
        return "multiple"
    elif doc.select_one("div.programme-details"):
        return "magazineArticle"
    return None


def get_search_results(doc: Any, check_only: bool = False) -> Optional[Dict[str, str]]:
    """Get search results from a multiple item page"""
    items = {}
    rows = doc.select("h2 > a.title")

    for row in rows:
        href = row.get("href")
        title = row.get_text(strip=True)
        if href and title:
            if check_only:
                return True
            items[href] = title

    return items if items else None


def scrape(doc: Any, url: str) -> List[Dict[str, Any]]:
    """Scrape a single item page - returns both magazine article and broadcast item"""
    items = []

    # Magazine article item
    mag_item = {
        "itemType": "magazineArticle",
        "title": "",
        "creators": [],
        "date": "",
        "publicationTitle": "The Radio Times",
        "ISSN": "0033-8060",
        "language": "en-GB",
        "url": url,
        "itemID": url + "#magazinArticle",
        "attachments": [{"title": "Snapshot", "mimeType": "text/html"}],
        "tags": [],
        "notes": [],
        "seeAlso": [],
    }

    # Get title
    title_elem = doc.select_one("h1")
    if title_elem:
        title_text = title_elem.get_text(strip=True)
        # Convert from uppercase if needed
        if title_text == title_text.upper():
            mag_item["title"] = title_text.title()
        else:
            mag_item["title"] = title_text

    # Get issue info from aside
    aside = doc.select_one("aside.issue p")
    if aside:
        aside_text = aside.get_text(strip=True)
        parts = aside_text.split("\n")
        if len(parts) > 0:
            mag_item["issue"] = parts[0].replace("Issue", "").strip()
        if len(parts) > 1:
            mag_item["date"] = parts[1].strip()
        if len(parts) > 2:
            mag_item["pages"] = parts[2].replace("Page", "").strip()

    # Get aired info
    aired_elem = doc.select_one(".primary-content a")
    url_program_elem = doc.select_one(".primary-content a")
    synopsis_elem = doc.select_one(".synopsis")

    aired = aired_elem.get_text(strip=True) if aired_elem else ""
    url_program = url_program_elem.get("href") if url_program_elem else ""
    synopsis = synopsis_elem.get_text(strip=True) if synopsis_elem else ""

    if aired:
        mag_item["notes"].append({"note": aired})
    if synopsis:
        mag_item["abstractNote"] = synopsis

    items.append(mag_item)

    # Broadcast item (radio or TV)
    if url_program:
        program = url_program.split("/")[2] if len(url_program.split("/")) > 2 else ""
        broadcast_type = "tvBroadcast" if program in TV_PROGRAMS else "radioBroadcast"

        broadcast_item = {
            "itemType": broadcast_type,
            "title": mag_item["title"],
            "creators": [],
            "date": "",
            "programTitle": "",
            "seeAlso": [mag_item["itemID"]],
            "attachments": [],
            "tags": [],
            "notes": [],
        }

        # Parse aired text: "BBC Radio 4 FM, 30 September 1967 6.35"
        if aired:
            pieces = aired.split(",")
            if len(pieces) > 0:
                broadcast_item["programTitle"] = pieces[0].strip()
            if len(pieces) > 1:
                time_elem = doc.select_one(".primary-content a span.time")
                time_text = time_elem.get_text(strip=True) if time_elem else ""

                # Add leading zero if needed
                if time_text and time_text.find(".") == 1:
                    time_text = "0" + time_text

                # Convert time format
                time_text = time_text.replace(".", ":")

                # Extract date
                date_part = pieces[1].strip()
                # Remove time from date_part if present
                date_part = re.sub(r"\s+\d+[\.:]\d+.*$", "", date_part)

                if time_text:
                    broadcast_item["date"] = f"{date_part}T{time_text}"
                else:
                    broadcast_item["date"] = date_part

        items.append(broadcast_item)

    return items


def do_web(doc: Any, url: str) -> List[Dict[str, Any]]:
    """Main entry point for the translator"""
    web_type = detect_web(doc, url)

    if web_type == "multiple":
        return []
    else:
        return scrape(doc, url)
