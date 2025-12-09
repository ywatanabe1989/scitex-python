"""
Translator: Australian Dictionary of Biography
Description: Australian Dictionary of Biography translator for Zotero
Translator ID: 0aea3026-a246-4201-a4b5-265f75b9a6a7
"""

from typing import Any, Dict, Optional, List
import re


TRANSLATOR_INFO = {
    "translator_id": "0aea3026-a246-4201-a4b5-265f75b9a6a7",
    "label": "Australian Dictionary of Biography",
    "creator": "Sebastian Karcher",
    "target": r"^https?://adb\.anu\.edu\.au/biograph(y|ies)/",
    "min_version": "3.0",
    "priority": 100,
    "in_repository": True,
    "translator_type": 4,
    "browser_support": "gcsibv",
    "last_updated": "2021-07-14 04:18:08",
}


def detect_web(doc: Any, url: str) -> Optional[str]:
    """Detect if the page is a single item or multiple items"""
    if "/biography/" in url:
        return "bookSection"
    elif get_search_results(doc, check_only=True):
        return "multiple"
    return None


def get_search_results(doc: Any, check_only: bool = False) -> Optional[Dict[str, str]]:
    """Get search results from a multiple item page"""
    items = {}
    rows = doc.select("a.name")

    for row in rows:
        href = row.get("href")
        title = row.get_text(strip=True)
        if href and title:
            if check_only:
                return True
            items[href] = title

    return items if items else None


def scrape(doc: Any, url: str) -> Dict[str, Any]:
    """Scrape a single item page"""
    item = {
        "itemType": "bookSection",
        "title": "",
        "creators": [],
        "date": "",
        "bookTitle": "Australian Dictionary of Biography",
        "numberOfVolumes": "18",
        "publisher": "National Centre of Biography, Australian National University",
        "place": "Canberra",
        "language": "en",
        "url": url,
        "attachments": [{"title": "Snapshot", "mimeType": "text/html"}],
        "tags": [],
        "notes": [],
        "seeAlso": [],
    }

    main = doc.select_one("#pageColumnMain")
    if not main:
        return item

    # Get title
    title_elem = main.select_one("h2")
    if title_elem:
        item["title"] = title_elem.get_text(strip=True)

    # Get abstract
    abstract_elem = main.select_one(".biographyContent p")
    if abstract_elem:
        item["abstractNote"] = abstract_elem.get_text(strip=True)

    # Get volume and date from notice
    notice_elem = main.select_one(".textNotice")
    if notice_elem:
        notice_text = notice_elem.get_text(strip=True)
        # Try to match: "volume X, (MUP), YYYY"
        match = re.search(r"([^,]+), \(MUP\), ([^,]+)", notice_text)
        if match:
            volume_text = match.group(1)
            item["volume"] = re.sub(
                r"^\s*volume\s*", "", volume_text, flags=re.IGNORECASE
            )
            item["date"] = match.group(2).strip()
        else:
            # Just use the text as date
            item["date"] = notice_text.strip()

    # Get authors
    author_links = main.select(".authorName a")
    for author_link in author_links:
        author_name = author_link.get_text(strip=True)
        parts = author_name.split()
        if len(parts) > 1:
            item["creators"].append(
                {
                    "firstName": " ".join(parts[:-1]),
                    "lastName": parts[-1],
                    "creatorType": "author",
                }
            )

    return item


def do_web(doc: Any, url: str) -> List[Dict[str, Any]]:
    """Main entry point for the translator"""
    web_type = detect_web(doc, url)

    if web_type == "multiple":
        items = get_search_results(doc, check_only=False)
        return []
    else:
        return [scrape(doc, url)]
