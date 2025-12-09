"""
Translator: Atlanta Journal-Constitution
Description: Atlanta Journal-Constitution translator for Zotero
Translator ID: 01322929-5782-4612-81f7-d861fb46d9f2
"""

from typing import Any, Dict, Optional, List
import re
import json


TRANSLATOR_INFO = {
    "translator_id": "01322929-5782-4612-81f7-d861fb46d9f2",
    "label": "Atlanta Journal-Constitution",
    "creator": "Abe Jellinek",
    "target": r"^https?://(www\.)?ajc\.com",
    "min_version": "3.0",
    "priority": 100,
    "in_repository": True,
    "translator_type": 4,
    "browser_support": "gcsibv",
    "last_updated": "2021-07-14 19:41:44",
}


def detect_web(doc: Any, url: str) -> Optional[str]:
    """Detect if the page is a single item or multiple items"""
    if doc.select_one(".c-articleContent") and doc.select_one(
        'script[type="application/ld+json"]'
    ):
        if "blog/" in url:
            return "blogPost"
        else:
            return "newspaperArticle"
    elif get_search_results(doc, check_only=True):
        return "multiple"
    return None


def get_search_results(doc: Any, check_only: bool = False) -> Optional[Dict[str, str]]:
    """Get search results from a multiple item page"""
    items = {}
    rows = doc.select("a.gs-title")

    for row in rows:
        href = row.get("href")
        title = row.get_text(strip=True)
        if href and title:
            if check_only:
                return True
            items[href] = title

    return items if items else None


def extract_place(lead_text: str) -> str:
    """Extract place from lead text"""
    place_re = re.compile(r"^\s*([A-Z\-\']{3,})\b")
    match = place_re.search(lead_text)
    if match:
        place = match.group(1)
        # Capitalize properly
        return place.title()
    return ""


def scrape(doc: Any, url: str) -> Dict[str, Any]:
    """Scrape a single item page"""
    json_ld_tag = doc.select_one('script[type="application/ld+json"]')
    if not json_ld_tag:
        return {}

    try:
        json_data = json.loads(json_ld_tag.string)
    except (json.JSONDecodeError, AttributeError):
        return {}

    item_type = "blogPost" if "blog/" in url else "newspaperArticle"

    item = {
        "itemType": item_type,
        "title": json_data.get("headline", ""),
        "creators": [],
        "date": json_data.get("datePublished", ""),
        "language": "English",
        "libraryCatalog": "AJC.com",
        "url": url,
        "attachments": [{"title": "Snapshot", "mimeType": "text/html"}],
        "tags": [],
        "notes": [],
        "seeAlso": [],
    }

    # Get abstract
    description_tag = doc.select_one('meta[name="description"]')
    item["abstractNote"] = json_data.get("description") or (
        description_tag.get("content") if description_tag else ""
    )

    # Extract place from abstract
    if item.get("abstractNote"):
        item["place"] = extract_place(item["abstractNote"])

    # Get section label
    section_label = doc.select_one(".section-label")
    section_text = section_label.get_text(strip=True) if section_label else ""

    if item_type == "blogPost":
        item["blogTitle"] = f"{section_text} (The Atlanta Journal-Constitution)"
    else:
        item["section"] = section_text
        item["publicationTitle"] = "The Atlanta Journal-Constitution"
        item["ISSN"] = "1539-7459"

    # Get language
    language_tag = doc.select_one('meta[name="language"]')
    if language_tag:
        item["language"] = language_tag.get("content", "English")

    # Get authors
    if json_data.get("author") and json_data["author"].get("name"):
        author_names = json_data["author"]["name"].split(", ")
        for author_name in author_names:
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
