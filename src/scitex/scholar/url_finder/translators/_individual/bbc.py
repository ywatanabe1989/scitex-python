"""
Translator: BBC
Description: BBC translator for Zotero
Translator ID: f4130157-93f7-4493-8f24-a7c85549013d
"""

from typing import Any, Dict, Optional, List
import re
import json
from datetime import datetime


TRANSLATOR_INFO = {
    "translator_id": "f4130157-93f7-4493-8f24-a7c85549013d",
    "label": "BBC",
    "creator": "Philipp Zumstein",
    "target": r"^https?://(www|news?)\. bbc\.(co\.uk|com)",
    "min_version": "3.0",
    "priority": 100,
    "in_repository": True,
    "translator_type": 4,
    "browser_support": "gcsibv",
    "last_updated": "2019-06-10 21:51:43",
}


def detect_web(doc: Any, url: str) -> Optional[str]:
    """Detect if the page is a newspaper article, blog post, video, or multiple items"""
    url = re.sub(r"[?#].+", "", url)

    if re.search(r"\d{8}$", url) or re.search(r"\d{7}\.(stm)$", url):
        page_node = doc.select_one("#page")
        if page_node:
            classes = page_node.get("class", [])
            if isinstance(classes, str):
                classes = classes.split()
            if any(cls in classes for cls in ["media-asset-page", "vxp-headlines"]):
                return "videoRecording"
        return "newspaperArticle"

    if "/newsbeat/article" in url:
        return "blogPost"

    if get_search_results(doc, check_only=True):
        return "multiple"

    return None


def get_search_results(doc: Any, check_only: bool = False) -> Optional[Dict[str, str]]:
    """Get search results from a multiple item page"""
    items = {}
    rows = doc.select("a:has(h3)")

    # For NewsBeat
    if not rows:
        rows = doc.select('article div h1[itemprop="headline"] a')

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
    url = re.sub(r"[?#].+", "", url)
    item_type = detect_web(doc, url)
    if not item_type or item_type == "multiple":
        return {}

    item = {
        "itemType": item_type,
        "title": "",
        "creators": [],
        "date": "",
        "abstractNote": "",
        "language": "en-GB",
        "url": url,
        "attachments": [{"title": "Snapshot", "mimeType": "text/html"}],
        "tags": [],
        "notes": [],
        "seeAlso": [],
    }

    # Try to get data from JSON-LD
    json_ld_tag = doc.select_one('script[type="application/ld+json"]')
    if json_ld_tag:
        try:
            json_data = json.loads(json_ld_tag.string)
            if json_data.get("datePublished"):
                item["date"] = json_data["datePublished"]
        except (json.JSONDecodeError, AttributeError):
            pass

    # Get date from data-seconds attribute
    if not item["date"]:
        seconds_elem = doc.select_one('div:has(h1, h2) *[class*="date"][data-seconds]')
        if seconds_elem:
            seconds = seconds_elem.get("data-seconds")
            if seconds:
                try:
                    timestamp = int(seconds)
                    dt = datetime.fromtimestamp(timestamp)
                    item["date"] = dt.isoformat()
                except (ValueError, TypeError):
                    pass

    # Try other date sources
    if not item["date"]:
        date_meta = doc.select_one('meta[property="rnews:datePublished"]')
        if date_meta:
            item["date"] = date_meta.get("content", "")
        else:
            timestamp_elem = doc.select_one("p.timestamp")
            if timestamp_elem:
                item["date"] = timestamp_elem.get_text(strip=True)
            else:
                date_meta2 = doc.select_one('meta[name="OriginalPublicationDate"]')
                if date_meta2:
                    item["date"] = date_meta2.get("content", "")

    # Get title from embedded metadata
    title_meta = doc.select_one('meta[property="og:title"]')
    if title_meta:
        item["title"] = title_meta.get("content", "")
    else:
        h1 = doc.select_one("h1")
        if h1:
            item["title"] = h1.get_text(strip=True)

    # For old .stm pages
    if url.endswith(".stm"):
        headline_meta = doc.select_one('meta[name="Headline"]')
        if headline_meta:
            item["title"] = headline_meta.get("content", "")

    # Get authors from byline
    byline_elem = doc.select_one("span.byline__name")
    if byline_elem:
        author_string = byline_elem.get_text(strip=True)
        author_string = author_string.replace("By", "").replace("...", "").strip()

        # Check if author is real or just webpage title
        h1_elem = doc.select_one("h1")
        webpage_title = h1_elem.get_text(strip=True).lower() if h1_elem else ""

        authors = author_string.split("&")
        for author in authors:
            author = author.strip()
            if webpage_title and author.lower() in webpage_title:
                continue
            parts = author.split()
            if len(parts) > 1:
                item["creators"].append(
                    {
                        "firstName": " ".join(parts[:-1]),
                        "lastName": parts[-1],
                        "creatorType": "author",
                    }
                )
    else:
        # Try old format
        byline_p = doc.select_one("p.byline")
        if byline_p:
            author_string = byline_p.get_text(strip=True)
            title_em = doc.select_one("em.title")
            if title_em:
                author_string = author_string.replace(title_em.get_text(), "")
            author_string = author_string.replace("By", "").strip()
            authors = author_string.split("&")
            for author in authors:
                author = author.strip()
                parts = author.split()
                if len(parts) > 1:
                    item["creators"].append(
                        {
                            "firstName": " ".join(parts[:-1]),
                            "lastName": parts[-1],
                            "creatorType": "author",
                        }
                    )

    # Get abstract
    desc_meta = doc.select_one('meta[property="og:description"]')
    if desc_meta:
        item["abstractNote"] = desc_meta.get("content", "")
    else:
        desc_meta2 = doc.select_one('meta[name="Description"]')
        if desc_meta2:
            item["abstractNote"] = desc_meta2.get("content", "")

    # Get publication title and section
    if item_type == "newspaperArticle":
        item["publicationTitle"] = "BBC News"
        section_meta = doc.select_one('meta[property="article:section"]')
        if section_meta:
            item["section"] = section_meta.get("content", "")

    elif item_type == "blogPost":
        item["blogTitle"] = "BBC Newsbeat"

    elif item_type == "videoRecording":
        item["studio"] = "BBC"

    # Get tags
    tags_meta = doc.select('meta[property="article:tag"]')
    for tag_meta in tags_meta:
        tag_content = tag_meta.get("content", "")
        if tag_content:
            # Capitalize first letter
            item["tags"].append(tag_content[0].upper() + tag_content[1:])

    return item


def do_web(doc: Any, url: str) -> List[Dict[str, Any]]:
    """Main entry point for the translator"""
    web_type = detect_web(doc, url)

    if web_type == "multiple":
        return []
    else:
        return [scrape(doc, url)]
