"""
Translator: Artforum
Description: Artforum translator for Zotero
Translator ID: a127f012-4ea4-4d05-a657-24d47f91b016
"""

from typing import Any, Dict, Optional, List
import re
import json


TRANSLATOR_INFO = {
    "translator_id": "a127f012-4ea4-4d05-a657-24d47f91b016",
    "label": "Artforum",
    "creator": "czar",
    "target": r"^https?://(www\.)?artforum\.com/",
    "min_version": "3.0",
    "priority": 100,
    "in_repository": True,
    "translator_type": 4,
    "browser_support": "gcsibv",
    "last_updated": "2021-09-02 00:33:38",
}


def detect_web(doc: Any, url: str) -> Optional[str]:
    """Detect if the page is a single item or multiple items"""
    if re.search(r"-\d{5,}([?#].*)?$", url):
        if doc.select_one("h3.print-article__issue-title"):
            return "magazineArticle"
        return "blogPost"
    elif get_search_results(doc, check_only=True):
        return "multiple"
    return None


def get_search_results(doc: Any, check_only: bool = False) -> Optional[Dict[str, str]]:
    """Get search results from a multiple item page"""
    items = {}
    rows = doc.select(
        "h1.results-list__h1, .toc-article__title, .news-list h1, "
        ".reviews-list h1, .article-list h1, p.hp-singlefeature-author__writer, "
        "h3.hp-news__title, h3.hp-twocolumn__title a, h3.hp-artguide__title, "
        "p.hp-bloglist__teaser a"
    )

    for row in rows:
        link = row.select_one("a")
        if not link:
            link = row if row.name == "a" else None
            if not link:
                parent = row.find_parent("a")
                if parent:
                    link = parent

        if link:
            href = link.get("href")
            title = row.get_text(strip=True)
            if href and title:
                if check_only:
                    return True
                items[href] = title

    return items if items else None


def scrape(doc: Any, url: str) -> Dict[str, Any]:
    """Scrape a single item page"""
    item_type = "magazineArticle" if "print/" in url else "blogPost"

    item = {
        "itemType": item_type,
        "title": "",
        "creators": [],
        "date": "",
        "publicationTitle": "Artforum",
        "language": "en-US",
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
            item["title"] = json_data.get("name", "")
            item["date"] = json_data.get("dateModified") or json_data.get(
                "datePublished", ""
            )

            if not item["creators"] and json_data.get("author"):
                author_name = json_data["author"].get("name", "")
                if author_name:
                    parts = author_name.split()
                    if len(parts) > 1:
                        item["creators"].append(
                            {
                                "firstName": " ".join(parts[:-1]),
                                "lastName": parts[-1],
                                "creatorType": "author",
                            }
                        )
        except (json.JSONDecodeError, KeyError):
            pass

    # Get authors from metadata
    author_links = doc.select(".contrib-link a")
    for link in author_links:
        author_name = link.get_text(strip=True)
        parts = author_name.split()
        if len(parts) > 1:
            item["creators"].append(
                {
                    "firstName": " ".join(parts[:-1]),
                    "lastName": parts[-1],
                    "creatorType": "author",
                }
            )

    # Handle magazine articles
    if item_type == "magazineArticle":
        item["ISSN"] = "0004-3532"
        issue_date_elem = doc.select_one("h3.print-article__issue-title")
        if issue_date_elem:
            date_text = issue_date_elem.get_text(strip=True).replace("PRINT ", "")
            item["date"] = date_text

            # Try to get volume and issue from linked page
            issue_link = issue_date_elem.select_one("a")
            if issue_link:
                # Note: In a real implementation, you would fetch this page
                # and extract volume/issue. For now, we'll leave them empty
                pass

    return item


def do_web(doc: Any, url: str) -> List[Dict[str, Any]]:
    """Main entry point for the translator"""
    web_type = detect_web(doc, url)

    if web_type == "multiple":
        items = get_search_results(doc, check_only=False)
        # In a real implementation, you would use Zotero.selectItems
        # For now, return empty list as we need user interaction
        return []
    else:
        return [scrape(doc, url)]
