# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/cbc.py
# --------------------------------------------------------------------------------
# """
# CBC Translator
# 
# Translates CBC (Canadian Broadcasting Corporation) articles to Zotero format.
# 
# Metadata:
#     translatorID: 03c4b906-8cb2-4850-a771-697cbd92c2a1
#     label: CBC
#     creator: Geoff Banh
#     target: ^https?://www\.cbc\.ca/
#     minVersion: 5.0
#     priority: 100
#     inRepository: True
#     translatorType: 4
#     browserSupport: gcsibv
#     lastUpdated: 2024-03-14 20:55:10
# """
# 
# from typing import Dict, Any, Optional
# from bs4 import BeautifulSoup
# import json
# import re
# 
# 
# class CBCTranslator:
#     """Translator for CBC (Canadian Broadcasting Corporation) articles."""
# 
#     METADATA = {
#         "translatorID": "03c4b906-8cb2-4850-a771-697cbd92c2a1",
#         "label": "CBC",
#         "creator": "Geoff Banh",
#         "target": r"^https?://www\.cbc\.ca/",
#         "minVersion": "5.0",
#         "priority": 100,
#         "inRepository": True,
#         "translatorType": 4,
#         "browserSupport": "gcsibv",
#         "lastUpdated": "2024-03-14 20:55:10",
#     }
# 
#     def detect_web(self, doc: BeautifulSoup, url: str) -> str:
#         """
#         Detect if the page is a CBC article or search results.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             url: URL of the page
# 
#         Returns:
#             Type of content detected
#         """
#         path = url.split("?")[0]  # Remove query params
# 
#         if "/search?" in url and self.get_search_results(doc, True):
#             return "multiple"
# 
#         # Check if it's a content page with LD+JSON
#         if any(
#             section in path
#             for section in [
#                 "news/",
#                 "sports/",
#                 "radio/",
#                 "books/",
#                 "arts/",
#                 "music/",
#                 "life/",
#                 "television/",
#                 "archives/",
#             ]
#         ):
#             if self.get_ld_json(doc):
#                 return "newspaperArticle"
# 
#         if "/player/" in path:
#             return "videoRecording"
# 
#         return ""
# 
#     def get_search_results(
#         self, doc: BeautifulSoup, check_only: bool = False
#     ) -> Dict[str, str]:
#         """
#         Get search results from the page.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             check_only: If True, return early on first match
# 
#         Returns:
#             Dictionary mapping URLs to titles
#         """
#         items = {}
#         rows = doc.select(".card.cardListing")
# 
#         for row in rows:
#             href = row.get("href")
#             title_elem = row.select_one("h3.headline")
#             title = title_elem.get_text(strip=True) if title_elem else None
# 
#             if not href or not title:
#                 continue
#             if check_only:
#                 return {"found": "true"}
#             items[href] = title
# 
#         return items
# 
#     def get_ld_json(self, doc: BeautifulSoup) -> Optional[Dict]:
#         """
#         Extract LD+JSON structured data from the page.
# 
#         Args:
#             doc: BeautifulSoup parsed document
# 
#         Returns:
#             Parsed JSON data or None
#         """
#         ld_script = doc.select_one('script[type="application/ld+json"]')
#         if ld_script and ld_script.string:
#             try:
#                 return json.loads(ld_script.string)
#             except json.JSONDecodeError:
#                 return None
#         return None
# 
#     def get_meta_content(
#         self, doc: BeautifulSoup, attribute: str, value: str
#     ) -> Optional[str]:
#         """
#         Get content from a meta tag.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             attribute: Attribute name (e.g., 'property', 'name')
#             value: Attribute value
# 
#         Returns:
#             Content attribute value or None
#         """
#         meta = doc.select_one(f'meta[{attribute}="{value}"]')
#         return meta.get("content") if meta else None
# 
#     def do_web(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
#         """
#         Extract article data from CBC page.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             url: URL of the page
# 
#         Returns:
#             Dictionary containing article metadata or search results
#         """
#         if self.detect_web(doc, url) == "multiple":
#             return self.get_search_results(doc)
#         return self.scrape(doc, url)
# 
#     def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
#         """
#         Scrape article data from the document.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             url: URL of the page
# 
#         Returns:
#             Dictionary containing article metadata
#         """
#         item_type = self.detect_web(doc, url)
# 
#         item = {
#             "itemType": item_type if item_type else "newspaperArticle",
#             "language": "en-CA",
#             "libraryCatalog": "CBC.ca",
#             "url": url,
#             "creators": [],
#             "tags": [],
#             "attachments": [],
#         }
# 
#         ld = self.get_ld_json(doc)
# 
#         if ld:
#             # URL
#             og_url = self.get_meta_content(doc, "property", "og:url")
#             if og_url:
#                 item["url"] = og_url
# 
#             # Title
#             if "headline" in ld:
#                 item["title"] = ld["headline"]
#             elif "name" in ld:
#                 item["title"] = ld["name"]
# 
#             # Date
#             if item_type == "videoRecording" and "uploadDate" in ld:
#                 item["date"] = (
#                     ld["uploadDate"][:10]
#                     if len(ld["uploadDate"]) >= 10
#                     else ld["uploadDate"]
#                 )
#             elif "datePublished" in ld:
#                 item["date"] = (
#                     ld["datePublished"][:10]
#                     if len(ld["datePublished"]) >= 10
#                     else ld["datePublished"]
#                 )
# 
#             # Abstract
#             if "description" in ld:
#                 item["abstractNote"] = ld["description"]
# 
#             # Authors
#             if "author" in ld and ld["author"]:
#                 authors = (
#                     ld["author"] if isinstance(ld["author"], list) else [ld["author"]]
#                 )
#                 for author in authors:
#                     if (
#                         isinstance(author, dict)
#                         and author.get("@type") != "Organization"
#                     ):
#                         author_name = author.get("name", "")
#                         if "," in author_name:
#                             # Multiple authors in one entry
#                             for name in author_name.split(","):
#                                 name = name.strip()
#                                 if name:
#                                     item["creators"].append(self._clean_author(name))
#                         else:
#                             if author_name:
#                                 item["creators"].append(self._clean_author(author_name))
# 
#             # Publication title
#             site_name = "CBC"
#             if item_type != "videoRecording":
#                 # Extract department from URL
#                 dept_match = re.search(r"\.ca/(\w+)(?=/)", url)
#                 if dept_match:
#                     dept = dept_match.group(1).capitalize()
#                     site_name += " " + dept
#             item["publicationTitle"] = site_name
# 
#             # Running time for videos
#             if item_type == "videoRecording" and "duration" in ld:
#                 duration = ld["duration"]
#                 # Parse ISO 8601 duration (e.g., PT82.849S)
#                 if duration.startswith("PT"):
#                     seconds_match = re.search(r"(\d+(?:\.\d+)?)S", duration)
#                     if seconds_match:
#                         item["runningTime"] = seconds_match.group(1)
# 
#         # Add snapshot attachment
#         item["attachments"].append(
#             {"title": "Snapshot", "mimeType": "text/html", "url": url}
#         )
# 
#         return item
# 
#     def _clean_author(self, name: str) -> Dict[str, Any]:
#         """
#         Parse author name into first and last name.
# 
#         Args:
#             name: Full author name
# 
#         Returns:
#             Dictionary with firstName and lastName
#         """
#         name = name.strip()
#         parts = name.split()
# 
#         if len(parts) >= 2:
#             return {
#                 "firstName": " ".join(parts[:-1]),
#                 "lastName": parts[-1],
#                 "creatorType": "author",
#             }
#         else:
#             return {"lastName": name, "creatorType": "author", "fieldMode": True}

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/cbc.py
# --------------------------------------------------------------------------------
