# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/blogger.py
# --------------------------------------------------------------------------------
# """
# Blogger Translator
# 
# Translates blog posts from Blogspot/Blogger.
# 
# Metadata:
#     translatorID: 6f9aa90d-6631-4459-81ef-a0758d2e3921
#     label: Blogger
#     creator: Adam Crymble
#     target: \.blogspot\.com
#     minVersion: 3.0
#     priority: 100
#     inRepository: True
#     translatorType: 4
#     browserSupport: gcsibv
#     lastUpdated: 2016-09-05 23:14:05
# """
# 
# from typing import Dict, Any, List, Optional
# from bs4 import BeautifulSoup
# import re
# 
# 
# class BloggerTranslator:
#     """Translator for Blogger/Blogspot blog posts."""
# 
#     METADATA = {
#         "translatorID": "6f9aa90d-6631-4459-81ef-a0758d2e3921",
#         "label": "Blogger",
#         "creator": "Adam Crymble",
#         "target": r"\.blogspot\.com",
#         "minVersion": "3.0",
#         "priority": 100,
#         "inRepository": True,
#         "translatorType": 4,
#         "browserSupport": "gcsibv",
#         "lastUpdated": "2016-09-05 23:14:05",
#     }
# 
#     def detect_web(self, doc: BeautifulSoup, url: str) -> str:
#         """Detect page type."""
#         entries = doc.select("h3.post-title.entry-title")
# 
#         if len(entries) > 1:
#             return "multiple"
#         elif len(entries) == 1:
#             return "blogPost"
#         return ""
# 
#     def _get_search_results(self, doc: BeautifulSoup) -> Dict[str, str]:
#         """Get search results."""
#         items = {}
#         rows = doc.select(
#             "h3.post-title.entry-title a, li.archivedate.expanded ul.posts li a"
#         )
# 
#         for row in rows:
#             href = row.get("href")
#             title = row.get_text(strip=True)
#             if href and title:
#                 items[href] = title
# 
#         return items
# 
#     def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
#         """Extract data from the page."""
#         page_type = self.detect_web(doc, url)
# 
#         if page_type == "multiple":
#             items = self._get_search_results(doc)
#             return [{"url": u} for u in items.keys()]
#         else:
#             return [self.scrape(doc, url)]
# 
#     def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
#         """Scrape blog post data from the document."""
#         item = {
#             "itemType": "blogPost",
#             "libraryCatalog": "Blogger",
#             "creators": [],
#             "tags": [],
#             "attachments": [],
#         }
# 
#         # Extract title
#         title_elem = doc.select_one("h3.post-title.entry-title a")
#         if title_elem:
#             item["title"] = title_elem.get_text(strip=True)
#         else:
#             item["title"] = doc.title.string if doc.title else ""
# 
#         # Extract author
#         author_elem = doc.select_one("span.post-author.vcard span.fn")
#         if author_elem:
#             author = author_elem.get_text(strip=True).lower()
# 
#             # Remove "by " prefix if present
#             if " by " in author:
#                 author = author[author.index(" by") + 3 :].strip()
# 
#             # Capitalize words
#             words = author.split()
#             author = " ".join(word.capitalize() for word in words)
# 
#             item["creators"].append(self._clean_author(author, "author"))
# 
#         # Extract date
#         date_elem = doc.select_one("h2.date-header")
#         if date_elem:
#             item["date"] = date_elem.get_text(strip=True)
# 
#         # Extract tags
#         tag_elems = doc.select("span.post-labels a")
#         for tag_elem in tag_elems:
#             tag_text = tag_elem.get_text(strip=True)
#             if tag_text:
#                 item["tags"].append({"tag": tag_text})
# 
#         # Extract blog title
#         if doc.title:
#             blog_parts = doc.title.string.split(":")
#             if blog_parts:
#                 item["blogTitle"] = blog_parts[0]
# 
#         # Clean URL (remove query and fragment)
#         clean_url = re.sub(r"[?#].+", "", url)
#         item["url"] = clean_url
# 
#         # Add snapshot
#         item["attachments"].append(
#             {"url": clean_url, "title": "Blogspot Snapshot", "mimeType": "text/html"}
#         )
# 
#         return item
# 
#     def _clean_author(self, name: str, creator_type: str) -> Dict[str, Any]:
#         """Parse author name into structured format."""
#         name = name.strip()
#         parts = name.split()
# 
#         if len(parts) >= 2:
#             return {
#                 "firstName": " ".join(parts[:-1]),
#                 "lastName": parts[-1],
#                 "creatorType": creator_type,
#             }
#         else:
#             return {"firstName": "", "lastName": name, "creatorType": creator_type}

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/blogger.py
# --------------------------------------------------------------------------------
