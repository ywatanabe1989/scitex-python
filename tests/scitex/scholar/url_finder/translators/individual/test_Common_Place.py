# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/Common_Place.py
# --------------------------------------------------------------------------------
# """
# Common-Place Translator
# 
# Translates Common-Place journal articles to Zotero format.
# 
# Metadata:
#     translatorID: c3edb423-f267-47a1-a8c2-158c247f87c2
#     label: Common-Place
#     creator: Frederick Gibbs, Philipp Zumstein
#     target: ^https?://(www\\.)?(common-place\\.org/|common-place-archives\\.org/)
#     minVersion: 3.0
#     priority: 100
#     inRepository: True
#     translatorType: 4
#     browserSupport: gcsibv
#     lastUpdated: 2016-09-10 09:34:34
# """
# 
# from typing import Dict, Any, Optional, List
# from bs4 import BeautifulSoup
# import re
# 
# 
# class CommonPlaceTranslator:
#     """Translator for Common-Place journal articles."""
# 
#     METADATA = {
#         "translatorID": "c3edb423-f267-47a1-a8c2-158c247f87c2",
#         "label": "Common-Place",
#         "creator": "Frederick Gibbs, Philipp Zumstein",
#         "target": r"^https?://(www\.)?(common-place\.org/|common-place-archives\.org/)",
#         "minVersion": "3.0",
#         "priority": 100,
#         "inRepository": True,
#         "translatorType": 4,
#         "browserSupport": "gcsibv",
#         "lastUpdated": "2016-09-10 09:34:34",
#     }
# 
#     def detect_web(self, doc: BeautifulSoup, url: str) -> str:
#         """
#         Detect if the page is a Common-Place article or search page.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             url: URL of the page
# 
#         Returns:
#             'journalArticle' if single article, 'multiple' if search results, empty string otherwise
#         """
#         if self._get_search_results(doc, True):
#             return "multiple"
#         elif (
#             "single-article" in doc.body.get("class", [])
#             or "common-place-archives.org" in url
#         ):
#             return "journalArticle"
#         return ""
# 
#     def _get_search_results(
#         self, doc: BeautifulSoup, check_only: bool = False
#     ) -> Dict[str, str]:
#         """
#         Get search results from the page.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             check_only: If True, return as soon as first result is found
# 
#         Returns:
#             Dictionary mapping URLs to titles, or False if none found
#         """
#         items = {}
#         found = False
#         rows = doc.select("h3.article-title > a, h2 > a")
# 
#         for row in rows:
#             href = row.get("href")
#             title = row.get_text(strip=True)
#             if not href or not title:
#                 continue
#             if check_only:
#                 return True
#             found = True
#             items[href] = title
# 
#         return items if found else False
# 
#     def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
#         """
#         Extract article data from Common-Place page.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             url: URL of the page
# 
#         Returns:
#             List of dictionaries containing article metadata
#         """
#         if self.detect_web(doc, url) == "multiple":
#             return []
#         else:
#             return [self.scrape(doc, url)]
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
#         item = {
#             "itemType": "journalArticle",
#             "publicationTitle": "Common-Place",
#             "url": url,
#             "creators": [],
#             "tags": [],
#             "attachments": [],
#             "notes": [],
#             "seeAlso": [],
#         }
# 
#         if "single-article" in doc.body.get("class", []):
#             # New format
#             title_elem = doc.select_one("article h1")
#             if title_elem:
#                 item["title"] = title_elem.get_text(strip=True)
# 
#             author_elem = doc.select_one("article h1 + p")
#             if author_elem:
#                 author = author_elem.get_text(strip=True)
#                 item["creators"].append(self._clean_author(author, "author"))
# 
#             abstract_elem = doc.select_one("article div.entry-excerpt")
#             if abstract_elem:
#                 item["abstractNote"] = abstract_elem.get_text(strip=True)
# 
#             # Extract date from breadcrumb
#             date_elem = doc.select_one("article ol.breadcrumb li")
#             if date_elem:
#                 # Get text nodes only
#                 date_text = "".join([t for t in date_elem.stripped_strings if t])
#                 item["date"] = date_text
# 
#             # Extract volume and issue
#             volno_elem = doc.select_one("article ol.breadcrumb li:first-child a")
#             if volno_elem:
#                 volno = volno_elem.get_text(strip=True)
#                 m = re.search(r"Vol\.\s*(\d+)\s+No\.\s*(\d+)", volno)
#                 if m:
#                     item["volume"] = m.group(1)
#                     item["issue"] = m.group(2)
# 
#         else:
#             # Old format
#             # Get issue year and month from body HTML
#             body_html = str(doc.body)
#             date_re = r'<a href="/vol-(\d+)/no-(\d+)/">([^<]*)</a>'
#             m = re.search(date_re, body_html)
#             if m:
#                 item["volume"] = m.group(1)
#                 item["issue"] = m.group(2)
#                 # Extract month and year from the third group
#                 issue_info = m.group(3)
#                 month_match = re.search(r"·\s*([\w\s]+)$", issue_info)
#                 if month_match:
#                     item["date"] = month_match.group(1)
# 
#             author = doc.select_one("div#content p span:nth-of-type(1)")
#             title = doc.select_one("div#content p span:nth-of-type(2)")
# 
#             if author and title:
#                 author_text = author.get_text(strip=True)
#                 title_text = title.get_text(strip=True)
# 
#                 # Determine if we have a book review
#                 if "Review by" in author_text:
#                     title_text = "Review of " + title_text
#                     author_text = author_text.replace("Review by", "").strip()
# 
#                 item["creators"].append(self._clean_author(author_text, "author"))
#                 item["title"] = title_text
#             else:
#                 # Older issue format
#                 review_elem = doc.select_one(
#                     "body > table tr > td:nth-child(2) > p:nth-child(2)"
#                 )
#                 if review_elem and "Review" in review_elem.get_text():
#                     title_elem = doc.select_one(
#                         "body > table tr > td:nth-child(2) > p i"
#                     )
#                     if title_elem:
#                         item["title"] = "Review of " + title_elem.get_text(strip=True)
#                     author_text = review_elem.get_text()[
#                         10:
#                     ].strip()  # Skip "Review by "
#                 else:
#                     title_elem = doc.select_one(
#                         "body > table tr > td:nth-child(2) > p b"
#                     )
#                     if title_elem:
#                         item["title"] = title_elem.get_text(strip=True)
#                     author_elem = doc.select_one(
#                         "body > table tr > td:nth-child(2) > p:first-child"
#                     )
#                     if author_elem:
#                         lines = author_elem.get_text().split("\n")
#                         author_text = lines[1].strip() if len(lines) > 1 else ""
# 
#                 if author_text:
#                     item["creators"].append(self._clean_author(author_text, "author"))
# 
#         # Add snapshot attachment
#         item["attachments"].append(
#             {"title": "Snapshot", "mimeType": "text/html", "url": url}
#         )
# 
#         return item
# 
#     def _clean_author(self, name: str, creator_type: str) -> Dict[str, Any]:
#         """
#         Parse author name into first and last name.
# 
#         Args:
#             name: Full author name
#             creator_type: Type of creator (author, editor, etc.)
# 
#         Returns:
#             Dictionary with firstName, lastName, and creatorType
#         """
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
#             return {"lastName": name, "creatorType": creator_type, "fieldMode": True}

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/Common_Place.py
# --------------------------------------------------------------------------------
