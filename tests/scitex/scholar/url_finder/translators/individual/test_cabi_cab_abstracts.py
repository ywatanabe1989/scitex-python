# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/cabi_cab_abstracts.py
# --------------------------------------------------------------------------------
# """
# CABI - CAB Abstracts Translator
# 
# Translates CABI CAB Abstracts articles to Zotero format.
# 
# Metadata:
#     translatorID: a29d22b3-c2e4-4cc0-ace4-6c2326144332
#     label: CABI - CAB Abstracts
#     creator: Sebastian Karcher
#     target: ^https?://(www\.)?cabidirect\.org/cabdirect
#     minVersion: 3.0.4
#     priority: 100
#     inRepository: True
#     translatorType: 4
#     browserSupport: gcsibv
#     lastUpdated: 2017-06-14 03:41:30
# """
# 
# from typing import Dict, Any, Optional
# from bs4 import BeautifulSoup
# import re
# 
# 
# class CABICABAbstractsTranslator:
#     """Translator for CABI CAB Abstracts database."""
# 
#     METADATA = {
#         "translatorID": "a29d22b3-c2e4-4cc0-ace4-6c2326144332",
#         "label": "CABI - CAB Abstracts",
#         "creator": "Sebastian Karcher",
#         "target": r"^https?://(www\.)?cabidirect\.org/cabdirect",
#         "minVersion": "3.0.4",
#         "priority": 100,
#         "inRepository": True,
#         "translatorType": 4,
#         "browserSupport": "gcsibv",
#         "lastUpdated": "2017-06-14 03:41:30",
#     }
# 
#     def detect_web(self, doc: BeautifulSoup, url: str) -> str:
#         """
#         Detect if the page is a CAB Abstracts article or search results.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             url: URL of the page
# 
#         Returns:
#             'journalArticle' for articles, 'multiple' for search, empty string otherwise
#         """
#         if "cabdirect/abstract/" in url or "cabdirect/FullTextPDF/" in url:
#             return "journalArticle"
#         if "cabdirect/search" in url and self.get_search_results(doc, True):
#             return "multiple"
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
#             Dictionary mapping URLs to titles, or empty dict
#         """
#         items = {}
#         rows = doc.select('div.list-content h2 a[href*="/abstract/"]')
# 
#         for row in rows:
#             href = row.get("href")
#             title = row.get_text(strip=True)
#             if not href or not title:
#                 continue
#             if check_only:
#                 return {"found": "true"}
#             items[href] = title
# 
#         return items
# 
#     def do_web(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
#         """
#         Extract article data from CAB Abstracts page.
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
#         item = {
#             "itemType": "journalArticle",
#             "url": url,
#             "creators": [],
#             "tags": [],
#             "attachments": [],
#             "libraryCatalog": "CABI - CAB Abstracts",
#             "archive": "",  # We don't want CAB in archive field
#         }
# 
#         # Extract PDF URL
#         pdf_elem = doc.select_one('p.btnabstract a[href$=".pdf"]')
#         if pdf_elem:
#             pdf_url = pdf_elem.get("href")
#             if pdf_url:
#                 item["attachments"].append(
#                     {
#                         "title": "Full Text PDF",
#                         "mimeType": "application/pdf",
#                         "url": pdf_url,
#                     }
#                 )
# 
#         # Extract abstract
#         abstract_elem = doc.select_one("div.abstract")
#         if abstract_elem:
#             item["abstractNote"] = abstract_elem.get_text(strip=True)
# 
#         # Extract editors
#         editor_elems = doc.select('p[id*="ulEditors"] a')
#         for editor in editor_elems:
#             editor_text = editor.get_text(strip=True)
#             if editor_text:
#                 item["creators"].append(self._clean_author(editor_text, "editor"))
# 
#         # Add snapshot attachment
#         item["attachments"].append(
#             {"title": "Snapshot", "mimeType": "text/html", "url": url}
#         )
# 
#         # Note: The actual translator uses RIS import which we can't fully replicate here
#         # This is a simplified version that extracts what we can from the HTML
# 
#         return item
# 
#     def _clean_author(self, name: str, creator_type: str = "author") -> Dict[str, Any]:
#         """
#         Parse author name into first and last name.
# 
#         Args:
#             name: Full author name
#             creator_type: Type of creator (author, editor, etc.)
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
#                 "creatorType": creator_type,
#             }
#         else:
#             return {"lastName": name, "creatorType": creator_type, "fieldMode": True}

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/cabi_cab_abstracts.py
# --------------------------------------------------------------------------------
