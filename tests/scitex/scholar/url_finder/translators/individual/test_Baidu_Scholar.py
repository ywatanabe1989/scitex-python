# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/Baidu_Scholar.py
# --------------------------------------------------------------------------------
# """
# Baidu Scholar Translator
# 
# Translates Baidu Scholar academic search results to Zotero format.
# 
# Metadata:
#     translatorID: e034d9be-c420-42cf-8311-23bca5735a32
#     label: Baidu Scholar
#     creator: Philipp Zumstein
#     target: ^https?://(www\\.)?xueshu\\.baidu\\.com/
#     minVersion: 3.0
#     priority: 100
#     inRepository: True
#     translatorType: 4
#     browserSupport: gcsibv
#     lastUpdated: 2021-06-16 17:43:54
# """
# 
# from typing import Dict, Any, List, Optional
# from bs4 import BeautifulSoup
# import re
# 
# 
# class BaiduScholarTranslator:
#     """Translator for Baidu Scholar (百度学术) academic search."""
# 
#     METADATA = {
#         "translatorID": "e034d9be-c420-42cf-8311-23bca5735a32",
#         "label": "Baidu Scholar",
#         "creator": "Philipp Zumstein",
#         "target": r"^https?://(www\.)?xueshu\.baidu\.com/",
#         "minVersion": "3.0",
#         "priority": 100,
#         "inRepository": True,
#         "translatorType": 4,
#         "browserSupport": "gcsibv",
#         "lastUpdated": "2021-06-16 17:43:54",
#     }
# 
#     def detect_web(self, doc: BeautifulSoup, url: str) -> str:
#         """
#         Detect if the page is a single paper or search results.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             url: URL of the page
# 
#         Returns:
#             'journalArticle' for single item, 'multiple' for list, empty string otherwise
#         """
#         if "paperid=" in url:
#             return "journalArticle"
#         elif self._get_search_results(doc, check_only=True):
#             return "multiple"
#         return ""
# 
#     def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
#         """
#         Extract paper data from Baidu Scholar page.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             url: URL of the page
# 
#         Returns:
#             List of dictionaries containing paper metadata
#         """
#         if self.detect_web(doc, url) == "multiple":
#             results = self._get_search_results(doc, check_only=False)
#             return [{"itemType": "multiple", "urls": results}]
#         else:
#             return [self.scrape(doc, url)]
# 
#     def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
#         """
#         Scrape paper data from the document.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             url: URL of the page
# 
#         Returns:
#             Dictionary containing paper metadata
#         """
#         item = {
#             "itemType": "journalArticle",
#             "creators": [],
#             "tags": [],
#             "attachments": [],
#         }
# 
#         # Extract paper ID from URL
#         paper_id_match = re.search(r"paperid=([a-z0-9]+)", url, re.IGNORECASE)
#         if not paper_id_match:
#             return item
# 
#         paper_id = paper_id_match.group(1)
# 
#         # Extract title - remove "百度学术" suffix
#         title_elem = doc.find("title")
#         if title_elem:
#             title = title_elem.get_text().replace("_百度学术", "").strip()
#             item["title"] = title
# 
#         # Extract tags/keywords
#         keyword_elems = doc.select("p.kw_main span a")
#         for elem in keyword_elems:
#             tag_text = elem.get_text().strip()
#             if tag_text:
#                 item["tags"].append({"tag": tag_text})
# 
#         # Extract data URL (actual paper URL)
#         data_url_elem = doc.select_one("i.reqdata")
#         if data_url_elem and data_url_elem.get("url"):
#             item["url"] = data_url_elem["url"]
# 
#         # Extract DOI from link
#         doi_link = doc.select_one('a.dl_item[data-url*="doi.org/"]')
#         if doi_link and doi_link.get("data-url"):
#             doi_url = doi_link["data-url"]
#             doi_start = doi_url.find("doi.org/") + 8
#             if doi_start > 7:
#                 item["DOI"] = doi_url[doi_start:]
# 
#         # Extract abstract
#         abstract_elem = doc.select_one("div.sc_abstract")
#         if not abstract_elem:
#             abstract_elem = doc.select_one("p.abstract")
#         if abstract_elem:
#             item["abstractNote"] = abstract_elem.get_text().strip()
# 
#         # Extract authors
#         author_elems = doc.select("p.author_text a")
#         for author_elem in author_elems:
#             author_name = author_elem.get_text().strip()
#             creator = self._clean_author(author_name)
#             item["creators"].append(creator)
# 
#         # Extract publication title
#         journal_elem = doc.select_one("a.journal_title")
#         if journal_elem and journal_elem.get("title"):
#             item["publicationTitle"] = journal_elem["title"]
# 
#         # Extract date
#         date_elem = doc.select_one("div.year_wr p.kw_main")
#         if date_elem:
#             item["date"] = date_elem.get_text().strip()
# 
#         # Extract DOI from text if not already found
#         if "DOI" not in item:
#             doi_elem = doc.select_one("div.doi_wr p.kw_main")
#             if doi_elem:
#                 item["DOI"] = doi_elem.get_text().strip()
# 
#         # Add snapshot attachment
#         item["attachments"].append(
#             {"title": "Snapshot", "mimeType": "text/html", "url": url}
#         )
# 
#         return item
# 
#     def _get_search_results(
#         self, doc: BeautifulSoup, check_only: bool = False
#     ) -> Optional[Dict[str, str]]:
#         """
#         Get search results from a page.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             check_only: If True, just check if results exist
# 
#         Returns:
#             Dictionary mapping URLs to titles, or True/None if check_only
#         """
#         rows = doc.select('h3 > a[href*="show?paperid="]')
# 
#         if not rows:
#             return None
# 
#         if check_only:
#             return True
# 
#         items = {}
#         for row in rows:
#             href = row.get("href")
#             title = row.get_text().strip()
#             if href and title:
#                 items[href] = title
# 
#         return items if items else None
# 
#     def _clean_author(self, name: str) -> Dict[str, Any]:
#         """
#         Parse author name into first and last name.
#         Handles Chinese names where first character is last name.
# 
#         Args:
#             name: Full author name
# 
#         Returns:
#             Dictionary with firstName and lastName
#         """
#         name = name.strip()
# 
#         # Check if name contains Latin characters or spaces
#         has_latin = bool(re.search(r"[A-Za-z]", name))
#         has_space = " " in name
# 
#         if has_latin or has_space:
#             # Western name format
#             parts = name.split()
#             if len(parts) >= 2:
#                 return {
#                     "firstName": " ".join(parts[:-1]),
#                     "lastName": parts[-1],
#                     "creatorType": "author",
#                 }
#             else:
#                 return {"lastName": name, "creatorType": "author", "fieldMode": True}
#         else:
#             # Chinese name: first character is last name, rest is first name
#             if len(name) > 1:
#                 return {
#                     "firstName": name[1:],
#                     "lastName": name[0],
#                     "creatorType": "author",
#                 }
#             else:
#                 return {"lastName": name, "creatorType": "author", "fieldMode": True}

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/Baidu_Scholar.py
# --------------------------------------------------------------------------------
