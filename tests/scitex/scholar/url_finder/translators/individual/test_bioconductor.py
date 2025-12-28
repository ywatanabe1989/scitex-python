# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/bioconductor.py
# --------------------------------------------------------------------------------
# """
# Bioconductor Translator
# 
# Translates Bioconductor package information.
# 
# Metadata:
#     translatorID: 21f62926-4343-4518-b6f2-a284e650e64a
#     label: Bioconductor
#     creator: Qiang Hu
#     target: https?://(www\.)?bioconductor\.org/(packages/.*/bioc/html|help/search)/
#     minVersion: 3.0
#     priority: 100
#     inRepository: True
#     translatorType: 4
#     browserSupport: gcsibv
#     lastUpdated: 2019-09-17 16:47:07
# """
# 
# from typing import Dict, Any, List, Optional
# from bs4 import BeautifulSoup
# import re
# 
# 
# class BioconductorTranslator:
#     """Translator for Bioconductor package pages."""
# 
#     METADATA = {
#         "translatorID": "21f62926-4343-4518-b6f2-a284e650e64a",
#         "label": "Bioconductor",
#         "creator": "Qiang Hu",
#         "target": r"https?://(www\.)?bioconductor\.org/(packages/.*/bioc/html|help/search)/",
#         "minVersion": "3.0",
#         "priority": 100,
#         "inRepository": True,
#         "translatorType": 4,
#         "browserSupport": "gcsibv",
#         "lastUpdated": "2019-09-17 16:47:07",
#     }
# 
#     def detect_web(self, doc: BeautifulSoup, url: str) -> str:
#         """Detect page type."""
#         if "/bioc/html/" in url:
#             return "computerProgram"
#         elif "/search/index.html" in url and self._get_search_results(
#             doc, check_only=True
#         ):
#             return "multiple"
#         return ""
# 
#     def _get_search_results(
#         self, doc: BeautifulSoup, check_only: bool = False
#     ) -> Dict[str, str]:
#         """Get search results."""
#         items = {}
#         rows = doc.select('dl > dt > a[href*="/bioc/html/"]')
# 
#         for row in rows:
#             href = row.get("href")
#             title = row.get_text(strip=True)
#             if not href or not title:
#                 continue
#             if check_only:
#                 return {"found": True}
#             items[href] = title
# 
#         return items
# 
#     def do_web(self, doc: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
#         """Extract data from the page."""
#         page_type = self.detect_web(doc, url)
# 
#         if page_type == "multiple":
#             items = self._get_search_results(doc, check_only=False)
#             return [{"url": u} for u in items.keys()]
#         elif page_type == "computerProgram":
#             return [self.scrape(doc, url)]
#         return []
# 
#     def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
#         """Scrape package data from the document."""
#         item = {
#             "itemType": "computerProgram",
#             "libraryCatalog": "Bioconductor",
#             "url": url,
#             "creators": [],
#             "tags": [],
#             "attachments": [],
#         }
# 
#         # Extract title
#         title_h1 = doc.select_one("#PageContent > h1")
#         if title_h1:
#             item["title"] = title_h1.get_text(strip=True)
# 
#             # Add subtitle if present
#             subtitle_h2 = doc.select_one("#PageContent > div.do_not_rebase > h2")
#             if subtitle_h2:
#                 item["title"] += ": " + subtitle_h2.get_text(strip=True)
# 
#         # Extract DOI
#         doi_link = doc.select_one('#PageContent div:nth-of-type(2) a[href*="doi.org"]')
#         if doi_link:
#             doi_text = doi_link.get_text(strip=True)
#             item["extra"] = "DOI: " + doi_text
# 
#         # Extract version, company, and abstract
#         paragraphs = doc.select("#PageContent > div.do_not_rebase > p")
#         for i, p in enumerate(paragraphs):
#             text = p.get_text(strip=True)
# 
#             if text.startswith("Bioconductor version:"):
#                 item["company"] = text
# 
#                 # Next paragraph is abstract
#                 if i + 1 < len(paragraphs):
#                     item["abstractNote"] = paragraphs[i + 1].get_text(strip=True)
# 
#             elif text.startswith("Author"):
#                 # Parse authors
#                 author_text = text.replace("Author:", "").strip()
#                 # Remove role indicators in brackets/parentheses
#                 author_text = re.sub(r"\[.+?\]", "", author_text)
#                 author_text = re.sub(r"\(.+?\)", "", author_text)
#                 # Split by comma or "and"
#                 authors = re.split(r",|and\s*", author_text)
# 
#                 for author_name in authors:
#                     author_name = author_name.strip()
#                     if author_name:
#                         item["creators"].append(
#                             self._clean_author(author_name, "programmer")
#                         )
# 
#         # Extract version number from table
#         version_cell = doc.select_one(
#             'table tbody tr td:-soup-contains("Version") + td'
#         )
#         if version_cell:
#             item["versionNumber"] = version_cell.get_text(strip=True)
# 
#         # Extract license
#         license_cell = doc.select_one(
#             'table tbody tr td:-soup-contains("License") + td'
#         )
#         if license_cell:
#             item["rights"] = license_cell.get_text(strip=True)
# 
#         # Extract short URL
#         url_cell = doc.select_one(
#             'table tbody tr td:-soup-contains("Package Short Url") + td'
#         )
#         if url_cell:
#             item["url"] = url_cell.get_text(strip=True)
# 
#         # Extract year from copyright notice
#         footer = doc.select_one('#SiteGlobalFooter div p:-soup-contains("Copyright")')
#         if footer:
#             footer_text = footer.get_text()
#             years = re.findall(r"\d{4}", footer_text)
#             if len(years) > 1:
#                 item["date"] = years[1]  # Second year is usually current
# 
#         # Extract tags from biocViews
#         tag_links = doc.select('td:-soup-contains("biocViews") + td a')
#         for tag_link in tag_links:
#             tag_text = tag_link.get_text(strip=True)
#             if tag_text:
#                 item["tags"].append({"tag": tag_text})
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
#             return {"lastName": name, "creatorType": creator_type, "fieldMode": True}

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/bioconductor.py
# --------------------------------------------------------------------------------
