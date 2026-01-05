# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/delpher.py
# --------------------------------------------------------------------------------
# """
# Delpher Translator
# 
# Translates Delpher (Dutch digital library) items to Zotero format.
# 
# Metadata:
#     translatorID: c4008cc5-9243-4d13-8b35-562cdd184558
#     label: Delpher
#     creator: Philipp Zumstein
#     target: ^https?://[^\\/]+\\.delpher\\.nl
#     minVersion: 3.0
#     priority: 100
#     inRepository: True
#     translatorType: 4
#     browserSupport: gcsibv
#     lastUpdated: 2022-01-20 14:35:30
# """
# 
# from typing import Dict, Any, Optional
# from bs4 import BeautifulSoup
# import re
# 
# 
# class DelpherTranslator:
#     """Translator for Delpher Dutch digital library."""
# 
#     METADATA = {
#         "translatorID": "c4008cc5-9243-4d13-8b35-562cdd184558",
#         "label": "Delpher",
#         "creator": "Philipp Zumstein",
#         "target": r"^https?://[^\/]+\.delpher\.nl",
#         "minVersion": "3.0",
#         "priority": 100,
#         "inRepository": True,
#         "translatorType": 4,
#         "browserSupport": "gcsibv",
#         "lastUpdated": "2022-01-20 14:35:30",
#     }
# 
#     def detect_web(self, doc: BeautifulSoup, url: str) -> str:
#         """Detect item type based on URL."""
#         if "/view" in url:
#             if "/boeken/" in url:
#                 return "book"
#             elif "/tijdschriften/" in url:
#                 return "journalArticle"
#             elif "/kranten/" in url:
#                 return "newspaperArticle"
#             elif "/radiobulletins/" in url:
#                 return "radioBroadcast"
#         elif self.get_search_results(doc, check_only=True):
#             return "multiple"
#         return ""
# 
#     def get_search_results(
#         self, doc: BeautifulSoup, check_only: bool = False
#     ) -> Optional[Dict[str, str]]:
#         """Extract search results."""
#         items = {}
#         rows = doc.select('article a.search-result__link[href^="/"]')
# 
#         for row in rows:
#             href = row.get("href")
#             title = row.get_text(strip=True)
# 
#             if not href or not title:
#                 continue
# 
#             if check_only:
#                 return {"found": "true"}
# 
#             items[href] = title
# 
#         return items if items else None
# 
#     def do_web(self, doc: BeautifulSoup, url: str) -> Any:
#         """Main extraction method."""
#         page_type = self.detect_web(doc, url)
# 
#         if page_type == "multiple":
#             return self.get_search_results(doc, check_only=False)
#         else:
#             return self.scrape(doc, url)
# 
#     def _get_metadata_field(
#         self, details: BeautifulSoup, field_name: str
#     ) -> Optional[str]:
#         """Extract metadata field value from details section."""
#         dt = details.find(
#             "dt",
#             class_="metadata__details-text",
#             string=lambda x: x and field_name in x,
#         )
#         if dt:
#             dd = dt.find_next_sibling("dd")
#             if dd:
#                 a_tag = dd.find("a")
#                 if a_tag:
#                     return a_tag.get_text(strip=True)
#                 return dd.get_text(strip=True)
#         return None
# 
#     def scrape(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
#         """Scrape item metadata."""
#         item_type = self.detect_web(doc, url)
#         item = {
#             "itemType": item_type,
#             "url": url,
#             "creators": [],
#             "tags": [],
#             "attachments": [],
#         }
# 
#         # Find metadata section
#         details = doc.select_one("dl.metadata__details-description-list")
#         if not details:
#             return item
# 
#         # Extract title
#         title = self._get_metadata_field(details, "Titel") or self._get_metadata_field(
#             details, "Kop"
#         )
#         if not title:
#             title = self._get_metadata_field(details, "Krantentitel")
#         if title:
#             item["title"] = title
# 
#         # Extract date
#         date = self._get_metadata_field(
#             details, "Publicatiedatum"
#         ) or self._get_metadata_field(details, "Datum")
#         if not date:
#             date = self._get_metadata_field(details, "Jaar van uitgave")
#         if date and len(date) > 4:
#             # Convert DD-MM-YYYY to YYYY-MM-DD
#             date = re.sub(r"(\d{2})-(\d{2})-(\d{4})", r"\3-\2-\1", date)
#         if date:
#             item["date"] = date
# 
#         # Extract other fields
#         item["libraryCatalog"] = (
#             self._get_metadata_field(details, "Herkomst") or "Delpher"
#         )
#         item["publisher"] = self._get_metadata_field(
#             details, "Drukker/Uitgever"
#         ) or self._get_metadata_field(details, "Uitgever")
#         item["callNumber"] = self._get_metadata_field(details, "PPN")
#         item["language"] = self._get_metadata_field(details, "Taal")
#         item["place"] = self._get_metadata_field(details, "Plaats van uitgave")
#         item["numPages"] = self._get_metadata_field(details, "Omvang")
# 
#         if item_type in ["newspaperArticle", "journalArticle"]:
#             item["publicationTitle"] = self._get_metadata_field(details, "Krantentitel")
#             item["issue"] = self._get_metadata_field(details, "Aflevering")
#             item["edition"] = self._get_metadata_field(details, "Editie")
# 
#         # Extract authors
#         author_dts = details.find_all(
#             "dt",
#             class_="metadata__details-text",
#             string=lambda x: x and ("Auteur" in x or "Coauteur" in x),
#         )
#         for dt in author_dts:
#             dd = dt.find_next_sibling("dd")
#             if dd:
#                 author_text = dd.get_text(strip=True)
#                 names = author_text.split()
#                 if len(names) >= 2:
#                     item["creators"].append(
#                         {
#                             "firstName": " ".join(names[:-1]),
#                             "lastName": names[-1],
#                             "creatorType": "author",
#                         }
#                     )
#                 else:
#                     item["creators"].append(
#                         {
#                             "lastName": author_text,
#                             "creatorType": "author",
#                             "fieldMode": True,
#                         }
#                     )
# 
#         # Extract tags
#         tag_dds = details.select('dt:contains("Onderwerp") ~ dd')
#         for dd in tag_dds:
#             tag_links = dd.find_all("a")
#             for link in tag_links:
#                 tag_text = link.get_text(strip=True)
#                 if tag_text:
#                     item["tags"].append({"tag": tag_text})
# 
#         # Extract URL from share input
#         url_input = doc.select_one(
#             "input.object-view-menu__share-links-details-input:last-of-type"
#         )
#         if url_input and url_input.get("value"):
#             item["url"] = url_input["value"]
# 
#         # Add attachments
#         item["attachments"].append(
#             {"title": "Snapshot", "mimeType": "text/html", "url": url}
#         )
# 
#         # Add PDF if available
#         pdf_link = doc.select_one('a.object-view-menu__downloads-link:contains("pdf")')
#         if pdf_link and pdf_link.get("href"):
#             item["attachments"].append(
#                 {
#                     "title": "Full Text PDF",
#                     "mimeType": "application/pdf",
#                     "url": pdf_link["href"],
#                 }
#             )
# 
#         # Add JPG if available
#         jpg_link = doc.select_one('a.object-view-menu__downloads-link:contains("jpg")')
#         if jpg_link and jpg_link.get("href"):
#             item["attachments"].append(
#                 {"title": "Image", "mimeType": "image/jpeg", "url": jpg_link["href"]}
#             )
# 
#         return item

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/delpher.py
# --------------------------------------------------------------------------------
