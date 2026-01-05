# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/Bangkok_Post.py
# --------------------------------------------------------------------------------
# """
# Bangkok Post Translator
# 
# Translates Bangkok Post articles to Zotero format.
# 
# Metadata:
#     translatorID: 7f74d823-d2ba-481c-b717-8b12c90ed874
#     label: Bangkok Post
#     creator: Matt Mayer
#     target: ^https://www\\.bangkokpost\\.com/[a-z0-9-]+/([a-z0-9-]+/)?[0-9]+
#     minVersion: 3.0
#     priority: 100
#     inRepository: True
#     translatorType: 4
#     browserSupport: gcsibv
#     lastUpdated: 2024-06-18 20:46:45
# """
# 
# from typing import Dict, Any, Optional
# from bs4 import BeautifulSoup
# 
# 
# class BangkokPostTranslator:
#     """Translator for Bangkok Post newspaper articles."""
# 
#     METADATA = {
#         "translatorID": "7f74d823-d2ba-481c-b717-8b12c90ed874",
#         "label": "Bangkok Post",
#         "creator": "Matt Mayer",
#         "target": r"^https://www\.bangkokpost\.com/[a-z0-9-]+/([a-z0-9-]+/)?[0-9]+",
#         "minVersion": "3.0",
#         "priority": 100,
#         "inRepository": True,
#         "translatorType": 4,
#         "browserSupport": "gcsibv",
#         "lastUpdated": "2024-06-18 20:46:45",
#     }
# 
#     def detect_web(self, doc: BeautifulSoup, url: str) -> str:
#         """
#         Detect if the page is a Bangkok Post article.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             url: URL of the page
# 
#         Returns:
#             'newspaperArticle' if detected, empty string otherwise
#         """
#         return "newspaperArticle"
# 
#     def do_web(self, doc: BeautifulSoup, url: str) -> Dict[str, Any]:
#         """
#         Extract article data from Bangkok Post page.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             url: URL of the page
# 
#         Returns:
#             Dictionary containing article metadata
#         """
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
#             "itemType": "newspaperArticle",
#             "publicationTitle": "Bangkok Post",
#             "url": url,
#             "creators": [],
#             "tags": [],
#             "attachments": [],
#         }
# 
#         # Extract title from meta tags
#         title_tag = doc.find("meta", {"property": "og:title"})
#         if title_tag and title_tag.get("content"):
#             item["title"] = title_tag["content"]
# 
#         # Extract author - try multiple sources
#         author_tag = doc.find("meta", {"name": "lead:author"})
#         if author_tag and author_tag.get("content"):
#             author_name = author_tag["content"]
#             item["creators"].append(self._clean_author(author_name))
#         else:
#             # Try alternate source
#             author_elem = doc.select_one(".info-opinion .columnnist-name a")
#             if author_elem and author_elem.get_text():
#                 author_name = author_elem.get_text().strip()
#                 item["creators"].append(self._clean_author(author_name))
# 
#         # Extract date - format like 2020-09-07T17:37:00+07:00
#         date_tag = doc.find("meta", {"name": "lead:published_at"})
#         if date_tag and date_tag.get("content"):
#             date_str = date_tag["content"]
#             # Extract YYYY-MM-DD from the timestamp
#             item["date"] = date_str[:10] if len(date_str) >= 10 else date_str
# 
#         # Extract abstract
#         abstract_tag = doc.find("meta", {"property": "og:description"})
#         if abstract_tag and abstract_tag.get("content"):
#             item["abstractNote"] = abstract_tag["content"]
# 
#         # Extract language
#         lang_tag = doc.find("meta", {"property": "og:locale"})
#         if lang_tag and lang_tag.get("content"):
#             item["language"] = lang_tag["content"]
# 
#         # Extract tags from keywords
#         keywords_tag = doc.find("meta", {"name": "keywords"})
#         if keywords_tag and keywords_tag.get("content"):
#             keywords = keywords_tag["content"].split(",")
#             item["tags"] = [{"tag": k.strip()} for k in keywords if k.strip()]
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/Bangkok_Post.py
# --------------------------------------------------------------------------------
