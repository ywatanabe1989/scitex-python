# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/Chicago_Journal_of_Theoretical_Computer_Science.py
# --------------------------------------------------------------------------------
# """
# Chicago Journal of Theoretical Computer Science Translator
# 
# Translates journal articles from CJTCS to Zotero format.
# 
# Metadata:
#     translatorID: 1e2a9aba-eb04-4398-9e3a-630e6132db13
#     label: Chicago Journal of Theoretical Computer Science
#     creator: Morgan Shirley
#     target: ^https?://cjtcs\.cs\.uchicago\.edu/articles
#     minVersion: 5.0
#     priority: 100
#     inRepository: True
#     translatorType: 4
#     browserSupport: gcsibv
#     lastUpdated: 2025-05-14 00:10:19
# """
# 
# from typing import Dict, Any
# from bs4 import BeautifulSoup
# import re
# 
# 
# class ChicagoJournalOfTheoreticalComputerScienceTranslator:
#     """Translator for Chicago Journal of Theoretical Computer Science."""
# 
#     METADATA = {
#         "translatorID": "1e2a9aba-eb04-4398-9e3a-630e6132db13",
#         "label": "Chicago Journal of Theoretical Computer Science",
#         "creator": "Morgan Shirley",
#         "target": r"^https?://cjtcs\.cs\.uchicago\.edu/articles",
#         "minVersion": "5.0",
#         "priority": 100,
#         "inRepository": True,
#         "translatorType": 4,
#         "browserSupport": "gcsibv",
#         "lastUpdated": "2025-05-14 00:10:19",
#     }
# 
#     SINGLE_RE = re.compile(
#         r"^https?://cjtcs\.cs\.uchicago\.edu/articles/\w+/\d+/contents\.html"
#     )
#     MULTIPLE_RE = re.compile(
#         r"^https?://cjtcs\.cs\.uchicago\.edu/articles/\w+/contents\.html"
#     )
# 
#     def detect_web(self, doc: BeautifulSoup, url: str) -> str:
#         """Detect if page is an article or volume contents."""
#         if self.MULTIPLE_RE.match(url):
#             # Check for article links
#             article_links = doc.select('ul > li > ul > li a[href*="contents.html"]')
#             if article_links:
#                 return "multiple"
#         elif self.SINGLE_RE.match(url):
#             return "journalArticle"
# 
#         return ""
# 
#     async def do_web(
#         self, doc: BeautifulSoup, url: str, request_text_func
#     ) -> Dict[str, Any]:
#         """Extract article data."""
#         return await self.scrape(doc, url, request_text_func)
# 
#     async def scrape(
#         self, doc: BeautifulSoup, url: str, request_text_func
#     ) -> Dict[str, Any]:
#         """
#         Scrape article data by fetching and parsing BibTeX.
# 
#         Args:
#             doc: BeautifulSoup parsed document
#             url: URL of the page
#             request_text_func: Function to request text content
# 
#         Returns:
#             Dictionary containing article metadata
#         """
#         item = {
#             "itemType": "journalArticle",
#             "creators": [],
#             "tags": [],
#             "attachments": [],
#         }
# 
#         # Find BibTeX link
#         bib_link = doc.select_one('a[href$=".bib"]')
#         if bib_link and bib_link.get("href"):
#             bib_url = bib_link["href"]
#             if not bib_url.startswith("http"):
#                 bib_url = "http://cjtcs.cs.uchicago.edu" + bib_url
# 
#             # Fetch BibTeX
#             bib_text = await request_text_func(bib_url)
#             if bib_text:
#                 # Fix author line if needed
#                 bib_text = self._fix_author_line(bib_text)
#                 # Parse BibTeX
#                 self._parse_bibtex(bib_text, item)
# 
#         # Extract DOI
#         doi_link = doc.select_one('a[href*="dx.doi.org"]')
#         if doi_link and doi_link.get("href"):
#             doi_url = doi_link["href"]
#             item["DOI"] = re.sub(r"https?://(?:dx\.)?doi\.org/", "", doi_url)
# 
#         # Add PDF attachment if available
#         pdf_link = doc.select_one('a[href$=".pdf"]')
#         if pdf_link and pdf_link.get("href"):
#             pdf_url = pdf_link["href"]
#             if not pdf_url.startswith("http"):
#                 pdf_url = "http://cjtcs.cs.uchicago.edu" + pdf_url
#             item["attachments"].append(
#                 {
#                     "url": pdf_url,
#                     "title": "Full Text PDF",
#                     "mimeType": "application/pdf",
#                 }
#             )
#         else:
#             # Add snapshot if no PDF
#             item["attachments"].append(
#                 {"title": "Snapshot", "mimeType": "text/html", "url": url}
#             )
# 
#         return item
# 
#     def _fix_author_line(self, bib_text: str) -> str:
#         """Fix BibTeX author line if missing equals sign."""
#         lines = bib_text.split("\n")
#         fixed_lines = []
#         for line in lines:
#             # Add equals sign after "author" if missing
#             fixed_line = re.sub(r"^(\s*author)(?!\s*=)", r"\1=", line)
#             fixed_lines.append(fixed_line)
#         return "\n".join(fixed_lines)
# 
#     def _parse_bibtex(self, bib_text: str, item: Dict[str, Any]):
#         """Parse BibTeX and fill item fields."""
#         # Simple BibTeX parser
#         # Extract title
#         title_match = re.search(
#             r'title\s*=\s*[{"](.*?)[}"]', bib_text, re.IGNORECASE | re.DOTALL
#         )
#         if title_match:
#             item["title"] = title_match.group(1).strip()
# 
#         # Extract authors
#         author_match = re.search(
#             r'author\s*=\s*[{"](.*?)[}"]', bib_text, re.IGNORECASE | re.DOTALL
#         )
#         if author_match:
#             authors = author_match.group(1).strip()
#             # Split by 'and'
#             for author in authors.split(" and "):
#                 author = author.strip()
#                 if author:
#                     item["creators"].append(self._parse_author(author))
# 
#         # Extract year/date
#         year_match = re.search(r'year\s*=\s*[{"](.*?)[}"]', bib_text, re.IGNORECASE)
#         if year_match:
#             item["date"] = year_match.group(1).strip()
# 
#         # Extract volume
#         volume_match = re.search(r'volume\s*=\s*[{"](.*?)[}"]', bib_text, re.IGNORECASE)
#         if volume_match:
#             item["volume"] = volume_match.group(1).strip()
# 
#         # Extract issue/number
#         issue_match = re.search(
#             r'(?:issue|number)\s*=\s*[{"](.*?)[}"]', bib_text, re.IGNORECASE
#         )
#         if issue_match:
#             item["issue"] = issue_match.group(1).strip()
# 
#         # Extract journal
#         journal_match = re.search(
#             r'journal\s*=\s*[{"](.*?)[}"]', bib_text, re.IGNORECASE
#         )
#         if journal_match:
#             item["publicationTitle"] = journal_match.group(1).strip()
# 
#     def _parse_author(self, author_str: str) -> Dict[str, Any]:
#         """Parse author name from BibTeX format."""
#         author_str = author_str.strip()
# 
#         # Handle "Last, First" format
#         if "," in author_str:
#             parts = author_str.split(",", 1)
#             return {
#                 "firstName": parts[1].strip() if len(parts) > 1 else "",
#                 "lastName": parts[0].strip(),
#                 "creatorType": "author",
#             }
#         else:
#             # Handle "First Last" format
#             parts = author_str.split()
#             if len(parts) >= 2:
#                 return {
#                     "firstName": " ".join(parts[:-1]),
#                     "lastName": parts[-1],
#                     "creatorType": "author",
#                 }
#             else:
#                 return {
#                     "lastName": author_str,
#                     "creatorType": "author",
#                     "fieldMode": True,
#                 }

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/Chicago_Journal_of_Theoretical_Computer_Science.py
# --------------------------------------------------------------------------------
