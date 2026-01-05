# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/pubmed.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """PubMed translator - extracts metadata from PubMed articles via E-utilities API.
# 
# PubMed is the primary free search interface for MEDLINE, the bibliographic database
# of life sciences and biomedical information maintained by the National Library of Medicine.
# 
# Strategy:
# 1. Extract PMID (PubMed ID) from page metadata or URL
# 2. Fetch structured XML data via NCBI E-utilities API
# 3. Parse XML to extract article metadata (title, authors, DOI, abstract, etc.)
# 4. Extract PDF URLs from publisher links when available
# """
# 
# import re
# import logging
# import httpx
# from typing import List, Optional, Dict, Any
# from playwright.async_api import Page
# from bs4 import BeautifulSoup
# from ..core.base import BaseTranslator
# 
# logger = logging.getLogger(__name__)
# 
# 
# class PubMedTranslator(BaseTranslator):
#     """PubMed - NCBI's biomedical literature database."""
# 
#     LABEL = "PubMed"
#     # Matches various PubMed URL patterns including:
#     # - pubmed.ncbi.nlm.nih.gov/<pmid>
#     # - ncbi.nlm.nih.gov/pubmed/<pmid>
#     # - pubmed search results and collections
#     # - books database
#     # - myncbi (My NCBI) collections and bibliography
#     URL_TARGET_PATTERN = r"^https?://([^/]+\.)?(www|preview)\.ncbi\.nlm\.nih\.gov[^/]*/(m/)?(books|pubmed|labs/pubmed|myncbi|sites/pubmed|sites/myncbi|sites/entrez|entrez/query\.fcgi\?.*db=PubMed|myncbi/browse/collection/?|myncbi/collections/)|^https?://pubmed\.ncbi\.nlm\.nih\.gov/(\d|\?|searches/|clipboard|collections/)"
# 
#     # NCBI E-utilities API endpoint
#     EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         """Check if URL matches PubMed patterns."""
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pmid_from_page(cls, page: Page) -> Optional[str]:
#         """Extract PMID from page metadata or URL.
# 
#         Strategy (matching JavaScript implementation):
#         1. Check meta tags: ncbi_uidlist, ncbi_article_id, uid
#         2. Check input element with id="absid"
#         3. Check canonical link or handheld link href
#         4. Check PubMed record links in bookshelf entries
#         5. Extract from URL pattern
#         """
#         try:
#             content = await page.content()
#             soup = BeautifulSoup(content, "html.parser")
# 
#             # Strategy 1: meta tags
#             for meta_name in ["ncbi_uidlist", "ncbi_article_id", "uid"]:
#                 meta = soup.find("meta", attrs={"name": meta_name})
#                 if meta and meta.get("content"):
#                     pmid = meta["content"].strip()
#                     if pmid.isdigit():
#                         return pmid
# 
#             # Strategy 2: input element
#             input_elem = soup.find("input", attrs={"id": "absid"})
#             if input_elem and input_elem.get("value"):
#                 pmid = input_elem["value"].strip()
#                 if pmid.isdigit():
#                     return pmid
# 
#             # Strategy 3: link href patterns
#             for rel in ["canonical", "handheld"]:
#                 link = soup.find("link", attrs={"rel": rel})
#                 if link and link.get("href"):
#                     match = re.search(r"/(\d+)(?:/|$)", link["href"])
#                     if match:
#                         return match.group(1)
# 
#             # Strategy 4: bookshelf PubMed record links
#             maincontent = soup.find(id="maincontent")
#             if maincontent:
#                 pubmed_link = maincontent.find(
#                     "a", attrs={"title": re.compile(r"PubMed record")}
#                 )
#                 if pubmed_link:
#                     pmid = pubmed_link.get_text(strip=True)
#                     if pmid.isdigit():
#                         return pmid
# 
#             # Strategy 5: URL extraction
#             url = page.url
#             match = re.search(r"/pubmed/(\d+)", url)
#             if match:
#                 return match.group(1)
# 
#             match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
#             if match:
#                 return match.group(1)
# 
#         except Exception as e:
#             logger.error(f"Error extracting PMID: {e}")
# 
#         return None
# 
#     @classmethod
#     async def fetch_pubmed_metadata(cls, pmid: str) -> Optional[Dict[str, Any]]:
#         """Fetch article metadata from NCBI E-utilities API.
# 
#         Args:
#             pmid: PubMed ID
# 
#         Returns:
#             Dictionary containing article metadata or None if fetch fails
#         """
#         try:
#             params = {
#                 "db": "PubMed",
#                 "tool": "ZoteroTranslatorsPython",
#                 "retmode": "xml",
#                 "rettype": "citation",
#                 "id": pmid,
#             }
# 
#             async with httpx.AsyncClient(timeout=30.0) as client:
#                 response = await client.get(cls.EUTILS_URL, params=params)
#                 response.raise_for_status()
#                 xml_text = response.text
# 
#                 # Check if valid PubMed data was returned (case-insensitive)
#                 if (
#                     "pubmedarticle" not in xml_text.lower()
#                     and "pubmedbookarticle" not in xml_text.lower()
#                 ):
#                     logger.warning(f"No PubMed data found for PMID {pmid}")
#                     return None
# 
#                 # Parse XML to extract metadata
#                 metadata = cls._parse_pubmed_xml(xml_text)
#                 metadata["pmid"] = pmid
#                 return metadata
# 
#         except Exception as e:
#             logger.error(f"Error fetching PubMed metadata for PMID {pmid}: {e}")
#             return None
# 
#     @classmethod
#     def _parse_pubmed_xml(cls, xml_text: str) -> Dict[str, Any]:
#         """Parse PubMed XML to extract article metadata.
# 
#         Args:
#             xml_text: PubMed XML response
# 
#         Returns:
#             Dictionary containing parsed metadata
#         """
#         soup = BeautifulSoup(xml_text, "xml")
#         metadata = {}
# 
#         try:
#             # Extract article title
#             article_title = soup.find("ArticleTitle")
#             if article_title:
#                 metadata["title"] = article_title.get_text(strip=True)
# 
#             # Extract authors
#             authors = []
#             for author in soup.find_all("Author"):
#                 last_name = author.find("LastName")
#                 fore_name = author.find("ForeName")
#                 if last_name:
#                     author_name = last_name.get_text(strip=True)
#                     if fore_name:
#                         author_name = f"{fore_name.get_text(strip=True)} {author_name}"
#                     authors.append(author_name)
#             metadata["authors"] = authors
# 
#             # Extract DOI
#             article_ids = soup.find_all("ArticleId", attrs={"IdType": "doi"})
#             if article_ids:
#                 metadata["doi"] = article_ids[0].get_text(strip=True)
# 
#             # Extract abstract
#             abstract = soup.find("AbstractText")
#             if abstract:
#                 metadata["abstract"] = abstract.get_text(strip=True)
# 
#             # Extract journal information
#             journal = soup.find("Journal")
#             if journal:
#                 journal_title = journal.find("Title")
#                 if journal_title:
#                     metadata["journal"] = journal_title.get_text(strip=True)
# 
#                 journal_abbrev = journal.find("ISOAbbreviation")
#                 if journal_abbrev:
#                     metadata["journal_abbreviation"] = journal_abbrev.get_text(
#                         strip=True
#                     )
# 
#             # Extract publication date
#             pub_date = soup.find("PubDate")
#             if pub_date:
#                 year = pub_date.find("Year")
#                 month = pub_date.find("Month")
#                 day = pub_date.find("Day")
#                 date_parts = []
#                 if year:
#                     date_parts.append(year.get_text(strip=True))
#                 if month:
#                     date_parts.append(month.get_text(strip=True))
#                 if day:
#                     date_parts.append(day.get_text(strip=True))
#                 metadata["date"] = "-".join(date_parts)
# 
#             # Extract volume and issue
#             volume = soup.find("Volume")
#             if volume:
#                 metadata["volume"] = volume.get_text(strip=True)
# 
#             issue = soup.find("Issue")
#             if issue:
#                 metadata["issue"] = issue.get_text(strip=True)
# 
#             # Extract pages
#             medline_pgn = soup.find("MedlinePgn")
#             if medline_pgn:
#                 metadata["pages"] = medline_pgn.get_text(strip=True)
# 
#         except Exception as e:
#             logger.error(f"Error parsing PubMed XML: {e}")
# 
#         return metadata
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         """Extract PDF URLs from PubMed page.
# 
#         PubMed doesn't host PDFs directly but links to publisher sites.
#         This method extracts links to full text on publisher websites.
#         """
#         pdf_urls = []
# 
#         try:
#             content = await page.content()
#             soup = BeautifulSoup(content, "html.parser")
# 
#             # Look for "Full Text Links" section
#             full_text_section = soup.find("div", class_=re.compile(r"full-text-links"))
#             if full_text_section:
#                 links = full_text_section.find_all("a", href=True)
#                 for link in links:
#                     href = link["href"]
#                     # Filter for likely PDF links
#                     if any(x in href.lower() for x in [".pdf", "download", "pdf"]):
#                         pdf_urls.append(href)
# 
#             # Look for LinkOut resources
#             linkout_section = soup.find("div", attrs={"id": "link-out"})
#             if linkout_section:
#                 links = linkout_section.find_all("a", href=True)
#                 for link in links:
#                     href = link["href"]
#                     if "pdf" in href.lower():
#                         pdf_urls.append(href)
# 
#             # PMC (PubMed Central) link - free full text
#             pmc_link = soup.find("a", attrs={"data-ga-action": "PMC article"})
#             if pmc_link and pmc_link.get("href"):
#                 pdf_urls.append(pmc_link["href"])
# 
#         except Exception as e:
#             logger.error(f"Error extracting PDF URLs: {e}")
# 
#         return pdf_urls
# 
# 
# # Demo usage
# async def main():
#     """Demonstrate PubMed translator usage."""
#     from playwright.async_api import async_playwright
# 
#     # Example PubMed URL
#     test_url = "https://pubmed.ncbi.nlm.nih.gov/20729678/"
# 
#     print(f"Testing PubMed translator with: {test_url}")
# 
#     if PubMedTranslator.matches_url(test_url):
#         print("✓ URL matches PubMed pattern")
# 
#         async with async_playwright() as p:
#             browser = await p.chromium.launch(headless=True)
#             page = await browser.new_page()
# 
#             try:
#                 await page.goto(test_url, wait_until="domcontentloaded", timeout=30000)
# 
#                 # Extract PMID
#                 pmid = await PubMedTranslator.extract_pmid_from_page(page)
#                 if pmid:
#                     print(f"✓ Extracted PMID: {pmid}")
# 
#                     # Fetch metadata
#                     metadata = await PubMedTranslator.fetch_pubmed_metadata(pmid)
#                     if metadata:
#                         print(f"✓ Title: {metadata.get('title', 'N/A')}")
#                         print(f"✓ Authors: {', '.join(metadata.get('authors', []))}")
#                         print(f"✓ DOI: {metadata.get('doi', 'N/A')}")
#                         print(f"✓ Journal: {metadata.get('journal', 'N/A')}")
# 
#                 # Extract PDF URLs
#                 pdf_urls = await PubMedTranslator.extract_pdf_urls_async(page)
#                 print(f"✓ Found {len(pdf_urls)} PDF links")
#                 for url in pdf_urls[:3]:
#                     print(f"  - {url}")
# 
#             finally:
#                 await browser.close()
#     else:
#         print("✗ URL does not match PubMed pattern")
# 
# 
# if __name__ == "__main__":
#     import asyncio
# 
#     asyncio.run(main())

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/pubmed.py
# --------------------------------------------------------------------------------
