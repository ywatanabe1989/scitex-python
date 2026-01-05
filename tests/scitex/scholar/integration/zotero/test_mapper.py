# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/integration/zotero/mapper.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Zotero ↔ SciTeX Scholar data mapping.
# 
# Maps between Zotero's item format and Scholar's Paper format.
# Handles field translations, data normalization, and metadata enrichment.
# """
# 
# from typing import Any, Dict, List, Optional
# from datetime import datetime
# 
# from scitex import logging
# from scitex.scholar.core.Paper import Paper
# 
# logger = logging.getLogger(__name__)
# 
# 
# class ZoteroMapper:
#     """Bidirectional mapping between Zotero and Scholar formats."""
# 
#     # Zotero item type mapping to Scholar paper type
#     ZOTERO_TYPE_MAP = {
#         "journalArticle": "article",
#         "conferencePaper": "conference",
#         "preprint": "preprint",
#         "book": "book",
#         "bookSection": "book_chapter",
#         "thesis": "thesis",
#         "report": "report",
#         "webpage": "webpage",
#         "patent": "patent",
#     }
# 
#     def __init__(self, config=None):
#         """Initialize mapper.
# 
#         Args:
#             config: Optional ScholarConfig instance
#         """
#         self.config = config
# 
#     def zotero_to_paper(self, zotero_item: Dict[str, Any]) -> Paper:
#         """Convert Zotero item to Scholar Paper.
# 
#         Args:
#             zotero_item: Zotero API item dict with 'data' key
# 
#         Returns:
#             Paper object
#         """
#         data = zotero_item.get("data", {})
# 
#         paper = Paper()
# 
#         # Basic metadata
#         paper.metadata.basic.title = data.get("title", "")
#         paper.metadata.basic.title_engines = ["zotero"]
# 
#         # Authors - Zotero format: [{"firstName": "John", "lastName": "Doe", "creatorType": "author"}]
#         creators = data.get("creators", [])
#         authors = []
#         for creator in creators:
#             if creator.get("creatorType") in ["author", "contributor"]:
#                 first = creator.get("firstName", "")
#                 last = creator.get("lastName", "")
#                 if first and last:
#                     authors.append(f"{last}, {first}")
#                 elif last:
#                     authors.append(last)
#                 elif first:
#                     authors.append(first)
# 
#         if authors:
#             paper.metadata.basic.authors = authors
#             paper.metadata.basic.authors_engines = ["zotero"]
# 
#         # Year - extract from date field
#         date_str = data.get("date", "")
#         if date_str:
#             year = self._extract_year(date_str)
#             if year:
#                 paper.metadata.basic.year = year
#                 paper.metadata.basic.year_engines = ["zotero"]
# 
#         # Abstract
#         abstract = data.get("abstractNote", "")
#         if abstract:
#             paper.metadata.basic.abstract = abstract
#             paper.metadata.basic.abstract_engines = ["zotero"]
# 
#         # Keywords/tags - Zotero format: [{"tag": "machine learning", "type": 1}]
#         tags = data.get("tags", [])
#         keywords = [tag.get("tag") for tag in tags if tag.get("tag")]
#         if keywords:
#             paper.metadata.basic.keywords = keywords
#             paper.metadata.basic.keywords_engines = ["zotero"]
# 
#         # Paper type
#         item_type = data.get("itemType", "")
#         if item_type:
#             paper_type = self.ZOTERO_TYPE_MAP.get(item_type, "article")
#             paper.metadata.basic.type = paper_type
#             paper.metadata.basic.type_engines = ["zotero"]
# 
#         # Identifiers
#         doi = data.get("DOI", "")
#         if doi:
#             paper.metadata.set_doi(doi)
#             paper.metadata.id.doi_engines = ["zotero"]
# 
#         # Publication details
#         journal = data.get("publicationTitle", "") or data.get(
#             "journalAbbreviation", ""
#         )
#         if journal:
#             paper.metadata.publication.journal = journal
#             paper.metadata.publication.journal_engines = ["zotero"]
# 
#         volume = data.get("volume", "")
#         if volume:
#             paper.metadata.publication.volume = volume
#             paper.metadata.publication.volume_engines = ["zotero"]
# 
#         issue = data.get("issue", "")
#         if issue:
#             paper.metadata.publication.issue = issue
#             paper.metadata.publication.issue_engines = ["zotero"]
# 
#         pages = data.get("pages", "")
#         if pages:
#             paper.metadata.publication.pages = pages
#             paper.metadata.publication.pages_engines = ["zotero"]
# 
#         publisher = data.get("publisher", "")
#         if publisher:
#             paper.metadata.publication.publisher = publisher
#             paper.metadata.publication.publisher_engines = ["zotero"]
# 
#         issn = data.get("ISSN", "")
#         if issn:
#             paper.metadata.publication.issn = issn
#             paper.metadata.publication.issn_engines = ["zotero"]
# 
#         # URL
#         url = data.get("url", "")
#         if url:
#             if "doi.org" in url:
#                 paper.metadata.url.doi = url
#                 paper.metadata.url.doi_engines = ["zotero"]
#             elif "arxiv.org" in url:
#                 paper.metadata.url.arxiv = url
#                 paper.metadata.url.arxiv_engines = ["zotero"]
#             else:
#                 paper.metadata.url.publisher = url
#                 paper.metadata.url.publisher_engines = ["zotero"]
# 
#         # Container metadata - track Zotero origin
#         paper.container.created_by = "zotero_import"
#         paper.container.created_at = datetime.now().isoformat()
#         paper.container.updated_at = datetime.now().isoformat()
# 
#         # Store Zotero-specific metadata
#         paper._zotero_key = zotero_item.get("key", "")
#         paper._zotero_version = zotero_item.get("version", 0)
#         paper._zotero_collections = data.get("collections", [])
#         paper._zotero_extra = data.get("extra", "")
# 
#         logger.debug(f"Converted Zotero item '{paper.metadata.basic.title}' to Paper")
# 
#         return paper
# 
#     def paper_to_zotero(self, paper: Paper) -> Dict[str, Any]:
#         """Convert Scholar Paper to Zotero item format.
# 
#         Args:
#             paper: Paper object
# 
#         Returns:
#             Dict in Zotero API item format
#         """
#         # Determine Zotero item type from Scholar type
#         paper_type = paper.metadata.basic.type or "article"
#         zotero_type = "journalArticle"  # default
#         for zot_type, scholar_type in self.ZOTERO_TYPE_MAP.items():
#             if scholar_type == paper_type:
#                 zotero_type = zot_type
#                 break
# 
#         # Build creators list
#         creators = []
#         if paper.metadata.basic.authors:
#             for author in paper.metadata.basic.authors:
#                 # Parse "Last, First" format
#                 parts = author.split(",")
#                 if len(parts) == 2:
#                     last_name = parts[0].strip()
#                     first_name = parts[1].strip()
#                     creators.append(
#                         {
#                             "creatorType": "author",
#                             "firstName": first_name,
#                             "lastName": last_name,
#                         }
#                     )
#                 else:
#                     # Single name - treat as lastName
#                     creators.append(
#                         {"creatorType": "author", "lastName": author.strip()}
#                     )
# 
#         # Build tags list
#         tags = []
#         if paper.metadata.basic.keywords:
#             tags = [{"tag": kw, "type": 1} for kw in paper.metadata.basic.keywords]
# 
#         # Build date string
#         date_str = ""
#         if paper.metadata.basic.year:
#             date_str = str(paper.metadata.basic.year)
# 
#         # Split pages if needed
#         pages_str = paper.metadata.publication.pages or ""
# 
#         # Build Zotero item
#         zotero_item = {
#             "itemType": zotero_type,
#             "title": paper.metadata.basic.title or "",
#             "creators": creators,
#             "abstractNote": paper.metadata.basic.abstract or "",
#             "publicationTitle": paper.metadata.publication.journal or "",
#             "volume": paper.metadata.publication.volume or "",
#             "issue": paper.metadata.publication.issue or "",
#             "pages": pages_str,
#             "date": date_str,
#             "DOI": paper.metadata.id.doi or "",
#             "url": paper.metadata.url.publisher or paper.metadata.url.doi or "",
#             "publisher": paper.metadata.publication.publisher or "",
#             "ISSN": paper.metadata.publication.issn or "",
#             "tags": tags,
#         }
# 
#         # Add collections if stored from previous import
#         if hasattr(paper, "_zotero_collections"):
#             zotero_item["collections"] = paper._zotero_collections
# 
#         # Add extra field with SciTeX metadata
#         extra_lines = []
#         if hasattr(paper, "_zotero_extra") and paper._zotero_extra:
#             extra_lines.append(paper._zotero_extra)
# 
#         # Add SciTeX identifiers to extra field
#         if paper.container.library_id:
#             extra_lines.append(f"SciTeX-ID: {paper.container.library_id}")
#         if paper.metadata.citation_count.total:
#             extra_lines.append(f"Citation-Count: {paper.metadata.citation_count.total}")
#         if paper.metadata.publication.impact_factor:
#             extra_lines.append(
#                 f"Impact-Factor: {paper.metadata.publication.impact_factor}"
#             )
# 
#         if extra_lines:
#             zotero_item["extra"] = "\n".join(extra_lines)
# 
#         logger.debug(f"Converted Paper '{paper.metadata.basic.title}' to Zotero item")
# 
#         return zotero_item
# 
#     def _extract_year(self, date_str: str) -> Optional[int]:
#         """Extract year from various date formats.
# 
#         Args:
#             date_str: Date string in various formats
# 
#         Returns:
#             Year as int, or None if not found
#         """
#         import re
# 
#         # Try to find 4-digit year
#         match = re.search(r"\b(19|20)\d{2}\b", date_str)
#         if match:
#             try:
#                 return int(match.group(0))
#             except ValueError:
#                 pass
# 
#         return None
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/integration/zotero/mapper.py
# --------------------------------------------------------------------------------
