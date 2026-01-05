# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/integration/mendeley/mapper.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Mendeley ↔ Scholar data mapping.
# """
# 
# from typing import Any, Dict, Optional
# from datetime import datetime
# 
# from scitex import logging
# from scitex.scholar.core.Paper import Paper
# from ..base import BaseMapper
# 
# logger = logging.getLogger(__name__)
# 
# 
# class MendeleyMapper(BaseMapper):
#     """Mendeley ↔ Scholar data mapping."""
# 
#     def external_to_paper(self, item: Dict[str, Any]) -> Paper:
#         """Convert Mendeley document to Scholar Paper.
# 
#         Args:
#             item: Mendeley document dict
# 
#         Returns:
#             Paper object
#         """
#         paper = Paper()
# 
#         # Basic metadata
#         paper.metadata.basic.title = item.get("title", "")
#         paper.metadata.basic.title_engines = ["mendeley"]
# 
#         # Authors - Mendeley format: [{"first_name": "John", "last_name": "Doe"}]
#         authors_data = item.get("authors", [])
#         authors = []
#         for author in authors_data:
#             first = author.get("first_name", "")
#             last = author.get("last_name", "")
#             if first and last:
#                 authors.append(f"{last}, {first}")
#             elif last:
#                 authors.append(last)
#             elif first:
#                 authors.append(first)
# 
#         if authors:
#             paper.metadata.basic.authors = authors
#             paper.metadata.basic.authors_engines = ["mendeley"]
# 
#         # Year
#         year = item.get("year")
#         if year:
#             paper.metadata.basic.year = int(year)
#             paper.metadata.basic.year_engines = ["mendeley"]
# 
#         # Abstract
#         abstract = item.get("abstract", "")
#         if abstract:
#             paper.metadata.basic.abstract = abstract
#             paper.metadata.basic.abstract_engines = ["mendeley"]
# 
#         # Keywords/tags
#         tags = item.get("tags", [])
#         keywords = item.get("keywords", [])
#         all_keywords = list(set(tags + keywords))
#         if all_keywords:
#             paper.metadata.basic.keywords = all_keywords
#             paper.metadata.basic.keywords_engines = ["mendeley"]
# 
#         # Type
#         doc_type = item.get("type", "")
#         if doc_type:
#             paper.metadata.basic.type = doc_type
#             paper.metadata.basic.type_engines = ["mendeley"]
# 
#         # Identifiers
#         doi = item.get("identifiers", {}).get("doi", "")
#         if not doi:
#             doi = item.get("doi", "")
#         if doi:
#             paper.metadata.set_doi(doi)
#             paper.metadata.id.doi_engines = ["mendeley"]
# 
#         pmid = item.get("identifiers", {}).get("pmid", "")
#         if pmid:
#             paper.metadata.id.pmid = pmid
#             paper.metadata.id.pmid_engines = ["mendeley"]
# 
#         # Publication details
#         source = item.get("source", "")
#         if source:
#             paper.metadata.publication.journal = source
#             paper.metadata.publication.journal_engines = ["mendeley"]
# 
#         volume = item.get("volume", "")
#         if volume:
#             paper.metadata.publication.volume = volume
#             paper.metadata.publication.volume_engines = ["mendeley"]
# 
#         issue = item.get("issue", "")
#         if issue:
#             paper.metadata.publication.issue = issue
#             paper.metadata.publication.issue_engines = ["mendeley"]
# 
#         pages = item.get("pages", "")
#         if pages:
#             paper.metadata.publication.pages = pages
#             paper.metadata.publication.pages_engines = ["mendeley"]
# 
#         publisher = item.get("publisher", "")
#         if publisher:
#             paper.metadata.publication.publisher = publisher
#             paper.metadata.publication.publisher_engines = ["mendeley"]
# 
#         issn = item.get("issn", "")
#         if issn:
#             paper.metadata.publication.issn = issn
#             paper.metadata.publication.issn_engines = ["mendeley"]
# 
#         # URLs
#         websites = item.get("websites", [])
#         if websites:
#             url = websites[0]
#             if "doi.org" in url:
#                 paper.metadata.url.doi = url
#                 paper.metadata.url.doi_engines = ["mendeley"]
#             else:
#                 paper.metadata.url.publisher = url
#                 paper.metadata.url.publisher_engines = ["mendeley"]
# 
#         # Container metadata
#         paper.container.created_by = "mendeley_import"
#         paper.container.created_at = datetime.now().isoformat()
#         paper.container.updated_at = datetime.now().isoformat()
# 
#         # Store Mendeley-specific metadata
#         paper._mendeley_id = item.get("id", "")
#         paper._mendeley_group_id = item.get("group_id", "")
#         paper._mendeley_profile_id = item.get("profile_id", "")
#         paper._mendeley_notes = item.get("notes", "")
# 
#         logger.debug(
#             f"Converted Mendeley document '{paper.metadata.basic.title}' to Paper"
#         )
# 
#         return paper
# 
#     def paper_to_external(self, paper: Paper) -> Dict[str, Any]:
#         """Convert Scholar Paper to Mendeley document format.
# 
#         Args:
#             paper: Paper object
# 
#         Returns:
#             Dict in Mendeley format
#         """
#         # Build authors list
#         authors = []
#         if paper.metadata.basic.authors:
#             for author in paper.metadata.basic.authors:
#                 parts = author.split(",")
#                 if len(parts) == 2:
#                     authors.append(
#                         {"first_name": parts[1].strip(), "last_name": parts[0].strip()}
#                     )
#                 else:
#                     authors.append({"last_name": author.strip()})
# 
#         # Build Mendeley document
#         mendeley_doc = {
#             "type": paper.metadata.basic.type or "journal",
#             "title": paper.metadata.basic.title or "",
#             "authors": authors,
#             "year": paper.metadata.basic.year,
#             "abstract": paper.metadata.basic.abstract or "",
#             "source": paper.metadata.publication.journal or "",
#             "volume": paper.metadata.publication.volume or "",
#             "issue": paper.metadata.publication.issue or "",
#             "pages": paper.metadata.publication.pages or "",
#             "publisher": paper.metadata.publication.publisher or "",
#             "issn": paper.metadata.publication.issn or "",
#             "keywords": paper.metadata.basic.keywords or [],
#             "tags": paper.metadata.basic.keywords or [],
#         }
# 
#         # Identifiers
#         identifiers = {}
#         if paper.metadata.id.doi:
#             identifiers["doi"] = paper.metadata.id.doi
#         if paper.metadata.id.pmid:
#             identifiers["pmid"] = paper.metadata.id.pmid
# 
#         if identifiers:
#             mendeley_doc["identifiers"] = identifiers
# 
#         # URLs
#         websites = []
#         if paper.metadata.url.doi:
#             websites.append(paper.metadata.url.doi)
#         elif paper.metadata.url.publisher:
#             websites.append(paper.metadata.url.publisher)
# 
#         if websites:
#             mendeley_doc["websites"] = websites
# 
#         logger.debug(
#             f"Converted Paper '{paper.metadata.basic.title}' to Mendeley document"
#         )
# 
#         return mendeley_doc
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/integration/mendeley/mapper.py
# --------------------------------------------------------------------------------
