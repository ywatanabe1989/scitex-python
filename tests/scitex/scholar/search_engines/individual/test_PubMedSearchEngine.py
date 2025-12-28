# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/search_engines/individual/PubMedSearchEngine.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: ./src/scitex/scholar/search_engines/individual/PubMedSearchEngine.py
# 
# """
# PubMed Search Engine - Keyword-based paper discovery via PubMed E-utilities.
# 
# Inherits from PubMedEngine for metadata retrieval capabilities and adds
# keyword search functionality.
# 
# Features:
#   - Keyword search across title and abstract
#   - Boolean operators (AND, OR, NOT)
#   - Year range filtering
#   - Batch results (up to 100 papers per query)
#   - Automatic rate limiting (3 requests/second)
# 
# Example:
#     >>> engine = PubMedSearchEngine()
#     >>> results = engine.search_by_keywords(
#     ...     query="hippocampus AND sharp wave",
#     ...     filters={'year_start': 2020, 'year_end': 2024},
#     ...     max_results=50
#     ... )
# """
# 
# import time
# import json
# from typing import List, Dict, Any, Optional
# from datetime import datetime
# 
# from scitex import logging
# from scitex.scholar.metadata_engines.individual.PubMedEngine import PubMedEngine
# from .._BaseSearchEngine import BaseSearchEngine
# 
# logger = logging.getLogger(__name__)
# 
# 
# class PubMedSearchEngine(PubMedEngine, BaseSearchEngine):
#     """PubMed search engine for keyword-based paper discovery."""
# 
#     def __init__(self, email: str = "research@example.com"):
#         """Initialize PubMed search engine.
# 
#         Args:
#             email: Email for NCBI E-utilities (required for polite usage)
#         """
#         super().__init__(email)
# 
#     def search_by_keywords(
#         self,
#         query: str,
#         filters: Optional[Dict[str, Any]] = None,
#         max_results: int = 100,
#     ) -> List[Dict[str, Any]]:
#         """Search PubMed by keywords across title and abstract.
# 
#         Args:
#             query: Keyword query (supports AND, OR, NOT operators)
#             filters: Optional filters:
#                 - year_start: int
#                 - year_end: int
#                 - negative_keywords: List[str] (applied post-API)
#             max_results: Maximum results (up to 100)
# 
#         Returns:
#             List of paper metadata dictionaries
# 
#         Example:
#             >>> results = engine.search_by_keywords(
#             ...     query="hippocampus sharp wave ripples",
#             ...     filters={'year_start': 2020},
#             ...     max_results=50
#             ... )
#         """
#         filters = filters or {}
# 
#         # Build PubMed query
#         pubmed_query = self._build_pubmed_query(query, filters)
# 
#         logger.info(f"{self.name}: Searching with query: {pubmed_query[:100]}...")
# 
#         # Step 1: Search for PMIDs
#         search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
#         search_params = {
#             "db": "pubmed",
#             "term": pubmed_query,
#             "retmode": "json",
#             "retmax": min(max_results, 100),  # PubMed API limit
#             "email": self.email,
#         }
# 
#         try:
#             response = self.session.get(search_url, params=search_params, timeout=30)
#             response.raise_for_status()
#             data = response.json()
#             pmids = data.get("esearchresult", {}).get("idlist", [])
# 
#             if not pmids:
#                 logger.info(f"{self.name}: No results found")
#                 return []
# 
#             logger.info(f"{self.name}: Found {len(pmids)} PMIDs")
# 
#             # Step 2: Fetch metadata for each PMID
#             results = []
#             for i, pmid in enumerate(pmids):
#                 try:
#                     # Rate limiting: NCBI requires 3 requests/second max
#                     if i > 0:
#                         time.sleep(0.34)
# 
#                     metadata = self._search_by_pmid(pmid, return_as="dict")
#                     if metadata:
#                         results.append(metadata)
# 
#                 except Exception as e:
#                     logger.warning(f"{self.name}: Failed to fetch PMID {pmid}: {e}")
#                     continue
# 
#             logger.success(f"{self.name}: Retrieved {len(results)}/{len(pmids)} papers")
# 
#             # Step 3: Apply post-filters (negative keywords, citations, etc.)
#             filtered_results = self._apply_post_filters(results, filters)
# 
#             logger.info(
#                 f"{self.name}: {len(filtered_results)} papers after post-filtering"
#             )
# 
#             return filtered_results
# 
#         except Exception as e:
#             logger.error(f"{self.name}: Search failed: {e}")
#             return []
# 
#     def _build_pubmed_query(self, query: str, filters: Dict[str, Any]) -> str:
#         """Build PubMed E-utilities query string.
# 
#         Args:
#             query: Base keyword query
#             filters: Filters to apply
# 
#         Returns:
#             PubMed query string with filters
# 
#         Example:
#             >>> _build_pubmed_query(
#             ...     "hippocampus sharp wave",
#             ...     {'year_start': 2020, 'year_end': 2024}
#             ... )
#             '(hippocampus sharp wave)[Title/Abstract] AND 2020:2024[pdat]'
#         """
#         parts = []
# 
#         # Main keyword query in Title/Abstract
#         # Clean and format query
#         query = query.strip()
# 
#         # If query contains explicit NOT, split and handle
#         if " NOT " in query.upper():
#             # User provided explicit Boolean query
#             parts.append(f"({query})[Title/Abstract]")
#         elif " AND " in query.upper() or " OR " in query.upper():
#             # User provided Boolean operators
#             parts.append(f"({query})[Title/Abstract]")
#         else:
#             # Simple keyword query - treat as phrase or AND all terms
#             parts.append(f"({query})[Title/Abstract]")
# 
#         # Year range filter
#         year_start = filters.get("year_start")
#         year_end = filters.get("year_end")
# 
#         if year_start and year_end:
#             parts.append(f"{year_start}:{year_end}[pdat]")
#         elif year_start:
#             current_year = datetime.now().year
#             parts.append(f"{year_start}:{current_year}[pdat]")
#         elif year_end:
#             parts.append(f"1900:{year_end}[pdat]")
# 
#         # Combine with AND
#         return " AND ".join(parts)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/search_engines/individual/PubMedSearchEngine.py
# --------------------------------------------------------------------------------
