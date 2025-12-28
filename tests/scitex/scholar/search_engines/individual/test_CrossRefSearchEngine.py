# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/search_engines/individual/CrossRefSearchEngine.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: ./src/scitex/scholar/search_engines/individual/CrossRefSearchEngine.py
# 
# """
# CrossRef Search Engine - Keyword-based paper discovery via CrossRef API.
# 
# Features:
#   - Keyword search across bibliographic data
#   - Year range filtering via API filters
#   - Large result sets (up to 1000 per query)
#   - No authentication required (polite pool with email)
# """
# 
# import time
# from typing import List, Dict, Any, Optional
# 
# from scitex import logging
# from scitex.scholar.metadata_engines.individual.CrossRefEngine import CrossRefEngine
# from .._BaseSearchEngine import BaseSearchEngine
# 
# logger = logging.getLogger(__name__)
# 
# 
# class CrossRefSearchEngine(CrossRefEngine, BaseSearchEngine):
#     """CrossRef search engine for keyword-based paper discovery."""
# 
#     def search_by_keywords(
#         self,
#         query: str,
#         filters: Optional[Dict[str, Any]] = None,
#         max_results: int = 100,
#     ) -> List[Dict[str, Any]]:
#         """Search CrossRef by keywords.
# 
#         Args:
#             query: Keyword query
#             filters: Optional filters (year_start, year_end)
#             max_results: Maximum results
# 
#         Returns:
#             List of paper metadata dictionaries
#         """
#         filters = filters or {}
# 
#         # Build CrossRef API parameters
#         params = {
#             "query.bibliographic": query,
#             "rows": min(max_results, 1000),
#             "mailto": self.email,
#         }
# 
#         # Add filters - only get articles, not journals or issues
#         filter_parts = ["type:journal-article"]  # Only journal articles
# 
#         if filters.get("year_start"):
#             filter_parts.append(f"from-pub-date:{filters['year_start']}")
#         if filters.get("year_end"):
#             filter_parts.append(f"until-pub-date:{filters['year_end']}")
# 
#         params["filter"] = ",".join(filter_parts)
# 
#         logger.info(f"{self.name}: Searching CrossRef with query: {query[:50]}...")
# 
#         try:
#             url = "https://api.crossref.org/works"
#             response = self.session.get(url, params=params, timeout=30)
#             response.raise_for_status()
#             data = response.json()
# 
#             items = data.get("message", {}).get("items", [])
# 
#             if not items:
#                 logger.info(f"{self.name}: No results found")
#                 return []
# 
#             logger.info(f"{self.name}: Found {len(items)} results")
# 
#             # Convert to standardized format
#             results = []
#             for item in items:
#                 metadata = self._crossref_to_standard_format(item)
#                 if metadata:
#                     results.append(metadata)
# 
#             # Apply post-filters
#             filtered_results = self._apply_post_filters(results, filters)
# 
#             return filtered_results
# 
#         except Exception as e:
#             logger.error(f"{self.name}: Search failed: {e}")
#             return []
# 
#     def _crossref_to_standard_format(self, item: Dict) -> Dict[str, Any]:
#         """Convert CrossRef item to standard metadata format."""
#         # Extract authors
#         authors = []
#         for author in item.get("author", []):
#             if "given" in author and "family" in author:
#                 authors.append(f"{author['given']} {author['family']}")
#             elif "family" in author:
#                 authors.append(author["family"])
# 
#         # Extract year
#         year = None
#         if "published-print" in item:
#             date_parts = item["published-print"].get("date-parts", [[]])[0]
#             if date_parts:
#                 year = date_parts[0]
#         elif "published-online" in item:
#             date_parts = item["published-online"].get("date-parts", [[]])[0]
#             if date_parts:
#                 year = date_parts[0]
# 
#         # Build metadata dict
#         metadata = {
#             "id": {
#                 "doi": item.get("DOI"),
#                 "doi_engines": [self.name] if item.get("DOI") else None,
#             },
#             "basic": {
#                 "title": item.get("title", [""])[0] if item.get("title") else None,
#                 "title_engines": [self.name] if item.get("title") else None,
#                 "authors": authors if authors else None,
#                 "authors_engines": [self.name] if authors else None,
#                 "abstract": item.get("abstract"),
#                 "abstract_engines": [self.name] if item.get("abstract") else None,
#             },
#             "publication": {
#                 "year": year,
#                 "year_engines": [self.name] if year else None,
#                 "journal": item.get("container-title", [""])[0]
#                 if item.get("container-title")
#                 else None,
#                 "journal_engines": [self.name] if item.get("container-title") else None,
#                 "volume": item.get("volume"),
#                 "issue": item.get("issue"),
#                 "issn": item.get("ISSN", [None])[0] if item.get("ISSN") else None,
#             },
#             "metrics": {
#                 "citation_count": item.get("is-referenced-by-count", 0),
#                 "is_open_access": item.get("link", [{}])[0].get("content-type")
#                 == "unspecified"
#                 if item.get("link")
#                 else False,
#             },
#             "urls": {
#                 "doi_url": f"https://doi.org/{item['DOI']}"
#                 if item.get("DOI")
#                 else None,
#                 "publisher": item.get("URL"),
#             },
#         }
# 
#         return metadata
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/search_engines/individual/CrossRefSearchEngine.py
# --------------------------------------------------------------------------------
