# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/metadata_engines/individual/CrossRefLocalEngine.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-09-30 07:29:16 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/scholar/engines/individual/CrossRefLocalEngine.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import json
# import time
# import urllib.parse
# from typing import Dict, List, Optional, Union
# 
# from scitex import logging
# 
# from ..utils import standardize_metadata
# from ._BaseDOIEngine import BaseDOIEngine
# 
# logger = logging.getLogger(__name__)
# 
# 
# class CrossRefLocalEngine(BaseDOIEngine):
#     """CrossRef Local Engine using local Django API or external public API
# 
#     Supports both:
#     - Internal API: http://crossref:3333 (Docker network)
#     - External API: https://scitex.ai/scholar/api/crossref (Public internet)
# 
#     Automatically detects API format and adjusts endpoints accordingly.
#     """
# 
#     def __init__(
#         self,
#         email: str = "research@example.com",
#         api_url: str = "http://127.0.0.1:3333",
#     ):
#         super().__init__(email)
#         self.api_url = api_url.rstrip("/")
# 
#         # Detect API type: external (public) vs internal (Docker/local)
#         self._is_external_api = (
#             "/api/crossref" in self.api_url or "scitex.ai" in self.api_url
#         )
# 
#     @property
#     def name(self) -> str:
#         return "CrossRefLocal"
# 
#     @property
#     def rate_limit_delay(self) -> float:
#         return 0.01
# 
#     def _build_endpoint_url(self, endpoint: str) -> str:
#         """Build the correct endpoint URL based on API type
# 
#         Args:
#             endpoint: Endpoint name (e.g., 'search', 'health', 'stats')
# 
#         Returns:
#             Full URL for the endpoint
# 
#         Examples:
#             Internal: http://crossref:3333/api/search/
#             External: https://scitex.ai/scholar/api/crossref/search/
#         """
#         if self._is_external_api:
#             # External API: base URL already includes /scholar/api/crossref
#             return f"{self.api_url}/{endpoint}/"
#         else:
#             # Internal API: need to add /api/ prefix
#             return f"{self.api_url}/api/{endpoint}/"
# 
#     def search(
#         self,
#         title: Optional[str] = None,
#         year: Optional[Union[int, str]] = None,
#         authors: Optional[List[str]] = None,
#         doi: Optional[str] = None,
#         max_results=1,
#         return_as: Optional[str] = "dict",
#         **kwargs,
#     ) -> Optional[Dict]:
#         """Search using local CrossRef API with all parameters"""
#         params = {}
# 
#         if doi:
#             doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
#             params["doi"] = doi
# 
#         if title:
#             params["title"] = title
# 
#         if year:
#             params["year"] = str(year)
# 
#         if authors:
#             if isinstance(authors, list):
#                 params["authors"] = " ".join(authors)
#             else:
#                 params["authors"] = str(authors)
# 
#         if not params:
#             return self._create_minimal_metadata(return_as=return_as)
# 
#         return self._make_search_request(params, return_as)
# 
#     def _make_search_request(self, params: dict, return_as: str) -> Optional[Dict]:
#         """Make search request to local or external API"""
#         url = self._build_endpoint_url("search")
# 
#         try:
#             assert return_as in [
#                 "dict",
#                 "json",
#             ], "return_as must be either 'dict' or 'json'"
# 
#             response = self.session.get(url, params=params, timeout=10)
#             response.raise_for_status()
#             data = response.json()
# 
#             if "doi" in params and isinstance(data, dict) and not data.get("error"):
#                 return self._extract_metadata_from_crossref_data(data, return_as)
# 
#             elif "results" in data and data["results"]:
#                 first_result = data["results"][0]
#                 if first_result.get("doi"):
#                     return self._search_by_doi_only(first_result["doi"], return_as)
# 
#             elif isinstance(data, dict) and not data.get("error"):
#                 return self._extract_metadata_from_crossref_data(data, return_as)
# 
#             return self._create_minimal_metadata(return_as=return_as)
# 
#         except Exception as e:
#             # Shorten verbose connection error messages
#             if "Connection refused" in str(e) or "Max retries exceeded" in str(e):
#                 logger.warning(
#                     f"CrossRef Local server not available at {self.api_url} (connection refused)"
#                 )
#             else:
#                 logger.warning(f"CrossRef Local search error: {e}")
#             return self._create_minimal_metadata(return_as=return_as)
# 
#     def _search_by_doi_only(self, doi: str, return_as: str) -> Optional[Dict]:
#         """Get full metadata for DOI"""
#         doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
#         url = self._build_endpoint_url("search")
#         params = {"doi": doi}
# 
#         try:
#             response = self.session.get(url, params=params, timeout=10)
#             response.raise_for_status()
#             data = response.json()
#             return self._extract_metadata_from_crossref_data(data, return_as)
#         except Exception as exc:
#             # Shorten verbose connection error messages
#             if "Connection refused" in str(exc) or "Max retries exceeded" in str(exc):
#                 logger.warning(
#                     f"CrossRef Local server not available at {self.api_url} (connection refused)"
#                 )
#             else:
#                 logger.warning(f"CrossRef Local DOI lookup error: {exc}")
#             return self._create_minimal_metadata(doi=doi, return_as=return_as)
# 
#     def _extract_metadata_from_crossref_data(
#         self, data, return_as: str
#     ) -> Optional[Dict]:
#         """Extract metadata from CrossRef JSON data"""
#         if not data or data.get("error"):
#             return self._create_minimal_metadata(return_as=return_as)
# 
#         title_list = data.get("title", [])
#         title = title_list[0] if title_list else None
#         if title and title.endswith("."):
#             title = title[:-1]
# 
#         pub_year = None
#         published = data.get("published-print") or data.get("published-online")
#         if published and published.get("date-parts"):
#             pub_year = published["date-parts"][0][0]
# 
#         extracted_authors = []
#         for author in data.get("author", []):
#             given = author.get("given", "")
#             family = author.get("family", "")
#             if family:
#                 if given:
#                     extracted_authors.append(f"{given} {family}")
#                 else:
#                     extracted_authors.append(family)
# 
#         container_titles = data.get("container-title", [])
#         short_container_titles = data.get("short-container-title", [])
#         journal = container_titles[0] if container_titles else None
#         short_journal = short_container_titles[0] if short_container_titles else None
# 
#         issn_list = data.get("ISSN", [])
#         issn = issn_list[0] if issn_list else None
# 
#         metadata = {
#             "id": {
#                 "doi": data.get("DOI"),
#                 "doi_engines": [self.name] if data.get("DOI") else None,
#             },
#             "basic": {
#                 "title": title if title else None,
#                 "title_engines": [self.name] if title else None,
#                 "year": pub_year if pub_year else None,
#                 "year_engines": [self.name] if pub_year else None,
#                 "authors": extracted_authors if extracted_authors else None,
#                 "authors_engines": [self.name] if extracted_authors else None,
#             },
#             "publication": {
#                 "journal": journal if journal else None,
#                 "journal_engines": [self.name] if journal else None,
#                 "short_journal": short_journal if short_journal else None,
#                 "short_journal_engines": ([self.name] if short_journal else None),
#                 "publisher": (data.get("publisher") if data.get("publisher") else None),
#                 "publisher_engines": ([self.name] if data.get("publisher") else None),
#                 "volume": data.get("volume") if data.get("volume") else None,
#                 "volume_engines": [self.name] if data.get("volume") else None,
#                 "issue": data.get("issue") if data.get("issue") else None,
#                 "issue_engines": [self.name] if data.get("issue") else None,
#                 "issn": issn if issn else None,
#                 "issn_engines": [self.name] if issn else None,
#             },
#             "url": {
#                 "doi": (
#                     f"https://doi.org/{data.get('DOI')}" if data.get("DOI") else None
#                 ),
#                 "doi_engines": [self.name] if data.get("DOI") else None,
#             },
#             "system": {
#                 f"searched_by_{self.name}": True,
#             },
#         }
# 
#         metadata = standardize_metadata(metadata)
# 
#         if return_as == "dict":
#             return metadata
#         if return_as == "json":
#             return json.dumps(metadata, indent=2)
# 
# 
# if __name__ == "__main__":
#     from pprint import pprint
# 
#     from scitex.scholar.metadata_engines.individual import CrossRefLocalEngine
# 
#     TITLE = "deep learning"
#     DOI = "10.1038/nature12373"
# 
#     # Example 1: Internal API (Docker network or localhost)
#     print("\n" + "=" * 60)
#     print("INTERNAL API EXAMPLE")
#     print("=" * 60)
#     engine_internal = CrossRefLocalEngine(
#         "test@example.com", api_url="http://crossref:3333"
#     )
#     print(f"API URL: {engine_internal.api_url}")
#     print(f"Is External: {engine_internal._is_external_api}")
#     print(f"Search endpoint: {engine_internal._build_endpoint_url('search')}")
# 
#     # Example 2: External API (public internet)
#     print("\n" + "=" * 60)
#     print("EXTERNAL API EXAMPLE")
#     print("=" * 60)
#     engine_external = CrossRefLocalEngine(
#         "test@example.com", api_url="https://scitex.ai/scholar/api/crossref"
#     )
#     print(f"API URL: {engine_external.api_url}")
#     print(f"Is External: {engine_external._is_external_api}")
#     print(f"Search endpoint: {engine_external._build_endpoint_url('search')}")
# 
#     # Test search (use external for demo)
#     print("\n" + "=" * 60)
#     print("SEARCH TEST")
#     print("=" * 60)
#     result = engine_external.search(doi=DOI)
#     if result:
#         print(f"Title: {result.get('basic', {}).get('title')}")
#         print(f"DOI: {result.get('id', {}).get('doi')}")
#         print(f"Year: {result.get('basic', {}).get('year')}")
#     else:
#         print("No results found")
# 
# 
# # Usage examples:
# #
# # Internal API (from NAS Docker):
# #   export SCITEX_SCHOLAR_CROSSREF_API_URL=http://crossref:3333
# #   python -m scitex.scholar.metadata_engines.individual.CrossRefLocalEngine
# #
# # External API (from anywhere):
# #   export SCITEX_SCHOLAR_CROSSREF_API_URL=https://scitex.ai/scholar/api/crossref
# #   python -m scitex.scholar.metadata_engines.individual.CrossRefLocalEngine
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/metadata_engines/individual/CrossRefLocalEngine.py
# --------------------------------------------------------------------------------
