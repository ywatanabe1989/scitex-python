#!/usr/bin/env python3
# Timestamp: "2026-02-03"
# File: src/scitex/scholar/metadata_engines/individual/OpenAlexLocalEngine.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import json
from typing import Dict, List, Optional, Union

from scitex import logging

from ..utils import standardize_metadata
from ._BaseDOIEngine import BaseDOIEngine

logger = logging.getLogger(__name__)


class OpenAlexLocalEngine(BaseDOIEngine):
    """OpenAlex Local Engine using local FastAPI or external public API

    Supports both:
    - Internal API: http://openalex:31292 (Docker network)
    - External API: https://scitex.ai/scholar/api/openalex (Public internet)

    Automatically detects API format and adjusts endpoints accordingly.
    """

    def __init__(
        self,
        email: str = "research@example.com",
        api_url: str = "http://127.0.0.1:31292",
    ):
        super().__init__(email)
        self.api_url = api_url.rstrip("/")

        # Detect API type: external (public) vs internal (Docker/local)
        self._is_external_api = (
            "/api/openalex" in self.api_url or "scitex.ai" in self.api_url
        )

    @property
    def name(self) -> str:
        return "OpenAlexLocal"

    @property
    def rate_limit_delay(self) -> float:
        return 0.01

    def _build_endpoint_url(self, endpoint: str) -> str:
        """Build the correct endpoint URL based on API type

        Args:
            endpoint: Endpoint name (e.g., 'works', 'info')

        Returns
        -------
            Full URL for the endpoint

        Examples
        --------
            Internal: http://openalex:31292/works
            External: https://scitex.ai/scholar/api/openalex/works
        """
        if self._is_external_api:
            # External API: base URL already includes /scholar/api/openalex
            return f"{self.api_url}/{endpoint}"
        else:
            # Internal API: direct endpoint
            return f"{self.api_url}/{endpoint}"

    def search(
        self,
        title: Optional[str] = None,
        year: Optional[Union[int, str]] = None,
        authors: Optional[List[str]] = None,
        doi: Optional[str] = None,
        max_results=1,
        return_as: Optional[str] = "dict",
        **kwargs,
    ) -> Optional[Dict]:
        """Search using local OpenAlex API with all parameters"""
        if doi:
            return self._search_by_doi(doi, return_as)
        else:
            return self._search_by_metadata(
                title, year, authors, max_results, return_as
            )

    def _search_by_doi(self, doi: str, return_as: str) -> Optional[Dict]:
        """Get work metadata by DOI"""
        doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        url = self._build_endpoint_url(f"works/{doi}")

        try:
            assert return_as in [
                "dict",
                "json",
            ], "return_as must be either 'dict' or 'json'"

            response = self.session.get(url, timeout=10)
            if response.status_code == 404:
                return self._create_minimal_metadata(doi=doi, return_as=return_as)
            response.raise_for_status()
            data = response.json()

            return self._extract_metadata_from_work(data, return_as)

        except Exception as e:
            if "Connection refused" in str(e) or "Max retries exceeded" in str(e):
                logger.warning(
                    f"OpenAlex Local server not available at {self.api_url} (connection refused)"
                )
            else:
                logger.warning(f"OpenAlex Local DOI search error: {e}")
            return self._create_minimal_metadata(doi=doi, return_as=return_as)

    def _search_by_metadata(
        self,
        title: Optional[str],
        year: Optional[Union[int, str]],
        authors: Optional[List[str]],
        max_results: int,
        return_as: str,
    ) -> Optional[Dict]:
        """Search by metadata (title, year, authors)"""
        if not title:
            return self._create_minimal_metadata(
                title=title,
                year=year,
                authors=authors,
                return_as=return_as,
            )

        # Build search query
        query = title
        if year:
            query = f"{query} {year}"
        if authors and isinstance(authors, list) and authors:
            query = f"{query} {authors[0]}"

        url = self._build_endpoint_url("works")
        params = {"q": query, "limit": max(5, max_results)}

        try:
            assert return_as in [
                "dict",
                "json",
            ], "return_as must be either 'dict' or 'json'"

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                return self._create_minimal_metadata(
                    title=title,
                    year=year,
                    authors=authors,
                    return_as=return_as,
                )

            # Find best matching result
            for work in results:
                work_title = work.get("title", "")
                if work_title and work_title.endswith("."):
                    work_title = work_title[:-1]
                if work_title and self._is_title_match(title, work_title):
                    return self._extract_metadata_from_work(work, return_as)

            # Return first result if no exact match
            return self._extract_metadata_from_work(results[0], return_as)

        except Exception as e:
            if "Connection refused" in str(e) or "Max retries exceeded" in str(e):
                logger.warning(
                    f"OpenAlex Local server not available at {self.api_url} (connection refused)"
                )
            else:
                logger.warning(f"OpenAlex Local search error: {e}")
            return self._create_minimal_metadata(
                title=title,
                year=year,
                authors=authors,
                return_as=return_as,
            )

    def _extract_metadata_from_work(self, data: dict, return_as: str) -> Optional[Dict]:
        """Extract metadata from OpenAlex work data"""
        if not data:
            return self._create_minimal_metadata(return_as=return_as)

        # Extract title
        title = data.get("title")
        if title and title.endswith("."):
            title = title[:-1]

        # Extract DOI
        doi = data.get("doi")
        if doi:
            doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

        # Extract authors
        authors = data.get("authors", [])
        if not authors:
            authors = []

        # Extract publication info
        year = data.get("year")
        journal = data.get("source")
        issn = data.get("issn")
        volume = data.get("volume")
        issue = data.get("issue")
        pages = data.get("pages")
        publisher = None  # Not directly available in response

        # Extract citation count
        cited_by_count = data.get("cited_by_count")

        # Extract concepts/keywords
        concepts = data.get("concepts", [])
        keywords = [c.get("name") for c in concepts if c.get("name")]

        # Extract OA info
        is_oa = data.get("is_oa", False)
        oa_url = data.get("oa_url")

        # Extract OpenAlex ID
        openalex_id = data.get("openalex_id")

        metadata = {
            "id": {
                "doi": doi if doi else None,
                "doi_engines": [self.name] if doi else None,
                "openalex_id": openalex_id if openalex_id else None,
                "openalex_id_engines": [self.name] if openalex_id else None,
            },
            "basic": {
                "title": title if title else None,
                "title_engines": [self.name] if title else None,
                "year": year if year else None,
                "year_engines": [self.name] if year else None,
                "authors": authors if authors else None,
                "authors_engines": [self.name] if authors else None,
                "keywords": keywords if keywords else None,
                "keywords_engines": [self.name] if keywords else None,
            },
            "publication": {
                "journal": journal if journal else None,
                "journal_engines": [self.name] if journal else None,
                "issn": issn if issn else None,
                "issn_engines": [self.name] if issn else None,
                "volume": volume if volume else None,
                "volume_engines": [self.name] if volume else None,
                "issue": issue if issue else None,
                "issue_engines": [self.name] if issue else None,
                "pages": pages if pages else None,
                "pages_engines": [self.name] if pages else None,
            },
            "citation_count": {
                "total": cited_by_count if cited_by_count else None,
                "total_engines": [self.name] if cited_by_count else None,
            },
            "url": {
                "doi": f"https://doi.org/{doi}" if doi else None,
                "doi_engines": [self.name] if doi else None,
                "pdf": oa_url if oa_url else None,
                "pdf_engines": [self.name] if oa_url else None,
            },
            "open_access": {
                "is_oa": is_oa,
                "is_oa_engines": [self.name] if is_oa is not None else None,
                "oa_url": oa_url if oa_url else None,
                "oa_url_engines": [self.name] if oa_url else None,
            },
            "system": {
                f"searched_by_{self.name}": True,
            },
        }

        metadata = standardize_metadata(metadata)

        if return_as == "dict":
            return metadata
        if return_as == "json":
            return json.dumps(metadata, indent=2)


if __name__ == "__main__":

    from scitex.scholar.metadata_engines.individual import OpenAlexLocalEngine

    TITLE = "deep learning"
    DOI = "10.1038/nature12373"

    # Example 1: Internal API (Docker network or localhost)
    print("\n" + "=" * 60)
    print("INTERNAL API EXAMPLE")
    print("=" * 60)
    engine_internal = OpenAlexLocalEngine(
        "test@example.com", api_url="http://openalex:31292"
    )
    print(f"API URL: {engine_internal.api_url}")
    print(f"Is External: {engine_internal._is_external_api}")
    print(f"Works endpoint: {engine_internal._build_endpoint_url('works')}")

    # Example 2: External API (public internet)
    print("\n" + "=" * 60)
    print("EXTERNAL API EXAMPLE")
    print("=" * 60)
    engine_external = OpenAlexLocalEngine(
        "test@example.com", api_url="https://scitex.ai/scholar/api/openalex"
    )
    print(f"API URL: {engine_external.api_url}")
    print(f"Is External: {engine_external._is_external_api}")
    print(f"Works endpoint: {engine_external._build_endpoint_url('works')}")

    # Test search (use internal for demo)
    print("\n" + "=" * 60)
    print("SEARCH TEST")
    print("=" * 60)
    engine = OpenAlexLocalEngine("test@example.com")
    result = engine.search(doi=DOI)
    if result:
        print(f"Title: {result.get('basic', {}).get('title')}")
        print(f"DOI: {result.get('id', {}).get('doi')}")
        print(f"Year: {result.get('basic', {}).get('year')}")
    else:
        print("No results found")


# Usage examples:
#
# Internal API (from NAS Docker):
#   export SCITEX_SCHOLAR_OPENALEX_API_URL=http://openalex:31292
#   python -m scitex.scholar.metadata_engines.individual.OpenAlexLocalEngine
#
# External API (from anywhere):
#   export SCITEX_SCHOLAR_OPENALEX_API_URL=https://scitex.ai/scholar/api/openalex
#   python -m scitex.scholar.metadata_engines.individual.OpenAlexLocalEngine

# EOF
