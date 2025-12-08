#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/scholar/search_engines/individual/OpenAlexSearchEngine.py

"""
OpenAlex Search Engine - Comprehensive academic search via OpenAlex API.

Features:
  - Full-text keyword search
  - Rich metadata (citations, open access, institutions)
  - Year range filtering
  - No API key required (polite pool with email)
  - Large result sets (up to 200 per page)
"""

from typing import List, Dict, Any, Optional

from scitex import logging
from scitex.scholar.metadata_engines.individual.OpenAlexEngine import OpenAlexEngine
from .._BaseSearchEngine import BaseSearchEngine

logger = logging.getLogger(__name__)


class OpenAlexSearchEngine(OpenAlexEngine, BaseSearchEngine):
    """OpenAlex search engine for comprehensive paper discovery."""

    def search_by_keywords(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search OpenAlex by keywords.

        Args:
            query: Keyword query
            filters: Optional filters (year_start, year_end, open_access)
            max_results: Maximum results

        Returns:
            List of paper metadata dictionaries
        """
        filters = filters or {}

        # Build OpenAlex API parameters
        params = {
            "search": query,
            "per-page": min(max_results, 200),  # OpenAlex max per page
            "mailto": self.email,
        }

        # Add filters
        filter_parts = []
        if filters.get("year_start"):
            filter_parts.append(f"publication_year:>{filters['year_start'] - 1}")
        if filters.get("year_end"):
            filter_parts.append(f"publication_year:<{filters['year_end'] + 1}")
        if filters.get("open_access"):
            filter_parts.append("is_oa:true")

        if filter_parts:
            params["filter"] = ",".join(filter_parts)

        logger.info(f"{self.name}: Searching OpenAlex with query: {query[:50]}...")

        try:
            url = "https://api.openalex.org/works"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            results_data = data.get("results", [])

            if not results_data:
                logger.info(f"{self.name}: No results found")
                return []

            logger.info(f"{self.name}: Found {len(results_data)} results")

            # Convert to standardized format
            results = []
            for item in results_data:
                metadata = self._openalex_to_standard_format(item)
                if metadata:
                    results.append(metadata)

            # Apply post-filters
            filtered_results = self._apply_post_filters(results, filters)

            return filtered_results

        except Exception as e:
            logger.error(f"{self.name}: Search failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return []

    def _openalex_to_standard_format(self, item: Dict) -> Dict[str, Any]:
        """Convert OpenAlex work to standard metadata format."""
        # Extract authors
        authors = []
        for authorship in item.get("authorships", []):
            author = authorship.get("author", {})
            display_name = author.get("display_name")
            if display_name:
                authors.append(display_name)

        # Extract identifiers
        doi = (
            item.get("doi", "").replace("https://doi.org/", "")
            if item.get("doi")
            else None
        )
        pmid = None
        for ext_id, value in item.get("ids", {}).items():
            if ext_id == "pmid":
                pmid = value.replace("https://pubmed.ncbi.nlm.nih.gov/", "")

        # Extract publication info
        year = item.get("publication_year")
        journal_name = None
        issn = None

        primary_location = item.get("primary_location", {})
        if primary_location:
            source = primary_location.get("source")
            if source:
                journal_name = source.get("display_name")
                issn_list = source.get("issn_l")
                if issn_list:
                    issn = issn_list[0] if isinstance(issn_list, list) else issn_list

        # Open access info
        open_access = item.get("open_access", {})
        is_oa = open_access.get("is_oa", False)
        oa_url = open_access.get("oa_url")

        # Build metadata dict
        metadata = {
            "id": {
                "doi": doi,
                "doi_engines": [self.name] if doi else None,
                "pmid": pmid,
                "pmid_engines": [self.name] if pmid else None,
                "openalex": item.get("id"),
            },
            "basic": {
                "title": item.get("title"),
                "title_engines": [self.name] if item.get("title") else None,
                "authors": authors if authors else None,
                "authors_engines": [self.name] if authors else None,
                "abstract": None,  # OpenAlex doesn't always provide abstracts
            },
            "publication": {
                "year": year,
                "year_engines": [self.name] if year else None,
                "journal": journal_name,
                "journal_engines": [self.name] if journal_name else None,
                "issn": issn,
            },
            "metrics": {
                "citation_count": item.get("cited_by_count", 0),
                "is_open_access": is_oa,
            },
            "urls": {
                "doi_url": f"https://doi.org/{doi}" if doi else None,
                "pdf": oa_url if is_oa else None,
                "publisher": item.get("primary_location", {}).get("landing_page_url"),
            },
        }

        return metadata


# EOF
