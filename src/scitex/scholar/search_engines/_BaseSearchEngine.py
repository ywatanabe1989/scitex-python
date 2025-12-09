#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/scholar/search_engines/_BaseSearchEngine.py

"""
Abstract base class for search engines.

Search engines perform keyword-based discovery of papers across title, abstract,
and full-text. They inherit from metadata engines to reuse connection/rate-limiting
logic but add keyword search capabilities.

Architecture:
  - BaseSearchEngine (abstract) ← defines search_by_keywords() interface
  - PubMedSearchEngine ← inherits PubMedEngine + implements search_by_keywords()
  - CrossRefSearchEngine ← inherits CrossRefEngine + implements search_by_keywords()
  - etc.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime


class BaseSearchEngine(ABC):
    """Abstract base class for academic search engines."""

    @abstractmethod
    def search_by_keywords(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search for papers by keywords.

        Args:
            query: Keyword query string (can include Boolean operators)
            filters: Optional filters dictionary with keys:
                - year_start: int - Start year
                - year_end: int - End year
                - open_access: bool - Filter for open access papers
                - document_type: str - Filter by document type
            max_results: Maximum number of results to return

        Returns:
            List of paper metadata dictionaries in standardized format

        Each result dict should contain:
            {
                'id': {
                    'doi': str,
                    'pmid': str,
                    'arxiv': str,
                    ...
                },
                'basic': {
                    'title': str,
                    'authors': List[str],
                    'abstract': str,
                    'keywords': List[str],
                    ...
                },
                'publication': {
                    'year': int,
                    'journal': str,
                    ...
                },
                'metrics': {
                    'citation_count': int,
                    'is_open_access': bool,
                    ...
                },
                'urls': {
                    'pdf': str,
                    'publisher': str,
                    'doi_url': str,
                    ...
                }
            }
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name for logging."""
        pass

    def _build_query_with_filters(
        self, base_query: str, filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Helper to build engine-specific query with filters.

        Override this in subclasses to implement engine-specific query syntax.

        Args:
            base_query: Base keyword query
            filters: Filters to apply

        Returns:
            Engine-specific query string
        """
        return base_query

    def _apply_post_filters(
        self, results: List[Dict[str, Any]], filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Apply client-side filters that couldn't be pushed to API.

        Args:
            results: List of paper metadata dicts
            filters: Filters to apply (negative_keywords, min_citations, etc.)

        Returns:
            Filtered list of results
        """
        if not filters:
            return results

        filtered = results

        # Filter by negative keywords in title/abstract
        negative_keywords = filters.get("negative_keywords", [])
        if negative_keywords:
            filtered = [
                paper
                for paper in filtered
                if not self._contains_negative_keywords(paper, negative_keywords)
            ]

        # Filter by minimum citations
        min_citations = filters.get("min_citations")
        if min_citations is not None:
            filtered = [
                paper
                for paper in filtered
                if self._get_citation_count(paper) >= min_citations
            ]

        # Filter by maximum citations
        max_citations = filters.get("max_citations")
        if max_citations is not None:
            filtered = [
                paper
                for paper in filtered
                if self._get_citation_count(paper) <= max_citations
            ]

        return filtered

    def _contains_negative_keywords(
        self, paper: Dict[str, Any], negative_keywords: List[str]
    ) -> bool:
        """Check if paper contains any negative keywords."""
        # Get searchable text
        text_parts = []

        if "basic" in paper:
            if paper["basic"].get("title"):
                text_parts.append(paper["basic"]["title"].lower())
            if paper["basic"].get("abstract"):
                text_parts.append(paper["basic"]["abstract"].lower())

        searchable_text = " ".join(text_parts)

        # Check for negative keywords
        for neg_kw in negative_keywords:
            if neg_kw.lower() in searchable_text:
                return True

        return False

    def _get_citation_count(self, paper: Dict[str, Any]) -> int:
        """Extract citation count from paper metadata."""
        if "metrics" in paper and paper["metrics"].get("citation_count"):
            return paper["metrics"]["citation_count"]
        return 0


# EOF
