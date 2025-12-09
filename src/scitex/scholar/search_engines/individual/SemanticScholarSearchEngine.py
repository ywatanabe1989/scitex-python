#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/scholar/search_engines/individual/SemanticScholarSearchEngine.py

"""
Semantic Scholar Search Engine - AI-powered academic search.

Features:
  - Keyword search with relevance ranking
  - Rich metadata (citations, influential citations, tldr)
  - Year range filtering
  - Field of study filtering
  - Rate limit: 100 requests per 5 minutes (API key recommended)
"""

from typing import List, Dict, Any, Optional

from scitex import logging
from scitex.scholar.metadata_engines.individual.SemanticScholarEngine import (
    SemanticScholarEngine,
)
from .._BaseSearchEngine import BaseSearchEngine

logger = logging.getLogger(__name__)


class SemanticScholarSearchEngine(SemanticScholarEngine, BaseSearchEngine):
    """Semantic Scholar search engine for AI-powered paper discovery."""

    def search_by_keywords(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search Semantic Scholar by keywords.

        Args:
            query: Keyword query
            filters: Optional filters (year_start, year_end)
            max_results: Maximum results

        Returns:
            List of paper metadata dictionaries
        """
        filters = filters or {}

        # Build Semantic Scholar API parameters
        params = {
            "query": query,
            "limit": min(max_results, 100),  # API max
            "fields": "paperId,title,authors,year,abstract,citationCount,openAccessPdf,externalIds,venue,publicationDate,fieldsOfStudy",
        }

        # Add year filter if specified
        if filters.get("year_start"):
            params["year"] = f"{filters['year_start']}-"
        if filters.get("year_end"):
            if "year" in params:
                params["year"] += str(filters["year_end"])
            else:
                params["year"] = f"-{filters['year_end']}"

        logger.info(
            f"{self.name}: Searching Semantic Scholar with query: {query[:50]}..."
        )

        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            papers = data.get("data", [])

            if not papers:
                logger.info(f"{self.name}: No results found")
                return []

            logger.info(f"{self.name}: Found {len(papers)} results")

            # Convert to standardized format
            results = []
            for paper in papers:
                metadata = self._semanticscholar_to_standard_format(paper)
                if metadata:
                    results.append(metadata)

            # Apply post-filters
            filtered_results = self._apply_post_filters(results, filters)

            return filtered_results

        except Exception as e:
            logger.error(f"{self.name}: Search failed: {e}")
            return []

    def _semanticscholar_to_standard_format(self, paper: Dict) -> Dict[str, Any]:
        """Convert Semantic Scholar paper to standard metadata format."""
        # Extract authors
        authors = []
        for author in paper.get("authors", []):
            if author.get("name"):
                authors.append(author["name"])

        # Extract external IDs
        external_ids = paper.get("externalIds", {})
        doi = external_ids.get("DOI")
        pmid = external_ids.get("PubMed")
        arxiv = external_ids.get("ArXiv")

        # Open access PDF
        oa_pdf = paper.get("openAccessPdf")
        pdf_url = oa_pdf.get("url") if oa_pdf else None

        # Build metadata dict
        metadata = {
            "id": {
                "doi": doi,
                "doi_engines": [self.name] if doi else None,
                "pmid": pmid,
                "pmid_engines": [self.name] if pmid else None,
                "arxiv": arxiv,
                "arxiv_engines": [self.name] if arxiv else None,
                "semanticscholar": paper.get("paperId"),
            },
            "basic": {
                "title": paper.get("title"),
                "title_engines": [self.name] if paper.get("title") else None,
                "authors": authors if authors else None,
                "authors_engines": [self.name] if authors else None,
                "abstract": paper.get("abstract"),
                "abstract_engines": [self.name] if paper.get("abstract") else None,
                "keywords": paper.get("fieldsOfStudy", []),
            },
            "publication": {
                "year": paper.get("year"),
                "year_engines": [self.name] if paper.get("year") else None,
                "journal": paper.get("venue"),
                "journal_engines": [self.name] if paper.get("venue") else None,
            },
            "metrics": {
                "citation_count": paper.get("citationCount", 0),
                "is_open_access": pdf_url is not None,
            },
            "urls": {
                "doi_url": f"https://doi.org/{doi}" if doi else None,
                "pdf": pdf_url,
                "publisher": f"https://www.semanticscholar.org/paper/{paper.get('paperId')}"
                if paper.get("paperId")
                else None,
            },
        }

        return metadata


# EOF
