#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/scholar/search_engines/individual/ArXivSearchEngine.py

"""
arXiv Search Engine - Keyword-based paper discovery via arXiv API.

Features:
  - Keyword search across title, abstract, authors
  - Category filtering
  - Date range filtering
  - Rate limit: 1 request per 3 seconds
"""

import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from urllib.parse import quote

from scitex import logging
from scitex.scholar.metadata_engines.individual.ArXivEngine import ArXivEngine
from .._BaseSearchEngine import BaseSearchEngine

logger = logging.getLogger(__name__)


class ArXivSearchEngine(ArXivEngine, BaseSearchEngine):
    """arXiv search engine for keyword-based paper discovery."""

    def search_by_keywords(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search arXiv by keywords.

        Args:
            query: Keyword query
            filters: Optional filters (year_start, year_end)
            max_results: Maximum results

        Returns:
            List of paper metadata dictionaries
        """
        filters = filters or {}

        # Build arXiv query
        # arXiv query format: all:keyword or ti:title or abs:abstract
        search_query = f"all:{query}"

        # Add date range if specified
        # arXiv uses submittedDate:[YYYYMMDD TO YYYYMMDD]
        # But the API query interface is different - we'll filter post-API

        url = f"http://export.arxiv.org/api/query?search_query={quote(search_query)}&start=0&max_results={max_results}"

        logger.info(f"{self.name}: Searching arXiv with query: {query[:50]}...")

        try:
            # Rate limiting: arXiv requires 1 request per 3 seconds
            time.sleep(3)

            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.content)

            # Namespace handling
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            entries = root.findall("atom:entry", ns)

            if not entries:
                logger.info(f"{self.name}: No results found")
                return []

            logger.info(f"{self.name}: Found {len(entries)} results")

            # Convert to standardized format
            results = []
            for entry in entries:
                metadata = self._arxiv_entry_to_standard_format(entry, ns)
                if metadata:
                    # Apply year filter post-API
                    if self._passes_year_filter(metadata, filters):
                        results.append(metadata)

            # Apply other post-filters
            filtered_results = self._apply_post_filters(results, filters)

            return filtered_results

        except Exception as e:
            logger.error(f"{self.name}: Search failed: {e}")
            return []

    def _arxiv_entry_to_standard_format(
        self, entry: ET.Element, ns: dict
    ) -> Dict[str, Any]:
        """Convert arXiv XML entry to standard metadata format."""
        # Extract arXiv ID
        arxiv_id = (
            entry.find("atom:id", ns).text.split("/abs/")[-1]
            if entry.find("atom:id", ns) is not None
            else None
        )

        # Extract title
        title_elem = entry.find("atom:title", ns)
        title = (
            title_elem.text.strip().replace("\n", " ")
            if title_elem is not None
            else None
        )

        # Extract authors
        authors = []
        for author_elem in entry.findall("atom:author", ns):
            name_elem = author_elem.find("atom:name", ns)
            if name_elem is not None:
                authors.append(name_elem.text)

        # Extract abstract
        summary_elem = entry.find("atom:summary", ns)
        abstract = (
            summary_elem.text.strip().replace("\n", " ")
            if summary_elem is not None
            else None
        )

        # Extract published date
        published_elem = entry.find("atom:published", ns)
        year = None
        if published_elem is not None:
            date_str = published_elem.text  # Format: 2024-01-15T12:00:00Z
            year = int(date_str[:4])

        # Extract DOI if available
        doi_elem = entry.find("arxiv:doi", ns)
        doi = doi_elem.text if doi_elem is not None else None

        # Build metadata dict
        metadata = {
            "id": {
                "arxiv": arxiv_id,
                "arxiv_engines": [self.name] if arxiv_id else None,
                "doi": doi,
                "doi_engines": [self.name] if doi else None,
            },
            "basic": {
                "title": title,
                "title_engines": [self.name] if title else None,
                "authors": authors if authors else None,
                "authors_engines": [self.name] if authors else None,
                "abstract": abstract,
                "abstract_engines": [self.name] if abstract else None,
            },
            "publication": {
                "year": year,
                "year_engines": [self.name] if year else None,
            },
            "urls": {
                "pdf": f"http://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None,
                "publisher": f"http://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
            },
        }

        return metadata

    def _passes_year_filter(
        self, metadata: Dict[str, Any], filters: Dict[str, Any]
    ) -> bool:
        """Check if paper passes year filter."""
        year = metadata.get("publication", {}).get("year")
        if year is None:
            return True

        year_start = filters.get("year_start")
        year_end = filters.get("year_end")

        if year_start and year < year_start:
            return False
        if year_end and year > year_end:
            return False

        return True


# EOF
