#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/scholar/pipelines/ScholarPipelineSearchSingle.py

"""
ScholarPipelineSearchSingle - Sequential academic paper search across multiple engines

Functionalities:
  - Searches multiple academic databases sequentially (one at a time)
  - Better for rate-limited scenarios
  - Aggregates and deduplicates results
  - Supports query-based search with filters

Pipeline Steps:
  1. Execute search queries across engines one by one
  2. Collect results from each engine
  3. Deduplicate by DOI/title
  4. Aggregate metadata
  5. Return ranked results

Dependencies:
  - API engines (PubMed, CrossRef, arXiv, Semantic Scholar, OpenAlex)

IO:
  - input: Search query string, filters, max_results
  - output: List of Paper objects matching the query
"""

import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime

from scitex import logging
from scitex.scholar.core import Paper
from scitex.scholar.search_engines.individual.PubMedSearchEngine import (
    PubMedSearchEngine,
)
from scitex.scholar.search_engines.individual.CrossRefSearchEngine import (
    CrossRefSearchEngine,
)
from scitex.scholar.search_engines.individual.ArXivSearchEngine import ArXivSearchEngine
from scitex.scholar.search_engines.individual.SemanticScholarSearchEngine import (
    SemanticScholarSearchEngine,
)
from scitex.scholar.search_engines.individual.OpenAlexSearchEngine import (
    OpenAlexSearchEngine,
)

logger = logging.getLogger(__name__)


class ScholarPipelineSearchSingle:
    """Orchestrates sequential academic paper search across multiple engines."""

    def __init__(
        self,
        use_cache: bool = True,
        email: str = None,
    ):
        """Initialize sequential search pipeline.

        Args:
            use_cache: Whether to use caching for API results
            email: User email for API rate limit benefits (PubMed, CrossRef, OpenAlex)
        """
        self.name = self.__class__.__name__
        self.use_cache = use_cache
        self.email = email or "research@scitex.io"

        # Initialize search engines with email for rate limit benefits
        self.engines = {
            "PubMed": PubMedSearchEngine(email=self.email),
            "CrossRef": CrossRefSearchEngine(email=self.email),
            "arXiv": ArXivSearchEngine(email=self.email),
            "Semantic_Scholar": SemanticScholarSearchEngine(email=self.email),
            "OpenAlex": OpenAlexSearchEngine(email=self.email),
        }

        # Statistics
        self.stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "cache_hits": 0,
            "total_time": 0.0,
            "engine_stats": {
                name: {
                    "attempts": 0,
                    "successes": 0,
                    "failures": 0,
                    "avg_response_time": 0.0,
                    "total_results": 0,
                }
                for name in self.engines.keys()
            },
        }

        logger.info(
            f"{self.name}: Initialized with engines: {', '.join(self.engines.keys())}"
        )

    async def search_async(
        self,
        query: str = None,
        title: str = None,
        doi: str = None,
        filters: Dict = None,
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """Search for papers across multiple engines sequentially.

        Args:
            query: Search query string
            title: Title to search for (alternative to query)
            doi: DOI to search for (alternative to query)
            filters: Additional filters (year_start, year_end, open_access, etc.)
            max_results: Maximum number of results to return

        Returns:
            Dict with:
                - results: List of paper dictionaries
                - metadata: Search metadata (engines used, timing, etc.)
        """
        start_time = datetime.now()
        self.stats["total_searches"] += 1

        # Determine search mode
        search_query = query or title or doi
        if not search_query:
            raise ValueError("Must provide query, title, or doi")

        logger.info(
            f"{self.name}: Starting sequential search: query='{search_query[:50]}...', "
            f"filters={filters}"
        )

        # Prepare search parameters
        filters = filters or {}
        per_engine_limit = max(10, max_results // len(self.engines))

        # Search engines one by one
        all_papers = []
        engines_used = []

        for engine_name, engine in self.engines.items():
            try:
                results = await self._search_engine(
                    engine_name=engine_name,
                    engine=engine,
                    query=search_query,
                    filters=filters,
                    max_results=per_engine_limit,
                )

                if results:
                    all_papers.extend(results)
                    engines_used.append(engine_name)
                    self.stats["engine_stats"][engine_name]["successes"] += 1
                    self.stats["engine_stats"][engine_name]["total_results"] += len(
                        results
                    )

                    logger.info(
                        f"{self.name}: {engine_name} returned {len(results)} results"
                    )

            except Exception as e:
                logger.error(f"{self.name}: {engine_name} search failed: {e}")
                self.stats["engine_stats"][engine_name]["failures"] += 1

        # Deduplicate by DOI and title
        unique_papers = self._deduplicate_papers(all_papers)

        # Enrich papers with impact factors
        unique_papers = self._enrich_impact_factors(unique_papers)

        # Sort if requested
        sort_by = filters.get("sort_by", "relevance")
        if sort_by and sort_by != "relevance":
            unique_papers = self._sort_papers(
                unique_papers, sort_by, filters.get("sort_order", "desc")
            )

        # Limit to max_results
        unique_papers = unique_papers[:max_results]

        # Convert to response format
        response_papers = [self._paper_to_dict(paper) for paper in unique_papers]

        # Calculate timing
        end_time = datetime.now()
        search_time = (end_time - start_time).total_seconds()
        self.stats["total_time"] += search_time
        self.stats["successful_searches"] += 1

        logger.success(
            f"{self.name}: Sequential search completed in {search_time:.2f}s, "
            f"{len(engines_used)}/{len(self.engines)} engines returned results, "
            f"found {len(response_papers)} unique papers"
        )

        return {
            "results": response_papers,
            "metadata": {
                "query": search_query,
                "filters": filters,
                "engines_used": engines_used,
                "total_engines": len(self.engines),
                "successful_engines": len(engines_used),
                "total_results": len(response_papers),
                "search_time": search_time,
                "timestamp": datetime.now().isoformat(),
            },
        }

    async def _search_engine(
        self,
        engine_name: str,
        engine: Any,
        query: str,
        filters: Dict,
        max_results: int,
    ) -> List[Paper]:
        """Search a single engine."""
        self.stats["engine_stats"][engine_name]["attempts"] += 1
        engine_start = datetime.now()

        try:
            # Run synchronous search in executor to make it async
            loop = asyncio.get_event_loop()

            # Prepare filters for API
            api_filters = {
                "year_start": filters.get("year_start"),
                "year_end": filters.get("year_end"),
                "open_access": filters.get("open_access"),
            }

            # Search with new search_by_keywords method
            results_list = await loop.run_in_executor(
                None,
                lambda: engine.search_by_keywords(
                    query=query, filters=api_filters, max_results=max_results
                ),
            )

            # Convert result dicts to Paper objects
            results = []
            for result in results_list or []:
                paper = Paper()

                # Populate paper metadata from result
                if "id" in result:
                    if result["id"].get("doi"):
                        paper.metadata.id.doi = result["id"]["doi"]
                        paper.metadata.id.doi_engines = result["id"].get(
                            "doi_engines", []
                        )
                    if result["id"].get("pmid"):
                        paper.metadata.id.pmid = result["id"]["pmid"]
                    if result["id"].get("arxiv"):
                        paper.metadata.id.arxiv_id = result["id"]["arxiv"]

                if "basic" in result:
                    if result["basic"].get("title"):
                        paper.metadata.basic.title = result["basic"]["title"]
                    if result["basic"].get("authors"):
                        paper.metadata.basic.authors = result["basic"]["authors"]
                    if result["basic"].get("abstract"):
                        paper.metadata.basic.abstract = result["basic"]["abstract"]
                    if result["basic"].get("keywords"):
                        paper.metadata.basic.keywords = result["basic"]["keywords"]

                if "publication" in result:
                    if result["publication"].get("year"):
                        paper.metadata.basic.year = result["publication"]["year"]
                    if result["publication"].get("journal"):
                        paper.metadata.publication.journal = result["publication"][
                            "journal"
                        ]

                if "metrics" in result:
                    if result["metrics"].get("citation_count"):
                        paper.metadata.citation_count.total = result["metrics"][
                            "citation_count"
                        ]
                    if "is_open_access" in result["metrics"]:
                        paper.metadata.access.is_open_access = result["metrics"][
                            "is_open_access"
                        ]
                        paper.metadata.access.is_open_access_engines = [engine_name]

                if "urls" in result:
                    if result["urls"].get("pdf"):
                        # pdfs is a list of dicts with url/source keys
                        paper.metadata.url.pdfs = [
                            {"url": result["urls"]["pdf"], "source": "search"}
                        ]
                        # If this is an open access paper, also store the PDF URL as oa_url
                        if paper.metadata.access.is_open_access:
                            paper.metadata.access.oa_url = result["urls"]["pdf"]
                            paper.metadata.access.oa_url_engines = [engine_name]
                    if result["urls"].get("publisher"):
                        paper.metadata.url.publisher = result["urls"]["publisher"]
                    if result["urls"].get("doi_url"):
                        paper.metadata.url.doi = result["urls"]["doi_url"]

                results.append(paper)

            # Update stats
            response_time = (datetime.now() - engine_start).total_seconds()
            stats = self.stats["engine_stats"][engine_name]
            n = stats["attempts"]
            stats["avg_response_time"] = (
                stats["avg_response_time"] * (n - 1) + response_time
            ) / n

            logger.info(
                f"{self.name}: {engine_name} returned {len(results)} results "
                f"in {response_time:.2f}s"
            )

            return results

        except Exception as e:
            logger.error(f"{self.name}: {engine_name} search failed: {e}")
            return []

    def _sort_papers(
        self, papers: List[Paper], sort_by: str, sort_order: str = "desc"
    ) -> List[Paper]:
        """Sort papers by specified criteria.

        Args:
            papers: List of Paper objects
            sort_by: Sort criterion ('citations', 'year', 'title')
            sort_order: Sort order ('asc' or 'desc')

        Returns:
            Sorted list of papers
        """

        def get_sort_key(paper):
            if sort_by == "citations":
                if hasattr(paper.metadata, "citation_count"):
                    return paper.metadata.citation_count.total or 0
                return 0
            elif sort_by == "year":
                if hasattr(paper.metadata, "basic"):
                    return paper.metadata.basic.year or 0
                return 0
            elif sort_by == "title":
                if hasattr(paper.metadata, "basic"):
                    return paper.metadata.basic.title or ""
                return ""
            return 0

        reverse = sort_order == "desc"
        sorted_papers = sorted(papers, key=get_sort_key, reverse=reverse)

        logger.info(
            f"{self.name}: Sorted {len(papers)} papers by {sort_by} ({sort_order})"
        )
        return sorted_papers

    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """Deduplicate papers by DOI and title."""
        seen_dois = set()
        seen_titles = set()
        unique_papers = []

        for paper in papers:
            # Get DOI
            doi = None
            if hasattr(paper, "metadata") and hasattr(paper.metadata, "id"):
                doi = paper.metadata.id.doi

            # Get title
            title = None
            if hasattr(paper, "metadata") and hasattr(paper.metadata, "basic"):
                title = paper.metadata.basic.title

            # Skip if duplicate
            if doi and doi in seen_dois:
                continue
            if title and title.lower().strip() in seen_titles:
                continue

            # Add to unique set
            if doi:
                seen_dois.add(doi)
            if title:
                seen_titles.add(title.lower().strip())

            unique_papers.append(paper)

        logger.info(
            f"{self.name}: Deduplicated {len(papers)} papers to {len(unique_papers)} unique"
        )

        return unique_papers

    def _enrich_impact_factors(self, papers: List[Paper]) -> List[Paper]:
        """Enrich papers with impact factors from JCR database.

        Args:
            papers: List of Paper objects

        Returns:
            Papers with enriched impact factors
        """
        from scitex.scholar.impact_factor import ImpactFactorEngine

        engine = ImpactFactorEngine()
        enriched = 0

        for paper in papers:
            # Skip if already has impact factor
            if (
                hasattr(paper.metadata, "publication")
                and paper.metadata.publication.impact_factor
            ):
                continue

            # Get journal name
            journal = None
            if (
                hasattr(paper.metadata, "publication")
                and paper.metadata.publication.journal
            ):
                journal = paper.metadata.publication.journal

            if journal:
                try:
                    metrics = engine.get_metrics(journal)
                    if metrics and metrics.get("impact_factor"):
                        paper.metadata.publication.impact_factor = metrics[
                            "impact_factor"
                        ]
                        if not hasattr(
                            paper.metadata.publication, "impact_factor_engines"
                        ):
                            paper.metadata.publication.impact_factor_engines = []
                        paper.metadata.publication.impact_factor_engines.append(
                            metrics.get("source", "ImpactFactorEngine")
                        )
                        enriched += 1
                except Exception:
                    pass  # Skip on error

        if enriched > 0:
            logger.info(f"{self.name}: Enriched {enriched} papers with impact factors")

        return papers

    def _paper_to_dict(self, paper: Paper) -> Dict[str, Any]:
        """Convert Paper object to dictionary for API response."""
        result = {
            "title": "",
            "authors": [],
            "year": None,
            "abstract": "",
            "journal": "",
            "impact_factor": None,
            "doi": "",
            "pmid": "",
            "arxiv_id": "",
            "citation_count": 0,
            "is_open_access": False,
            "pdf_url": "",
            "external_url": "",
            "document_type": "article",
            "keywords": [],
            "source_engines": [],
        }

        if not hasattr(paper, "metadata"):
            return result

        meta = paper.metadata

        # Basic info
        if hasattr(meta, "basic"):
            result["title"] = meta.basic.title or ""
            result["authors"] = meta.basic.authors or []
            result["abstract"] = meta.basic.abstract or ""
            result["keywords"] = meta.basic.keywords or []
            result["year"] = meta.basic.year

        # IDs
        if hasattr(meta, "id"):
            result["doi"] = meta.id.doi or ""
            result["pmid"] = meta.id.pmid or ""
            result["arxiv_id"] = meta.id.arxiv_id or ""
            result["source_engines"] = meta.id.doi_engines or []

        # Publication info
        if hasattr(meta, "publication"):
            result["journal"] = meta.publication.journal or ""
            result["impact_factor"] = meta.publication.impact_factor

        # Metrics
        if hasattr(meta, "citation_count"):
            result["citation_count"] = meta.citation_count.total or 0

        # Access metadata
        if hasattr(meta, "access"):
            result["is_open_access"] = meta.access.is_open_access or False
            result["oa_status"] = meta.access.oa_status
            result["oa_url"] = meta.access.oa_url
        else:
            result["is_open_access"] = False

        # URLs
        if hasattr(meta, "url"):
            # Extract first PDF URL from pdfs list
            result["pdf_url"] = meta.url.pdfs[0]["url"] if meta.url.pdfs else ""
            result["external_url"] = meta.url.publisher or meta.url.doi or ""

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self.stats,
            "success_rate": (
                100 * self.stats["successful_searches"] / self.stats["total_searches"]
                if self.stats["total_searches"] > 0
                else 0
            ),
            "average_time": (
                self.stats["total_time"] / self.stats["successful_searches"]
                if self.stats["successful_searches"] > 0
                else 0
            ),
        }

    def get_engine_capabilities(self, engine_name: str) -> Dict[str, Any]:
        """Get capabilities of a specific engine."""
        if engine_name not in self.engines:
            return {}

        return {
            "name": engine_name,
            "supports_query_search": True,
            "supports_doi_lookup": True,
            "supports_filters": True,
            "max_results": 1000,
            "stats": self.stats["engine_stats"].get(engine_name, {}),
        }


# EOF
