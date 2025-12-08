#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/scholar/pipelines/ScholarPipelineSearchParallel.py

"""
ScholarPipelineSearchParallel - Parallel academic paper search across multiple engines

Functionalities:
  - Searches multiple academic databases in parallel (PubMed, CrossRef, arXiv, etc.)
  - Aggregates and deduplicates results
  - Fast parallel execution with controlled concurrency
  - Supports query-based search with filters

Pipeline Steps:
  1. Execute search queries across all engines in parallel
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
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime

from scitex import logging
from scitex.scholar.core import Paper
from scitex.scholar.core import normalize_journal_name
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


class ScholarPipelineSearchParallel:
    """Orchestrates parallel academic paper search across multiple engines."""

    def __init__(
        self,
        max_workers: int = 5,
        timeout_per_engine: float = 30.0,
        use_cache: bool = True,
        email: str = None,
    ):
        """Initialize parallel search pipeline.

        Args:
            max_workers: Maximum number of parallel engine queries
            timeout_per_engine: Timeout for each engine in seconds
            use_cache: Whether to use caching for API results
            email: User email for API rate limit benefits (PubMed, CrossRef, OpenAlex)
        """
        self.name = self.__class__.__name__
        self.max_workers = max_workers
        self.timeout_per_engine = timeout_per_engine
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
            "parallel_speedup": 0.0,
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
            f"{self.name}: Initialized with {max_workers} workers, "
            f"engines: {', '.join(self.engines.keys())}"
        )

    async def search_async(
        self,
        query: str = None,
        title: str = None,
        doi: str = None,
        search_fields: List[str] = None,
        filters: Dict = None,
        max_results: int = 100,
        on_progress: Optional[Callable[[str, str, int], None]] = None,
    ) -> Dict[str, Any]:
        """Search for papers across multiple engines in parallel.

        Args:
            query: Search query string
            title: Title to search for (alternative to query)
            doi: DOI to search for (alternative to query)
            search_fields: Fields to search in ['title', 'abstract', 'keywords']
            filters: Additional filters (year_start, year_end, open_access, etc.)
            max_results: Maximum number of results to return
            on_progress: Optional callback(engine_name, status, count) called for each engine
                - engine_name: Name of the search engine (str)
                - status: 'loading', 'completed', or 'error' (str)
                - count: Number of results found (int)

        Returns:
            Dict with:
                - results: List of paper dictionaries
                - metadata: Search metadata (engines used, timing, engine_counts, etc.)
        """
        start_time = datetime.now()
        self.stats["total_searches"] += 1

        # Determine search mode
        search_query = query or title or doi
        if not search_query:
            raise ValueError("Must provide query, title, or doi")

        logger.info(
            f"{self.name}: Starting parallel search: query='{search_query[:50]}...', "
            f"fields={search_fields}, filters={filters}"
        )

        # Prepare search parameters
        filters = filters or {}
        search_fields = search_fields or ["title", "abstract"]
        per_engine_limit = max(10, max_results // len(self.engines))

        # Initialize engine counts dictionary
        engine_counts = {}

        # Create search tasks for each engine
        tasks = []
        engine_names = []

        for engine_name, engine in self.engines.items():
            task = self._search_engine(
                engine_name=engine_name,
                engine=engine,
                query=search_query,
                search_fields=search_fields,
                filters=filters,
                max_results=per_engine_limit,
                on_progress=on_progress,
                engine_counts=engine_counts,
            )
            tasks.append(task)
            engine_names.append(engine_name)

        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        all_papers = []
        engines_used = []

        for engine_name, result in zip(engine_names, results):
            if isinstance(result, Exception):
                logger.error(f"{self.name}: {engine_name} search failed: {result}")
                self.stats["engine_stats"][engine_name]["failures"] += 1
                continue

            logger.info(
                f"{self.name}: {engine_name} returned {len(result) if result else 0} results (type: {type(result).__name__})"
            )

            if result:
                all_papers.extend(result)
                engines_used.append(engine_name)
                self.stats["engine_stats"][engine_name]["successes"] += 1
                self.stats["engine_stats"][engine_name]["total_results"] += len(result)

        # Deduplicate by DOI and title
        unique_papers = self._deduplicate_papers(all_papers)

        # Enrich papers with missing citation counts (e.g., from PubMed)
        unique_papers = await self._enrich_citations(unique_papers)

        # Enrich papers with impact factors
        unique_papers = self._enrich_impact_factors(unique_papers)

        # Apply threshold filters
        unique_papers = self._apply_threshold_filters(unique_papers, filters)

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
            f"{self.name}: Parallel search completed in {search_time:.2f}s, "
            f"{len(engines_used)}/{len(self.engines)} engines returned results, "
            f"found {len(response_papers)} unique papers"
        )

        return {
            "results": response_papers,
            "metadata": {
                "query": search_query,
                "search_fields": search_fields,
                "filters": filters,
                "engines_used": engines_used,
                "total_engines": len(self.engines),
                "successful_engines": len(engines_used),
                "total_results": len(response_papers),
                "search_time": search_time,
                "timestamp": datetime.now().isoformat(),
                "engine_counts": engine_counts,  # Per-engine result counts
            },
        }

    async def _search_engine(
        self,
        engine_name: str,
        engine: Any,
        query: str,
        search_fields: List[str],
        filters: Dict,
        max_results: int,
        on_progress: Optional[Callable[[str, str, int], None]] = None,
        engine_counts: Dict[str, int] = None,
    ) -> List[Paper]:
        """Search a single engine with timeout and progress callback.

        Args:
            engine_name: Name of the search engine
            engine: Engine instance
            query: Search query
            search_fields: Fields to search in
            filters: Search filters
            max_results: Maximum results to return
            on_progress: Optional callback(engine_name, status, count)
            engine_counts: Shared dictionary to store per-engine counts
        """
        self.stats["engine_stats"][engine_name]["attempts"] += 1
        engine_start = datetime.now()

        # Notify start of search
        if on_progress:
            try:
                on_progress(engine_name, "loading", 0)
                logger.info(
                    f"{self.name}: Called on_progress for {engine_name}: loading, 0"
                )
            except Exception as e:
                logger.error(
                    f"{self.name}: Progress callback error for {engine_name}: {e}"
                )

        try:
            # Run synchronous search in executor to make it async
            loop = asyncio.get_event_loop()

            # Prepare filters for API
            api_filters = {
                "year_start": filters.get("year_start"),
                "year_end": filters.get("year_end"),
                "open_access": filters.get("open_access"),
            }

            # Search with timeout using new search_by_keywords method
            results_list = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: engine.search_by_keywords(
                        query=query, filters=api_filters, max_results=max_results
                    ),
                ),
                timeout=self.timeout_per_engine,
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

            # Store engine count
            if engine_counts is not None:
                engine_counts[engine_name] = len(results)

            # Notify completion
            if on_progress:
                try:
                    on_progress(engine_name, "completed", len(results))
                    logger.info(
                        f"{self.name}: Called on_progress for {engine_name}: completed, {len(results)}"
                    )
                except Exception as e:
                    logger.error(
                        f"{self.name}: Progress callback error for {engine_name}: {e}"
                    )

            logger.info(
                f"{self.name}: {engine_name} returned {len(results)} results "
                f"in {response_time:.2f}s"
            )

            return results

        except asyncio.TimeoutError:
            logger.warning(
                f"{self.name}: {engine_name} timed out after {self.timeout_per_engine}s"
            )
            # Notify error
            if on_progress:
                try:
                    on_progress(engine_name, "error", 0)
                except Exception:
                    pass
            return []
        except Exception as e:
            logger.error(f"{self.name}: {engine_name} search failed: {e}")
            import traceback

            logger.error(
                f"{self.name}: {engine_name} traceback:\n{traceback.format_exc()}"
            )
            # Notify error
            if on_progress:
                try:
                    on_progress(engine_name, "error", 0)
                except Exception:
                    pass
            return []

    async def _enrich_citations(self, papers: List[Paper]) -> List[Paper]:
        """Enrich papers with citation counts from OpenAlex (fast and comprehensive).

        Args:
            papers: List of Paper objects

        Returns:
            Papers with enriched citation counts
        """
        enriched = 0
        for paper in papers:
            # Skip if already has citations
            if (
                hasattr(paper.metadata, "citation_count")
                and paper.metadata.citation_count.total
                and paper.metadata.citation_count.total > 0
            ):
                continue

            # Try to enrich via DOI using OpenAlex (fastest)
            doi = None
            if hasattr(paper.metadata, "id") and paper.metadata.id.doi:
                doi = paper.metadata.id.doi

            if doi:
                try:
                    # Use OpenAlex engine for citation lookup
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: self.engines["OpenAlex"].get_metadata_by_doi(doi),
                        ),
                        timeout=2.0,  # Quick timeout for enrichment
                    )

                    if result and result.get("metrics", {}).get("citation_count"):
                        paper.metadata.citation_count.total = result["metrics"][
                            "citation_count"
                        ]
                        enriched += 1

                except asyncio.TimeoutError:
                    pass  # Skip on timeout
                except Exception:
                    pass  # Skip on error

        if enriched > 0:
            logger.info(f"{self.name}: Enriched {enriched} papers with citation counts")

        return papers

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

    def _apply_threshold_filters(
        self, papers: List[Paper], filters: Dict[str, Any]
    ) -> List[Paper]:
        """Apply threshold filters to papers.

        Args:
            papers: List of Paper objects
            filters: Dictionary containing threshold filters:
                - min_year: Minimum publication year
                - max_year: Maximum publication year
                - min_citations: Minimum citation count
                - min_impact_factor: Minimum journal impact factor

        Returns:
            Filtered list of papers
        """
        filtered_papers = []
        filter_stats = {
            "min_year": 0,
            "max_year": 0,
            "min_citations": 0,
            "min_impact_factor": 0,
        }

        for paper in papers:
            # Check year range
            if "min_year" in filters:
                try:
                    min_year = int(filters["min_year"])
                    year = (
                        paper.metadata.basic.year
                        if hasattr(paper.metadata, "basic")
                        else None
                    )
                    if not year or year < min_year:
                        filter_stats["min_year"] += 1
                        continue
                except (ValueError, AttributeError):
                    pass

            if "max_year" in filters:
                try:
                    max_year = int(filters["max_year"])
                    year = (
                        paper.metadata.basic.year
                        if hasattr(paper.metadata, "basic")
                        else None
                    )
                    if not year or year > max_year:
                        filter_stats["max_year"] += 1
                        continue
                except (ValueError, AttributeError):
                    pass

            # Check minimum citations
            if "min_citations" in filters:
                try:
                    min_citations = int(filters["min_citations"])
                    citations = 0
                    if hasattr(paper.metadata, "citation_count"):
                        citations = paper.metadata.citation_count.total or 0
                    if citations < min_citations:
                        filter_stats["min_citations"] += 1
                        continue
                except (ValueError, AttributeError):
                    pass

            # Check minimum impact factor
            if "min_impact_factor" in filters:
                try:
                    min_if = float(filters["min_impact_factor"])
                    impact_factor = 0.0
                    if (
                        hasattr(paper.metadata, "publication")
                        and paper.metadata.publication.impact_factor
                    ):
                        impact_factor = paper.metadata.publication.impact_factor or 0.0
                    if impact_factor < min_if:
                        filter_stats["min_impact_factor"] += 1
                        continue
                except (ValueError, AttributeError):
                    pass

            # Paper passed all filters
            filtered_papers.append(paper)

        # Log filter statistics
        total_filtered = sum(filter_stats.values())
        if total_filtered > 0:
            filter_msgs = [
                f"{count} by {name}"
                for name, count in filter_stats.items()
                if count > 0
            ]
            logger.info(
                f"{self.name}: Filtered {total_filtered} papers ({', '.join(filter_msgs)}), "
                f"{len(filtered_papers)} remain"
            )

        return filtered_papers

    def _sort_papers(
        self, papers: List[Paper], sort_by: str, sort_order: str = "desc"
    ) -> List[Paper]:
        """Sort papers by specified criteria (supports multi-level sorting).

        Args:
            papers: List of Paper objects
            sort_by: Sort criterion string. Can be:
                - Single: 'citations', 'year', 'title'
                - Multi-level (comma-separated): 'citations:desc,year:desc,title:asc'
            sort_order: Default sort order ('asc' or 'desc') - used when not specified in sort_by

        Returns:
            Sorted list of papers
        """
        # Parse sort criteria (supports both old single format and new multi-level format)
        sort_criteria = []

        if "," in sort_by:
            # Multi-level sorting: "citations:desc,year:desc,title:asc"
            for criterion in sort_by.split(","):
                criterion = criterion.strip()
                if ":" in criterion:
                    field, order = criterion.split(":", 1)
                    sort_criteria.append((field.strip(), order.strip()))
                else:
                    sort_criteria.append((criterion, sort_order))
        else:
            # Single-level sorting (backward compatible)
            sort_criteria = [(sort_by, sort_order)]

        def get_sort_value(paper, field):
            """Get sortable value for a specific field."""
            if field == "citations":
                if hasattr(paper.metadata, "citation_count"):
                    return paper.metadata.citation_count.total or 0
                return 0
            elif field == "year":
                if hasattr(paper.metadata, "basic"):
                    return paper.metadata.basic.year or 0
                return 0
            elif field == "title":
                if hasattr(paper.metadata, "basic"):
                    return (paper.metadata.basic.title or "").lower()
                return ""
            elif field == "impact_factor":
                if (
                    hasattr(paper.metadata, "publication")
                    and paper.metadata.publication.impact_factor
                ):
                    return paper.metadata.publication.impact_factor or 0
                return 0
            return 0

        def get_sort_key(paper):
            """Generate multi-level sort key tuple."""
            key_tuple = []
            for field, order in sort_criteria:
                value = get_sort_value(paper, field)
                # Negate numeric values for descending order (so higher values come first)
                if order == "desc" and isinstance(value, (int, float)):
                    value = -value
                # For strings, we'll handle reverse in the sorted() call
                key_tuple.append(value)
            return tuple(key_tuple)

        # Sort by the multi-level key
        # String fields need special handling for desc order
        has_string_fields = any(field in ["title"] for field, _ in sort_criteria)

        if has_string_fields:
            # Use multiple sorted() passes for mixed numeric/string fields (stable sort)
            sorted_papers = papers[:]
            for field, order in reversed(sort_criteria):
                reverse = order == "desc"
                sorted_papers = sorted(
                    sorted_papers,
                    key=lambda p: get_sort_value(p, field),
                    reverse=reverse,
                )
        else:
            # Pure numeric sorting can use single pass
            sorted_papers = sorted(papers, key=get_sort_key)

        if len(sort_criteria) > 1:
            criteria_str = ", then by ".join(
                [f"{field} ({order})" for field, order in sort_criteria]
            )
            logger.info(f"{self.name}: Sorted {len(papers)} papers by {criteria_str}")
        else:
            field, order = sort_criteria[0]
            logger.info(
                f"{self.name}: Sorted {len(papers)} papers by {field} ({order})"
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
            journal_raw = meta.publication.journal or ""
            result["journal"] = (
                normalize_journal_name(journal_raw) if journal_raw else ""
            )
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
