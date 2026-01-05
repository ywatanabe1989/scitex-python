# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/search_engines/ScholarSearchEngine.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: ./src/scitex/scholar/search_engines/ScholarSearchEngine.py
# 
# """
# ScholarSearchEngine - Unified academic paper search interface
# 
# Functionalities:
#   - Unified interface for academic paper search
#   - Supports both parallel and sequential search modes
#   - Query parsing with advanced filters
#   - Aggregates results from multiple academic databases
# 
# Features:
#   - Advanced query syntax (negative keywords, year ranges, filters)
#   - Automatic mode selection (parallel vs sequential)
#   - Result deduplication and ranking
#   - Rich metadata aggregation
# 
# Dependencies:
#   - ScholarPipelineSearchParallel (fast parallel search)
#   - ScholarPipelineSearchSingle (sequential search for rate-limited scenarios)
#   - SearchQueryParser (advanced query parsing)
# 
# IO:
#   - input: Query string with optional filters and search mode
#   - output: List of paper dictionaries with metadata
# """
# 
# from typing import List, Dict, Any, Optional, Literal
# from datetime import datetime
# 
# from scitex import logging
# from scitex.scholar.pipelines.ScholarPipelineSearchParallel import (
#     ScholarPipelineSearchParallel,
# )
# from scitex.scholar.pipelines.ScholarPipelineSearchSingle import (
#     ScholarPipelineSearchSingle,
# )
# from scitex.scholar.pipelines.SearchQueryParser import SearchQueryParser
# 
# logger = logging.getLogger(__name__)
# 
# 
# class ScholarSearchEngine:
#     """Unified academic paper search engine with multiple database support."""
# 
#     def __init__(
#         self,
#         default_mode: Literal["parallel", "single"] = "parallel",
#         use_cache: bool = True,
#         email: str = None,
#     ):
#         """Initialize unified search engine.
# 
#         Args:
#             default_mode: Default search mode ('parallel' or 'single')
#             use_cache: Whether to use caching for API results
#             email: User email for API rate limit benefits (PubMed, CrossRef, OpenAlex)
#         """
#         self.name = self.__class__.__name__
#         self.default_mode = default_mode
#         self.use_cache = use_cache
#         self.email = email
# 
#         # Initialize both pipeline modes with email for rate limit benefits
#         self.parallel_pipeline = ScholarPipelineSearchParallel(
#             max_workers=5,
#             timeout_per_engine=30.0,
#             use_cache=use_cache,
#             email=email,
#         )
# 
#         self.single_pipeline = ScholarPipelineSearchSingle(
#             use_cache=use_cache,
#             email=email,
#         )
# 
#         # Statistics
#         self.stats = {
#             "total_searches": 0,
#             "parallel_searches": 0,
#             "single_searches": 0,
#             "total_results": 0,
#             "avg_search_time": 0.0,
#         }
# 
#         logger.info(
#             f"{self.name}: Initialized with default mode '{default_mode}', "
#             f"cache={'enabled' if use_cache else 'disabled'}"
#         )
# 
#     async def search(
#         self,
#         query: str,
#         mode: Optional[Literal["parallel", "single"]] = None,
#         filters: Optional[Dict[str, Any]] = None,
#         max_results: int = 100,
#         parse_query: bool = True,
#     ) -> Dict[str, Any]:
#         """Search for academic papers across multiple databases.
# 
#         Args:
#             query: Search query string (supports advanced syntax if parse_query=True)
#             mode: Search mode ('parallel' or 'single'), defaults to default_mode
#             filters: Additional filters (merged with parsed query filters)
#             max_results: Maximum number of results to return
#             parse_query: Whether to parse query for advanced syntax
# 
#         Returns:
#             Dict with:
#                 - results: List of paper dictionaries
#                 - metadata: Search metadata (query, filters, timing, etc.)
#                 - stats: Search statistics
# 
#         Examples:
#             # Simple query
#             results = await engine.search("hippocampus")
# 
#             # Advanced query with filters
#             results = await engine.search(
#                 "hippocampus sharp wave -seizure year:2020-2024 if:>5"
#             )
# 
#             # Explicit filters (merged with parsed filters)
#             results = await engine.search(
#                 "epilepsy",
#                 filters={'year_start': 2020, 'open_access': True}
#             )
#         """
#         start_time = datetime.now()
#         self.stats["total_searches"] += 1
# 
#         # Determine search mode
#         search_mode = mode or self.default_mode
# 
#         # Parse query if requested
#         if parse_query:
#             parser = SearchQueryParser(query)
#             parsed_filters = parser.get_filters()
# 
#             # Merge filters (explicit filters override parsed)
#             combined_filters = {**parsed_filters, **(filters or {})}
# 
#             # Build clean query from positive keywords
#             clean_query = " ".join(parsed_filters.get("positive_keywords", [query]))
# 
#             logger.info(
#                 f"{self.name}: Parsed query '{query}' -> "
#                 f"keywords='{clean_query}', filters={combined_filters}"
#             )
#         else:
#             clean_query = query
#             combined_filters = filters or {}
# 
#         # Select pipeline
#         if search_mode == "parallel":
#             pipeline = self.parallel_pipeline
#             self.stats["parallel_searches"] += 1
#             logger.info(f"{self.name}: Using parallel search mode")
#         else:
#             pipeline = self.single_pipeline
#             self.stats["single_searches"] += 1
#             logger.info(f"{self.name}: Using single (sequential) search mode")
# 
#         # Execute search
#         try:
#             result = await pipeline.search_async(
#                 query=clean_query,
#                 filters=combined_filters,
#                 max_results=max_results,
#             )
# 
#             # Update statistics
#             search_time = (datetime.now() - start_time).total_seconds()
#             n = self.stats["total_searches"]
#             self.stats["avg_search_time"] = (
#                 self.stats["avg_search_time"] * (n - 1) + search_time
#             ) / n
#             self.stats["total_results"] += len(result.get("results", []))
# 
#             # Add search engine metadata
#             result["metadata"]["search_mode"] = search_mode
#             result["metadata"]["parsed_query"] = clean_query if parse_query else None
#             result["metadata"]["original_query"] = query
# 
#             logger.success(
#                 f"{self.name}: Search completed in {search_time:.2f}s, "
#                 f"found {len(result.get('results', []))} papers"
#             )
# 
#             return result
# 
#         except Exception as e:
#             logger.error(f"{self.name}: Search failed: {e}")
#             raise
# 
#     async def search_by_doi(
#         self,
#         doi: str,
#         mode: Optional[Literal["parallel", "single"]] = None,
#     ) -> Dict[str, Any]:
#         """Search for a paper by DOI.
# 
#         Args:
#             doi: DOI identifier
#             mode: Search mode (defaults to default_mode)
# 
#         Returns:
#             Dict with single paper result or empty results
#         """
#         logger.info(f"{self.name}: Searching by DOI: {doi}")
#         return await self.search(
#             query=doi,
#             mode=mode,
#             filters={},
#             max_results=1,
#             parse_query=False,
#         )
# 
#     async def search_by_title(
#         self,
#         title: str,
#         mode: Optional[Literal["parallel", "single"]] = None,
#         max_results: int = 10,
#     ) -> Dict[str, Any]:
#         """Search for papers by title.
# 
#         Args:
#             title: Paper title
#             mode: Search mode (defaults to default_mode)
#             max_results: Maximum results
# 
#         Returns:
#             Dict with matching papers
#         """
#         logger.info(f"{self.name}: Searching by title: {title[:50]}...")
#         return await self.search(
#             query=title,
#             mode=mode,
#             filters={},
#             max_results=max_results,
#             parse_query=False,
#         )
# 
#     def get_statistics(self) -> Dict[str, Any]:
#         """Get search engine statistics.
# 
#         Returns:
#             Dict with statistics from engine and pipelines
#         """
#         return {
#             "engine_stats": self.stats,
#             "parallel_pipeline_stats": self.parallel_pipeline.get_statistics(),
#             "single_pipeline_stats": self.single_pipeline.get_statistics(),
#         }
# 
#     def get_supported_engines(self) -> List[str]:
#         """Get list of supported academic databases.
# 
#         Returns:
#             List of engine names
#         """
#         return list(self.parallel_pipeline.engines.keys())
# 
#     def get_engine_capabilities(self, engine_name: str) -> Dict[str, Any]:
#         """Get capabilities of a specific engine.
# 
#         Args:
#             engine_name: Name of the engine
# 
#         Returns:
#             Dict with engine capabilities
#         """
#         return self.parallel_pipeline.get_engine_capabilities(engine_name)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/search_engines/ScholarSearchEngine.py
# --------------------------------------------------------------------------------
