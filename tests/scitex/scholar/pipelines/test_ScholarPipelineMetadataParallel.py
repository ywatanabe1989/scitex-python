# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/pipelines/ScholarPipelineMetadataParallel.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: ./src/scitex/scholar/pipelines/ScholarPipelineMetadataParallel.py
# 
# """
# ScholarPipelineMetadataParallel - Parallel metadata enrichment (API-only)
# 
# Functionalities:
#   - Enriches multiple papers in parallel with metadata using APIs ONLY
#   - NO browser automation, NO PDF downloads
#   - Fast and lightweight for BibTeX enrichment
#   - Controlled concurrency with semaphore
# 
# Pipeline Steps:
#   1. Create worker tasks with semaphore control
#   2. Each worker uses ScholarPipelineMetadataSingle
#   3. Parallel API calls for metadata enrichment
#   4. Impact factor enrichment via ImpactFactorEngine
#   5. Aggregate results
# 
# Dependencies:
#   - API engines only (no playwright/browser)
# 
# IO:
#   - input: List of Papers or DOI/title strings
#   - output: List of enriched Paper objects (metadata only, no PDFs)
# """
# 
# import asyncio
# from typing import List, Optional, Callable, Dict, Any
# 
# from scitex import logging
# from scitex.scholar.core import Paper, Papers
# from scitex.scholar.pipelines.ScholarPipelineMetadataSingle import (
#     ScholarPipelineMetadataSingle,
# )
# from scitex.scholar.config import ScholarConfig
# 
# logger = logging.getLogger(__name__)
# 
# 
# class ScholarPipelineMetadataParallel:
#     """Orchestrates parallel metadata enrichment using multiple workers (API-only)."""
# 
#     def __init__(
#         self,
#         num_workers: int = 4,
#         config: Optional[ScholarConfig] = None,
#     ):
#         """Initialize parallel metadata enrichment pipeline.
# 
#         Args:
#             num_workers: Number of parallel workers for API calls (default: 4)
#             config: ScholarConfig instance (optional)
#         """
#         self.name = self.__class__.__name__
#         self.num_workers = num_workers
#         self.config = config or ScholarConfig()
# 
#         logger.info(
#             f"{self.name}: Initialized with {num_workers} workers (API-only, no browser)"
#         )
# 
#     async def enrich_papers_async(
#         self,
#         papers: List[Paper],
#         force: bool = False,
#         on_progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
#     ) -> List[Paper]:
#         """Enrich multiple papers in parallel using API calls only.
# 
#         Args:
#             papers: List of Paper objects to enrich
#             force: If True, re-fetch even if metadata exists
#             on_progress: Optional callback(current, total, info) called after each paper
#                 - current: Number of papers processed so far (1-indexed)
#                 - total: Total number of papers
#                 - info: Dict with keys:
#                     - 'title': Paper title (str)
#                     - 'success': Whether enrichment succeeded (bool)
#                     - 'error': Error message if failed (str or None)
#                     - 'index': 0-indexed paper position (int)
# 
#         Returns:
#             List of enriched Paper objects
#         """
#         if not papers:
#             logger.warning(f"{self.name}: No papers to enrich")
#             return []
# 
#         total = len(papers)
#         effective_workers = min(self.num_workers, total)
# 
#         logger.info(
#             f"{self.name}: Enriching {total} papers with {effective_workers} workers"
#         )
#         logger.info(
#             f"{self.name}: on_progress callback={'PROVIDED' if on_progress else 'NOT PROVIDED'}"
#         )
# 
#         # Create semaphore for controlled parallelism
#         semaphore = asyncio.Semaphore(effective_workers)
# 
#         # Counter for completed papers (for progress callback)
#         completed_count = 0
#         progress_lock = asyncio.Lock()
# 
#         async def enrich_with_semaphore(paper: Paper, index: int) -> Paper:
#             """Enrich one paper with semaphore control."""
#             nonlocal completed_count
# 
#             async with semaphore:
#                 worker_id = index % effective_workers
#                 logger.info(
#                     f"{self.name}: [{index + 1}/{total}] Worker {worker_id} processing..."
#                 )
# 
#                 # Create single pipeline for this paper
#                 single_pipeline = ScholarPipelineMetadataSingle(config=self.config)
# 
#                 # Enrich the paper
#                 error_msg = None
#                 success = False
#                 try:
#                     enriched_paper = await single_pipeline.enrich_paper_async(
#                         paper, force=force
#                     )
#                     success = True
#                 except Exception as e:
#                     enriched_paper = paper  # Return original on error
#                     error_msg = str(e)
#                     logger.error(f"{self.name}: Error enriching paper {index}: {e}")
# 
#                 # Call progress callback if provided
#                 if on_progress:
#                     logger.info(
#                         f"{self.name}: About to invoke callback for paper {index + 1}/{total}"
#                     )
#                     async with progress_lock:
#                         completed_count += 1
# 
#                         # Get paper title safely
#                         title = "Untitled"
#                         if hasattr(paper, "metadata") and hasattr(
#                             paper.metadata, "basic"
#                         ):
#                             if paper.metadata.basic.title:
#                                 title = paper.metadata.basic.title
# 
#                         # Invoke callback (synchronous callback from async context)
#                         try:
#                             logger.info(
#                                 f"{self.name}: Calling on_progress({completed_count}, {total}, title={title[:30]}...)"
#                             )
#                             on_progress(
#                                 completed_count,
#                                 total,
#                                 {
#                                     "title": title,
#                                     "success": success,
#                                     "error": error_msg,
#                                     "index": index,
#                                 },
#                             )
#                             logger.info(f"{self.name}: Callback completed successfully")
#                         except Exception as cb_error:
#                             logger.error(
#                                 f"{self.name}: Progress callback error: {cb_error}"
#                             )
# 
#                 return enriched_paper
# 
#         # Create tasks for all papers
#         tasks = [enrich_with_semaphore(paper, i) for i, paper in enumerate(papers)]
# 
#         # Process with controlled parallelism
#         enriched_papers = await asyncio.gather(*tasks, return_exceptions=True)
# 
#         # Filter and count results
#         successful = []
#         errors = 0
# 
#         for i, result in enumerate(enriched_papers):
#             if isinstance(result, Exception):
#                 logger.error(f"{self.name}: Paper {i + 1} raised exception: {result}")
#                 errors += 1
#                 # Return original paper on error
#                 successful.append(papers[i])
#             elif result:
#                 successful.append(result)
#             else:
#                 # None result - return original
#                 successful.append(papers[i])
# 
#         logger.success(
#             f"{self.name}: Enriched {total - errors}/{total} papers successfully"
#         )
# 
#         if errors > 0:
#             logger.warning(f"{self.name}: {errors} papers had errors")
# 
#         return successful
# 
#     async def enrich_from_doi_or_title_list_async(
#         self,
#         doi_or_title_list: List[str],
#         force: bool = False,
#     ) -> List[Paper]:
#         """Enrich papers from a list of DOI or title strings.
# 
#         Args:
#             doi_or_title_list: List of DOI or title strings
#             force: If True, re-fetch even if cached
# 
#         Returns:
#             List of enriched Paper objects
#         """
#         if not doi_or_title_list:
#             logger.warning(f"{self.name}: Empty input list")
#             return []
# 
#         logger.info(
#             f"{self.name}: Creating papers from {len(doi_or_title_list)} queries"
#         )
# 
#         # Convert queries to Paper objects
#         papers = []
#         for query in doi_or_title_list:
#             paper = Paper()
# 
#             # Check if DOI or title
#             is_doi = query.strip().startswith("10.")
# 
#             if is_doi:
#                 paper.metadata.id.doi = query.strip()
#                 paper.metadata.id.doi_engines = ["user_input"]
#             else:
#                 paper.metadata.basic.title = query.strip()
# 
#             papers.append(paper)
# 
#         # Enrich all papers
#         return await self.enrich_papers_async(papers, force=force)
# 
#     async def enrich_papers_collection_async(
#         self,
#         papers_collection: Papers,
#         force: bool = False,
#     ) -> Papers:
#         """Enrich a Papers collection with metadata.
# 
#         Args:
#             papers_collection: Papers collection to enrich
#             force: If True, re-fetch even if metadata exists
# 
#         Returns:
#             Enriched Papers collection
#         """
#         # Extract list of papers
#         paper_list = list(papers_collection)
# 
#         # Enrich all papers
#         enriched_list = await self.enrich_papers_async(paper_list, force=force)
# 
#         # Return as Papers collection
#         return Papers(enriched_list, project=papers_collection.project)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/pipelines/ScholarPipelineMetadataParallel.py
# --------------------------------------------------------------------------------
