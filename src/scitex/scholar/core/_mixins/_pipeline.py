#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/core/_mixins/_pipeline.py

"""
Pipeline mixin for Scholar class.

Provides paper processing pipeline functionality for single and batch operations.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, List, Optional, Union

from scitex import logging
from scitex.scholar.pdf_download.ScholarPDFDownloader import ScholarPDFDownloader
from scitex.scholar.url_finder.ScholarURLFinder import ScholarURLFinder

if TYPE_CHECKING:
    from ..Paper import Paper
    from ..Papers import Papers

logger = logging.getLogger(__name__)


class PipelineMixin:
    """Mixin providing paper processing pipeline methods."""

    async def process_paper_async(
        self,
        title: Optional[str] = None,
        doi: Optional[str] = None,
        project: Optional[str] = None,
    ) -> Paper:
        """Complete sequential pipeline for processing a single paper.

        Accepts either title OR doi. Uses storage-first approach:
        each stage checks storage before processing.

        Workflow:
          Stage 0: Resolve DOI from title (if needed)
          Stage 1: Load or create Paper from storage
          Stage 2: Find PDF URLs -> save to storage
          Stage 3: Download PDF -> save to storage
          Stage 4: Update project symlinks

        Args:
            title: Paper title (will resolve DOI using engine)
            doi: DOI of the paper (preferred if available)
            project: Project name (uses self.project if None)

        Returns
        -------
            Fully processed Paper object

        Examples
        --------
            paper = await scholar.process_paper_async(doi="10.1038/s41598-017-02626-y")
            paper = await scholar.process_paper_async(title="Attention Is All You Need")
        """
        from ..Paper import Paper

        if not title and not doi:
            raise ValueError("Must provide either title or doi")

        project = project or self.project

        logger.info(f"{'=' * 60}")
        logger.info("Processing paper")
        if title:
            logger.info(f"Title: {title[:50]}...")
        if doi:
            logger.info(f"DOI: {doi}")
        logger.info(f"{'=' * 60}")

        # Stage 0: Resolve DOI from title (if needed)
        if not doi and title:
            logger.info("Stage 0: Resolving DOI from title...")
            results = await self._scholar_engine.search_async(title=title)

            if results and results.get("id", {}).get("doi"):
                doi = results["id"]["doi"]
                logger.success(f"Resolved DOI: {doi}")
            else:
                logger.error(f"Could not resolve DOI from title: {title}")
                raise ValueError(f"Could not resolve DOI from title: {title}")

        paper_id = self.config.path_manager._generate_paper_id(doi=doi)
        storage_path = self.config.get_library_master_dir() / paper_id

        logger.info(f"Paper ID: {paper_id}")
        logger.info(f"Storage: {storage_path}")

        # Stage 1: Load or create Paper from storage
        logger.info("\nStage 1: Loading/creating metadata...")
        if self._library_manager.has_metadata(paper_id):
            paper = self._library_manager.load_paper_from_id(paper_id)
            logger.info("Loaded existing metadata from storage")
        else:
            paper = Paper()
            paper.metadata.set_doi(doi)
            paper.container.scitex_id = paper_id

            if title:
                paper.metadata.basic.title = title

            self._library_manager.save_paper_incremental(paper_id, paper)
            logger.success("Created new paper entry in storage")

        # Stage 2: Check/find URLs
        logger.info("\nStage 2: Checking/finding PDF URLs...")
        if not self._library_manager.has_urls(paper_id):
            logger.info(f"Finding PDF URLs for DOI: {doi}")
            (
                browser,
                context,
            ) = await self._browser_manager.get_authenticated_browser_and_context_async()
            try:
                url_finder = ScholarURLFinder(context, config=self.config)
                urls = await url_finder.find_pdf_urls(doi)

                paper.metadata.url.pdfs = urls
                self._library_manager.save_paper_incremental(paper_id, paper)
                logger.success(f"Found {len(urls)} PDF URLs, saved to storage")
            finally:
                await self._browser_manager.close()
        else:
            logger.info(
                f"PDF URLs already in storage ({len(paper.metadata.url.pdfs)} URLs)"
            )

        # Stage 3: Check/download PDF
        logger.info("\nStage 3: Checking/downloading PDF...")
        if not self._library_manager.has_pdf(paper_id):
            logger.info("Downloading PDF...")
            if paper.metadata.url.pdfs:
                (
                    browser,
                    context,
                ) = await self._browser_manager.get_authenticated_browser_and_context_async()
                try:
                    downloader = ScholarPDFDownloader(context, config=self.config)

                    pdf_url = (
                        paper.metadata.url.pdfs[0]["url"]
                        if isinstance(paper.metadata.url.pdfs[0], dict)
                        else paper.metadata.url.pdfs[0]
                    )
                    temp_path = storage_path / "main.pdf"

                    result = await downloader.download_from_url(
                        pdf_url, temp_path, doi=doi
                    )
                    if result and result.exists():
                        paper.metadata.path.pdfs.append(str(result))
                        self._library_manager.save_paper_incremental(paper_id, paper)
                        logger.success(f"{self.name}: Downloaded PDF, saved to storage")
                    else:
                        logger.warning(f"{self.name}: Failed to download PDF")
                finally:
                    await self._browser_manager.close()
            else:
                logger.warning(f"{self.name}: No PDF URLs available for download")
        else:
            logger.info(f"{self.name}: PDF already in storage")

        # Stage 4: Update project symlinks
        if project and project not in ["master", "MASTER"]:
            logger.info(f"{self.name}: \nStage 4: Updating project symlinks...")
            self._library_manager.update_symlink(
                master_storage_path=storage_path,
                project=project,
            )
            logger.success(f"{self.name}: Updated symlink in project: {project}")

        logger.info(f"\n{'=' * 60}")
        logger.success(f"{self.name}: Paper processing complete")
        logger.info(f"{'=' * 60}\n")

        return paper

    def process_paper(
        self,
        title: Optional[str] = None,
        doi: Optional[str] = None,
        project: Optional[str] = None,
    ) -> Paper:
        """Synchronous wrapper for process_paper_async.

        See process_paper_async() for full documentation.
        """
        return asyncio.run(
            self.process_paper_async(title=title, doi=doi, project=project)
        )

    async def process_papers_async(
        self,
        papers: Union[Papers, List[str]],
        project: Optional[str] = None,
        max_concurrent: int = 3,
    ) -> Papers:
        """Process multiple papers with controlled parallelism.

        Each paper goes through complete sequential pipeline.
        Semaphore controls how many papers process concurrently.

        Architecture:
          - Parallel papers (max_concurrent at a time)
          - Sequential stages per paper
          - Storage checks before each stage

        Args:
            papers: Papers collection or list of DOIs
            project: Project name (uses self.project if None)
            max_concurrent: Maximum concurrent papers (default: 3)
                           Set to 1 for purely sequential processing

        Returns
        -------
            Papers collection with processed papers

        Examples
        --------
            papers = scholar.load_bibtex("papers.bib")
            processed = await scholar.process_papers_async(papers, max_concurrent=3)

            dois = ["10.1038/...", "10.1016/...", "10.1109/..."]
            processed = await scholar.process_papers_async(dois, max_concurrent=1)
        """
        from ..Paper import Paper
        from ..Papers import Papers

        project = project or self.project

        if isinstance(papers, list):
            papers_list = []
            for doi in papers:
                p = Paper()
                p.metadata.set_doi(doi)
                papers_list.append(p)
            papers = Papers(papers_list, project=project, config=self.config)

        total = len(papers)
        logger.info(f"{self.name}: \n{'=' * 60}")
        logger.info(
            f"{self.name}: Processing {total} papers (max_concurrent={max_concurrent})"
        )
        logger.info(f"{self.name}: Project: {project}")
        logger.info(f"{self.name}: {'=' * 60}\n")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(paper, index):
            """Process one paper with semaphore control."""
            async with semaphore:
                logger.info(f"{self.name}: \n[{index}/{total}] Starting paper...")
                try:
                    result = await self.process_paper_async(
                        title=paper.metadata.basic.title,
                        doi=paper.metadata.id.doi,
                        project=project,
                    )
                    logger.success(f"{self.name}: [{index}/{total}] Completed")
                    return result
                except Exception as e:
                    logger.error(f"{self.name}: [{index}/{total}] Failed: {e}")
                    return None

        tasks = [process_with_semaphore(paper, i + 1) for i, paper in enumerate(papers)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_papers = []
        errors = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"{self.name}: Paper {i + 1} raised exception: {result}")
                errors += 1
            elif result is not None:
                processed_papers.append(result)

        logger.info(f"{self.name}: \n{'=' * 60}")
        logger.info(f"{self.name}: Batch Processing Complete")
        logger.info(f"{self.name}:   Total: {total}")
        logger.info(f"{self.name}:   Successful: {len(processed_papers)}")
        logger.info(f"{self.name}:   Failed: {total - len(processed_papers)}")
        logger.info(f"{self.name}:   Errors: {errors}")
        logger.info(f"{self.name}: {'=' * 60}\n")

        return Papers(processed_papers, project=project, config=self.config)

    def process_papers(
        self,
        papers: Union[Papers, List[str]],
        project: Optional[str] = None,
        max_concurrent: int = 3,
    ) -> Papers:
        """Synchronous wrapper for process_papers_async.

        See process_papers_async() for full documentation.
        """
        return asyncio.run(
            self.process_papers_async(
                papers=papers,
                project=project,
                max_concurrent=max_concurrent,
            )
        )


# EOF
