#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-01-22 16:32:41 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/pipelines/ScholarPipelineBibTeX.py


"""
Functionalities:
  - Processes BibTeX files through parallel paper acquisition pipeline
  - Loads papers from BibTeX parallel download update BibTeX with results
  - Supports resumable processing (skips already downloaded papers)
  - Updates BibTeX with enriched metadata and download status

Dependencies:
  - packages:
    - playwright
    - asyncio

IO:
  - input-files:
    - BibTeX files (.bib)

  - output-files:
    - library/MASTER/{paper_id}/metadata.json (multiple papers)
    - library/MASTER/{paper_id}/main.pdf (multiple papers)
    - library/{project}/{paper_id} -> ../MASTER/{paper_id} (multiple symlinks)
    - {input_bibtex}_processed.bib (enriched BibTeX with download status)
"""

import asyncio
from pathlib import Path
from typing import Optional
from typing import Union

import scitex as stx
from scitex import logging
from scitex.scholar.core import Papers
from scitex.scholar.pipelines.ScholarPipelineParallel import (
    ScholarPipelineParallel,
)
from scitex.scholar.storage import BibTeXHandler

logger = logging.getLogger(__name__)

"""Functions & Classes"""
class ScholarPipelineBibTeX:
    """Processes BibTeX files through parallel paper acquisition pipeline"""

    def __init__(
        self,
        num_workers: int = 4,
        browser_mode: str = "stealth",
        base_chrome_profile: str = "system",
    ):
        """Initialize BibTeX pipeline.

        Args:
            num_workers: Number of parallel workers (default: 4)
            browser_mode: Browser mode for all workers ('stealth' or 'interactive')
            base_chrome_profile: Base chrome profile to sync from (default: 'system')
        """
        self.name = self.__class__.__name__
        self.num_workers = num_workers
        self.browser_mode = browser_mode
        self.base_chrome_profile = base_chrome_profile

        # Initialize parallel pipeline
        self.parallel_pipeline = ScholarPipelineParallel(
            num_workers=num_workers,
            browser_mode=browser_mode,
            base_chrome_profile=base_chrome_profile,
        )

        logger.info(
            f"{self.name}: Initialized with {num_workers} workers, mode={browser_mode}"
        )

    async def process_bibtex_file_async(
        self,
        bibtex_path: Union[str, Path],
        project: Optional[str] = None,
        output_bibtex_path: Optional[Union[str, Path]] = None,
    ) -> Papers:
        """Process all papers from a BibTeX file in parallel.

        Args:
            bibtex_path: Path to input BibTeX file
            project: Project name for symlinking (optional)
            output_bibtex_path: Path to save enriched BibTeX (optional, defaults to {input}_processed.bib)

        Returns
        -------
            Papers collection with processed papers
        """
        bibtex_path = Path(bibtex_path)
        if not bibtex_path.exists():
            raise FileNotFoundError(f"BibTeX file not found: {bibtex_path}")

        logger.info(f"{self.name}: Processing BibTeX file: {bibtex_path}")

        # Step 1: Load papers from BibTeX
        bibtex_handler = BibTeXHandler(project=project)
        papers = bibtex_handler.papers_from_bibtex(bibtex_path)

        if not papers:
            logger.warning(f"{self.name}: No papers found in BibTeX file")
            return Papers([], project=project)

        logger.info(f"{self.name}: Loaded {len(papers)} papers from BibTeX")

        # Step 2: Process papers in parallel using ScholarPipelineParallel
        papers_collection = Papers(papers, project=project)
        processed_papers = (
            await self.parallel_pipeline.process_papers_from_collection_async(
                papers=papers_collection,
                project=project,
            )
        )

        # Step 3: Save enriched BibTeX with processing results
        if output_bibtex_path is None:
            # Default: save as {input}_processed.bib
            output_bibtex_path = (
                bibtex_path.parent / f"{bibtex_path.stem}_processed.bib"
            )
        else:
            output_bibtex_path = Path(output_bibtex_path)

        # Convert processed papers to Papers collection
        processed_collection = Papers(processed_papers, project=project)

        # Save enriched BibTeX
        bibtex_handler.papers_to_bibtex(
            processed_collection,
            output_path=output_bibtex_path,
        )

        logger.success(
            f"{self.name}: Processed {len(processed_papers)}/{len(papers)} papers"
        )
        logger.success(
            f"{self.name}: Saved enriched BibTeX: {output_bibtex_path}"
        )

        # Update project bibliography if project specified
        if project:
            try:
                from scitex.scholar.config import ScholarConfig

                config = ScholarConfig()
                bib_handler = BibTeXHandler(project=project, config=config)

                # Setup bibliography with original and processed files
                bib_handler.setup_project_bibliography(
                    project=project,
                    bibtex_files=[bibtex_path, output_bibtex_path],
                )

                logger.success(
                    f"{self.name}: Updated project bibliography: {project}"
                )
            except Exception as e:
                logger.warning(f"Failed to update bibliography: {e}")

        return processed_collection

    async def process_bibtex_text_async(
        self,
        bibtex_text: str,
        project: Optional[str] = None,
        output_bibtex_path: Optional[Union[str, Path]] = None,
    ) -> Papers:
        """Process papers from BibTeX text content in parallel.

        Args:
            bibtex_text: BibTeX content as string
            project: Project name for symlinking (optional)
            output_bibtex_path: Path to save enriched BibTeX (optional)

        Returns
        -------
            Papers collection with processed papers
        """
        logger.info(f"{self.name}: Processing BibTeX text content")

        # Step 1: Load papers from BibTeX text
        bibtex_handler = BibTeXHandler(project=project)
        papers = bibtex_handler.papers_from_bibtex(bibtex_text)

        if not papers:
            logger.warning(f"{self.name}: No papers found in BibTeX text")
            return Papers([], project=project)

        logger.info(
            f"{self.name}: Loaded {len(papers)} papers from BibTeX text"
        )

        # Step 2: Process papers in parallel
        papers_collection = Papers(papers, project=project)
        processed_papers = (
            await self.parallel_pipeline.process_papers_from_collection_async(
                papers=papers_collection,
                project=project,
            )
        )

        # Step 3: Save enriched BibTeX if path provided
        processed_collection = Papers(processed_papers, project=project)

        if output_bibtex_path:
            output_bibtex_path = Path(output_bibtex_path)
            bibtex_handler.papers_to_bibtex(
                processed_collection,
                output_path=output_bibtex_path,
            )
            logger.success(
                f"{self.name}: Saved enriched BibTeX: {output_bibtex_path}"
            )

        logger.success(
            f"{self.name}: Processed {len(processed_papers)}/{len(papers)} papers"
        )

        return processed_collection


@stx.session
def main(
    bibtex: str = None,
    project: str = None,
    output: str = None,
    num_workers: int = 4,
    browser_mode: str = "stealth",
    chrome_profile: str = "system",
    CONFIG=stx.INJECTED,
    logger=stx.INJECTED,
) -> int:
    """Process BibTeX files through parallel paper acquisition pipeline.

    Parameters
    ----------
    bibtex : str
        Path to BibTeX file (required)
    project : str
        Project name for symlinking (optional)
    output : str
        Output BibTeX path (default: {input}_processed.bib)
    num_workers : int
        Number of parallel workers (default: 4)
    browser_mode : str
        Browser mode: 'stealth' or 'interactive' (default: stealth)
    chrome_profile : str
        Base Chrome profile name to sync from (default: system)

    Returns
    -------
    int
        Exit status code (0 for success)
    """
    if not bibtex:
        logger.error("No BibTeX file provided. Use --bibtex")
        return 1

    bibtex_path = Path(bibtex)
    if not bibtex_path.exists():
        logger.error(f"BibTeX file not found: {bibtex_path}")
        return 1

    logger.info(f"Processing BibTeX file: {bibtex_path}")
    logger.info(f"Workers: {num_workers}")
    logger.info(f"Project: {project or 'None'}")

    # Create BibTeX pipeline
    bibtex_pipeline = ScholarPipelineBibTeX(
        num_workers=num_workers,
        browser_mode=browser_mode,
        base_chrome_profile=chrome_profile,
    )

    # Run pipeline
    papers = asyncio.run(
        bibtex_pipeline.process_bibtex_file_async(
            bibtex_path=bibtex_path,
            project=project,
            output_bibtex_path=output,
        )
    )

    logger.success(
        f"BibTeX processing complete: {len(papers)} papers processed"
    )
    return 0


if __name__ == "__main__":
    main()

"""
python -m scitex.scholar.pipelines.ScholarPipelineBibTeX \
    --bibtex ./data/scholar/bib_files/neurovista.bib \
    --project neurovista \
    --num-workers 8 \
    --chrome-profile system \
    --browser-mode interactive
"""

# EOF
