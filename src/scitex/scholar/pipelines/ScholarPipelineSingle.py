#!/usr/bin/env python3
# Timestamp: "2026-01-22 (ywatanabe)"
# File: src/scitex/scholar/pipelines/ScholarPipelineSingle.py
"""
Single paper acquisition pipeline orchestrator.

Functionalities:
  - Orchestrates full paper acquisition pipeline from query to storage
  - Single command: query (DOI/title) + project -> complete paper in library
  - Coordinates all workers: metadata, URLs, download, extraction, storage

IO:
  - output-files:
    - library/MASTER/{paper_id}/metadata.json
    - library/MASTER/{paper_id}/main.pdf
    - library/MASTER/{paper_id}/content.txt
    - library/MASTER/{paper_id}/tables.json
    - library/MASTER/{paper_id}/images/
    - library/{project}/{paper_id} -> ../MASTER/{paper_id}
"""

from __future__ import annotations

import asyncio
from typing import Optional

import scitex as stx
from scitex import logging
from scitex.scholar.storage import PaperIO

from ._single_steps import PipelineHelpersMixin, PipelineStepsMixin

logger = logging.getLogger(__name__)


class ScholarPipelineSingle(PipelineStepsMixin, PipelineHelpersMixin):
    """Orchestrates full paper acquisition pipeline."""

    def __init__(
        self, browser_mode: str = "interactive", chrome_profile: str = "system"
    ):
        self.name = self.__class__.__name__
        self.browser_mode = browser_mode
        self.chrome_profile = chrome_profile

    async def process_single_paper(
        self,
        doi_or_title: str,
        project: Optional[str] = None,
        force: bool = False,
    ):
        """Process single paper from query (DOI or Title) to complete storage.

        Pipeline:
        1. Normalize as DOI
        2. Create Paper object (resolve DOI from title if needed)
        3. Add paper ID (8-digit hash)
        4. Resolve metadata (ScholarEngine)
        5. Setup browser
        6. Find PDF URLs
        7. Download PDF
        8. Extract content (text, tables, images)
        9. Link to project (if specified)
        10. Log final status

        Args:
            doi_or_title: DOI or title string
            project: Optional project name for symlinking
            force: If True, ignore existing files and force fresh processing

        Returns
        -------
            Tuple of (Complete Paper object, symlink_path)
        """
        # Step 1-3: Initialize
        doi = self._step_01_normalize_as_doi(doi_or_title)
        paper = await self._step_02_create_paper(doi, doi_or_title)
        paper = self._step_03_add_paper_id(paper)

        io = PaperIO(paper)
        logger.info(f"{self.name}: Paper directory: {io.paper_dir}")

        with logger.to(io.paper_dir / "logs" / "pipeline.log"):
            # Step 4: Metadata
            paper = await self._step_04_resolve_metadata(paper, io, force)

            # Steps 5-7: Browser and PDF
            browser_manager, context, auth_gateway = await self._step_05_setup_browser(
                paper, io
            )
            if context:
                await self._step_06_find_pdf_urls(
                    paper, io, context, auth_gateway, force, browser_manager
                )
                await self._step_07_download_pdf(
                    paper, io, context, auth_gateway, force, browser_manager
                )
            if browser_manager:
                await browser_manager.close()

            # Step 8: Content extraction
            self._step_08_extract_content(io, force)

            # Step 9-10: Finalize
            symlink_path = self._step_09_link_to_project(paper, io, project)
            self._step_10_log_final_status(io)

            return paper, symlink_path


@stx.session
def main(
    doi_or_title: str = None,
    project: str = None,
    browser_mode: str = "stealth",
    chrome_profile: str = "system",
    force: bool = False,
    CONFIG=stx.INJECTED,
    logger=stx.INJECTED,
) -> int:
    """Orchestrate full paper acquisition pipeline.

    Parameters
    ----------
    doi_or_title : str
        DOI or paper title (required)
    project : str
        Project name for symlinking (optional)
    browser_mode : str
        Browser mode: 'stealth' or 'interactive' (default: stealth)
    chrome_profile : str
        Chrome profile name (default: system)
    force : bool
        Force fresh processing (default: False)

    Returns
    -------
    int
        Exit status code (0 for success)
    """
    if not doi_or_title:
        logger.error("--doi-or-title is required")
        return 1

    pipeline = ScholarPipelineSingle(
        browser_mode=browser_mode, chrome_profile=chrome_profile
    )
    paper, symlink_path = asyncio.run(
        pipeline.process_single_paper(
            doi_or_title=doi_or_title,
            project=project,
            force=force,
        )
    )
    return 0


if __name__ == "__main__":
    main()

# Usage:
# python -m scitex.scholar.pipelines.ScholarPipelineSingle \
#     --doi-or-title "10.1038/nature12373" \
#     --project test \
#     --chrome-profile system \
#     --browser-mode stealth

# EOF
