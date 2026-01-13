#!/usr/bin/env python3
# Timestamp: "2026-01-14 (ywatanabe)"
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

import argparse
import asyncio
from typing import Optional

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
                    paper, io, context, auth_gateway, force
                )
                await self._step_07_download_pdf(
                    paper, io, context, auth_gateway, force
                )
            if browser_manager:
                await browser_manager.close()

            # Step 8: Content extraction
            self._step_08_extract_content(io, force)

            # Step 9-10: Finalize
            symlink_path = self._step_09_link_to_project(paper, io, project)
            self._step_10_log_final_status(io)

            return paper, symlink_path


def main(args):
    """Run single paper pipeline."""
    pipeline = ScholarPipelineSingle(
        browser_mode=args.browser_mode, chrome_profile=args.chrome_profile
    )
    paper, symlink_path = asyncio.run(
        pipeline.process_single_paper(
            doi_or_title=args.doi_or_title,
            project=args.project,
            force=args.force,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Orchestrate full paper acquisition pipeline"
    )
    parser.add_argument(
        "--doi-or-title", type=str, required=True, help="DOI or paper title"
    )
    parser.add_argument(
        "--project", type=str, default=None, help="Project name for symlinking"
    )
    parser.add_argument(
        "--browser-mode",
        type=str,
        choices=["stealth", "interactive"],
        default="stealth",
        help="Browser mode (default: stealth)",
    )
    parser.add_argument(
        "--chrome-profile",
        type=str,
        required=True,
        help="Chrome profile name (default: system)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        default=False,
        help="Force fresh processing",
    )
    return parser.parse_args()


def run_main() -> None:
    """Initialize scitex framework, run main function, and cleanup."""
    import sys

    import matplotlib.pyplot as plt

    import scitex as stx

    args = parse_args()
    CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
        sys, plt, args=args, file=__file__, sdir_suffix=None, verbose=False, agg=True
    )
    exit_status = main(args)
    stx.session.close(
        CONFIG, verbose=False, notify=False, message="", exit_status=exit_status
    )


if __name__ == "__main__":
    run_main()

# Usage:
# python -m scitex.scholar.pipelines.ScholarPipelineSingle \
#     --doi-or-title "10.1038/nature12373" \
#     --project test \
#     --chrome-profile system \
#     --browser-mode stealth

# EOF
