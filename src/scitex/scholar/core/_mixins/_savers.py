#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/core/_mixins/_savers.py

"""
Saver mixin for Scholar class.

Provides methods for saving papers to library and BibTeX format.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from scitex import logging

if TYPE_CHECKING:
    from ..Papers import Papers

logger = logging.getLogger(__name__)


class SaverMixin:
    """Mixin providing paper saving methods."""

    def save_papers_to_library(self, papers: Papers) -> List[str]:
        """Save papers collection to library.

        Args:
            papers: Papers collection to save

        Returns
        -------
            List of paper IDs saved
        """
        saved_ids = []
        for paper in papers:
            try:
                paper_id = self._library.save_paper(paper)
                saved_ids.append(paper_id)
            except Exception as e:
                logger.warning(f"{self.name}: Failed to save paper: {e}")

        logger.info(
            f"{self.name}: Saved {len(saved_ids)}/{len(papers)} papers to library"
        )
        return saved_ids

    def save_papers_as_bibtex(
        self, papers: Papers, output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """Save papers to BibTeX format with enrichment metadata.

        Args:
            papers: Papers collection to save
            output_path: Optional path to save the BibTeX file

        Returns
        -------
            BibTeX content as string with enrichment metadata included
        """
        from ..storage.BibTeXHandler import BibTeXHandler

        bibtex_handler = BibTeXHandler(project=self.project, config=self.config)
        return bibtex_handler.papers_to_bibtex(papers, output_path)


# EOF
