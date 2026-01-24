#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/core/_mixins/_loaders.py

"""
Loader mixin for Scholar class.

Provides methods for loading papers from projects and BibTeX files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from scitex import logging

if TYPE_CHECKING:
    from ..Papers import Papers

logger = logging.getLogger(__name__)


class LoaderMixin:
    """Mixin providing paper loading methods."""

    def load_project(self, project: Optional[str] = None) -> Papers:
        """Load papers from a project using library manager service.

        Args:
            project: Project name (uses self.project if None)

        Returns
        -------
            Papers collection from the project
        """
        from ..Paper import Paper
        from ..Papers import Papers

        project_name = project or self.project
        if not project_name:
            raise ValueError("No project specified")

        logger.info(f"{self.name}: Loading papers from project: {project_name}")

        library_dir = self.config.path_manager.library_dir
        project_dir = library_dir / project_name

        if not project_dir.exists():
            logger.warning(
                f"{self.name}: Project directory does not exist: {project_dir}"
            )
            return Papers([], project=project_name)

        papers = []
        for item in project_dir.iterdir():
            if item.name in ["info", "project_metadata.json", "README.md"]:
                continue

            if item.is_symlink():
                master_path = item.resolve()
                if master_path.exists():
                    metadata_file = master_path / "metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file) as f:
                                metadata = json.load(f)

                            paper = Paper.from_dict(metadata)
                            papers.append(paper)
                        except Exception as e:
                            logger.warning(
                                f"{self.name}: Failed to load metadata "
                                f"from {metadata_file}: {e}"
                            )

        logger.info(
            f"{self.name}: Loaded {len(papers)} papers from project: {project_name}"
        )
        return Papers(papers, project=project_name)

    def load_bibtex(self, bibtex_input: Union[str, Path]) -> Papers:
        """Load Papers collection from BibTeX file or content.

        Args:
            bibtex_input: BibTeX file path or content string

        Returns
        -------
            Papers collection
        """
        from ..Papers import Papers

        papers = self._library.papers_from_bibtex(bibtex_input)

        papers_collection = Papers(papers, config=self.config, project=self.project)
        papers_collection.library = self._library

        return papers_collection


# EOF
