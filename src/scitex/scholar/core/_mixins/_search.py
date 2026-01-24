#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/core/_mixins/_search.py

"""
Search mixin for Scholar class.

Provides search functionality for local library and across projects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from scitex import logging

if TYPE_CHECKING:
    from ..Papers import Papers

logger = logging.getLogger(__name__)


class SearchMixin:
    """Mixin providing search methods."""

    def search_library(self, query: str, project: Optional[str] = None) -> Papers:
        """Search papers in local library.

        For new literature search (not in library), use AI2 Scholar QA:
        https://scholarqa.allen.ai/chat/ then process with:
        papers = scholar.load_bibtex('file.bib') followed by scholar.enrich(papers)

        Args:
            query: Search query
            project: Project filter (uses self.project if None)

        Returns
        -------
            Papers collection matching the query
        """
        from ..Papers import Papers

        logger.info(f"{self.name}: Searching library for: {query}")
        return Papers([], project=project or self.project)

    def search_across_projects(
        self, query: str, projects: Optional[List[str]] = None
    ) -> Papers:
        """Search for papers across multiple projects or the entire library.

        Args:
            query: Search query
            projects: List of project names to search (None for all)

        Returns
        -------
            Papers collection with search results
        """
        from ..Paper import Paper
        from ..Papers import Papers

        if projects is None:
            all_projects = [p["name"] for p in self.list_projects()]
        else:
            all_projects = projects

        all_papers = []
        for project in all_projects:
            try:
                project_dir = self.config.get_library_project_dir(project)
                for item in project_dir.iterdir():
                    if item.is_symlink() or item.is_dir():
                        paper_dir = item.resolve() if item.is_symlink() else item
                        metadata_file = paper_dir / "metadata.json"
                        if metadata_file.exists():
                            try:
                                paper = Paper.model_validate_json(
                                    metadata_file.read_text()
                                )
                                query_lower = query.lower()
                                title = (paper.metadata.basic.title or "").lower()
                                abstract = (paper.metadata.basic.abstract or "").lower()
                                authors = paper.metadata.basic.authors or []
                                if (
                                    query_lower in title
                                    or query_lower in abstract
                                    or any(
                                        query_lower in (a or "").lower()
                                        for a in authors
                                    )
                                ):
                                    all_papers.append(paper)
                            except Exception as e:
                                logger.debug(
                                    f"{self.name}: Failed to load {metadata_file}: {e}"
                                )
            except Exception as e:
                logger.debug(f"{self.name}: Failed to search project {project}: {e}")

        return Papers(all_papers, config=self.config, project="search_results")


# EOF
