#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/core/Scholar.py

"""
Unified Scholar class for scientific literature management.

This is the main entry point for all scholar functionality, providing:
- Simple, intuitive API
- Smart defaults
- Method chaining
- Progressive disclosure of advanced features
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from scitex import logging
from scitex.scholar.config import ScholarConfig

from ._mixins import (
    EnricherMixin,
    LibraryHandlerMixin,
    LoaderMixin,
    PDFDownloadMixin,
    PipelineMixin,
    ProjectHandlerMixin,
    SaverMixin,
    SearchMixin,
    ServiceMixin,
    URLFindingMixin,
)

logger = logging.getLogger(__name__)


class Scholar(
    EnricherMixin,
    URLFindingMixin,
    PDFDownloadMixin,
    LoaderMixin,
    SearchMixin,
    SaverMixin,
    ProjectHandlerMixin,
    LibraryHandlerMixin,
    PipelineMixin,
    ServiceMixin,
):
    """
    Main interface for SciTeX Scholar - scientific literature management made simple.

    By default, papers are automatically enriched with:
    - Journal impact factors from impact_factor package (2024 JCR data)
    - Citation counts from Semantic Scholar (via DOI/title matching)

    Example usage:
        # Basic search with automatic enrichment
        scholar = Scholar()
        papers = scholar.search("deep learning neuroscience")
        # Papers now have impact_factor and citation_count populated
        papers.save("my_pac.bib")

        # Disable automatic enrichment if needed
        config = ScholarConfig(enable_auto_enrich=False)
        scholar = Scholar(config=config)

        # Search specific source
        papers = scholar.search("transformer models", sources='arxiv')

        # Advanced workflow
        papers = scholar.search("transformer models", year_min=2020) \\
                      .filter(min_citations=50) \\
                      .sort_by("impact_factor") \\
                      .save("transformers.bib")

        # Local library
        scholar._index_local_pdfs("./my_papers")
        local_papers = scholar.search_local("attention mechanism")
    """

    @property
    def name(self):
        """Class name for logging."""
        return self.__class__.__name__

    def __init__(
        self,
        config: Optional[Union[ScholarConfig, str, Path]] = None,
        project: Optional[str] = None,
        project_description: Optional[str] = None,
        browser_mode: Optional[str] = None,
    ):
        """Initialize Scholar with configuration.

        Args:
            config: Can be:
                   - ScholarConfig instance
                   - Path to YAML config file (str or Path)
                   - None (uses ScholarConfig.load() to find config)
            project: Default project name for operations
            project_description: Optional description for the project
            browser_mode: Browser mode ('stealth', 'interactive', 'manual')
        """
        self.config = self._init_config(config)
        self.browser_mode = browser_mode or "stealth"

        self.project = self.config.resolve("project", project, "default")
        self.workspace_dir = self.config.path_manager.get_workspace_dir()

        if project:
            self._ensure_project_exists(project, project_description)

        library_path = self.config.get_library_project_dir(self.project)
        if project:
            project_path = library_path / project
            logger.info(
                f"Scholar initialized with project '{project}' at {project_path}"
            )
        else:
            logger.info(f"{self.name}: Scholar initialized (library: {library_path})")


__all__ = ["Scholar"]


if __name__ == "__main__":
    from .Paper import Paper
    from .Papers import Papers

    def main():
        """Demonstrate Scholar class usage - Clean API Demo."""
        print("\n" + "=" * 60)
        print("Scholar Module Demo - Clean API")
        print("=" * 60 + "\n")

        # 1. Initialize Scholar
        print("1. Initialize Scholar")
        print("-" * 60)
        scholar = Scholar(
            project="demo_project",
            project_description="Demo project for testing Scholar API",
        )
        print("Scholar initialized")
        print(f"  Project: {scholar.project}")
        print()

        # 2. Project Management
        print("2. Project Management:")
        try:
            project_dir = scholar._create_project_metadata(
                "neural_networks_2024",
                description="Collection of neural network papers from 2024",
            )
            print("   Created project: neural_networks_2024")
            print(f"   Project directory: {project_dir}")

            projects = scholar.list_projects()
            print(f"   Total projects in library: {len(projects)}")
            for proj in projects[:3]:
                print(f"      - {proj['name']}: {proj.get('description', 'No desc')}")
            if len(projects) > 3:
                print(f"      ... and {len(projects) - 3} more")

        except Exception as e:
            print(f"   Project management demo skipped: {e}")
        print()

        # 3. Library Statistics
        print("3. Library Statistics:")
        try:
            stats = scholar.get_library_statistics()
            print(f"   Total projects: {stats['total_projects']}")
            print(f"   Total papers: {stats['total_papers']}")
            print(f"   Storage usage: {stats['storage_mb']:.2f} MB")
            print(f"   Library path: {stats['library_path']}")

        except Exception as e:
            print(f"   Library statistics demo skipped: {e}")
        print()

        # 4. Working with Papers
        print("4. Working with Papers:")
        p1 = Paper()
        p1.metadata.basic.title = "Vision Transformer: An Image Is Worth 16x16 Words"
        p1.metadata.basic.authors = ["Dosovitskiy, Alexey", "Beyer, Lucas"]
        p1.metadata.basic.year = 2021
        p1.metadata.publication.journal = "ICLR"
        p1.metadata.set_doi("10.48550/arXiv.2010.11929")

        sample_papers = [p1]
        papers = Papers(
            sample_papers,
            project="neural_networks_2024",
            config=scholar.config,
        )
        print(f"   Created collection with {len(papers)} papers")
        print()

        # 5. Configuration
        print("5. Configuration Management:")
        print(f"   Scholar directory: {scholar.config.paths.scholar_dir}")
        print(f"   Library directory: {scholar.config.get_library_project_dir()}")
        print()

        # 6. Service Components
        print("6. Service Components (Internal):")
        print(f"   Scholar Engine: {type(scholar._scholar_engine).__name__}")
        print(f"   Auth Manager: {type(scholar._auth_manager).__name__}")
        print(f"   Browser Manager: {type(scholar._browser_manager).__name__}")
        print(f"   Library Manager: {type(scholar._library_manager).__name__}")
        print()

        print("Scholar demo completed!")

    main()


# EOF
