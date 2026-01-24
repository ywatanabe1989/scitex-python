#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/storage/_mixins/_library_operations.py

"""
Library operations mixin for LibraryManager.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from scitex import logging

logger = logging.getLogger(__name__)


class LibraryOperationsMixin:
    """Mixin providing library operation methods."""

    def update_library_metadata(
        self,
        paper_id: str,
        project: str,
        doi: str,
        metadata: Dict[str, Any],
        create_structure: bool = True,
    ) -> bool:
        """Update Scholar library metadata.json with resolved DOI."""
        try:
            library_path = self.config.path_manager.library_dir
            paper_dir = library_path / project / paper_id
            metadata_file = paper_dir / "metadata.json"

            if create_structure and not paper_dir.exists():
                self.config.path_manager._ensure_directory(paper_dir)
                logger.info(f"Created Scholar library structure: {paper_dir}")

            existing_metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file) as file_:
                        existing_metadata = json.load(file_)
                except Exception as exc_:
                    logger.warning(f"Error loading existing metadata: {exc_}")

            updated_metadata = {
                **existing_metadata,
                **metadata,
                "doi": doi,
                "doi_resolved_at": datetime.now().isoformat(),
                "doi_source": "batch_doi_resolver",
            }

            with open(metadata_file, "w") as file_:
                json.dump(updated_metadata, file_, indent=2)

            logger.success(f"Updated metadata for {paper_id}: DOI {doi}")
            return True

        except Exception as exc_:
            logger.error(f"Error updating library metadata for {paper_id}: {exc_}")
            return False

    def create_writer_directory_structure(self, paper_id: str, project: str) -> Path:
        """Create basic paper directory structure."""
        library_path = self.config.path_manager.library_dir
        paper_dir = library_path / project / paper_id

        self.config.path_manager._ensure_directory(paper_dir)

        for subdir in ["attachments", "screenshots"]:
            subdir_path = paper_dir / subdir
            self.config.path_manager._ensure_directory(subdir_path)

        logger.info(f"Created Scholar library structure: {paper_dir}")
        return paper_dir

    def validate_library_structure(self, project: str) -> Dict[str, Any]:
        """Validate existing library structure for a project."""
        validation = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "paper_count": 0,
            "missing_metadata": [],
        }

        library_path = self.config.path_manager.library_dir
        project_dir = library_path / project

        if not project_dir.exists():
            validation["errors"].append(
                f"Project directory does not exist: {project_dir}"
            )
            validation["valid"] = False
            return validation

        for paper_dir in project_dir.iterdir():
            if paper_dir.is_dir() and len(paper_dir.name) == 8:
                validation["paper_count"] += 1

                metadata_file = paper_dir / "metadata.json"
                if not metadata_file.exists():
                    validation["missing_metadata"].append(paper_dir.name)
                    validation["warnings"].append(
                        f"Missing metadata.json: {paper_dir.name}"
                    )

        return validation

    def resolve_and_update_library(
        self,
        papers_with_ids: List[Dict[str, Any]],
        project: str,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Resolve DOIs and update Scholar library metadata.json files."""
        if not self.single_doi_resolver:
            raise ValueError("SingleDOIResolver is required for resolving DOIs")

        results = {}
        for paper in papers_with_ids:
            paper_id = paper.get("paper_id")
            if not paper_id:
                logger.warning(
                    f"Skipping paper without paper_id: {paper.get('title', 'Unknown')}"
                )
                continue

            title = paper.get("title")
            if not title:
                logger.warning(f"Skipping paper {paper_id} without title")
                continue

            logger.info(f"Resolving DOI for {paper_id}: {title[:50]}...")

            try:
                result = asyncio.run(
                    self.single_doi_resolver.metadata2doi_async(
                        title=title,
                        year=paper.get("year"),
                        authors=paper.get("authors"),
                        sources=sources,
                    )
                )

                if result and isinstance(result, dict) and result.get("doi"):
                    doi = result["doi"]

                    success = self.update_library_metadata(
                        paper_id=paper_id,
                        project=project,
                        doi=doi,
                        metadata={
                            "title": title,
                            "title_source": "input",
                            "year": paper.get("year"),
                            "year_source": "input" if paper.get("year") else None,
                            "authors": paper.get("authors"),
                            "authors_source": "input" if paper.get("authors") else None,
                            "journal": paper.get("journal"),
                            "journal_source": "input" if paper.get("journal") else None,
                            "doi_resolution_source": result.get("source"),
                        },
                    )

                    if success:
                        results[paper_id] = doi
                        logger.success(f"{paper_id}: {doi}")
                    else:
                        logger.error(
                            f"{paper_id}: DOI resolved but metadata update failed"
                        )
                else:
                    logger.warning(f"{paper_id}: No DOI found")

            except Exception as exc_:
                logger.error(f"{paper_id}: Error during resolution: {exc_}")

        logger.success(
            f"Resolved {len(results)}/{len(papers_with_ids)} DOIs and updated library metadata"
        )
        return results

    def resolve_and_create_library_structure(
        self,
        papers: List[Dict[str, Any]],
        project: str,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Synchronous wrapper for resolve_and_create_library_structure_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot run synchronous version in async context. "
                    "Use resolve_and_create_library_structure_async() instead."
                )
            else:
                return loop.run_until_complete(
                    self.resolve_and_create_library_structure_async(
                        papers, project, sources
                    )
                )
        except RuntimeError:
            return asyncio.run(
                self.resolve_and_create_library_structure_async(
                    papers, project, sources
                )
            )


# EOF
