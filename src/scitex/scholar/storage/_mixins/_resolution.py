#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/storage/_mixins/_resolution.py

"""
DOI resolution mixin for LibraryManager.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from scitex import logging
from scitex.scholar._utils import TextNormalizer

logger = logging.getLogger(__name__)


class ResolutionMixin:
    """Mixin providing DOI resolution methods."""

    def check_library_for_doi(
        self, title: str, year: Optional[int] = None
    ) -> Optional[str]:
        """Check if DOI already exists in master Scholar library."""
        try:
            for paper_dir in self.library_master_dir.iterdir():
                if not paper_dir.is_dir():
                    continue

                metadata_file = paper_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as file_:
                            metadata = json.load(file_)

                        stored_title = metadata.get("title", "")
                        stored_year = metadata.get("year")
                        stored_doi = metadata.get("doi")

                        title_match = self._is_title_similar(title, stored_title)
                        year_match = (
                            not year
                            or not stored_year
                            or abs(int(stored_year) - int(year)) <= 1
                            if isinstance(stored_year, (int, str))
                            and str(stored_year).isdigit()
                            else stored_year == year
                        )

                        if title_match and year_match and stored_doi:
                            logger.info(
                                f"DOI found in master Scholar library: {stored_doi} (paper_id: {paper_dir.name})"
                            )
                            return stored_doi

                    except (json.JSONDecodeError, KeyError, ValueError) as exc_:
                        logger.debug(
                            f"Error reading metadata from {metadata_file}: {exc_}"
                        )
                        continue

            return None

        except Exception as exc_:
            logger.debug(f"Error checking master Scholar library: {exc_}")
            return None

    async def resolve_and_create_library_structure_async(
        self,
        papers: List[Dict[str, Any]],
        project: str,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Resolve DOIs and create full Scholar library structure with proper paths."""
        if not self.single_doi_resolver:
            raise ValueError("SingleDOIResolver is required for resolving DOIs")

        results = {}
        for paper in papers:
            title = paper.get("title")
            if not title:
                logger.warning(f"Skipping paper without title: {paper}")
                continue

            logger.info(f"Processing: {title[:50]}...")

            try:
                doi_result = await self.single_doi_resolver.metadata2doi_async(
                    title=title,
                    year=paper.get("year"),
                    authors=paper.get("authors"),
                    sources=sources,
                )

                enhanced_metadata = self._extract_enhanced_metadata(doi_result, paper)
                paper_info = {**paper, **enhanced_metadata}

                storage_paths = self._call_path_manager_get_storage_paths(
                    paper_info=paper_info, collection_name="MASTER"
                )
                paper_id = storage_paths["unique_id"]
                storage_path = storage_paths["storage_path"]
                metadata_file = storage_path / "metadata.json"

                complete_metadata = self._create_complete_metadata(
                    paper, doi_result, paper_id, enhanced_metadata
                )

                with open(metadata_file, "w") as file_:
                    json.dump(complete_metadata, file_, indent=2)

                logger.success(
                    f"Saved metadata.json for {paper_id} ({len(complete_metadata)} fields)"
                )

                project_symlink_path = self._create_project_symlink(
                    master_storage_path=storage_path,
                    project=project,
                    readable_name=storage_paths["readable_name"],
                )

                bibtex_source_filename = getattr(self, "_source_filename", "papers")
                info_dir = self._create_bibtex_info_structure(
                    project=project,
                    paper_info={**paper, **enhanced_metadata},
                    complete_metadata=complete_metadata,
                    bibtex_source_filename=bibtex_source_filename,
                )

                results[title] = {
                    "scitex_id": paper_id,
                    "scholar_id": paper_id,
                    "doi": complete_metadata.get("doi"),
                    "master_storage_path": str(storage_path),
                    "project_symlink_path": str(project_symlink_path)
                    if project_symlink_path
                    else None,
                    "readable_name": storage_paths["readable_name"],
                    "metadata_file": str(metadata_file),
                    "info_dir": str(info_dir) if info_dir else None,
                }

                logger.info(f"Created library entry: {paper_id}")
                if complete_metadata.get("doi"):
                    logger.info(f"   DOI: {complete_metadata['doi']}")
                logger.info(f"   Storage: {storage_path}")

            except Exception as exc_:
                logger.error(f"Error processing '{title[:30]}...': {exc_}")

        logger.success(
            f"Created Scholar library entries for {len(results)}/{len(papers)} papers"
        )
        return results

    async def resolve_and_create_library_structure_with_source_async(
        self,
        papers: List[Dict[str, Any]],
        project: str,
        sources: Optional[List[str]] = None,
        bibtex_source_filename: str = "papers",
    ) -> Dict[str, Dict[str, str]]:
        """Enhanced version that passes source filename for BibTeX structure."""
        self._source_filename = bibtex_source_filename
        return await self.resolve_and_create_library_structure_async(
            papers=papers, project=project, sources=sources
        )

    def _extract_enhanced_metadata(
        self, doi_result: Optional[Dict], paper: Dict
    ) -> Dict[str, Any]:
        """Extract enhanced metadata from DOI resolution result."""
        enhanced = {}
        if doi_result and isinstance(doi_result, dict):
            metadata_source = doi_result.get("metadata", {})
            enhanced.update(
                {
                    "doi": doi_result.get("doi"),
                    "journal": metadata_source.get("journal")
                    or doi_result.get("journal")
                    or paper.get("journal"),
                    "authors": metadata_source.get("authors")
                    or doi_result.get("authors")
                    or paper.get("authors"),
                    "year": metadata_source.get("year")
                    or doi_result.get("year")
                    or paper.get("year"),
                    "title": metadata_source.get("title")
                    or doi_result.get("title")
                    or paper.get("title"),
                    "abstract": metadata_source.get("abstract")
                    or doi_result.get("abstract"),
                    "publisher": metadata_source.get("publisher")
                    or doi_result.get("publisher"),
                    "volume": metadata_source.get("volume") or doi_result.get("volume"),
                    "issue": metadata_source.get("issue") or doi_result.get("issue"),
                    "pages": metadata_source.get("pages") or doi_result.get("pages"),
                    "issn": metadata_source.get("issn") or doi_result.get("issn"),
                    "short_journal": metadata_source.get("short_journal")
                    or doi_result.get("short_journal"),
                }
            )

            if doi_result.get("doi"):
                logger.success(
                    f"Enhanced metadata from DOI source: {dict(metadata_source)}"
                )

        return enhanced

    def _create_complete_metadata(
        self,
        paper: Dict,
        doi_result: Optional[Dict],
        paper_id: str,
        enhanced_metadata: Dict,
    ) -> Dict[str, Any]:
        """Create complete metadata dictionary with source tracking."""
        raw_title = enhanced_metadata.get("title") or paper.get("title")
        clean_title = TextNormalizer.clean_metadata_text(raw_title) if raw_title else ""
        raw_abstract = None
        if enhanced_metadata.get("abstract"):
            raw_abstract = TextNormalizer.clean_metadata_text(
                enhanced_metadata["abstract"]
            )

        doi_source_value = self._get_doi_source_value(doi_result)

        complete_metadata = {
            "title": clean_title,
            "title_source": doi_source_value
            if enhanced_metadata.get("title") != paper.get("title")
            else "manual",
            "authors": enhanced_metadata.get("authors") or paper.get("authors"),
            "authors_source": doi_source_value
            if enhanced_metadata.get("authors") != paper.get("authors")
            else ("manual" if paper.get("authors") else None),
            "year": enhanced_metadata.get("year") or paper.get("year"),
            "year_source": doi_source_value
            if enhanced_metadata.get("year") != paper.get("year")
            else ("manual" if paper.get("year") else None),
            "journal": enhanced_metadata.get("journal") or paper.get("journal"),
            "journal_source": doi_source_value
            if enhanced_metadata.get("journal") != paper.get("journal")
            else ("manual" if paper.get("journal") else None),
            "abstract": raw_abstract,
            "abstract_source": doi_source_value
            if enhanced_metadata.get("abstract")
            else None,
            "scitex_id": paper_id,
            "created_at": datetime.now().isoformat(),
            "created_by": "SciTeX Scholar",
        }

        if doi_result and isinstance(doi_result, dict):
            safe_fields = [
                "publisher",
                "volume",
                "issue",
                "pages",
                "issn",
                "short_journal",
            ]
            for field in safe_fields:
                value = enhanced_metadata.get(field)
                if value is not None:
                    complete_metadata[field] = value
                    complete_metadata[f"{field}_source"] = (
                        doi_source_value or "unknown_api"
                    )

        if doi_result and doi_result.get("doi"):
            complete_metadata.update(
                {"doi": doi_result["doi"], "doi_source": doi_source_value}
            )
            logger.success(f"DOI resolved for {paper_id}: {doi_result['doi']}")
        else:
            complete_metadata.update(
                {"doi": None, "doi_source": None, "doi_resolution_failed": True}
            )
            logger.warning(
                f"DOI resolution failed for {paper_id}: {paper.get('title', '')[:40]}..."
            )

        self._add_standard_fields(complete_metadata)

        storage_paths = self._call_path_manager_get_storage_paths(
            paper_info={**paper, **enhanced_metadata}, collection_name="MASTER"
        )
        storage_path = storage_paths["storage_path"]

        complete_metadata.update(
            {
                "master_storage_path": str(storage_path),
                "readable_name": storage_paths["readable_name"],
                "metadata_file": str(storage_path / "metadata.json"),
            }
        )

        return complete_metadata

    def _get_doi_source_value(self, doi_result: Optional[Dict]) -> Optional[str]:
        """Get normalized DOI source value."""
        if not doi_result or not doi_result.get("source"):
            return None

        source = doi_result["source"]
        if "crossref" in source.lower():
            return "crossref"
        elif "semantic" in source.lower():
            return "semantic_scholar"
        elif "pubmed" in source.lower():
            return "pubmed"
        elif "openalex" in source.lower():
            return "openalex"
        return source

    def _add_standard_fields(self, complete_metadata: Dict) -> None:
        """Add standard fields with None defaults."""
        standard_fields = {
            "keywords": None,
            "references": None,
            "venue": None,
            "publisher": None,
            "volume": None,
            "issue": None,
            "pages": None,
            "issn": None,
            "short_journal": None,
        }

        missing_fields = []
        for field, default_value in standard_fields.items():
            if field not in complete_metadata or complete_metadata[field] is None:
                complete_metadata[field] = default_value
                missing_fields.append(field)

        if missing_fields:
            logger.info(
                f"Missing fields for future enhancement: {', '.join(missing_fields)}"
            )

    def _is_title_similar(
        self, title1: str, title2: str, threshold: float = 0.7
    ) -> bool:
        """Check if two titles are similar enough to be considered the same paper."""
        if not title1 or not title2:
            return False

        def normalize_title(title: str) -> str:
            title = title.lower()
            title = re.sub(r"[^\w\s]", " ", title)
            title = re.sub(r"\s+", " ", title)
            return title.strip()

        norm_title1 = normalize_title(title1)
        norm_title2 = normalize_title(title2)

        words1 = set(norm_title1.split())
        words2 = set(norm_title2.split())

        if not words1 or not words2:
            return False

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        similarity = intersection / union if union > 0 else 0.0

        return similarity >= threshold


# EOF
