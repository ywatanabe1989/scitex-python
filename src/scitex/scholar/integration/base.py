#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base classes for reference manager integrations.

Provides common interface and functionality for all reference managers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime

from scitex import logging
from scitex.scholar.core.Paper import Paper
from scitex.scholar.core.Papers import Papers
from scitex.scholar.storage import LibraryManager, BibTeXHandler

logger = logging.getLogger(__name__)


class BaseMapper(ABC):
    """Base class for data mapping between reference manager and Scholar formats."""

    def __init__(self, config=None):
        """Initialize mapper.

        Args:
            config: Optional ScholarConfig instance
        """
        self.config = config

    @abstractmethod
    def external_to_paper(self, item: Dict[str, Any]) -> Paper:
        """Convert external format to Scholar Paper.

        Args:
            item: Item in external format

        Returns:
            Paper object
        """
        pass

    @abstractmethod
    def paper_to_external(self, paper: Paper) -> Dict[str, Any]:
        """Convert Scholar Paper to external format.

        Args:
            paper: Paper object

        Returns:
            Dict in external format
        """
        pass

    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extract year from date string.

        Args:
            date_str: Date string in various formats

        Returns:
            Year as int or None
        """
        import re

        match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if match:
            try:
                return int(match.group(0))
            except ValueError:
                pass
        return None


class BaseImporter(ABC):
    """Base class for importing from reference managers."""

    def __init__(
        self,
        credentials: Optional[Dict[str, str]] = None,
        project: str = "default",
        config=None,
    ):
        """Initialize importer.

        Args:
            credentials: Authentication credentials
            project: Scholar project name
            config: Optional ScholarConfig instance
        """
        self.credentials = credentials or {}
        self.project = project
        self.config = config

        self.library_manager = LibraryManager(project=project, config=config)
        self.mapper = self._create_mapper()

    @abstractmethod
    def _create_mapper(self) -> BaseMapper:
        """Create mapper instance for this reference manager.

        Returns:
            Mapper instance
        """
        pass

    @abstractmethod
    def import_collection(
        self,
        collection_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> Papers:
        """Import items from a collection.

        Args:
            collection_id: Collection ID
            collection_name: Collection name
            limit: Maximum items to import
            **kwargs: Additional platform-specific options

        Returns:
            Papers collection
        """
        pass

    @abstractmethod
    def import_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        limit: Optional[int] = None,
        **kwargs,
    ) -> Papers:
        """Import items with specific tags.

        Args:
            tags: List of tag names
            match_all: Require all tags (AND) vs any tag (OR)
            limit: Maximum items to import
            **kwargs: Additional platform-specific options

        Returns:
            Papers collection
        """
        pass

    def import_all(self, limit: Optional[int] = None, **kwargs) -> Papers:
        """Import all items from library.

        Args:
            limit: Maximum items to import
            **kwargs: Additional platform-specific options

        Returns:
            Papers collection
        """
        return self.import_collection(collection_id=None, limit=limit, **kwargs)

    def import_to_library(
        self,
        papers: Union[Papers, List[Paper]],
        update_existing: bool = True,
    ) -> Dict[str, str]:
        """Import Papers into Scholar library storage.

        Args:
            papers: Papers collection or list
            update_existing: Update existing papers

        Returns:
            Dict mapping titles to library IDs
        """
        if isinstance(papers, Papers):
            paper_list = papers.papers
        else:
            paper_list = papers

        results = {}

        for i, paper in enumerate(paper_list, 1):
            try:
                paper_id = self.library_manager.save_resolved_paper(
                    paper_data=paper, project=self.project
                )

                results[paper.metadata.basic.title or f"paper_{i}"] = paper_id

                logger.info(f"[{i}/{len(paper_list)}] Saved: {paper_id}")

            except Exception as e:
                logger.error(
                    f"Failed to save paper: {paper.metadata.basic.title[:50]}... - {e}"
                )
                continue

        logger.success(
            f"Imported {len(results)}/{len(paper_list)} papers to Scholar library"
        )

        return results


class BaseExporter(ABC):
    """Base class for exporting to reference managers."""

    def __init__(
        self,
        credentials: Optional[Dict[str, str]] = None,
        project: str = "default",
        config=None,
    ):
        """Initialize exporter.

        Args:
            credentials: Authentication credentials
            project: Scholar project name
            config: Optional ScholarConfig instance
        """
        self.credentials = credentials or {}
        self.project = project
        self.config = config

        self.bibtex_handler = BibTeXHandler(project=project, config=config)
        self.mapper = self._create_mapper()

    @abstractmethod
    def _create_mapper(self) -> BaseMapper:
        """Create mapper instance for this reference manager.

        Returns:
            Mapper instance
        """
        pass

    @abstractmethod
    def export_papers(
        self,
        papers: Union[Papers, List[Paper]],
        collection_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Export Papers to reference manager.

        Args:
            papers: Papers collection or list
            collection_name: Target collection name
            **kwargs: Additional platform-specific options

        Returns:
            Dict mapping titles to external IDs
        """
        pass

    def export_as_bibtex(
        self,
        papers: Union[Papers, List[Paper]],
        output_path: Union[str, Path],
    ) -> Path:
        """Export Papers as BibTeX file.

        Args:
            papers: Papers collection or list
            output_path: Output file path

        Returns:
            Path to created file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.bibtex_handler.papers_to_bibtex(papers, output_path=output_path)

        logger.success(f"Exported BibTeX: {output_path}")
        return output_path

    def export_as_ris(
        self,
        papers: Union[Papers, List[Paper]],
        output_path: Union[str, Path],
    ) -> Path:
        """Export Papers as RIS file.

        Args:
            papers: Papers collection or list
            output_path: Output file path

        Returns:
            Path to created file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(papers, Papers):
            paper_list = papers.papers
        else:
            paper_list = papers

        # Generate RIS
        ris_lines = []
        for paper in paper_list:
            ris_lines.append("TY  - JOUR")

            if paper.metadata.basic.title:
                ris_lines.append(f"TI  - {paper.metadata.basic.title}")

            if paper.metadata.basic.authors:
                for author in paper.metadata.basic.authors:
                    ris_lines.append(f"AU  - {author}")

            if paper.metadata.basic.year:
                ris_lines.append(f"PY  - {paper.metadata.basic.year}")

            if paper.metadata.publication.journal:
                ris_lines.append(f"JO  - {paper.metadata.publication.journal}")

            if paper.metadata.id.doi:
                ris_lines.append(f"DO  - {paper.metadata.id.doi}")

            if paper.metadata.basic.abstract:
                ris_lines.append(f"AB  - {paper.metadata.basic.abstract}")

            ris_lines.append("ER  - ")
            ris_lines.append("")

        output_path.write_text("\n".join(ris_lines))

        logger.success(f"Exported RIS: {output_path}")
        return output_path


class BaseLinker(ABC):
    """Base class for live synchronization with reference managers."""

    def __init__(
        self,
        credentials: Optional[Dict[str, str]] = None,
        project: str = "default",
        config=None,
        sync_interval: int = 60,
    ):
        """Initialize linker.

        Args:
            credentials: Authentication credentials
            project: Scholar project name
            config: Optional ScholarConfig instance
            sync_interval: Seconds between syncs
        """
        self.credentials = credentials or {}
        self.project = project
        self.config = config
        self.sync_interval = sync_interval

        self.importer = self._create_importer()
        self.exporter = self._create_exporter()
        self.mapper = self._create_mapper()
        self.library_manager = LibraryManager(project=project, config=config)

        self._callbacks: List[Callable] = []
        self._running = False

    @abstractmethod
    def _create_importer(self) -> BaseImporter:
        """Create importer instance.

        Returns:
            Importer instance
        """
        pass

    @abstractmethod
    def _create_exporter(self) -> BaseExporter:
        """Create exporter instance.

        Returns:
            Exporter instance
        """
        pass

    @abstractmethod
    def _create_mapper(self) -> BaseMapper:
        """Create mapper instance.

        Returns:
            Mapper instance
        """
        pass

    def register_callback(self, callback: Callable[[str, Paper], None]):
        """Register callback for sync events.

        Args:
            callback: Function(event_type, paper)
        """
        self._callbacks.append(callback)
        logger.info(f"Registered callback: {callback.__name__}")

    def insert_citation(
        self,
        paper: Paper,
        format: str = "bibtex",
        style: str = "apa",
    ) -> str:
        """Generate formatted citation.

        Args:
            paper: Paper object
            format: Citation format ('bibtex', 'ris', 'text')
            style: Citation style ('apa', 'mla', 'chicago')

        Returns:
            Formatted citation string
        """
        if format == "bibtex":
            authors = paper.metadata.basic.authors or ["Unknown"]
            first_author = authors[0].split(",")[0] if authors else "Unknown"
            year = paper.metadata.basic.year or "NoYear"
            key = f"{first_author}{year}"

            lines = [f"@article{{{key},"]
            if paper.metadata.basic.title:
                lines.append(f"  title = {{{paper.metadata.basic.title}}},")
            if authors:
                lines.append(f"  author = {{" + " and ".join(authors) + "},")
            if paper.metadata.basic.year:
                lines.append(f"  year = {{{paper.metadata.basic.year}}},")
            if paper.metadata.publication.journal:
                lines.append(f"  journal = {{{paper.metadata.publication.journal}}},")
            if paper.metadata.id.doi:
                lines.append(f"  doi = {{{paper.metadata.id.doi}}},")
            lines.append("}")

            return "\n".join(lines)

        elif format == "ris":
            lines = ["TY  - JOUR"]
            if paper.metadata.basic.title:
                lines.append(f"TI  - {paper.metadata.basic.title}")
            if paper.metadata.basic.authors:
                for author in paper.metadata.basic.authors:
                    lines.append(f"AU  - {author}")
            if paper.metadata.basic.year:
                lines.append(f"PY  - {paper.metadata.basic.year}")
            if paper.metadata.id.doi:
                lines.append(f"DO  - {paper.metadata.id.doi}")
            lines.append("ER  - ")

            return "\n".join(lines)

        elif format == "text":
            authors = paper.metadata.basic.authors or []
            year = paper.metadata.basic.year or "n.d."
            title = paper.metadata.basic.title or "Untitled"
            journal = paper.metadata.publication.journal or ""

            if style == "apa":
                if not authors:
                    author_str = "Unknown"
                elif len(authors) == 1:
                    author_str = authors[0]
                elif len(authors) == 2:
                    author_str = f"{authors[0]} & {authors[1]}"
                else:
                    author_str = f"{authors[0]} et al."

                citation = f"{author_str} ({year}). {title}."
                if journal:
                    citation += f" {journal}."
                return citation

            elif style == "mla":
                author_str = authors[0] if authors else "Unknown"
                citation = f'{author_str}. "{title}."'
                if journal:
                    citation += f" {journal},"
                citation += f" {year}."
                return citation

            elif style == "chicago":
                author_str = authors[0] if authors else "Unknown"
                citation = f'{author_str}. "{title}."'
                if journal:
                    citation += f" {journal}"
                citation += f" ({year})."
                return citation

        return ""


# EOF
