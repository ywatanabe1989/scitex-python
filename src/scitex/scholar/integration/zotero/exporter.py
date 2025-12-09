#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zotero exporter - Export data from Scholar library to Zotero.

Features:
- Export papers to Zotero collections
- Export manuscripts as preprint entries
- Export project metadata
- Export citation files (.bib, .ris)
- Batch export with progress tracking
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

from scitex import logging
from scitex.scholar.core.Paper import Paper
from scitex.scholar.core.Papers import Papers
from scitex.scholar.storage import BibTeXHandler
from .mapper import ZoteroMapper

logger = logging.getLogger(__name__)


class ZoteroExporter:
    """Export data from Scholar library to Zotero."""

    def __init__(
        self,
        library_id: Optional[str] = None,
        library_type: str = "user",
        api_key: Optional[str] = None,
        project: str = "default",
        config=None,
    ):
        """Initialize Zotero exporter.

        Args:
            library_id: Zotero library ID (user ID or group ID)
            library_type: 'user' or 'group'
            api_key: Zotero API key (from https://www.zotero.org/settings/keys)
            project: Scholar project name
            config: Optional ScholarConfig instance
        """
        self.library_id = library_id
        self.library_type = library_type
        self.api_key = api_key
        self.project = project
        self.config = config

        self.mapper = ZoteroMapper(config=config)
        self.bibtex_handler = BibTeXHandler(project=project, config=config)

        self._zot = None  # Lazy-initialized pyzotero client

    def _get_client(self):
        """Get or create pyzotero client."""
        if self._zot is None:
            if not self.library_id or not self.api_key:
                raise ValueError(
                    "library_id and api_key required for Zotero API access. "
                    "Get API key from: https://www.zotero.org/settings/keys"
                )

            try:
                from pyzotero import zotero

                self._zot = zotero.Zotero(
                    library_id=self.library_id,
                    library_type=self.library_type,
                    api_key=self.api_key,
                )
                logger.info(
                    f"Connected to Zotero {self.library_type} library: {self.library_id}"
                )
            except ImportError:
                raise ImportError(
                    "pyzotero not installed. Install with: pip install pyzotero"
                )

        return self._zot

    def export_papers(
        self,
        papers: Union[Papers, List[Paper]],
        collection_name: Optional[str] = None,
        create_collection: bool = True,
        update_existing: bool = True,
    ) -> Dict[str, str]:
        """Export Papers to Zotero library.

        Args:
            papers: Papers collection or list of Paper objects
            collection_name: Name of Zotero collection to add papers to
            create_collection: Create collection if it doesn't exist
            update_existing: Update existing items with same DOI/title

        Returns:
            Dict mapping paper titles to Zotero item keys
        """
        zot = self._get_client()

        if isinstance(papers, Papers):
            paper_list = papers.papers
        else:
            paper_list = papers

        # Get or create collection
        collection_key = None
        if collection_name:
            collection_key = self._get_or_create_collection(
                zot, collection_name, create=create_collection
            )

        results = {}

        for i, paper in enumerate(paper_list, 1):
            try:
                # Convert to Zotero format
                zotero_item = self.mapper.paper_to_zotero(paper)

                # Check if item already exists
                existing_key = None
                if update_existing:
                    existing_key = self._find_existing_item(zot, paper)

                if existing_key:
                    # Update existing item
                    logger.info(
                        f"[{i}/{len(paper_list)}] Updating existing item: {existing_key}"
                    )
                    zotero_item["version"] = self._get_item_version(zot, existing_key)
                    zot.update_item(zotero_item, existing_key)
                    item_key = existing_key
                else:
                    # Create new item
                    logger.info(
                        f"[{i}/{len(paper_list)}] Creating: {paper.metadata.basic.title[:50]}..."
                    )

                    # Add to collection if specified
                    if collection_key:
                        zotero_item["collections"] = [collection_key]

                    created = zot.create_items([zotero_item])
                    if created and "successful" in created and created["successful"]:
                        item_key = created["successful"]["0"]["key"]
                    else:
                        logger.error(f"Failed to create item: {created}")
                        continue

                results[paper.metadata.basic.title or f"paper_{i}"] = item_key

                logger.success(f"[{i}/{len(paper_list)}] Exported: {item_key}")

            except Exception as e:
                logger.error(
                    f"Failed to export paper: {paper.metadata.basic.title[:50]}... - {e}"
                )
                continue

        logger.success(f"Exported {len(results)}/{len(paper_list)} papers to Zotero")

        return results

    def export_as_bibtex(
        self,
        papers: Union[Papers, List[Paper]],
        output_path: Union[str, Path],
    ) -> Path:
        """Export Papers as BibTeX file compatible with Zotero import.

        Args:
            papers: Papers collection or list of Paper objects
            output_path: Path to save BibTeX file

        Returns:
            Path to created BibTeX file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use BibTeXHandler to create properly formatted BibTeX
        bibtex_content = self.bibtex_handler.papers_to_bibtex(
            papers, output_path=output_path
        )

        logger.success(f"Exported BibTeX file for Zotero import: {output_path}")

        return output_path

    def export_as_ris(
        self,
        papers: Union[Papers, List[Paper]],
        output_path: Union[str, Path],
    ) -> Path:
        """Export Papers as RIS file compatible with Zotero import.

        Args:
            papers: Papers collection or list of Paper objects
            output_path: Path to save RIS file

        Returns:
            Path to created RIS file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(papers, Papers):
            paper_list = papers.papers
        else:
            paper_list = papers

        # Generate RIS format
        ris_lines = []
        for paper in paper_list:
            # RIS format for journal article
            ris_lines.append("TY  - JOUR")  # Type: Journal Article

            if paper.metadata.basic.title:
                ris_lines.append(f"TI  - {paper.metadata.basic.title}")

            if paper.metadata.basic.authors:
                for author in paper.metadata.basic.authors:
                    ris_lines.append(f"AU  - {author}")

            if paper.metadata.basic.year:
                ris_lines.append(f"PY  - {paper.metadata.basic.year}")

            if paper.metadata.publication.journal:
                ris_lines.append(f"JO  - {paper.metadata.publication.journal}")

            if paper.metadata.publication.volume:
                ris_lines.append(f"VL  - {paper.metadata.publication.volume}")

            if paper.metadata.publication.issue:
                ris_lines.append(f"IS  - {paper.metadata.publication.issue}")

            if paper.metadata.publication.pages:
                ris_lines.append(f"SP  - {paper.metadata.publication.pages}")

            if paper.metadata.id.doi:
                ris_lines.append(f"DO  - {paper.metadata.id.doi}")

            if paper.metadata.basic.abstract:
                ris_lines.append(f"AB  - {paper.metadata.basic.abstract}")

            if paper.metadata.basic.keywords:
                for keyword in paper.metadata.basic.keywords:
                    ris_lines.append(f"KW  - {keyword}")

            if paper.metadata.url.doi:
                ris_lines.append(f"UR  - {paper.metadata.url.doi}")

            # End of record
            ris_lines.append("ER  - ")
            ris_lines.append("")  # Blank line between records

        # Write to file
        ris_content = "\n".join(ris_lines)
        output_path.write_text(ris_content)

        logger.success(f"Exported RIS file for Zotero import: {output_path}")

        return output_path

    def _get_or_create_collection(
        self, zot, collection_name: str, create: bool = True
    ) -> Optional[str]:
        """Get collection key by name or create if it doesn't exist.

        Args:
            zot: Pyzotero client
            collection_name: Name of collection
            create: Whether to create if not found

        Returns:
            Collection key or None
        """
        # Search for existing collection
        collections = zot.collections()
        for coll in collections:
            if coll["data"]["name"] == collection_name:
                logger.info(f"Found collection '{collection_name}': {coll['key']}")
                return coll["key"]

        # Create if requested
        if create:
            logger.info(f"Creating new collection: {collection_name}")
            template = zot.collection_template()
            template["name"] = collection_name
            created = zot.create_collections([template])

            if created and "successful" in created and created["successful"]:
                collection_key = created["successful"]["0"]["key"]
                logger.success(f"Created collection: {collection_key}")
                return collection_key

        logger.warning(f"Collection not found: {collection_name}")
        return None

    def _find_existing_item(self, zot, paper: Paper) -> Optional[str]:
        """Find existing Zotero item for a paper.

        Searches by DOI first, then by title.

        Args:
            zot: Pyzotero client
            paper: Paper object

        Returns:
            Zotero item key if found, None otherwise
        """
        # Search by DOI
        if paper.metadata.id.doi:
            items = zot.items(q=f"doi:{paper.metadata.id.doi}")
            if items:
                logger.debug(f"Found existing item by DOI: {items[0]['key']}")
                return items[0]["key"]

        # Search by title
        if paper.metadata.basic.title:
            # Exact title search
            items = zot.items(q=paper.metadata.basic.title)
            for item in items:
                if (
                    item["data"].get("title", "").lower()
                    == paper.metadata.basic.title.lower()
                ):
                    logger.debug(f"Found existing item by title: {item['key']}")
                    return item["key"]

        return None

    def _get_item_version(self, zot, item_key: str) -> int:
        """Get current version number of an item.

        Args:
            zot: Pyzotero client
            item_key: Zotero item key

        Returns:
            Version number
        """
        try:
            item = zot.item(item_key)
            return item.get("version", 0)
        except Exception as e:
            logger.warning(f"Failed to get item version for {item_key}: {e}")
            return 0


# EOF
