#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zotero importer - Import data from Zotero to Scholar library.

Features:
- Import bibliography with collections and tags
- Import PDF annotations and notes
- Import metadata enrichment
- Batch import with progress tracking
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

from scitex import logging
from scitex.scholar.core.Paper import Paper
from scitex.scholar.core.Papers import Papers
from scitex.scholar.storage import LibraryManager
from .mapper import ZoteroMapper

logger = logging.getLogger(__name__)


class ZoteroImporter:
    """Import data from Zotero to Scholar library."""

    def __init__(
        self,
        library_id: Optional[str] = None,
        library_type: str = "user",
        api_key: Optional[str] = None,
        project: str = "default",
        config=None,
    ):
        """Initialize Zotero importer.

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
        self.library_manager = LibraryManager(project=project, config=config)

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

    def import_collection(
        self,
        collection_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        limit: Optional[int] = None,
        include_pdfs: bool = True,
        include_annotations: bool = True,
    ) -> Papers:
        """Import items from a Zotero collection.

        Args:
            collection_id: Zotero collection ID (takes precedence over name)
            collection_name: Collection name (searches for matching collection)
            limit: Maximum number of items to import (None = all)
            include_pdfs: Whether to download PDF attachments
            include_annotations: Whether to import PDF annotations

        Returns:
            Papers collection with imported papers
        """
        zot = self._get_client()

        # Get collection ID from name if needed
        if not collection_id and collection_name:
            collections = zot.collections()
            for coll in collections:
                if coll["data"]["name"] == collection_name:
                    collection_id = coll["key"]
                    logger.info(
                        f"Found collection '{collection_name}': {collection_id}"
                    )
                    break

            if not collection_id:
                raise ValueError(f"Collection not found: {collection_name}")

        # Fetch items
        if collection_id:
            logger.info(f"Importing from collection: {collection_id}")
            items = zot.collection_items(collection_id, limit=limit)
        else:
            logger.info("Importing all top-level items")
            items = zot.top(limit=limit)

        logger.info(f"Retrieved {len(items)} items from Zotero")

        # Convert to Papers
        papers = []
        for i, item in enumerate(items, 1):
            try:
                # Skip attachments and notes (import separately)
                if item["data"]["itemType"] in ["attachment", "note"]:
                    continue

                paper = self.mapper.zotero_to_paper(item)

                # Mark as part of this collection
                if collection_id:
                    paper.container.projects.append(f"zotero_{collection_id}")

                # Import attachments if requested
                if include_pdfs:
                    self._import_attachments(zot, item["key"], paper)

                # Import annotations if requested
                if include_annotations:
                    self._import_annotations(zot, item["key"], paper)

                papers.append(paper)

                logger.info(
                    f"[{i}/{len(items)}] Imported: {paper.metadata.basic.title[:50]}..."
                )

            except Exception as e:
                logger.error(f"Failed to import item {item.get('key')}: {e}")
                continue

        papers_collection = Papers(papers, project=self.project)
        logger.success(f"Imported {len(papers)} papers from Zotero")

        return papers_collection

    def import_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        limit: Optional[int] = None,
        include_pdfs: bool = True,
        include_annotations: bool = True,
    ) -> Papers:
        """Import items with specific tags.

        Args:
            tags: List of tag names
            match_all: If True, require all tags; if False, require any tag
            limit: Maximum number of items to import
            include_pdfs: Whether to download PDF attachments
            include_annotations: Whether to import PDF annotations

        Returns:
            Papers collection with imported papers
        """
        zot = self._get_client()

        # Search by tags
        query_parts = []
        for tag in tags:
            query_parts.append(f"tag:{tag}")

        operator = " && " if match_all else " || "
        query = operator.join(query_parts)

        logger.info(f"Searching Zotero with query: {query}")
        items = zot.items(q=query, limit=limit)

        logger.info(f"Found {len(items)} items matching tags: {tags}")

        # Convert to Papers
        papers = []
        for i, item in enumerate(items, 1):
            try:
                if item["data"]["itemType"] in ["attachment", "note"]:
                    continue

                paper = self.mapper.zotero_to_paper(item)

                if include_pdfs:
                    self._import_attachments(zot, item["key"], paper)

                if include_annotations:
                    self._import_annotations(zot, item["key"], paper)

                papers.append(paper)

                logger.info(
                    f"[{i}/{len(items)}] Imported: {paper.metadata.basic.title[:50]}..."
                )

            except Exception as e:
                logger.error(f"Failed to import item {item.get('key')}: {e}")
                continue

        papers_collection = Papers(papers, project=self.project)
        logger.success(f"Imported {len(papers)} papers by tags")

        return papers_collection

    def import_all(
        self,
        limit: Optional[int] = None,
        include_pdfs: bool = True,
        include_annotations: bool = True,
    ) -> Papers:
        """Import all items from Zotero library.

        Args:
            limit: Maximum number of items to import
            include_pdfs: Whether to download PDF attachments
            include_annotations: Whether to import PDF annotations

        Returns:
            Papers collection with all imported papers
        """
        return self.import_collection(
            collection_id=None,
            limit=limit,
            include_pdfs=include_pdfs,
            include_annotations=include_annotations,
        )

    def _import_attachments(self, zot, item_key: str, paper: Paper) -> None:
        """Import PDF attachments for an item.

        Args:
            zot: Pyzotero client
            item_key: Zotero item key
            paper: Paper object to update
        """
        try:
            children = zot.children(item_key)

            for child in children:
                if child["data"]["itemType"] != "attachment":
                    continue

                content_type = child["data"].get("contentType", "")
                if "pdf" not in content_type.lower():
                    continue

                # Get attachment URL
                attachment_key = child["key"]
                attachment_url = f"https://api.zotero.org/users/{self.library_id}/items/{attachment_key}/file"

                # Store URL in paper metadata
                pdf_entry = {
                    "url": attachment_url,
                    "source": "zotero",
                    "zotero_key": attachment_key,
                }

                if pdf_entry not in paper.metadata.url.pdfs:
                    paper.metadata.url.pdfs.append(pdf_entry)
                    paper.metadata.url.pdfs_engines.append("zotero")

                logger.debug(f"Added PDF attachment: {attachment_key}")

        except Exception as e:
            logger.warning(f"Failed to import attachments for {item_key}: {e}")

    def _import_annotations(self, zot, item_key: str, paper: Paper) -> None:
        """Import PDF annotations for an item.

        Args:
            zot: Pyzotero client
            item_key: Zotero item key
            paper: Paper object to update
        """
        try:
            children = zot.children(item_key)

            annotations = []
            for child in children:
                if child["data"]["itemType"] != "annotation":
                    continue

                annotation_data = {
                    "type": child["data"].get("annotationType", ""),
                    "text": child["data"].get("annotationText", ""),
                    "comment": child["data"].get("annotationComment", ""),
                    "color": child["data"].get("annotationColor", ""),
                    "page": child["data"].get("annotationPageLabel", ""),
                    "date": child["data"].get("dateModified", ""),
                }

                annotations.append(annotation_data)

            if annotations:
                # Store annotations as special field
                paper._zotero_annotations = annotations
                logger.debug(f"Imported {len(annotations)} annotations")

        except Exception as e:
            logger.warning(f"Failed to import annotations for {item_key}: {e}")

    def import_to_library(
        self,
        papers: Union[Papers, List[Paper]],
        update_existing: bool = True,
    ) -> Dict[str, str]:
        """Import Papers into Scholar library storage.

        Args:
            papers: Papers collection or list of Paper objects
            update_existing: Whether to update existing papers

        Returns:
            Dict mapping paper titles to library IDs
        """
        if isinstance(papers, Papers):
            paper_list = papers.papers
        else:
            paper_list = papers

        results = {}

        for i, paper in enumerate(paper_list, 1):
            try:
                # Save to library
                paper_id = self.library_manager.save_resolved_paper(
                    paper_data=paper, project=self.project
                )

                results[paper.metadata.basic.title or f"paper_{i}"] = paper_id

                logger.info(f"[{i}/{len(paper_list)}] Saved to library: {paper_id}")

            except Exception as e:
                logger.error(
                    f"Failed to save paper to library: {paper.metadata.basic.title[:50]}... - {e}"
                )
                continue

        logger.success(
            f"Imported {len(results)}/{len(paper_list)} papers to Scholar library"
        )

        return results


# EOF
