#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zotero linker - Live synchronization between Zotero and Scholar library.

Features:
- Monitor Zotero library for changes
- Automatically sync new/updated items
- Live citation insertion
- Auto-update on library changes
- Tag-based organization
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable
import json

from scitex import logging
from scitex.scholar.core.Paper import Paper
from scitex.scholar.storage import LibraryManager
from .importer import ZoteroImporter
from .exporter import ZoteroExporter
from .mapper import ZoteroMapper

logger = logging.getLogger(__name__)


class ZoteroLinker:
    """Bidirectional live synchronization with Zotero."""

    def __init__(
        self,
        library_id: Optional[str] = None,
        library_type: str = "user",
        api_key: Optional[str] = None,
        project: str = "default",
        config=None,
        sync_interval: int = 60,  # seconds
    ):
        """Initialize Zotero linker.

        Args:
            library_id: Zotero library ID (user ID or group ID)
            library_type: 'user' or 'group'
            api_key: Zotero API key
            project: Scholar project name
            config: Optional ScholarConfig instance
            sync_interval: Seconds between sync checks (default: 60)
        """
        self.library_id = library_id
        self.library_type = library_type
        self.api_key = api_key
        self.project = project
        self.config = config
        self.sync_interval = sync_interval

        self.importer = ZoteroImporter(
            library_id=library_id,
            library_type=library_type,
            api_key=api_key,
            project=project,
            config=config,
        )
        self.exporter = ZoteroExporter(
            library_id=library_id,
            library_type=library_type,
            api_key=api_key,
            project=project,
            config=config,
        )
        self.mapper = ZoteroMapper(config=config)
        self.library_manager = LibraryManager(project=project, config=config)

        # State tracking
        self._last_sync_version = 0
        self._synced_items: Set[str] = set()  # Zotero item keys
        self._callbacks: List[Callable] = []
        self._running = False

    def register_callback(self, callback: Callable[[str, Paper], None]):
        """Register callback for sync events.

        Callback signature: callback(event_type: str, paper: Paper)
        Event types: 'added', 'updated', 'deleted'

        Args:
            callback: Function to call on sync events
        """
        self._callbacks.append(callback)
        logger.info(f"Registered sync callback: {callback.__name__}")

    def start_sync(
        self,
        bidirectional: bool = True,
        auto_import: bool = True,
        auto_export: bool = False,
    ):
        """Start continuous synchronization.

        Args:
            bidirectional: Sync both directions (Zotero ↔ Scholar)
            auto_import: Automatically import new Zotero items
            auto_export: Automatically export new Scholar items
        """
        logger.info(
            f"Starting Zotero sync (interval: {self.sync_interval}s, "
            f"bidirectional: {bidirectional}, auto_import: {auto_import}, "
            f"auto_export: {auto_export})"
        )

        self._running = True

        try:
            while self._running:
                try:
                    # Check for Zotero updates
                    if auto_import:
                        self._sync_from_zotero()

                    # Check for Scholar updates
                    if bidirectional and auto_export:
                        self._sync_to_zotero()

                    # Wait before next sync
                    time.sleep(self.sync_interval)

                except KeyboardInterrupt:
                    logger.info("Sync interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Sync error: {e}")
                    time.sleep(self.sync_interval)

        finally:
            self._running = False
            logger.info("Stopped Zotero sync")

    def stop_sync(self):
        """Stop continuous synchronization."""
        self._running = False
        logger.info("Stopping Zotero sync...")

    def sync_once(
        self,
        bidirectional: bool = True,
        auto_import: bool = True,
        auto_export: bool = False,
    ) -> Dict[str, int]:
        """Perform single synchronization.

        Args:
            bidirectional: Sync both directions
            auto_import: Import new Zotero items
            auto_export: Export new Scholar items

        Returns:
            Dict with sync statistics
        """
        stats = {"imported": 0, "exported": 0, "updated": 0, "errors": 0}

        try:
            if auto_import:
                imported = self._sync_from_zotero()
                stats["imported"] = imported

            if bidirectional and auto_export:
                exported = self._sync_to_zotero()
                stats["exported"] = exported

        except Exception as e:
            logger.error(f"Sync error: {e}")
            stats["errors"] += 1

        logger.success(
            f"Sync complete: {stats['imported']} imported, "
            f"{stats['exported']} exported, {stats['errors']} errors"
        )

        return stats

    def _sync_from_zotero(self) -> int:
        """Sync updates from Zotero to Scholar.

        Returns:
            Number of items synced
        """
        try:
            zot = self.importer._get_client()

            # Get library version
            current_version = zot.last_modified_version()

            if current_version <= self._last_sync_version:
                logger.debug("No Zotero updates since last sync")
                return 0

            logger.info(
                f"Zotero library updated (version: {self._last_sync_version} → {current_version})"
            )

            # Get items modified since last sync
            items = zot.items(since=self._last_sync_version)

            logger.info(f"Found {len(items)} modified items in Zotero")

            synced_count = 0
            for item in items:
                try:
                    # Skip attachments and notes
                    if item["data"]["itemType"] in ["attachment", "note"]:
                        continue

                    item_key = item["key"]

                    # Convert to Paper
                    paper = self.mapper.zotero_to_paper(item)

                    # Determine if new or updated
                    event_type = (
                        "added" if item_key not in self._synced_items else "updated"
                    )

                    # Save to library
                    paper_id = self.library_manager.save_resolved_paper(
                        paper_data=paper, project=self.project
                    )

                    # Track synced item
                    self._synced_items.add(item_key)
                    synced_count += 1

                    # Notify callbacks
                    for callback in self._callbacks:
                        try:
                            callback(event_type, paper)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

                    logger.info(
                        f"[{event_type}] {paper.metadata.basic.title[:50]}... → {paper_id}"
                    )

                except Exception as e:
                    logger.error(f"Failed to sync item {item.get('key')}: {e}")
                    continue

            # Update last sync version
            self._last_sync_version = current_version

            logger.success(f"Synced {synced_count} items from Zotero")

            return synced_count

        except Exception as e:
            logger.error(f"Failed to sync from Zotero: {e}")
            return 0

    def _sync_to_zotero(self) -> int:
        """Sync updates from Scholar to Zotero.

        Returns:
            Number of items synced
        """
        # This would require tracking Scholar library changes
        # Implementation depends on how Scholar tracks modifications
        logger.info("Scholar → Zotero sync not yet implemented")
        return 0

    def insert_citation(
        self,
        paper: Paper,
        format: str = "bibtex",
        style: str = "apa",
    ) -> str:
        """Generate formatted citation for a paper.

        Args:
            paper: Paper object
            format: Citation format ('bibtex', 'ris', 'text')
            style: Citation style ('apa', 'mla', 'chicago', etc.)

        Returns:
            Formatted citation string
        """
        if format == "bibtex":
            # Generate BibTeX entry
            zotero_item = self.mapper.paper_to_zotero(paper)

            # Create citation key
            authors = paper.metadata.basic.authors or ["Unknown"]
            first_author = authors[0].split(",")[0] if authors else "Unknown"
            year = paper.metadata.basic.year or "NoYear"
            key = f"{first_author}{year}"

            # Build BibTeX entry
            lines = [f"@article{{{key},"]
            lines.append(f"  title = {{{zotero_item.get('title', '')}}},")

            if authors:
                authors_str = " and ".join(authors)
                lines.append(f"  author = {{{authors_str}}},")

            if paper.metadata.basic.year:
                lines.append(f"  year = {{{paper.metadata.basic.year}}},")

            if paper.metadata.publication.journal:
                lines.append(f"  journal = {{{paper.metadata.publication.journal}}},")

            if paper.metadata.id.doi:
                lines.append(f"  doi = {{{paper.metadata.id.doi}}},")

            lines.append("}")

            return "\n".join(lines)

        elif format == "ris":
            # Generate RIS format
            lines = ["TY  - JOUR"]

            if paper.metadata.basic.title:
                lines.append(f"TI  - {paper.metadata.basic.title}")

            if paper.metadata.basic.authors:
                for author in paper.metadata.basic.authors:
                    lines.append(f"AU  - {author}")

            if paper.metadata.basic.year:
                lines.append(f"PY  - {paper.metadata.basic.year}")

            if paper.metadata.publication.journal:
                lines.append(f"JO  - {paper.metadata.publication.journal}")

            if paper.metadata.id.doi:
                lines.append(f"DO  - {paper.metadata.id.doi}")

            lines.append("ER  - ")

            return "\n".join(lines)

        elif format == "text":
            # Generate formatted text citation
            authors = paper.metadata.basic.authors or []
            year = paper.metadata.basic.year or "n.d."
            title = paper.metadata.basic.title or "Untitled"
            journal = paper.metadata.publication.journal or ""

            if style == "apa":
                # APA format
                if len(authors) == 0:
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
                # MLA format
                if authors:
                    author_str = authors[0]
                else:
                    author_str = "Unknown"

                citation = f'{author_str}. "{title}."'
                if journal:
                    citation += f" {journal},"
                citation += f" {year}."

                return citation

            elif style == "chicago":
                # Chicago format
                if authors:
                    author_str = authors[0]
                else:
                    author_str = "Unknown"

                citation = f'{author_str}. "{title}."'
                if journal:
                    citation += f" {journal}"
                citation += f" ({year})."

                return citation

        return ""

    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status.

        Returns:
            Dict with sync status information
        """
        return {
            "running": self._running,
            "last_sync_version": self._last_sync_version,
            "synced_items_count": len(self._synced_items),
            "sync_interval": self.sync_interval,
            "callbacks_registered": len(self._callbacks),
        }


# EOF
