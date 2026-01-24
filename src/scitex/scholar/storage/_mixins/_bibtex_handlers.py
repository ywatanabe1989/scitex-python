#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/storage/_mixins/_bibtex_handlers.py

"""
BibTeX handling mixin for LibraryManager.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from scitex import logging

logger = logging.getLogger(__name__)


class BibtexHandlersMixin:
    """Mixin providing BibTeX handling methods."""

    def _create_bibtex_info_structure(
        self,
        project: str,
        paper_info: Dict[str, Any],
        complete_metadata: Dict[str, Any],
        bibtex_source_filename: str = "papers",
    ) -> Optional[Path]:
        """Create info/papers_bib/pac.bib structure."""
        try:
            project_dir = self.config.path_manager.get_library_project_dir(project)
            info_dir = project_dir / "info" / f"{bibtex_source_filename}_bib"
            info_dir.mkdir(parents=True, exist_ok=True)

            bibtex_file = info_dir / f"{bibtex_source_filename}.bib"
            unresolved_dir = info_dir / "unresolved"
            unresolved_dir.mkdir(parents=True, exist_ok=True)

            first_author = "unknown"
            if complete_metadata.get("authors"):
                authors = complete_metadata["authors"]
                if isinstance(authors, list) and authors:
                    first_author = str(authors[0]).split()[-1].lower()
                elif isinstance(authors, str):
                    first_author = authors.split()[-1].lower()

            year = complete_metadata.get("year", "unknown")
            entry_key = f"{first_author}{year}"

            bibtex_entry = self._generate_bibtex_entry(complete_metadata, entry_key)

            if bibtex_file.exists():
                with open(bibtex_file, "a", encoding="utf-8") as file_:
                    file_.write(f"\n{bibtex_entry}")
            else:
                with open(bibtex_file, "w", encoding="utf-8") as file_:
                    file_.write(bibtex_entry)

            if not complete_metadata.get("doi"):
                unresolved_file = unresolved_dir / f"{entry_key}.json"
                unresolved_data = {
                    "title": complete_metadata.get("title", ""),
                    "authors": complete_metadata.get("authors", []),
                    "year": complete_metadata.get("year", ""),
                    "journal": complete_metadata.get("journal", ""),
                    "scholar_id": complete_metadata.get("scholar_id", ""),
                    "resolution_failed": True,
                    "timestamp": complete_metadata.get("created_at", ""),
                }
                with open(unresolved_file, "w", encoding="utf-8") as file_:
                    json.dump(unresolved_data, file_, indent=2)
                logger.info(f"Added unresolved entry: {unresolved_file}")

            logger.success(f"Updated BibTeX info structure: {bibtex_file}")
            return info_dir

        except Exception as exc_:
            logger.warning(f"Failed to create BibTeX info structure: {exc_}")
            return None

    def _generate_bibtex_entry(self, metadata: Dict[str, Any], entry_key: str) -> str:
        """Generate BibTeX entry from metadata."""
        entry_type = "article"
        if metadata.get("journal"):
            entry_type = "article"
        elif metadata.get("booktitle"):
            entry_type = "inproceedings"
        elif metadata.get("publisher") and not metadata.get("journal"):
            entry_type = "book"

        bibtex = f"@{entry_type}{{{entry_key},\n"

        field_mappings = {
            "title": "title",
            "authors": "author",
            "year": "year",
            "journal": "journal",
            "doi": "doi",
            "volume": "volume",
            "issue": "number",
            "pages": "pages",
            "publisher": "publisher",
            "booktitle": "booktitle",
            "abstract": "abstract",
        }

        for meta_field, bibtex_field in field_mappings.items():
            value = metadata.get(meta_field)
            if value:
                if isinstance(value, list):
                    value = " and ".join(str(val_) for val_ in value)
                value_escaped = str(value).replace("{", "\\{").replace("}", "\\}")
                bibtex += f"  {bibtex_field} = {{{value_escaped}}},\n"

                source_field = f"{meta_field}_source"
                if source_field in metadata:
                    bibtex += f"  % {bibtex_field}_source = {metadata[source_field]}\n"

        bibtex += f"  % scholar_id = {metadata.get('scholar_id', 'unknown')},\n"
        bibtex += f"  % created_at = {metadata.get('created_at', 'unknown')},\n"
        bibtex += f"  % created_by = {metadata.get('created_by', 'unknown')},\n"
        bibtex += "}\n"

        return bibtex


# EOF
