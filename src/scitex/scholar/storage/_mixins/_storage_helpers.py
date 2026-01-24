#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/storage/_mixins/_storage_helpers.py

"""
Storage helper mixin for LibraryManager.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Optional

from scitex import logging

if TYPE_CHECKING:
    from scitex.scholar.core.Paper import Paper

logger = logging.getLogger(__name__)


class StorageHelpersMixin:
    """Mixin providing storage helper methods."""

    def has_metadata(self, paper_id: str) -> bool:
        """Check if metadata.json exists for paper."""
        metadata_file = self.library_master_dir / paper_id / "metadata.json"
        return metadata_file.exists()

    def has_urls(self, paper_id: str) -> bool:
        """Check if PDF URLs exist in metadata."""
        if not self.has_metadata(paper_id):
            return False

        metadata_file = self.library_master_dir / paper_id / "metadata.json"
        try:
            with open(metadata_file) as f:
                data = json.load(f)

            urls = data.get("metadata", {}).get("url", {}).get("pdfs", [])
            return len(urls) > 0
        except Exception:
            return False

    def has_pdf(self, paper_id: str) -> bool:
        """Check if PDF file exists in storage."""
        paper_dir = self.library_master_dir / paper_id
        if not paper_dir.exists():
            return False

        pdf_files = list(paper_dir.glob("*.pdf"))
        return len(pdf_files) > 0

    def load_paper_from_id(self, paper_id: str) -> Optional[Paper]:
        """Load Paper object from storage by ID."""
        from scitex.scholar.core.Paper import Paper

        metadata_file = self.library_master_dir / paper_id / "metadata.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file) as f:
                data = json.load(f)

            paper = Paper.from_dict(data)
            return paper

        except Exception as e:
            logger.error(f"Failed to load paper {paper_id}: {e}")
            return None

    def save_paper_incremental(self, paper_id: str, paper: Paper) -> None:
        """Save Paper object to storage (incremental update)."""
        storage_path = self.library_master_dir / paper_id
        storage_path.mkdir(parents=True, exist_ok=True)

        metadata_file = storage_path / "metadata.json"

        existing_data = {}
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    existing_data = json.load(f)
            except Exception:
                pass

        new_data = paper.model_dump()
        merged_data = self._merge_metadata(existing_data, new_data)

        if "container" not in merged_data:
            merged_data["container"] = {}
        merged_data["container"]["updated_at"] = datetime.now().isoformat()

        with open(metadata_file, "w") as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved paper {paper_id} to storage")

    def _merge_metadata(self, existing: Dict, new: Dict) -> Dict:
        """Recursively merge metadata dicts, preferring new non-None values."""
        result = existing.copy()

        for key, new_value in new.items():
            if key not in result:
                result[key] = new_value
            elif new_value is None:
                pass
            elif isinstance(new_value, dict) and isinstance(result[key], dict):
                result[key] = self._merge_metadata(result[key], new_value)
            elif isinstance(new_value, list) and len(new_value) > 0:
                result[key] = new_value
            elif new_value:
                result[key] = new_value

        return result


# EOF
