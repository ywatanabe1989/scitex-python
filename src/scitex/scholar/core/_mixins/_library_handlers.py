#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/core/_mixins/_library_handlers.py

"""
Library handler mixin for Scholar class.

Provides library-wide statistics and backup functionality.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

from scitex import logging
from scitex.logging import ScholarError

logger = logging.getLogger(__name__)


class LibraryHandlerMixin:
    """Mixin providing library management methods."""

    def get_library_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the entire Scholar library.

        Returns
        -------
            Dictionary with library-wide statistics
        """
        master_dir = self.config.get_library_master_dir()
        projects = self.list_projects()

        stats = {
            "total_projects": len(projects),
            "total_papers": (
                len(list(master_dir.glob("*"))) if master_dir.exists() else 0
            ),
            "projects": projects,
            "library_path": str(self.config.path_manager.library_dir),
            "master_path": str(master_dir),
        }

        if master_dir.exists():
            total_size = sum(
                f.stat().st_size for f in master_dir.rglob("*") if f.is_file()
            )
            stats["storage_mb"] = total_size / (1024 * 1024)
        else:
            stats["storage_mb"] = 0

        return stats

    def backup_library(self, backup_path: Union[str, Path]) -> Dict[str, Any]:
        """Create a backup of the Scholar library.

        Args:
            backup_path: Path for the backup

        Returns
        -------
            Dictionary with backup information
        """
        backup_path = Path(backup_path)
        library_path = self.config.path_manager.library_dir

        if not library_path.exists():
            raise ScholarError("Library directory does not exist")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = backup_path / f"scholar_library_backup_{timestamp}"

        logger.info(f"{self.name}: Creating library backup at {backup_dir}")
        shutil.copytree(library_path, backup_dir)

        backup_info = {
            "timestamp": timestamp,
            "source": str(library_path),
            "backup": str(backup_dir),
            "size_mb": sum(
                f.stat().st_size for f in backup_dir.rglob("*") if f.is_file()
            )
            / (1024 * 1024),
        }

        metadata_file = backup_dir / "backup_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(backup_info, f, indent=2)

        logger.info(
            f"{self.name}: Library backup completed: {backup_info['size_mb']:.2f} MB"
        )
        return backup_info


# EOF
