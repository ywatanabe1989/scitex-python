#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/core/_mixins/_project_handlers.py

"""
Project handler mixin for Scholar class.

Provides project management functionality.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from scitex import logging

logger = logging.getLogger(__name__)


class ProjectHandlerMixin:
    """Mixin providing project management methods."""

    def _ensure_project_exists(
        self, project: str, description: Optional[str] = None
    ) -> Path:
        """Ensure project directory exists, create if needed (PRIVATE).

        Args:
            project: Project name
            description: Optional project description

        Returns
        -------
            Path to the project directory
        """
        project_dir = self.config.get_library_project_dir(project)
        info_dir = project_dir / "info"

        if not project_dir.exists():
            project_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"{self.name}: Auto-created project directory: {project}")

        info_dir.mkdir(parents=True, exist_ok=True)

        old_metadata_file = project_dir / "project_metadata.json"
        metadata_file = info_dir / "project_metadata.json"

        if old_metadata_file.exists() and not metadata_file.exists():
            shutil.move(str(old_metadata_file), str(metadata_file))
            logger.info(f"{self.name}: Moved project metadata to info directory")

        if not metadata_file.exists():
            metadata = {
                "name": project,
                "description": description or f"Papers for {project} project",
                "created": datetime.now().isoformat(),
                "created_by": "SciTeX Scholar",
                "auto_created": True,
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"{self.name}: Created project metadata in info directory: {project}"
            )

        return project_dir

    def _create_project_metadata(
        self, project: str, description: Optional[str] = None
    ) -> Path:
        """Create project directory and metadata (PRIVATE).

        DEPRECATED: Use _ensure_project_exists instead.

        Args:
            project: Project name
            description: Optional project description

        Returns
        -------
            Path to the created project directory
        """
        return self._ensure_project_exists(project, description)

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects in the Scholar library.

        Returns
        -------
            List of project information dictionaries
        """
        library_dir = self.config.path_manager.library_dir
        projects = []

        for item in library_dir.iterdir():
            if item.is_dir() and item.name != "MASTER":
                project_info = {
                    "name": item.name,
                    "path": str(item),
                    "papers_count": len(list(item.glob("*"))),
                    "created": None,
                    "description": None,
                }

                metadata_file = item / "project_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        project_info.update(metadata)
                    except Exception as e:
                        logger.debug(f"Failed to load metadata for {item.name}: {e}")

                projects.append(project_info)

        return sorted(projects, key=lambda x: x["name"])


# EOF
