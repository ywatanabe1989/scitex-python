#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/template.py

"""
Writer project template management.

Provides functions to create and copy writer project templates.
"""

import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def copy_template(source: Path, dest: Path, name: str = None) -> Path:
    """
    Copy writer template to destination.

    Args:
        source: Source template directory (e.g., /tmp/scitex-writer)
        dest: Destination directory
        name: Project name (optional, defaults to directory name)

    Returns:
        Path to created project directory

    Raises:
        FileNotFoundError: If source doesn't exist
        FileExistsError: If destination already exists
    """
    source = Path(source)
    dest = Path(dest)

    if not source.exists():
        raise FileNotFoundError(f"Template not found: {source}")

    if dest.exists():
        raise FileExistsError(f"Destination already exists: {dest}")

    logger.info(f"Copying writer template from {source} to {dest}")

    # Copy entire directory structure
    shutil.copytree(source, dest, symlinks=True)

    # Update project name if provided
    if name:
        _update_project_name(dest, name)

    logger.info(f"Created writer project at {dest}")
    return dest


def create_writer_project(
    dest: Path,
    name: str,
    template_source: Optional[Path] = None
) -> Path:
    """
    Create new writer project from template.

    Args:
        dest: Destination directory for new project
        name: Project name
        template_source: Optional custom template source (default: auto-detect)

    Returns:
        Path to created project directory

    Examples:
        >>> from pathlib import Path
        >>> project_dir = create_writer_project(
        ...     Path("/path/to/my-paper"),
        ...     name="My Paper"
        ... )
    """
    dest = Path(dest)

    # Find template source if not provided
    if template_source is None:
        template_locations = [
            Path("/tmp/scitex-writer"),
            Path.home() / "proj" / "scitex-writer",
        ]

        for location in template_locations:
            if location.exists():
                template_source = location
                break

        if template_source is None:
            raise FileNotFoundError(
                "scitex-writer template not found. "
                "Please clone to /tmp/scitex-writer or ~/proj/scitex-writer"
            )

    return copy_template(template_source, dest, name)


def _update_project_name(project_dir: Path, name: str) -> None:
    """
    Update project name in template files.

    Args:
        project_dir: Project directory
        name: New project name
    """
    # Update shared/title.tex if it exists
    title_file = project_dir / "shared" / "title.tex"
    if title_file.exists():
        try:
            content = title_file.read_text()
            # Simple replacement - adjust based on actual template format
            if "Template Title" in content:
                content = content.replace("Template Title", name)
                title_file.write_text(content)
                logger.debug(f"Updated title in {title_file}")
        except Exception as e:
            logger.warning(f"Could not update title: {e}")


__all__ = [
    'copy_template',
    'create_writer_project',
]

# EOF
