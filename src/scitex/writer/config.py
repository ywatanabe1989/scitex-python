#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/config.py

"""
Writer configuration and utilities.

Provides configuration management and helper functions for scitex.writer.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class WriterConfig:
    """Configuration for scitex.writer."""

    project_dir: Path
    """Root directory of writer project"""

    manuscript_dir: Path
    """Directory for manuscript (01_manuscript/)"""

    supplementary_dir: Path
    """Directory for supplementary (02_supplementary/)"""

    revision_dir: Path
    """Directory for revision (03_revision/)"""

    shared_dir: Path
    """Directory for shared resources (shared/)"""

    compile_script: Optional[Path] = None
    """Path to compile script (auto-detected if None)"""

    @classmethod
    def from_directory(cls, project_dir: Path) -> 'WriterConfig':
        """
        Create config from project directory.

        Args:
            project_dir: Path to writer project root

        Returns:
            WriterConfig instance

        Examples:
            >>> config = WriterConfig.from_directory(Path("/path/to/project"))
            >>> print(config.manuscript_dir)
        """
        project_dir = Path(project_dir)

        return cls(
            project_dir=project_dir,
            manuscript_dir=project_dir / "01_manuscript",
            supplementary_dir=project_dir / "02_supplementary",
            revision_dir=project_dir / "03_revision",
            shared_dir=project_dir / "shared",
        )

    def validate(self) -> bool:
        """
        Validate that required directories exist.

        Returns:
            True if valid writer project structure

        Raises:
            ValueError: If invalid structure
        """
        if not self.project_dir.exists():
            raise ValueError(f"Project directory not found: {self.project_dir}")

        # Check for at least one document directory
        doc_dirs = [
            self.manuscript_dir,
            self.supplementary_dir,
            self.revision_dir
        ]

        if not any(d.exists() for d in doc_dirs):
            raise ValueError(
                f"No document directories found in {self.project_dir}. "
                "Expected: 01_manuscript/, 02_supplementary/, or 03_revision/"
            )

        return True


def find_writer_root(start_path: Path = None) -> Optional[Path]:
    """
    Find writer project root by searching upward for marker files/directories.

    Args:
        start_path: Starting directory (default: current directory)

    Returns:
        Path to writer project root, or None if not found

    Examples:
        >>> root = find_writer_root()
        >>> if root:
        ...     print(f"Found writer project at {root}")
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path)

    # Marker indicators of writer project root
    markers = [
        'compile',  # Main compile script
        '01_manuscript',
        'shared',
        'config'
    ]

    current = start_path.resolve()

    # Search upward
    for _ in range(10):  # Limit search depth
        # Check if this directory has writer markers
        found_markers = sum(1 for m in markers if (current / m).exists())

        if found_markers >= 2:  # Need at least 2 markers
            logger.debug(f"Found writer root at {current}")
            return current

        # Move up one level
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    logger.debug("Writer project root not found")
    return None


def get_bib_file(project_dir: Path) -> Optional[Path]:
    """
    Get path to bibliography file.

    Args:
        project_dir: Writer project directory

    Returns:
        Path to bibliography.bib or None if not found
    """
    bib_locations = [
        project_dir / "shared" / "bib_files" / "bibliography.bib",
        project_dir / "shared" / "bibliography.bib",
    ]

    for bib_path in bib_locations:
        if bib_path.exists():
            return bib_path

    return None


def get_output_pdf(project_dir: Path, doc_type: str = 'manuscript') -> Optional[Path]:
    """
    Get path to output PDF.

    Args:
        project_dir: Writer project directory
        doc_type: Document type ('manuscript', 'supplementary', 'revision')

    Returns:
        Path to PDF or None if not found
    """
    doc_map = {
        'manuscript': '01_manuscript',
        'supplementary': '02_supplementary',
        'revision': '03_revision'
    }

    doc_dir = project_dir / doc_map.get(doc_type, '01_manuscript')
    pdf_path = doc_dir / f"{doc_type}.pdf"

    if pdf_path.exists():
        return pdf_path

    return None


__all__ = [
    'WriterConfig',
    'find_writer_root',
    'get_bib_file',
    'get_output_pdf',
]

# EOF
