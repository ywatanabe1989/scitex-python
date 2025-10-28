#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:08:47 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/_WriterConfig.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/writer/dataclasses/_WriterConfig.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
WriterConfig - dataclass for writer configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    def from_directory(cls, project_dir: Path) -> "WriterConfig":
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
            raise ValueError(
                f"Project directory not found: {self.project_dir}"
            )

        # Check for at least one document directory
        doc_dirs = [
            self.manuscript_dir,
            self.supplementary_dir,
            self.revision_dir,
        ]

        if not any(d.exists() for d in doc_dirs):
            raise ValueError(
                f"No document directories found in {self.project_dir}. "
                "Expected: 01_manuscript/, 02_supplementary/, or 03_revision/"
            )

        return True


__all__ = ["WriterConfig"]

# EOF
