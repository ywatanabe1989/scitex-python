#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-28 17:16:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/tree/_ConfigTree.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/tree/_ConfigTree.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
ConfigTree - dataclass for project config directory structure.

Represents the config/ directory with configuration files for different documents.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ..core import DocumentSection


@dataclass
class ConfigTree:
    """Config directory structure (config/)."""

    root: Path
    git_root: Optional[Path] = None

    # Configuration files
    config_manuscript: DocumentSection = None
    config_supplementary: DocumentSection = None
    config_revision: DocumentSection = None
    load_config: DocumentSection = None

    def __post_init__(self):
        """Initialize all DocumentSection instances."""
        if self.config_manuscript is None:
            self.config_manuscript = DocumentSection(
                self.root / "config_manuscript.yaml", self.git_root
            )
        if self.config_supplementary is None:
            self.config_supplementary = DocumentSection(
                self.root / "config_supplementary.yaml", self.git_root
            )
        if self.config_revision is None:
            self.config_revision = DocumentSection(
                self.root / "config_revision.yaml", self.git_root
            )
        if self.load_config is None:
            self.load_config = DocumentSection(
                self.root / "load_config.sh", self.git_root
            )

    def verify_structure(self) -> tuple[bool, list[str]]:
        """
        Verify config structure has required files.

        Returns:
            (is_valid, list_of_missing_files_with_paths)
        """
        required = [
            ("config_manuscript.yaml", self.config_manuscript),
            ("config_supplementary.yaml", self.config_supplementary),
            ("config_revision.yaml", self.config_revision),
        ]

        missing = []
        for name, section in required:
            if not section.path.exists():
                expected_path = (
                    section.path.relative_to(self.git_root)
                    if self.git_root
                    else section.path
                )
                missing.append(f"{name} (expected at: {expected_path})")

        return len(missing) == 0, missing


__all__ = ["ConfigTree"]

# EOF
