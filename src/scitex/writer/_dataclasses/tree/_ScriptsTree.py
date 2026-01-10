#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-28 17:16:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/tree/_ScriptsTree.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/tree/_ScriptsTree.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
ScriptsTree - dataclass for scripts directory structure.

Represents the scripts/ directory with compilation and helper scripts.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ..core import DocumentSection


@dataclass
class ScriptsTree:
    """Scripts directory structure (scripts/)."""

    root: Path
    git_root: Optional[Path] = None

    # Subdirectories
    examples: Path = None
    installation: Path = None
    powershell: Path = None
    python: Path = None
    shell: Path = None

    # Shell compilation scripts
    compile_manuscript: DocumentSection = None
    compile_supplementary: DocumentSection = None
    compile_revision: DocumentSection = None
    watch_compile: DocumentSection = None

    def __post_init__(self):
        """Initialize all Path and DocumentSection instances."""
        if self.examples is None:
            self.examples = self.root / "examples"
        if self.installation is None:
            self.installation = self.root / "installation"
        if self.powershell is None:
            self.powershell = self.root / "powershell"
        if self.python is None:
            self.python = self.root / "python"
        if self.shell is None:
            self.shell = self.root / "shell"

        # Initialize compilation scripts
        if self.compile_manuscript is None:
            self.compile_manuscript = DocumentSection(
                self.shell / "compile_manuscript.sh", self.git_root
            )
        if self.compile_supplementary is None:
            self.compile_supplementary = DocumentSection(
                self.shell / "compile_supplementary.sh", self.git_root
            )
        if self.compile_revision is None:
            self.compile_revision = DocumentSection(
                self.shell / "compile_revision.sh", self.git_root
            )
        if self.watch_compile is None:
            self.watch_compile = DocumentSection(
                self.shell / "watch_compile.sh", self.git_root
            )

    def verify_structure(self) -> tuple[bool, list[str]]:
        """
        Verify scripts structure has required directories and scripts.

        Returns:
            (is_valid, list_of_missing_items_with_paths)
        """
        missing = []

        # Check required directories
        required_dirs = [
            ("python", self.python),
            ("shell", self.shell),
        ]
        for name, path in required_dirs:
            if not path.exists():
                expected_path = (
                    path.relative_to(self.git_root) if self.git_root else path
                )
                missing.append(f"Missing {name}/ (expected at: {expected_path})")

        # Check required compilation scripts
        required_scripts = [
            ("compile_manuscript.sh", self.compile_manuscript),
            ("compile_supplementary.sh", self.compile_supplementary),
            ("compile_revision.sh", self.compile_revision),
        ]
        for name, section in required_scripts:
            if not section.path.exists():
                expected_path = (
                    section.path.relative_to(self.git_root)
                    if self.git_root
                    else section.path
                )
                missing.append(f"Missing {name} (expected at: {expected_path})")

        return len(missing) == 0, missing


__all__ = ["ScriptsTree"]

# EOF
