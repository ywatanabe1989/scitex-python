#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-28 16:27:14 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/writer.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/writer/writer.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Writer class for manuscript LaTeX compilation.

Provides object-oriented interface to scitex-writer functionality.
"""

from pathlib import Path
from typing import Optional
from typing import Callable

from .compile import compile_manuscript
from .compile import compile_supplementary
from .compile import compile_revision
from .compile import CompilationResult
from .watch import watch_manuscript
from ._init_directory import init_directory as _init_directory
from .types import (
    DocumentSection,
    Document,
    ManuscriptDocument,
    SupplementaryDocument,
    RevisionDocument,
)
from scitex import logging

logger = logging.getLogger(__name__)


class Writer:
    """LaTeX manuscript compiler."""

    def __init__(
        self,
        project_dir: Path,
        name: Optional[str] = None,
        git_strategy: Optional[str] = "child",
    ):
        """
        Initialize for project directory.

        If directory doesn't exist, creates new project.

        Args:
            project_dir: Path to project directory
            name: Project name (used if creating new project)
            git_strategy: Git initialization strategy
                - 'child': Create isolated git in project directory (default)
                - 'parent': Use parent git repository (error if not found)
                - None: Disable git initialization
        """
        self.project_dir = Path(project_dir)
        self.git_strategy = git_strategy

        # Create new project if directory doesn't exist
        if not self.project_dir.exists():
            project_name = name or self.project_dir.name
            target_dir = self.project_dir.parent

            # Choose creation method based on git_strategy
            if self.git_strategy == "child":
                # Use full template (includes git initialization)
                _init_directory(project_name, str(target_dir))
            else:
                # Just create minimal directory structure
                # (parent or None strategy - no template needed)
                self.project_dir.mkdir(parents=True, exist_ok=True)
                for subdir in [
                    "01_manuscript/contents",
                    "02_supplementary/contents",
                    "03_revision/contents",
                    "shared",
                ]:
                    (self.project_dir / subdir).mkdir(
                        parents=True, exist_ok=True
                    )

        # Initialize git repo based on strategy
        self.git_root = self._init_git_repo()

        # Document accessors (pass git_root for efficiency)
        self.manuscript = ManuscriptDocument(
            self.project_dir / "01_manuscript", git_root=self.git_root
        )
        self.supplementary = SupplementaryDocument(
            self.project_dir / "02_supplementary", git_root=self.git_root
        )
        self.revision = RevisionDocument(
            self.project_dir / "03_revision", git_root=self.git_root
        )

    def _find_parent_git(self) -> Optional[Path]:
        """
        Find parent git repository by walking up directory tree.

        Returns:
            Path to parent git root, or None if not found
        """
        current = self.project_dir.absolute()
        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent
        return None

    def _create_child_git(self) -> Optional[Path]:
        """
        Create isolated git repository in project directory.

        Returns:
            Path to git root (project_dir), or None on failure
        """
        import subprocess

        try:
            # Check if already a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.project_dir,
                capture_output=True,
                timeout=5,
            )

            if result.returncode == 0:
                # Already a git repo
                return self.project_dir

            # Initialize new git repo
            subprocess.run(
                ["git", "init"],
                cwd=self.project_dir,
                capture_output=True,
                timeout=5,
            )

            # Initial commit
            subprocess.run(
                ["git", "add", "."],
                cwd=self.project_dir,
                capture_output=True,
                timeout=5,
            )

            subprocess.run(
                ["git", "commit", "-m", "Initial commit from scitex-writer"],
                cwd=self.project_dir,
                capture_output=True,
                timeout=5,
            )

            return self.project_dir
        except Exception:
            return None

    def _init_git_repo(self) -> Optional[Path]:
        """
        Initialize or detect git repository based on git_strategy with graceful degradation.

        Returns:
            Path to git repository root, or None if disabled

        Strategy details:
        - None: Git disabled, returns None
        - 'child': Creates isolated git repo in project directory
        - 'parent': Tries to use parent git, degrades to 'child' if not found
        """
        # Strategy: disabled
        if self.git_strategy is None:
            return None

        # Strategy: parent (with graceful degradation to child)
        if self.git_strategy == "parent":
            parent_git = self._find_parent_git()

            if parent_git:
                logger.info(f"Using parent git repository: {parent_git}")
                return parent_git

            # Graceful degradation: no parent git found, use child strategy
            logger.warning(
                f"No parent git repository found for {self.project_dir}. "
                f"Degrading to 'child' strategy (isolated git repo)."
            )
            return self._create_child_git()

        # Strategy: child (create isolated repo)
        if self.git_strategy == "child":
            return self._create_child_git()

        # Unknown strategy
        raise ValueError(
            f"Unknown git_strategy: {self.git_strategy}. "
            f"Expected 'parent', 'child', or None"
        )

    def compile_manuscript(self, timeout: int = 300) -> CompilationResult:
        """Compile manuscript."""
        return compile_manuscript(self.project_dir, timeout=timeout)

    def compile_supplementary(self, timeout: int = 300) -> CompilationResult:
        """Compile supplementary materials."""
        return compile_supplementary(self.project_dir, timeout=timeout)

    def compile_revision(
        self, track_changes: bool = False, timeout: int = 300
    ) -> CompilationResult:
        """Compile revision."""
        return compile_revision(
            self.project_dir, track_changes=track_changes, timeout=timeout
        )

    def watch(self, on_compile: Optional[Callable] = None) -> None:
        """Auto-recompile on file changes."""
        watch_manuscript(self.project_dir, on_compile=on_compile)

    def get_pdf(self, doc_type: str = "manuscript") -> Optional[Path]:
        """Get output PDF path (Read)."""
        doc_map = {
            "manuscript": "01_manuscript",
            "supplementary": "02_supplementary",
            "revision": "03_revision",
        }
        pdf = self.project_dir / doc_map[doc_type] / f"{doc_type}.pdf"
        return pdf if pdf.exists() else None

    def delete(self) -> bool:
        """Delete project directory (Delete)."""
        import shutil

        try:
            shutil.rmtree(self.project_dir)
            return True
        except Exception:
            return False


__all__ = ["Writer"]

# EOF
