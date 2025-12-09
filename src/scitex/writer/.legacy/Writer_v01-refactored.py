#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/Writer.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/Writer.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Writer class for manuscript LaTeX compilation.

Provides object-oriented interface to scitex-writer functionality.
"""

from pathlib import Path
from typing import Optional, Callable

from scitex.logging import getLogger

# Delegate to specialized modules
from ._project import ensure_project_exists, validate_structure, create_document_trees
from ._git import initialize_git
from ._compile import run_compile
from .dataclasses import CompilationResult
from .dataclasses.config import DOC_TYPE_DIRS
from .utils._watch import watch_manuscript

logger = getLogger(__name__)


class Writer:
    """
    LaTeX manuscript compiler.

    Simplified initialization that delegates to specialized modules.
    """

    def __init__(
        self,
        project_dir: Path,
        name: Optional[str] = None,
        git_strategy: Optional[str] = "child",
    ):
        """
        Initialize for project directory.

        If directory doesn't exist, creates new project.

        Parameters
        ----------
        project_dir : Path
            Path to project directory
        name : str, optional
            Project name (used if creating new project)
        git_strategy : str or None
            Git initialization strategy:
            - 'child': Create isolated git in project directory (default)
            - 'parent': Use parent git repository
            - 'origin': Preserve template's original git history
            - None: Disable git initialization
        """
        self.project_name = name or Path(project_dir).name
        self.project_dir = Path(project_dir)
        self.git_strategy = git_strategy

        logger.info(
            f"Writer: Initializing with:\n"
            f"    Project Name: {self.project_name}\n"
            f"    Project Directory: {self.project_dir.absolute()}\n"
            f"    Git Strategy: {self.git_strategy}..."
        )

        # Delegate to specialized modules
        self.project_dir = ensure_project_exists(
            self.project_dir, self.project_name, self.git_strategy
        )
        validate_structure(self.project_dir)
        self.git_root = initialize_git(self.project_dir, self.git_strategy)

        # Create document trees
        self.manuscript, self.supplementary, self.revision, self.scripts = (
            create_document_trees(self.project_dir, self.git_root)
        )

        logger.success(f"Writer: Initialization complete for {self.project_name}")

    def compile_manuscript(self, timeout: int = 300) -> CompilationResult:
        """
        Compile manuscript to PDF.

        Runs scripts/shell/compile_manuscript.sh with configured settings.

        Parameters
        ----------
        timeout : int
            Maximum compilation time in seconds (default: 300)

        Returns
        -------
        CompilationResult
            Compilation status and outputs

        Examples
        --------
        >>> writer = Writer(Path("my_paper"))
        >>> result = writer.compile_manuscript()
        >>> if result.success:
        ...     print(f"PDF created: {result.output_pdf}")
        """
        return run_compile("manuscript", self.project_dir, timeout=timeout)

    def compile_supplementary(self, timeout: int = 300) -> CompilationResult:
        """
        Compile supplementary materials to PDF.

        Runs scripts/shell/compile_supplementary.sh with configured settings.

        Parameters
        ----------
        timeout : int
            Maximum compilation time in seconds (default: 300)

        Returns
        -------
        CompilationResult
            Compilation status and outputs
        """
        return run_compile("supplementary", self.project_dir, timeout=timeout)

    def compile_revision(
        self, track_changes: bool = False, timeout: int = 300
    ) -> CompilationResult:
        """
        Compile revision document with optional change tracking.

        Runs scripts/shell/compile_revision.sh with configured settings.

        Parameters
        ----------
        track_changes : bool
            Enable change tracking in compiled PDF (default: False)
        timeout : int
            Maximum compilation time in seconds (default: 300)

        Returns
        -------
        CompilationResult
            Compilation status and outputs
        """
        return run_compile(
            "revision", self.project_dir, timeout=timeout, track_changes=track_changes
        )

    def watch(self, on_compile: Optional[Callable] = None) -> None:
        """Auto-recompile on file changes."""
        watch_manuscript(self.project_dir, on_compile=on_compile)

    def get_pdf(self, doc_type: str = "manuscript") -> Optional[Path]:
        """Get output PDF path."""
        pdf = self.project_dir / DOC_TYPE_DIRS[doc_type] / f"{doc_type}.pdf"
        return pdf if pdf.exists() else None

    def delete(self) -> bool:
        """Delete project directory."""
        import shutil

        try:
            logger.warning(
                f"Writer: Deleting project directory at {self.project_dir.absolute()}"
            )
            shutil.rmtree(self.project_dir)
            logger.success(
                f"Writer: Successfully deleted project at {self.project_dir.absolute()}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Writer: Failed to delete project directory at {self.project_dir.absolute()}: {e}"
            )
            return False


__all__ = ["Writer"]

# EOF
