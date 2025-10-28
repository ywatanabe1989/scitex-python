#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:13:07 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_Writer.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/writer/_Writer.py"
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
from git import Repo
from git import InvalidGitRepositoryError

from ._compile import compile_manuscript
from ._compile import compile_supplementary
from ._compile import compile_revision
from ._compile import CompilationResult
from .utils._watch import watch_manuscript
from ._clone_writer_project import (
    clone_writer_project as _clone_writer_project,
)
from .dataclasses.config import DOC_TYPE_DIRS
from .dataclasses import ManuscriptTree
from .dataclasses import SupplementaryTree
from .dataclasses import RevisionTree
from .dataclasses.tree import ScriptsTree
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

        # Create or attach to project
        self.project_dir = self._attach_or_create_project(name)

        # Initialize git repository based on strategy
        self.git_root = self._init_git_repo()

        # Document accessors (pass git_root for efficiency)
        self.manuscript = ManuscriptTree(
            self.project_dir / "01_manuscript", git_root=self.git_root
        )
        self.supplementary = SupplementaryTree(
            self.project_dir / "02_supplementary", git_root=self.git_root
        )
        self.revision = RevisionTree(
            self.project_dir / "03_revision", git_root=self.git_root
        )
        self.scripts = ScriptsTree(
            self.project_dir / "scripts", git_root=self.git_root
        )

        logger.success(
            f"Writer: Initialization complete for {self.project_name}"
        )

    def _attach_or_create_project(self, name: Optional[str] = None) -> Path:
        """
        Create new project or attach to existing one.

        If project directory doesn't exist, creates it based on git_strategy:
        - 'child': Full template with git initialization
        - 'parent'/'None': Minimal directory structure

        Args:
            name: Project name (used if creating new project)

        Returns:
            Path to the project directory
        """
        if self.project_dir.exists():
            logger.info(
                f"Writer: Attached to existing project at {self.project_dir.absolute()}"
            )
            # Verify existing project structure
            self._verify_project_structure()
            return self.project_dir

        project_name = name or self.project_dir.name
        target_dir = self.project_dir.parent

        logger.info(
            f"Writer: Creating new project '{project_name}' in {target_dir}"
        )

        # Initialize project directory structure
        success = _clone_writer_project(
            project_name, str(target_dir), self.git_strategy
        )

        if not success:
            logger.error(
                f"Writer: Failed to initialize project directory for {project_name}"
            )
            raise RuntimeError(
                f"Could not create project directory at {self.project_dir}"
            )

        # Verify target directory exists
        if not target_dir.exists():
            logger.error(
                f"Writer: Target directory {target_dir} does not exist"
            )
            raise RuntimeError(
                f"Target directory {target_dir} was not created"
            )

        # Verify project directory was created
        if not self.project_dir.exists():
            logger.error(
                f"Writer: Project directory {self.project_dir} was not created"
            )
            raise RuntimeError(
                f"Project directory {self.project_dir} was not created"
            )

        logger.success(
            f"Writer: Successfully created project at {self.project_dir.absolute()}"
        )
        return self.project_dir

    def _verify_project_structure(self) -> None:
        """
        Verify attached project has expected structure.

        Checks:
        - Required directories exist (01_manuscript, 02_supplementary, 03_revision)
        - .git exists (for git-enabled strategies)

        Raises:
            RuntimeError: If structure is invalid
        """
        required_dirs = [
            self.project_dir / "01_manuscript",
            self.project_dir / "02_supplementary",
            self.project_dir / "03_revision",
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.error(f"Writer: Expected directory missing: {dir_path}")
                raise RuntimeError(
                    f"Project structure invalid: missing {dir_path.name} directory"
                )

        logger.success(
            f"Writer: Project structure verified at {self.project_dir.absolute()}"
        )

    def _remove_child_git(self) -> None:
        """
        Remove project's local .git folder when using parent git strategy.

        When parent git is found, the project's own .git/ is redundant and
        should be removed to avoid nested git repository issues.

        Logs warning/success/error as appropriate.
        """
        child_git = self.project_dir / ".git"

        if not child_git.exists():
            logger.info(f"Writer: No child .git found at {self.project_dir}")
            return

        try:
            import shutil

            logger.info(
                f"Writer: Removing project's child .git to use parent repository..."
            )
            shutil.rmtree(child_git)
            logger.success(
                f"Writer: Removed child .git from {self.project_dir}"
            )
        except Exception as e:
            logger.warning(
                f"Writer: Failed to remove child .git from {self.project_dir}: {e}"
            )

    def _find_parent_git(self) -> Optional[Path]:
        """
        Find parent git repository by walking up directory tree.

        Uses GitPython to search parent directories for git repo.

        Returns:
            Path to parent git root, or None if not found
        """
        try:
            # search_parent_directories=True searches up the tree
            repo_parent = Repo(
                self.project_dir.parent, search_parent_directories=True
            )
            return Path(repo_parent.git_dir).parent
        except InvalidGitRepositoryError:
            return None

    def _create_child_git(self) -> Optional[Path]:
        """
        Create isolated git repository in project directory.

        Uses GitPython to initialize and make initial commit.

        Returns:
            Path to git root (project_dir), or None on failure
        """
        try:
            # Check if already a git repo
            try:
                repo = Repo(self.project_dir)
                logger.info(
                    f"Writer: Project is already a git repository at {self.project_dir}"
                )
                return self.project_dir
            except InvalidGitRepositoryError:
                # Not yet a repo, initialize one
                logger.info(
                    f"Writer: Initializing new git repository at {self.project_dir}"
                )
                repo = Repo.init(self.project_dir)

            # Stage and commit
            repo.index.add(["."])
            repo.index.commit("Initial commit from scitex-writer")

            logger.success(
                f"Writer: Git repository initialized at {self.project_dir}"
            )
            return self.project_dir
        except Exception as e:
            logger.warning(
                f"Writer: Failed to create child git repository: {e}"
            )
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
        - 'origin': Preserves template's original git history (handled by clone)
        """
        # Strategy: disabled
        if self.git_strategy is None:
            logger.info(
                "Writer: Git initialization disabled (git_strategy=None)"
            )
            return None

        # Strategy: parent (with graceful degradation to child)
        if self.git_strategy == "parent":
            logger.info(
                f"Writer: Using 'parent' git strategy, searching for parent repository..."
            )
            parent_git = self._find_parent_git()

            if parent_git:
                logger.info(
                    f"Writer: Found parent git repository: {parent_git}"
                )
                # Remove cloned project's .git if it exists (will use parent git instead)
                self._remove_child_git()
                return parent_git

            # Graceful degradation: no parent git found, use child strategy
            logger.warning(
                f"Writer: No parent git repository found for {self.project_dir}. "
                f"Degrading to 'child' strategy (isolated git repo)."
            )
            return self._create_child_git()

        # Strategy: child (create isolated repo)
        if self.git_strategy == "child":
            logger.info(
                f"Writer: Using 'child' git strategy, creating isolated repository..."
            )
            return self._create_child_git()

        # Strategy: origin (preserve template git history)
        if self.git_strategy == "origin":
            logger.info(
                f"Writer: Using 'origin' git strategy, template git history preserved..."
            )
            try:
                repo = Repo(self.project_dir)
                logger.info(
                    f"Writer: Found git repository at {self.project_dir}"
                )
                return self.project_dir
            except InvalidGitRepositoryError:
                logger.warning(
                    f"Writer: No git repository found at {self.project_dir}. "
                    f"Degrading to 'child' strategy."
                )
                return self._create_child_git()

        # Unknown strategy
        raise ValueError(
            f"Writer: Unknown git_strategy: {self.git_strategy}. "
            f"Expected 'parent', 'child', 'origin', or None"
        )

    def compile_manuscript(self, timeout: int = 300) -> CompilationResult:
        """
        Compile manuscript to PDF.

        Runs scripts/shell/compile_manuscript.sh with configured settings.

        Args:
            timeout: Maximum compilation time in seconds (default: 300)

        Returns:
            CompilationResult with success status, PDF path, and errors/warnings

        Shell script options (see scripts/shell/compile_manuscript.sh):
            -nf,  --no_figs       Exclude figures for quick compilation
            -p2t, --ppt2tif       Convert PowerPoint to TIF on WSL
            -c,   --crop_tif      Crop TIF images to remove excess whitespace
            -q,   --quiet         Do not show detailed logs for LaTeX compilation
            -v,   --verbose       Show verbose LaTeX compilation output
            -f,   --force         Force recompilation (ignore cache)

        Example:
            >>> writer = Writer(Path("my_paper"))
            >>> result = writer.compile_manuscript()
            >>> if result.success:
            ...     print(f"PDF created: {result.output_pdf}")
        """
        return compile_manuscript(self.project_dir, timeout=timeout)

    def compile_supplementary(self, timeout: int = 300) -> CompilationResult:
        """
        Compile supplementary materials to PDF.

        Runs scripts/shell/compile_supplementary.sh with configured settings.

        Args:
            timeout: Maximum compilation time in seconds (default: 300)

        Returns:
            CompilationResult with success status, PDF path, and errors/warnings

        Shell script options (see scripts/shell/compile_supplementary.sh):
            -nf,  --no_figs       Exclude figures for quick compilation
            -q,   --quiet         Do not show detailed logs for LaTeX compilation
            -v,   --verbose       Show verbose LaTeX compilation output
            -f,   --force         Force recompilation (ignore cache)

        Example:
            >>> writer = Writer(Path("my_paper"))
            >>> result = writer.compile_supplementary()
            >>> if result.success:
            ...     print(f"PDF created: {result.output_pdf}")
        """
        return compile_supplementary(self.project_dir, timeout=timeout)

    def compile_revision(
        self, track_changes: bool = False, timeout: int = 300
    ) -> CompilationResult:
        """
        Compile revision document with optional change tracking.

        Runs scripts/shell/compile_revision.sh with configured settings.

        Args:
            track_changes: Enable change tracking in compiled PDF (default: False)
            timeout: Maximum compilation time in seconds (default: 300)

        Returns:
            CompilationResult with success status, PDF path, and errors/warnings

        Shell script options (see scripts/shell/compile_revision.sh):
            -tc, --track_changes  Show tracked changes in output PDF
            -nf, --no_figs        Exclude figures for quick compilation
            -q,  --quiet          Do not show detailed logs for LaTeX compilation
            -v,  --verbose        Show verbose LaTeX compilation output
            -f,  --force          Force recompilation (ignore cache)

        Example:
            >>> writer = Writer(Path("my_paper"))
            >>> result = writer.compile_revision(track_changes=True)
            >>> if result.success:
            ...     print(f"Revision PDF: {result.output_pdf}")
        """
        return compile_revision(
            self.project_dir, track_changes=track_changes, timeout=timeout
        )

    def watch(self, on_compile: Optional[Callable] = None) -> None:
        """Auto-recompile on file changes."""
        watch_manuscript(self.project_dir, on_compile=on_compile)

    def get_pdf(self, doc_type: str = "manuscript") -> Optional[Path]:
        """Get output PDF path (Read)."""
        pdf = self.project_dir / DOC_TYPE_DIRS[doc_type] / f"{doc_type}.pdf"
        return pdf if pdf.exists() else None

    def delete(self) -> bool:
        """Delete project directory (Delete)."""
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
