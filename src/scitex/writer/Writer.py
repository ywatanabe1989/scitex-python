#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:13:07 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_Writer.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_Writer.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Writer class for manuscript LaTeX compilation.

Provides object-oriented interface to scitex-writer functionality.
"""


from pathlib import Path
from typing import Optional
from typing import Callable

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
from .dataclasses.tree import SharedTree
from scitex import logging
from scitex.git import init_git_repo

logger = logging.getLogger(__name__)


class Writer:
    """LaTeX manuscript compiler."""

    def __init__(
        self,
        project_dir: Path,
        name: Optional[str] = None,
        git_strategy: Optional[str] = "child",
        branch: Optional[str] = None,
        tag: Optional[str] = None,
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
            branch: Specific branch of template repository to clone (optional)
                If None, clones the default branch. Mutually exclusive with tag.
            tag: Specific tag/release of template repository to clone (optional)
                If None, clones the default branch. Mutually exclusive with branch.
        """
        self.project_name = name or Path(project_dir).name
        self.project_dir = Path(project_dir)
        self.git_strategy = git_strategy
        self.branch = branch
        self.tag = tag

        ref_info = ""
        if branch:
            ref_info = f" (branch: {branch})"
        elif tag:
            ref_info = f" (tag: {tag})"
        logger.info(
            f"Writer: Initializing with:\n"
            f"    Project Name: {self.project_name}\n"
            f"    Project Directory: {self.project_dir.absolute()}\n"
            f"    Git Strategy: {self.git_strategy}{ref_info}..."
        )

        # Create or attach to project
        self.project_dir = self._attach_or_create_project(name)

        # Initialize git repository based on strategy (delegates to template module)
        self.git_root = init_git_repo(self.project_dir, self.git_strategy)

        # Document accessors (pass git_root for efficiency)
        self.shared = SharedTree(self.project_dir / "00_shared", git_root=self.git_root)
        self.manuscript = ManuscriptTree(
            self.project_dir / "01_manuscript", git_root=self.git_root
        )
        self.supplementary = SupplementaryTree(
            self.project_dir / "02_supplementary", git_root=self.git_root
        )
        self.revision = RevisionTree(
            self.project_dir / "03_revision", git_root=self.git_root
        )
        self.scripts = ScriptsTree(self.project_dir / "scripts", git_root=self.git_root)

        logger.success(f"Writer: Initialization complete for {self.project_name}")

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

        logger.info(
            f"Writer: Creating new project '{project_name}' at {self.project_dir.absolute()}"
        )

        # Initialize project directory structure
        success = _clone_writer_project(
            str(self.project_dir), self.git_strategy, self.branch, self.tag
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
            logger.error(f"Writer: Target directory {target_dir} does not exist")
            raise RuntimeError(f"Target directory {target_dir} was not created")

        # Verify project directory was created
        if not self.project_dir.exists():
            logger.error(
                f"Writer: Project directory {self.project_dir} was not created"
            )
            raise RuntimeError(f"Project directory {self.project_dir} was not created")

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

    def compile_manuscript(
        self,
        timeout: int = 300,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> CompilationResult:
        """
        Compile manuscript to PDF with optional live callbacks.

        Runs scripts/shell/compile_manuscript.sh with configured settings.

        Args:
            timeout: Maximum compilation time in seconds (default: 300)
            log_callback: Called with each log line: log_callback("Running pdflatex...")
            progress_callback: Called with progress: progress_callback(50, "Pass 2/3")

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

            >>> # With callbacks for live updates
            >>> def on_log(msg):
            ...     append_to_job_log(job_id, msg)
            >>> def on_progress(percent, step):
            ...     update_job_progress(job_id, percent, step)
            >>> result = writer.compile_manuscript(
            ...     log_callback=on_log,
            ...     progress_callback=on_progress
            ... )
        """
        return compile_manuscript(
            self.project_dir,
            timeout=timeout,
            log_callback=log_callback,
            progress_callback=progress_callback,
        )

    def compile_supplementary(
        self,
        timeout: int = 300,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> CompilationResult:
        """
        Compile supplementary materials to PDF with optional live callbacks.

        Runs scripts/shell/compile_supplementary.sh with configured settings.

        Args:
            timeout: Maximum compilation time in seconds (default: 300)
            log_callback: Called with each log line
            progress_callback: Called with progress updates

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
        return compile_supplementary(
            self.project_dir,
            timeout=timeout,
            log_callback=log_callback,
            progress_callback=progress_callback,
        )

    def compile_revision(
        self,
        track_changes: bool = False,
        timeout: int = 300,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> CompilationResult:
        """
        Compile revision document with optional change tracking and live callbacks.

        Runs scripts/shell/compile_revision.sh with configured settings.

        Args:
            track_changes: Enable change tracking in compiled PDF (default: False)
            timeout: Maximum compilation time in seconds (default: 300)
            log_callback: Called with each log line
            progress_callback: Called with progress updates

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
            self.project_dir,
            track_changes=track_changes,
            timeout=timeout,
            log_callback=log_callback,
            progress_callback=progress_callback,
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


def run_session() -> None:
    """Initialize scitex framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt, rng
    import sys
    import matplotlib.pyplot as plt
    import scitex as stx

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        sdir_suffix=None,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

    stx.session.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


def main(args):
    if args.action == "create":
        writer = Writer.create(
            args.project_name,
            git_strategy=args.git_strategy,
        )
        if writer:
            print(f"Created writer project: {writer.project_dir}")
            return 0
        else:
            print("Failed to create writer project")
            return 1

    elif args.action == "compile":
        writer = Writer(Path(args.dir) if args.dir else Path.cwd())

        if args.document == "manuscript":
            result = writer.compile_manuscript()
        elif args.document == "supplementary":
            result = writer.compile_supplementary()
        elif args.document == "revision":
            result = writer.compile_revision(track_changes=args.track_changes)

        if result.success:
            print(f"Compilation successful: {result.output_pdf}")
            return 0
        else:
            print(f"Compilation failed (exit code {result.exit_code})")
            return result.exit_code

    elif args.action == "info":
        writer = Writer(Path(args.dir) if args.dir else Path.cwd())
        print(f"Project: {writer.project_dir}")
        print(f"\nDocuments:")
        print(f"  - Manuscript: {writer.manuscript}")
        print(f"  - Supplementary: {writer.supplementary}")
        print(f"  - Revision: {writer.revision}")
        return 0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Writer project management and compilation"
    )
    parser.add_argument(
        "--action",
        "-a",
        type=str,
        choices=["create", "compile", "info"],
        default="info",
        help="Action to perform (default: info)",
    )
    parser.add_argument(
        "--project-name",
        "-n",
        type=str,
        help="Project name (for create action)",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        help="Project directory (for compile/info)",
    )
    parser.add_argument(
        "--document",
        "-t",
        type=str,
        choices=["manuscript", "supplementary", "revision"],
        default="manuscript",
        help="Document to compile (default: manuscript)",
    )
    parser.add_argument(
        "--git-strategy",
        "-g",
        type=str,
        choices=["child", "parent", "origin", "none"],
        default="child",
        help="Git strategy (for create action, default: child)",
    )
    parser.add_argument(
        "--track-changes",
        action="store_true",
        help="Enable change tracking (revision only)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_session()


__all__ = ["Writer"]

# python -m scitex.writer.Writer --action create --project-name my_paper
# python -m scitex.writer.Writer --action compile --dir ./my_paper --document manuscript
# python -m scitex.writer.Writer --action info --dir ./my_paper

# EOF
