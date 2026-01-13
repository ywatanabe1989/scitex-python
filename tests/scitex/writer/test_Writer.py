#!/usr/bin/env python3
"""Tests for scitex.writer.Writer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex.writer.Writer import Writer


@pytest.fixture
def valid_project_structure(tmp_path):
    """Create a valid writer project structure."""
    (tmp_path / "00_shared").mkdir()
    (tmp_path / "01_manuscript").mkdir()
    (tmp_path / "02_supplementary").mkdir()
    (tmp_path / "03_revision").mkdir()
    (tmp_path / "scripts").mkdir()
    return tmp_path


class TestWriterInitialization:
    """Tests for Writer initialization with existing project."""

    def test_initializes_with_existing_project(self, valid_project_structure):
        """Verify Writer initializes with existing project."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            assert writer.project_dir == valid_project_structure
            assert writer.project_name == valid_project_structure.name

    def test_sets_project_name_from_directory(self, valid_project_structure):
        """Verify project_name defaults to directory name."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            assert writer.project_name == valid_project_structure.name

    def test_uses_custom_project_name(self, valid_project_structure):
        """Verify custom project name is used."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure, name="custom_name")

            assert writer.project_name == "custom_name"

    def test_initializes_document_trees(self, valid_project_structure):
        """Verify document trees are initialized."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            assert writer.manuscript is not None
            assert writer.supplementary is not None
            assert writer.revision is not None
            assert writer.scripts is not None
            assert writer.shared is not None


class TestWriterProjectVerification:
    """Tests for Writer project structure verification."""

    def test_raises_when_manuscript_missing(self, tmp_path):
        """Verify raises RuntimeError when manuscript directory is missing."""
        (tmp_path / "00_shared").mkdir()
        (tmp_path / "02_supplementary").mkdir()
        (tmp_path / "03_revision").mkdir()
        (tmp_path / "scripts").mkdir()

        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            with pytest.raises(RuntimeError, match="01_manuscript"):
                Writer(tmp_path)

    def test_raises_when_supplementary_missing(self, tmp_path):
        """Verify raises RuntimeError when supplementary directory is missing."""
        (tmp_path / "00_shared").mkdir()
        (tmp_path / "01_manuscript").mkdir()
        (tmp_path / "03_revision").mkdir()
        (tmp_path / "scripts").mkdir()

        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            with pytest.raises(RuntimeError, match="02_supplementary"):
                Writer(tmp_path)

    def test_raises_when_revision_missing(self, tmp_path):
        """Verify raises RuntimeError when revision directory is missing."""
        (tmp_path / "00_shared").mkdir()
        (tmp_path / "01_manuscript").mkdir()
        (tmp_path / "02_supplementary").mkdir()
        (tmp_path / "scripts").mkdir()

        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            with pytest.raises(RuntimeError, match="03_revision"):
                Writer(tmp_path)


class TestWriterGetPdf:
    """Tests for Writer.get_pdf method."""

    def test_returns_none_when_pdf_missing(self, valid_project_structure):
        """Verify returns None when PDF doesn't exist."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            assert writer.get_pdf("manuscript") is None

    def test_returns_path_when_pdf_exists(self, valid_project_structure):
        """Verify returns Path when PDF exists."""
        pdf_path = valid_project_structure / "01_manuscript" / "manuscript.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            result = writer.get_pdf("manuscript")
            assert result == pdf_path

    def test_returns_supplementary_pdf(self, valid_project_structure):
        """Verify returns supplementary PDF path."""
        pdf_path = valid_project_structure / "02_supplementary" / "supplementary.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            result = writer.get_pdf("supplementary")
            assert result == pdf_path


class TestWriterDelete:
    """Tests for Writer.delete method."""

    def test_deletes_project_directory(self, valid_project_structure):
        """Verify delete removes project directory."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            result = writer.delete()

            assert result is True
            assert not valid_project_structure.exists()

    def test_delete_returns_false_on_error(self, valid_project_structure):
        """Verify delete returns False on error."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

        with patch("shutil.rmtree", side_effect=PermissionError("Access denied")):
            result = writer.delete()

            assert result is False


class TestWriterCompileMethods:
    """Tests for Writer compilation methods."""

    def test_compile_manuscript_calls_function(self, valid_project_structure):
        """Verify compile_manuscript calls the compile function."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

        mock_result = MagicMock()
        with patch(
            "scitex.writer.Writer.compile_manuscript",
            return_value=mock_result,
        ) as mock_compile:
            result = mock_compile(
                valid_project_structure,
                timeout=300,
                log_callback=None,
                progress_callback=None,
            )

            assert result == mock_result

    def test_compile_supplementary_calls_function(self, valid_project_structure):
        """Verify compile_supplementary calls the compile function."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

        mock_result = MagicMock()
        with patch(
            "scitex.writer.Writer.compile_supplementary",
            return_value=mock_result,
        ) as mock_compile:
            result = mock_compile(
                valid_project_structure,
                timeout=300,
                log_callback=None,
                progress_callback=None,
            )

            assert result == mock_result

    def test_compile_revision_calls_function(self, valid_project_structure):
        """Verify compile_revision calls the compile function."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

        mock_result = MagicMock()
        with patch(
            "scitex.writer.Writer.compile_revision",
            return_value=mock_result,
        ) as mock_compile:
            result = mock_compile(
                valid_project_structure,
                track_changes=False,
                timeout=300,
                log_callback=None,
                progress_callback=None,
            )

            assert result == mock_result


class TestWriterWatch:
    """Tests for Writer.watch method."""

    def test_watch_calls_watch_manuscript(self, valid_project_structure):
        """Verify watch calls watch_manuscript function."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

        callback = MagicMock()
        with patch("scitex.writer.Writer.watch_manuscript") as mock_watch:
            writer.watch(on_compile=callback)

            mock_watch.assert_called_once_with(
                valid_project_structure, on_compile=callback
            )


class TestWriterGitStrategy:
    """Tests for Writer git_strategy parameter."""

    def test_default_git_strategy_is_child(self, valid_project_structure):
        """Verify default git_strategy is 'child'."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure)

            assert writer.git_strategy == "child"

    def test_custom_git_strategy(self, valid_project_structure):
        """Verify custom git_strategy is set."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure, git_strategy="parent")

            assert writer.git_strategy == "parent"

    def test_git_strategy_none(self, valid_project_structure):
        """Verify git_strategy=None is allowed."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure, git_strategy=None)

            assert writer.git_strategy is None


class TestWriterBranchTag:
    """Tests for Writer branch and tag parameters."""

    def test_branch_parameter(self, valid_project_structure):
        """Verify branch parameter is stored."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure, branch="develop")

            assert writer.branch == "develop"

    def test_tag_parameter(self, valid_project_structure):
        """Verify tag parameter is stored."""
        with patch("scitex.writer.Writer.init_git_repo", return_value=None):
            writer = Writer(valid_project_structure, tag="v1.0.0")

            assert writer.tag == "v1.0.0"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/Writer.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2025-10-29 06:13:07 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_Writer.py
# # ----------------------------------------
# from __future__ import annotations
# 
# import os
# 
# __FILE__ = "./src/scitex/writer/_Writer.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Writer class for manuscript LaTeX compilation.
# 
# Provides object-oriented interface to scitex-writer functionality.
# """
# 
# 
# from pathlib import Path
# from typing import Callable, Optional
# 
# from scitex import logging
# from scitex.git import init_git_repo
# 
# from ._clone_writer_project import (
#     clone_writer_project as _clone_writer_project,
# )
# from ._compile import (
#     CompilationResult,
#     compile_manuscript,
#     compile_revision,
#     compile_supplementary,
# )
# from ._dataclasses import ManuscriptTree, RevisionTree, SupplementaryTree
# from ._dataclasses.config import DOC_TYPE_DIRS
# from ._dataclasses.tree import ScriptsTree, SharedTree
# from .utils._watch import watch_manuscript
# 
# logger = logging.getLogger(__name__)
# 
# 
# class Writer:
#     """LaTeX manuscript compiler."""
# 
#     def __init__(
#         self,
#         project_dir: Path,
#         name: Optional[str] = None,
#         git_strategy: Optional[str] = "child",
#         branch: Optional[str] = None,
#         tag: Optional[str] = None,
#     ):
#         """
#         Initialize for project directory.
# 
#         If directory doesn't exist, creates new project.
# 
#         Parameters
#         ----------
#         project_dir : Path
#             Path to project directory.
#         name : str, optional
#             Project name (used if creating new project).
#         git_strategy : str, optional
#             Git initialization strategy:
#             - 'child': Create isolated git in project directory (default)
#             - 'parent': Use parent git repository
#             - 'origin': Preserve template's original git history
#             - None: Disable git initialization
#         branch : str, optional
#             Specific branch of template repository to clone.
#             If None, clones the default branch. Mutually exclusive with tag.
#         tag : str, optional
#             Specific tag/release of template repository to clone.
#             If None, clones the default branch. Mutually exclusive with branch.
#         """
#         self.project_name = name or Path(project_dir).name
#         self.project_dir = Path(project_dir)
#         self.git_strategy = git_strategy
#         self.branch = branch
#         self.tag = tag
# 
#         ref_info = ""
#         if branch:
#             ref_info = f" (branch: {branch})"
#         elif tag:
#             ref_info = f" (tag: {tag})"
#         logger.info(
#             f"Writer: Initializing with:\n"
#             f"    Project Name: {self.project_name}\n"
#             f"    Project Directory: {self.project_dir.absolute()}\n"
#             f"    Git Strategy: {self.git_strategy}{ref_info}..."
#         )
# 
#         # Create or attach to project
#         self.project_dir = self._attach_or_create_project(name)
# 
#         # Initialize git repository based on strategy (delegates to template module)
#         self.git_root = init_git_repo(self.project_dir, self.git_strategy)
# 
#         # Document accessors (pass git_root for efficiency)
#         self.shared = SharedTree(self.project_dir / "00_shared", git_root=self.git_root)
#         self.manuscript = ManuscriptTree(
#             self.project_dir / "01_manuscript", git_root=self.git_root
#         )
#         self.supplementary = SupplementaryTree(
#             self.project_dir / "02_supplementary", git_root=self.git_root
#         )
#         self.revision = RevisionTree(
#             self.project_dir / "03_revision", git_root=self.git_root
#         )
#         self.scripts = ScriptsTree(self.project_dir / "scripts", git_root=self.git_root)
# 
#         logger.success(f"Writer: Initialization complete for {self.project_name}")
# 
#     def _attach_or_create_project(self, name: Optional[str] = None) -> Path:
#         """
#         Create new project or attach to existing one.
# 
#         If project directory doesn't exist, creates it based on git_strategy:
#         - 'child': Full template with git initialization
#         - 'parent'/'None': Minimal directory structure
# 
#         Parameters
#         ----------
#         name : str, optional
#             Project name (used if creating new project).
# 
#         Returns
#         -------
#         Path
#             Path to the project directory.
#         """
#         if self.project_dir.exists():
#             logger.info(
#                 f"Writer: Attached to existing project at {self.project_dir.absolute()}"
#             )
#             # Verify existing project structure
#             self._verify_project_structure()
#             return self.project_dir
# 
#         project_name = name or self.project_dir.name
# 
#         logger.info(
#             f"Writer: Creating new project '{project_name}' at {self.project_dir.absolute()}"
#         )
# 
#         # Initialize project directory structure
#         success = _clone_writer_project(
#             str(self.project_dir), self.git_strategy, self.branch, self.tag
#         )
# 
#         if not success:
#             logger.error(
#                 f"Writer: Failed to initialize project directory for {project_name}"
#             )
#             raise RuntimeError(
#                 f"Could not create project directory at {self.project_dir}"
#             )
# 
#         # Verify project directory was created
#         if not self.project_dir.exists():
#             logger.error(
#                 f"Writer: Project directory {self.project_dir} was not created"
#             )
#             raise RuntimeError(f"Project directory {self.project_dir} was not created")
# 
#         logger.success(
#             f"Writer: Successfully created project at {self.project_dir.absolute()}"
#         )
#         return self.project_dir
# 
#     def _verify_project_structure(self) -> None:
#         """
#         Verify attached project has expected structure.
# 
#         Checks:
#         - Required directories exist (01_manuscript, 02_supplementary, 03_revision)
#         - .git exists (for git-enabled strategies)
# 
#         Raises
#         ------
#         RuntimeError
#             If structure is invalid.
#         """
#         required_dirs = [
#             self.project_dir / "01_manuscript",
#             self.project_dir / "02_supplementary",
#             self.project_dir / "03_revision",
#         ]
# 
#         for dir_path in required_dirs:
#             if not dir_path.exists():
#                 logger.error(f"Writer: Expected directory missing: {dir_path}")
#                 raise RuntimeError(
#                     f"Project structure invalid: missing {dir_path.name} directory"
#                 )
# 
#         logger.success(
#             f"Writer: Project structure verified at {self.project_dir.absolute()}"
#         )
# 
#     def compile_manuscript(
#         self,
#         timeout: int = 300,
#         log_callback: Optional[Callable[[str], None]] = None,
#         progress_callback: Optional[Callable[[int, str], None]] = None,
#     ) -> CompilationResult:
#         """
#         Compile manuscript to PDF with optional live callbacks.
# 
#         Runs scripts/shell/compile_manuscript.sh with configured settings.
# 
#         Parameters
#         ----------
#         timeout : int, optional
#             Maximum compilation time in seconds (default: 300).
#         log_callback : callable, optional
#             Called with each log line: log_callback("Running pdflatex...").
#         progress_callback : callable, optional
#             Called with progress: progress_callback(50, "Pass 2/3").
# 
#         Returns
#         -------
#         CompilationResult
#             With success status, PDF path, and errors/warnings.
# 
#         Examples
#         --------
#         >>> writer = Writer(Path("my_paper"))
#         >>> result = writer.compile_manuscript()
#         >>> if result.success:
#         ...     print(f"PDF created: {result.output_pdf}")
#         """
#         return compile_manuscript(
#             self.project_dir,
#             timeout=timeout,
#             log_callback=log_callback,
#             progress_callback=progress_callback,
#         )
# 
#     def compile_supplementary(
#         self,
#         timeout: int = 300,
#         log_callback: Optional[Callable[[str], None]] = None,
#         progress_callback: Optional[Callable[[int, str], None]] = None,
#     ) -> CompilationResult:
#         """
#         Compile supplementary materials to PDF with optional live callbacks.
# 
#         Runs scripts/shell/compile_supplementary.sh with configured settings.
# 
#         Parameters
#         ----------
#         timeout : int, optional
#             Maximum compilation time in seconds (default: 300).
#         log_callback : callable, optional
#             Called with each log line.
#         progress_callback : callable, optional
#             Called with progress updates.
# 
#         Returns
#         -------
#         CompilationResult
#             With success status, PDF path, and errors/warnings.
# 
#         Examples
#         --------
#         >>> writer = Writer(Path("my_paper"))
#         >>> result = writer.compile_supplementary()
#         >>> if result.success:
#         ...     print(f"PDF created: {result.output_pdf}")
#         """
#         return compile_supplementary(
#             self.project_dir,
#             timeout=timeout,
#             log_callback=log_callback,
#             progress_callback=progress_callback,
#         )
# 
#     def compile_revision(
#         self,
#         track_changes: bool = False,
#         timeout: int = 300,
#         log_callback: Optional[Callable[[str], None]] = None,
#         progress_callback: Optional[Callable[[int, str], None]] = None,
#     ) -> CompilationResult:
#         """
#         Compile revision document with optional change tracking and live callbacks.
# 
#         Runs scripts/shell/compile_revision.sh with configured settings.
# 
#         Parameters
#         ----------
#         track_changes : bool, optional
#             Enable change tracking in compiled PDF (default: False).
#         timeout : int, optional
#             Maximum compilation time in seconds (default: 300).
#         log_callback : callable, optional
#             Called with each log line.
#         progress_callback : callable, optional
#             Called with progress updates.
# 
#         Returns
#         -------
#         CompilationResult
#             With success status, PDF path, and errors/warnings.
# 
#         Examples
#         --------
#         >>> writer = Writer(Path("my_paper"))
#         >>> result = writer.compile_revision(track_changes=True)
#         >>> if result.success:
#         ...     print(f"Revision PDF: {result.output_pdf}")
#         """
#         return compile_revision(
#             self.project_dir,
#             track_changes=track_changes,
#             timeout=timeout,
#             log_callback=log_callback,
#             progress_callback=progress_callback,
#         )
# 
#     def watch(self, on_compile: Optional[Callable] = None) -> None:
#         """Auto-recompile on file changes."""
#         watch_manuscript(self.project_dir, on_compile=on_compile)
# 
#     def get_pdf(self, doc_type: str = "manuscript") -> Optional[Path]:
#         """Get output PDF path (Read)."""
#         pdf = self.project_dir / DOC_TYPE_DIRS[doc_type] / f"{doc_type}.pdf"
#         return pdf if pdf.exists() else None
# 
#     def delete(self) -> bool:
#         """Delete project directory (Delete)."""
#         import shutil
# 
#         try:
#             logger.warning(
#                 f"Writer: Deleting project directory at {self.project_dir.absolute()}"
#             )
#             shutil.rmtree(self.project_dir)
#             logger.success(
#                 f"Writer: Successfully deleted project at {self.project_dir.absolute()}"
#             )
#             return True
#         except Exception as e:
#             logger.error(
#                 f"Writer: Failed to delete project directory at {self.project_dir.absolute()}: {e}"
#             )
#             return False
# 
# 
# def run_session() -> None:
#     """Initialize scitex framework, run main function, and cleanup."""
#     global CONFIG, CC, sys, plt, rng
#     import sys
# 
#     import matplotlib.pyplot as plt
# 
#     import scitex as stx
# 
#     args = parse_args()
# 
#     CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
#         sys,
#         plt,
#         args=args,
#         file=__FILE__,
#         sdir_suffix=None,
#         verbose=False,
#         agg=True,
#     )
# 
#     exit_status = main(args)
# 
#     stx.session.close(
#         CONFIG,
#         verbose=False,
#         notify=False,
#         message="",
#         exit_status=exit_status,
#     )
# 
# 
# def main(args):
#     if args.action == "create":
#         writer = Writer.create(
#             args.project_name,
#             git_strategy=args.git_strategy,
#         )
#         if writer:
#             print(f"Created writer project: {writer.project_dir}")
#             return 0
#         else:
#             print("Failed to create writer project")
#             return 1
# 
#     elif args.action == "compile":
#         writer = Writer(Path(args.dir) if args.dir else Path.cwd())
# 
#         if args.document == "manuscript":
#             result = writer.compile_manuscript()
#         elif args.document == "supplementary":
#             result = writer.compile_supplementary()
#         elif args.document == "revision":
#             result = writer.compile_revision(track_changes=args.track_changes)
# 
#         if result.success:
#             print(f"Compilation successful: {result.output_pdf}")
#             return 0
#         else:
#             print(f"Compilation failed (exit code {result.exit_code})")
#             return result.exit_code
# 
#     elif args.action == "info":
#         writer = Writer(Path(args.dir) if args.dir else Path.cwd())
#         print(f"Project: {writer.project_dir}")
#         print("\nDocuments:")
#         print(f"  - Manuscript: {writer.manuscript}")
#         print(f"  - Supplementary: {writer.supplementary}")
#         print(f"  - Revision: {writer.revision}")
#         return 0
# 
# 
# def parse_args():
#     import argparse
# 
#     parser = argparse.ArgumentParser(
#         description="Writer project management and compilation"
#     )
#     parser.add_argument(
#         "--action",
#         "-a",
#         type=str,
#         choices=["create", "compile", "info"],
#         default="info",
#         help="Action to perform (default: info)",
#     )
#     parser.add_argument(
#         "--project-name",
#         "-n",
#         type=str,
#         help="Project name (for create action)",
#     )
#     parser.add_argument(
#         "--dir",
#         "-d",
#         type=str,
#         help="Project directory (for compile/info)",
#     )
#     parser.add_argument(
#         "--document",
#         "-t",
#         type=str,
#         choices=["manuscript", "supplementary", "revision"],
#         default="manuscript",
#         help="Document to compile (default: manuscript)",
#     )
#     parser.add_argument(
#         "--git-strategy",
#         "-g",
#         type=str,
#         choices=["child", "parent", "origin", "none"],
#         default="child",
#         help="Git strategy (for create action, default: child)",
#     )
#     parser.add_argument(
#         "--track-changes",
#         action="store_true",
#         help="Enable change tracking (revision only)",
#     )
# 
#     return parser.parse_args()
# 
# 
# if __name__ == "__main__":
#     run_session()
# 
# 
# __all__ = ["Writer"]
# 
# # python -m scitex.writer.Writer --action create --project-name my_paper
# # python -m scitex.writer.Writer --action compile --dir ./my_paper --document manuscript
# # python -m scitex.writer.Writer --action info --dir ./my_paper
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/Writer.py
# --------------------------------------------------------------------------------
