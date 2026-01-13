#!/usr/bin/env python3
"""Tests for scitex.writer._clone_writer_project."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex.writer._clone_writer_project import clone_writer_project


class TestCloneWriterProjectSuccess:
    """Tests for clone_writer_project success cases."""

    def test_calls_clone_writer_directory(self, tmp_path):
        """Verify clone_writer_project calls clone_writer_directory."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ) as mock_clone:
            result = clone_writer_project(project_dir)

            mock_clone.assert_called_once_with(project_dir, "child", None, None)
            assert result is True

    def test_passes_git_strategy(self, tmp_path):
        """Verify git_strategy is passed to clone_writer_directory."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ) as mock_clone:
            clone_writer_project(project_dir, git_strategy="parent")

            mock_clone.assert_called_once_with(project_dir, "parent", None, None)

    def test_passes_branch(self, tmp_path):
        """Verify branch is passed to clone_writer_directory."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ) as mock_clone:
            clone_writer_project(project_dir, branch="develop")

            mock_clone.assert_called_once_with(project_dir, "child", "develop", None)

    def test_passes_tag(self, tmp_path):
        """Verify tag is passed to clone_writer_directory."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ) as mock_clone:
            clone_writer_project(project_dir, tag="v1.0.0")

            mock_clone.assert_called_once_with(project_dir, "child", None, "v1.0.0")

    def test_returns_true_on_success(self, tmp_path):
        """Verify returns True when clone succeeds."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ):
            result = clone_writer_project(project_dir)

            assert result is True


class TestCloneWriterProjectFailure:
    """Tests for clone_writer_project failure cases."""

    def test_returns_false_when_clone_fails(self, tmp_path):
        """Verify returns False when clone_writer_directory returns False."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=False,
        ):
            result = clone_writer_project(project_dir)

            assert result is False

    def test_returns_false_on_exception(self, tmp_path):
        """Verify returns False when exception is raised."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            side_effect=RuntimeError("Clone failed"),
        ):
            result = clone_writer_project(project_dir)

            assert result is False


class TestCloneWriterProjectGitStrategy:
    """Tests for clone_writer_project git_strategy parameter."""

    def test_default_git_strategy_is_child(self, tmp_path):
        """Verify default git_strategy is 'child'."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ) as mock_clone:
            clone_writer_project(project_dir)

            call_args = mock_clone.call_args[0]
            assert call_args[1] == "child"

    def test_git_strategy_none(self, tmp_path):
        """Verify git_strategy=None is passed correctly."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ) as mock_clone:
            clone_writer_project(project_dir, git_strategy=None)

            mock_clone.assert_called_once_with(project_dir, None, None, None)

    def test_git_strategy_origin(self, tmp_path):
        """Verify git_strategy='origin' is passed correctly."""
        project_dir = str(tmp_path / "new_paper")

        with patch(
            "scitex.template.clone_writer_directory",
            return_value=True,
        ) as mock_clone:
            clone_writer_project(project_dir, git_strategy="origin")

            mock_clone.assert_called_once_with(project_dir, "origin", None, None)


class TestCloneWriterProjectExport:
    """Tests for module exports."""

    def test_function_is_exported(self):
        """Verify clone_writer_project is importable."""
        from scitex.writer._clone_writer_project import clone_writer_project

        assert callable(clone_writer_project)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_clone_writer_project.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-11-18 15:41:26 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_clone_writer_project.py
# 
# import os
# 
# """
# Writer project template management.
# 
# Provides functions to create and copy writer project templates.
# """
# 
# from typing import Optional
# 
# from scitex.logging import getLogger
# 
# logger = getLogger(__name__)
# 
# 
# def clone_writer_project(
#     project_dir: str,
#     git_strategy: Optional[str] = "child",
#     branch: Optional[str] = None,
#     tag: Optional[str] = None,
# ) -> bool:
#     """
#     Initialize a new writer project directory from scitex-writer template.
# 
#     Convenience wrapper for scitex.template.clone_writer_directory.
#     Handles template cloning and project setup automatically.
# 
#     Args:
#         project_dir: Path to project directory (will be created)
#         git_strategy: Git initialization strategy (optional)
#             - 'child': Create isolated git in project directory (default)
#             - 'parent': Use parent git repository
#             - 'origin': Preserve template's original git history
#             - None: Disable git initialization
#         branch: Specific branch of the template repository to clone (optional)
#             If None, clones the default branch. Mutually exclusive with tag.
#         tag: Specific tag/release of the template repository to clone (optional)
#             If None, clones the default branch. Mutually exclusive with branch.
# 
#     Returns:
#         True if successful, False otherwise
# 
#     Examples:
#         >>> from scitex.writer import clone_writer_project
#         >>> clone_writer_project("my_paper")
#         >>> clone_writer_project("./papers/my_paper")
#         >>> clone_writer_project("my_paper", git_strategy="parent")
#         >>> clone_writer_project("my_paper", branch="develop")
#         >>> clone_writer_project("my_paper", tag="v1.0.0")
#     """
#     from scitex.template import clone_writer_directory
# 
#     try:
#         result = clone_writer_directory(project_dir, git_strategy, branch, tag)
#         return result
#     except Exception as e:
#         logger.error(f"Failed to initialize writer directory: {e}")
#         return False
# 
# 
# def run_session() -> None:
#     """Initialize scitex framework, run main function, and cleanup."""
#     global CONFIG, CC, sys, plt, rng
#     import sys
#     import matplotlib.pyplot as plt
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
#     result = clone_writer_project(
#         args.project_dir,
#         git_strategy=args.git_strategy,
#         branch=args.branch,
#         tag=args.tag,
#     )
# 
#     if result:
#         print(f"Successfully created writer project: {args.project_dir}")
#         return 0
#     else:
#         print(f"Failed to create writer project: {args.project_dir}")
#         return 1
# 
# 
# def parse_args():
#     import argparse
# 
#     parser = argparse.ArgumentParser(description="Clone scitex writer project template")
#     parser.add_argument(
#         "project_dir",
#         type=str,
#         help="Path to project directory (will be created)",
#     )
#     parser.add_argument(
#         "--git-strategy",
#         "-g",
#         type=str,
#         choices=["child", "parent", "origin", "none"],
#         default="child",
#         help="Git initialization strategy (default: child)",
#     )
#     parser.add_argument(
#         "--branch",
#         "-b",
#         type=str,
#         default=None,
#         help="Specific branch of template to clone (mutually exclusive with --tag)",
#     )
#     parser.add_argument(
#         "--tag",
#         "-t",
#         type=str,
#         default=None,
#         help="Specific tag/release of template to clone (mutually exclusive with --branch)",
#     )
# 
#     return parser.parse_args()
# 
# 
# if __name__ == "__main__":
#     run_session()
# 
# 
# __all__ = [
#     "clone_writer_project",
# ]
# 
# # python -m scitex.writer._clone_writer_project my_paper --git-strategy child
# # python -m scitex.writer._clone_writer_project ./papers/my_paper
# # python -m scitex.writer._clone_writer_project my_paper --branch develop
# # python -m scitex.writer._clone_writer_project my_paper --tag v1.0.0
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_clone_writer_project.py
# --------------------------------------------------------------------------------
