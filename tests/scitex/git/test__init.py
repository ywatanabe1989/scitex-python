#!/usr/bin/env python3
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/git/test_init.py

"""Tests for git init module."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("git")
from git import Repo

from scitex.git._init import (
    create_child_git,
    find_parent_git,
    init_git_repo,
    remove_child_git,
)


class TestFindParentGit:
    def test_find_parent_no_git(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "project"
            project_dir.mkdir()

            result = find_parent_git(project_dir)
            assert result is None

    def test_find_parent_with_git(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent_dir = Path(temp_dir)
            Repo.init(parent_dir)

            child_dir = parent_dir / "child"
            child_dir.mkdir()

            result = find_parent_git(child_dir)
            assert result == parent_dir


class TestRemoveChildGit:
    def test_remove_child_git_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            git_dir = project_dir / ".git"
            git_dir.mkdir()

            remove_child_git(project_dir)
            assert not git_dir.exists()

    def test_remove_child_git_not_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            git_dir = project_dir / ".git"

            remove_child_git(project_dir)
            assert not git_dir.exists()


class TestCreateChildGit:
    def test_create_child_git_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            result = create_child_git(project_dir)
            assert result == project_dir
            assert (project_dir / ".git").exists()

    def test_create_child_git_already_exists(self):
        """When repo already exists, validate_tree_structures is called.

        Note: In temp directory without proper project structure,
        validate_tree_structures fails and returns None.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            first_result = create_child_git(project_dir)
            second_result = create_child_git(project_dir)

            assert first_result == project_dir
            # validate_tree_structures fails without proper project structure
            assert second_result is None


class TestInitGitRepo:
    def test_init_git_repo_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            result = init_git_repo(project_dir, git_strategy=None)
            assert result is None

    def test_init_git_repo_child(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            result = init_git_repo(project_dir, git_strategy="child")
            assert result == project_dir
            assert (project_dir / ".git").exists()

    def test_init_git_repo_parent_no_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            result = init_git_repo(project_dir, git_strategy="parent")
            assert result == project_dir
            assert (project_dir / ".git").exists()

    def test_init_git_repo_invalid_strategy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)

            with pytest.raises(ValueError, match="invalid"):
                init_git_repo(project_dir, git_strategy="invalid")

    def test_find_parent_git_nested_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            Repo.init(root)

            nested = root / "a" / "b" / "c" / "d"
            nested.mkdir(parents=True)

            result = find_parent_git(nested)
            assert result == root

    def test_init_git_repo_parent_with_existing_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent_dir = Path(temp_dir)
            Repo.init(parent_dir)

            child_dir = parent_dir / "child"
            child_dir.mkdir()

            result = init_git_repo(child_dir, git_strategy="parent")
            assert result == parent_dir
            assert not (child_dir / ".git").exists()
            assert (parent_dir / ".git").exists()


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_init.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/git/init.py
#
# """
# Git initialization utilities.
#
# Provides functions to initialize, find, and manage git repositories
# with different strategies (child, parent, origin).
# """
#
# from pathlib import Path
# from typing import Optional
# from git import Repo, InvalidGitRepositoryError
#
# from scitex.logging import getLogger
#
# logger = getLogger(__name__)
#
#
# def find_parent_git(project_dir: Path) -> Optional[Path]:
#     """
#     Find parent git repository by walking up directory tree.
#
#     Args:
#         project_dir: Starting directory to search from
#
#     Returns:
#         Path to parent git root, or None if not found
#     """
#     try:
#         repo_parent = Repo(project_dir.parent, search_parent_directories=True)
#         return Path(repo_parent.git_dir).parent
#     except InvalidGitRepositoryError:
#         return None
#
#
# def remove_child_git(project_dir: Path) -> bool:
#     """
#     Remove project's local .git folder.
#
#     When parent git is found, the project's own .git/ is redundant and
#     should be removed to avoid nested git repository issues.
#
#     Args:
#         project_dir: Project directory containing .git to remove
#
#     Returns:
#         True if removed successfully or doesn't exist, False on error
#     """
#     child_git = project_dir / ".git"
#
#     if not child_git.exists():
#         logger.info(f"No child .git found at {project_dir}")
#         return True
#
#     try:
#         import shutil
#
#         logger.info(f"Removing child .git to use parent repository...")
#         shutil.rmtree(child_git)
#         logger.success(f"Removed child .git from {project_dir}")
#         return True
#     except PermissionError as e:
#         logger.error(f"Permission denied removing .git from {project_dir}: {e}")
#         return False
#     except Exception as e:
#         logger.error(f"Failed to remove child .git from {project_dir}: {e}")
#         return False
#
#
# def create_child_git(project_dir: Path) -> Optional[Path]:
#     """
#     Create isolated git repository in project directory.
#
#     Uses GitPython to initialize and make initial commit.
#
#     Args:
#         project_dir: Directory to initialize as git repo
#
#     Returns:
#         Path to git root (project_dir), or None on failure
#     """
#     try:
#         try:
#             repo = Repo(project_dir)
#             logger.info(f"Project is already a git repository at {project_dir}")
#             # Validate project structure even if repo already exists
#             from scitex.writer._validate_tree_structures import validate_tree_structures
#
#             validate_tree_structures(project_dir)
#             return project_dir
#         except InvalidGitRepositoryError:
#             logger.info(f"Initializing new git repository at {project_dir}")
#             repo = Repo.init(project_dir)
#
#         repo.index.add(["."])
#         repo.index.commit("Initial commit from scitex template")
#
#         logger.success(f"Git repository initialized at {project_dir}")
#         return project_dir
#     except PermissionError as e:
#         logger.error(f"Permission denied creating git repository at {project_dir}: {e}")
#         return None
#     except OSError as e:
#         logger.error(f"IO error creating git repository at {project_dir}: {e}")
#         return None
#     except Exception as e:
#         logger.error(f"Failed to create child git repository at {project_dir}: {e}")
#         return None
#
#
# def init_git_repo(
#     project_dir: Path, git_strategy: Optional[str] = "child"
# ) -> Optional[Path]:
#     """
#     Initialize or detect git repository based on git_strategy.
#
#     Args:
#         project_dir: Project directory
#         git_strategy: Git initialization strategy
#             - None: Git disabled, returns None
#             - 'child': Creates isolated git repo in project directory
#             - 'parent': Tries to use parent git, degrades to 'child' if not found
#             - 'origin': Preserves template's original git history (handled by clone)
#
#     Returns:
#         Path to git repository root, or None if disabled
#     """
#     if git_strategy is None:
#         logger.info("Git initialization disabled (git_strategy=None)")
#         return None
#
#     if git_strategy == "parent":
#         logger.info("Using 'parent' git strategy, searching for parent repository...")
#         parent_git = find_parent_git(project_dir)
#
#         if parent_git:
#             logger.info(f"Found parent git repository: {parent_git}")
#             remove_child_git(project_dir)
#             return parent_git
#
#         logger.warning(
#             f"No parent git repository found for {project_dir}. "
#             f"Degrading to 'child' strategy (isolated git repo)."
#         )
#         return create_child_git(project_dir)
#
#     if git_strategy == "child":
#         logger.info("Using 'child' git strategy, creating isolated repository...")
#         return create_child_git(project_dir)
#
#     if git_strategy == "origin":
#         logger.info("Using 'origin' git strategy, template git history preserved...")
#         try:
#             repo = Repo(project_dir)
#             logger.info(f"Found git repository at {project_dir}")
#             return project_dir
#         except InvalidGitRepositoryError:
#             logger.warning(
#                 f"No git repository found at {project_dir}. "
#                 f"Degrading to 'child' strategy."
#             )
#             return create_child_git(project_dir)
#
#     raise ValueError(
#         f"Unknown git_strategy: {git_strategy}. "
#         f"Expected 'parent', 'child', 'origin', or None"
#     )
#
#
# __all__ = [
#     "find_parent_git",
#     "remove_child_git",
#     "create_child_git",
#     "init_git_repo",
# ]
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_init.py
# --------------------------------------------------------------------------------
