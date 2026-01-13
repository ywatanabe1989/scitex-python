#!/usr/bin/env python3

"""Tests for git validation utilities."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("git")

from scitex.git._validation import (
    validate_branch_name,
    validate_commit_message,
    validate_path,
)


class TestValidateBranchName:
    """Tests for validate_branch_name function."""

    def test_valid_simple_name(self):
        valid, error = validate_branch_name("main")
        assert valid is True
        assert error == ""

    def test_valid_feature_branch(self):
        valid, error = validate_branch_name("feature/new-feature")
        assert valid is True
        assert error == ""

    def test_valid_with_numbers(self):
        valid, error = validate_branch_name("feature/issue-123")
        assert valid is True
        assert error == ""

    def test_valid_with_dots(self):
        valid, error = validate_branch_name("release/v1.0.0")
        assert valid is True
        assert error == ""

    def test_invalid_empty_name(self):
        valid, error = validate_branch_name("")
        assert valid is False
        assert "empty" in error.lower()

    def test_invalid_whitespace_only(self):
        valid, error = validate_branch_name("   ")
        assert valid is False
        assert "empty" in error.lower()

    def test_invalid_starts_with_dash(self):
        valid, error = validate_branch_name("-feature")
        assert valid is False
        assert "'-'" in error

    def test_invalid_ends_with_lock(self):
        valid, error = validate_branch_name("branch.lock")
        assert valid is False
        assert ".lock" in error

    def test_invalid_contains_double_dots(self):
        valid, error = validate_branch_name("feature..branch")
        assert valid is False
        assert ".." in error

    def test_invalid_contains_tilde(self):
        valid, error = validate_branch_name("feature~1")
        assert valid is False
        assert "~" in error

    def test_invalid_contains_caret(self):
        valid, error = validate_branch_name("branch^2")
        assert valid is False
        assert "^" in error

    def test_invalid_contains_colon(self):
        valid, error = validate_branch_name("branch:name")
        assert valid is False
        assert ":" in error

    def test_invalid_contains_question_mark(self):
        valid, error = validate_branch_name("branch?name")
        assert valid is False
        assert "?" in error

    def test_invalid_contains_asterisk(self):
        valid, error = validate_branch_name("branch*name")
        assert valid is False
        assert "*" in error

    def test_invalid_contains_bracket(self):
        valid, error = validate_branch_name("branch[1]")
        assert valid is False
        assert "[" in error

    def test_invalid_contains_backslash(self):
        valid, error = validate_branch_name("branch\\name")
        assert valid is False
        assert "\\" in error

    def test_invalid_contains_space(self):
        valid, error = validate_branch_name("branch name")
        assert valid is False
        assert " " in error

    def test_invalid_contains_tab(self):
        valid, error = validate_branch_name("branch\tname")
        assert valid is False

    def test_invalid_ends_with_slash(self):
        valid, error = validate_branch_name("feature/")
        assert valid is False
        assert "/" in error

    def test_invalid_starts_with_slash(self):
        valid, error = validate_branch_name("/feature")
        assert valid is False
        assert "/" in error

    def test_invalid_consecutive_slashes(self):
        valid, error = validate_branch_name("feature//branch")
        assert valid is False
        assert "consecutive" in error.lower() or "//" in error


class TestValidateCommitMessage:
    """Tests for validate_commit_message function."""

    def test_valid_simple_message(self):
        valid, error = validate_commit_message("Initial commit")
        assert valid is True
        assert error == ""

    def test_valid_multiline_message(self):
        valid, error = validate_commit_message(
            "Add feature\n\nThis adds a new feature."
        )
        assert valid is True
        assert error == ""

    def test_valid_with_special_chars(self):
        valid, error = validate_commit_message("Fix bug #123: Handle edge case")
        assert valid is True
        assert error == ""

    def test_invalid_empty_message(self):
        valid, error = validate_commit_message("")
        assert valid is False
        assert "empty" in error.lower()

    def test_invalid_whitespace_only(self):
        valid, error = validate_commit_message("   ")
        assert valid is False
        assert "empty" in error.lower()

    def test_invalid_newlines_only(self):
        valid, error = validate_commit_message("\n\n")
        assert valid is False
        assert "empty" in error.lower()


class TestValidatePath:
    """Tests for validate_path function."""

    def test_valid_existing_path(self, tmp_path):
        valid, error = validate_path(tmp_path, must_exist=True)
        assert valid is True
        assert error == ""

    def test_valid_new_path(self, tmp_path):
        new_path = tmp_path / "new_directory"
        valid, error = validate_path(new_path, must_exist=False)
        assert valid is True
        assert error == ""

    def test_invalid_nonexistent_must_exist(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist"
        valid, error = validate_path(nonexistent, must_exist=True)
        assert valid is False
        assert "does not exist" in error.lower()

    def test_valid_file_path(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        valid, error = validate_path(test_file, must_exist=True)
        assert valid is True
        assert error == ""

    def test_valid_nested_path(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        valid, error = validate_path(nested, must_exist=True)
        assert valid is True
        assert error == ""


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_validation.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/git/validation.py
# 
# """
# Input validation utilities for git operations.
# """
# 
# import re
# from pathlib import Path
# from typing import Tuple
# 
# 
# def validate_branch_name(name: str) -> Tuple[bool, str]:
#     """
#     Validate git branch name according to git naming rules.
# 
#     Parameters
#     ----------
#     name : str
#         Branch name to validate
# 
#     Returns
#     -------
#     Tuple[bool, str]
#         (is_valid, error_message)
#     """
#     if not name or not name.strip():
#         return False, "Branch name cannot be empty"
# 
#     name = name.strip()
# 
#     if name.startswith("-"):
#         return False, "Branch name cannot start with '-'"
# 
#     if name.endswith(".lock"):
#         return False, "Branch name cannot end with '.lock'"
# 
#     if ".." in name:
#         return False, "Branch name cannot contain '..'"
# 
#     invalid_chars = ["~", "^", ":", "?", "*", "[", "\\", " ", "\t"]
#     for char in invalid_chars:
#         if char in name:
#             return False, f"Branch name cannot contain '{char}'"
# 
#     if name.endswith("/"):
#         return False, "Branch name cannot end with '/'"
# 
#     if name.startswith("/"):
#         return False, "Branch name cannot start with '/'"
# 
#     if "//" in name:
#         return False, "Branch name cannot contain consecutive slashes"
# 
#     return True, ""
# 
# 
# def validate_commit_message(message: str) -> Tuple[bool, str]:
#     """
#     Validate git commit message.
# 
#     Parameters
#     ----------
#     message : str
#         Commit message to validate
# 
#     Returns
#     -------
#     Tuple[bool, str]
#         (is_valid, error_message)
#     """
#     if not message or not message.strip():
#         return False, "Commit message cannot be empty"
# 
#     return True, ""
# 
# 
# def validate_path(path: Path, must_exist: bool = False) -> Tuple[bool, str]:
#     """
#     Validate path for git operations.
# 
#     Parameters
#     ----------
#     path : Path
#         Path to validate
#     must_exist : bool
#         Whether path must exist
# 
#     Returns
#     -------
#     Tuple[bool, str]
#         (is_valid, error_message)
#     """
#     try:
#         resolved = path.resolve()
# 
#         if must_exist and not resolved.exists():
#             return False, f"Path does not exist: {path}"
# 
#         path_str = str(resolved)
#         if ".." in path.parts:
#             return False, "Path contains parent directory references"
# 
#         return True, ""
# 
#     except (OSError, RuntimeError) as e:
#         return False, f"Invalid path: {e}"
# 
# 
# __all__ = [
#     "validate_branch_name",
#     "validate_commit_message",
#     "validate_path",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_validation.py
# --------------------------------------------------------------------------------
