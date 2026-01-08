#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for DocumentSection class.

Tests cover:
- File read/write operations
- Git operations (commit, history, diff, checkout)
- diff_between() for comparing two references
- Human-readable reference resolution
- Relative/absolute time specifications
- Error handling and edge cases
- Integration with git repository
"""

import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
pytest.importorskip("git")

from scitex.writer.dataclasses.core._DocumentSection import DocumentSection


class TestDocumentSectionReadWrite:
    """Test file read/write operations."""

    @pytest.fixture
    def temp_file(self):
        """Create temporary file."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_doc_")
        temp_file = Path(temp_dir) / "test.tex"
        temp_file.write_text("Initial content\n")

        yield temp_file

        # Cleanup
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    def test_read_existing_file(self, temp_file):
        """Test reading existing file."""
        section = DocumentSection(temp_file)
        content = section.read()

        assert content is not None
        assert "Initial content" in str(content)

    def test_read_nonexistent_file(self):
        """Test reading non-existent file returns None."""
        section = DocumentSection(Path("/nonexistent/file.tex"))
        content = section.read()

        assert content is None

    def test_write_file(self, temp_file):
        """Test writing to file."""
        section = DocumentSection(temp_file)
        result = section.write("New content\n")

        assert result is True
        assert temp_file.read_text() == "New content\n"

    def test_write_with_list(self, temp_file):
        """Test writing list of lines."""
        section = DocumentSection(temp_file)
        result = section.write(["Line 1", "Line 2", "Line 3"])

        assert result is True
        content = temp_file.read_text()
        assert "Line 1\nLine 2\nLine 3" in content

    def test_write_nonexistent_directory(self):
        """Test writing to non-existent directory fails gracefully."""
        section = DocumentSection(Path("/nonexistent/dir/file.tex"))
        result = section.write("content")

        assert result is False

    def test_read_write_roundtrip(self, temp_file):
        """Test reading and writing back same content."""
        section = DocumentSection(temp_file)
        original = section.read()

        section.write(original)
        readback = section.read()

        assert original == readback


class TestDocumentSectionGitOperations:
    """Test git-based operations."""

    @pytest.fixture
    def git_repo_with_file(self):
        """Create git repository with test file."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_git_")
        repo_path = Path(temp_dir)

        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Configure git
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Create test file
        test_file = repo_path / "test.tex"
        test_file.write_text("Initial content\n")

        # Initial commit
        subprocess.run(
            ["git", "add", "test.tex"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        yield test_file, repo_path

        # Cleanup
        if repo_path.exists():
            shutil.rmtree(repo_path)

    def test_commit_file(self, git_repo_with_file):
        """Test committing file to git."""
        test_file, repo_path = git_repo_with_file

        section = DocumentSection(test_file, git_root=repo_path)
        section.write("Updated content\n")
        result = section.commit("Update test file")

        assert result is True

        # Verify commit was made
        log_result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Update test file" in log_result.stdout

    def test_history_shows_commits(self, git_repo_with_file):
        """Test getting file history."""
        test_file, repo_path = git_repo_with_file

        section = DocumentSection(test_file, git_root=repo_path)
        section.write("Update 1\n")
        section.commit("Commit 1")
        section.write("Update 2\n")
        section.commit("Commit 2")

        history = section.history()

        assert len(history) >= 2
        assert any("Commit 1" in h for h in history)
        assert any("Commit 2" in h for h in history)

    def test_diff_shows_changes(self, git_repo_with_file):
        """Test getting diff of changes."""
        test_file, repo_path = git_repo_with_file

        section = DocumentSection(test_file, git_root=repo_path)
        section.write("Changed content\n")

        diff = section.diff()

        assert len(diff) > 0
        assert "Changed content" in diff or "-" in diff

    def test_checkout_restores_file(self, git_repo_with_file):
        """Test checking out file from git."""
        test_file, repo_path = git_repo_with_file

        section = DocumentSection(test_file, git_root=repo_path)

        # Modify file
        section.write("Modified content\n")
        modified = section.read()

        # Restore from HEAD
        result = section.checkout("HEAD")

        assert result is True
        restored = section.read()
        assert restored != modified
        assert "Initial content" in str(restored)

    def test_history_empty_without_git(self):
        """Test history returns empty list without git repo."""
        temp_file = Path(tempfile.mktemp(suffix=".tex"))
        temp_file.write_text("content\n")

        section = DocumentSection(temp_file)
        history = section.history()

        assert history == []

        # Cleanup
        temp_file.unlink()

    def test_diff_empty_without_git(self):
        """Test diff returns empty string without git repo."""
        temp_file = Path(tempfile.mktemp(suffix=".tex"))
        temp_file.write_text("content\n")

        section = DocumentSection(temp_file)
        diff = section.diff()

        assert diff == ""

        # Cleanup
        temp_file.unlink()

    def test_commit_fails_without_git(self):
        """Test commit returns False without git repo."""
        temp_file = Path(tempfile.mktemp(suffix=".tex"))
        temp_file.write_text("content\n")

        section = DocumentSection(temp_file)
        result = section.commit("test message")

        assert result is False

        # Cleanup
        temp_file.unlink()


class TestDiffBetween:
    """Test diff_between() for comparing two references."""

    @pytest.fixture
    def git_repo_with_history(self):
        """Create git repository with multiple commits."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_diff_between_")
        repo_path = Path(temp_dir)

        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Configure git
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Create test file with initial content
        test_file = repo_path / "test.tex"
        test_file.write_text("Initial content\n")

        # Initial commit
        subprocess.run(
            ["git", "add", "test.tex"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Commit 2: Add first section
        test_file.write_text("Initial content\n\nSection 1\n")
        subprocess.run(
            ["git", "add", "test.tex"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Add section 1"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Commit 3: Add second section
        test_file.write_text("Initial content\n\nSection 1\n\nSection 2\n")
        subprocess.run(
            ["git", "add", "test.tex"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Add section 2"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Create a tag
        subprocess.run(
            ["git", "tag", "v1.0"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        yield test_file, repo_path

        # Cleanup
        if repo_path.exists():
            shutil.rmtree(repo_path)

    def test_diff_between_commits(self, git_repo_with_history):
        """Test comparing two commits."""
        test_file, repo_path = git_repo_with_history

        section = DocumentSection(test_file, git_root=repo_path)

        # Compare first commit to last
        diff = section.diff_between("HEAD~2", "HEAD")

        assert len(diff) > 0
        assert "Section 1" in diff or "+" in diff

    def test_diff_between_tag_and_head(self, git_repo_with_history):
        """Test comparing tag to HEAD."""
        test_file, repo_path = git_repo_with_history

        section = DocumentSection(test_file, git_root=repo_path)

        # Compare v1.0 tag to HEAD
        diff = section.diff_between("v1.0", "HEAD")

        # Should show nothing changed after tag
        # (because we didn't add commits after creating tag in this fixture)
        assert isinstance(diff, str)

    def test_diff_between_empty_when_no_changes(self, git_repo_with_history):
        """Test diff_between returns empty when commits are identical."""
        test_file, repo_path = git_repo_with_history

        section = DocumentSection(test_file, git_root=repo_path)

        # Compare HEAD to itself
        diff = section.diff_between("HEAD", "HEAD")

        assert diff == ""

    def test_diff_between_invalid_ref1(self, git_repo_with_history):
        """Test diff_between with invalid first reference."""
        test_file, repo_path = git_repo_with_history

        section = DocumentSection(test_file, git_root=repo_path)

        # Try with non-existent ref
        diff = section.diff_between("nonexistent123", "HEAD")

        assert diff == ""


class TestRefResolution:
    """Test _resolve_ref() for human-readable references."""

    @pytest.fixture
    def simple_git_repo(self):
        """Create simple git repository with one file."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_ref_resolve_")
        repo_path = Path(temp_dir)

        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Configure git
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Create and commit file
        test_file = repo_path / "test.tex"
        test_file.write_text("Test\n")

        subprocess.run(
            ["git", "add", "test.tex"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        yield test_file, repo_path

        # Cleanup
        if repo_path.exists():
            shutil.rmtree(repo_path)

    def test_resolve_head(self, simple_git_repo):
        """Test resolving HEAD reference."""
        test_file, repo_path = simple_git_repo

        section = DocumentSection(test_file, git_root=repo_path)
        resolved = section._resolve_ref("HEAD")

        assert resolved == "HEAD"

    def test_resolve_now_to_head(self, simple_git_repo):
        """Test resolving 'now' to HEAD."""
        test_file, repo_path = simple_git_repo

        section = DocumentSection(test_file, git_root=repo_path)
        resolved = section._resolve_ref("now")

        assert resolved == "HEAD"

    def test_resolve_invalid_ref(self, simple_git_repo):
        """Test resolving invalid reference returns None."""
        test_file, repo_path = simple_git_repo

        section = DocumentSection(test_file, git_root=repo_path)
        resolved = section._resolve_ref("nonexistent_ref_xyz")

        assert resolved is None


class TestTimeResolution:
    """Test time-based reference resolution."""

    @pytest.fixture
    def timed_git_repo(self):
        """Create git repository with time-spaced commits."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_time_resolve_")
        repo_path = Path(temp_dir)

        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Configure git
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Create initial commit
        test_file = repo_path / "test.tex"
        test_file.write_text("v1\n")

        subprocess.run(
            ["git", "add", "test.tex"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "v1"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Wait a bit, then create second commit
        time.sleep(1)
        test_file.write_text("v2\n")
        subprocess.run(
            ["git", "add", "test.tex"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "v2"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        yield test_file, repo_path

        # Cleanup
        if repo_path.exists():
            shutil.rmtree(repo_path)

    def test_parse_relative_time_days(self, timed_git_repo):
        """Test parsing '2 days ago' specification."""
        test_file, repo_path = timed_git_repo

        section = DocumentSection(test_file, git_root=repo_path)
        dt = section._parse_relative_time("2 days ago")

        assert dt is not None
        assert isinstance(dt, datetime)
        # Should be about 2 days in the past
        now = datetime.now()
        delta = (now - dt).total_seconds()
        assert 48 * 3600 - 60 < delta < 48 * 3600 + 60  # Within 1 minute of 2 days

    def test_parse_relative_time_hours(self, timed_git_repo):
        """Test parsing '24 hours ago' specification."""
        test_file, repo_path = timed_git_repo

        section = DocumentSection(test_file, git_root=repo_path)
        dt = section._parse_relative_time("24 hours ago")

        assert dt is not None
        assert isinstance(dt, datetime)

    def test_parse_relative_time_invalid(self, timed_git_repo):
        """Test parsing invalid time specification."""
        test_file, repo_path = timed_git_repo

        section = DocumentSection(test_file, git_root=repo_path)
        dt = section._parse_relative_time("invalid time spec")

        assert dt is None

    def test_parse_absolute_date_ymd(self):
        """Test parsing absolute date in YYYY-MM-DD format."""
        section = DocumentSection(Path("/tmp/test.tex"))
        dt = section._parse_absolute_date("2025-10-28")

        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 10
        assert dt.day == 28

    def test_parse_absolute_date_ymd_hm(self):
        """Test parsing absolute date with time in YYYY-MM-DD HH:MM format."""
        section = DocumentSection(Path("/tmp/test.tex"))
        dt = section._parse_absolute_date("2025-10-28 14:30")

        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 10
        assert dt.day == 28
        assert dt.hour == 14
        assert dt.minute == 30

    def test_parse_absolute_date_invalid(self):
        """Test parsing invalid date specification."""
        section = DocumentSection(Path("/tmp/test.tex"))
        dt = section._parse_absolute_date("not a date")

        assert dt is None


class TestIsValidGitRef:
    """Test _is_valid_git_ref() validation."""

    @pytest.fixture
    def simple_git_repo(self):
        """Create simple git repository."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_valid_ref_")
        repo_path = Path(temp_dir)

        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        test_file = repo_path / "test.tex"
        test_file.write_text("Test\n")

        subprocess.run(
            ["git", "add", "test.tex"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        yield test_file, repo_path

        if repo_path.exists():
            shutil.rmtree(repo_path)

    def test_valid_ref_head(self, simple_git_repo):
        """Test that HEAD is a valid reference."""
        test_file, repo_path = simple_git_repo

        section = DocumentSection(test_file, git_root=repo_path)
        is_valid = section._is_valid_git_ref("HEAD")

        assert is_valid is True

    def test_invalid_ref_nonexistent(self, simple_git_repo):
        """Test that nonexistent reference is invalid."""
        test_file, repo_path = simple_git_repo

        section = DocumentSection(test_file, git_root=repo_path)
        is_valid = section._is_valid_git_ref("nonexistent_xyz")

        assert is_valid is False


class TestDocumentSectionRepresentation:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ shows filename."""
        section = DocumentSection(Path("/path/to/introduction.tex"))
        repr_str = repr(section)

        assert "DocumentSection" in repr_str
        assert "introduction.tex" in repr_str


class TestDocumentSectionPathHandling:
    """Test path handling."""

    def test_path_attribute(self):
        """Test path attribute is stored."""
        path = Path("/path/to/file.tex")
        section = DocumentSection(path)

        assert section.path == path

    def test_git_root_passed_explicitly(self):
        """Test git_root is stored when passed."""
        git_root = Path("/path/to/git")
        section = DocumentSection(Path("/path/to/file.tex"), git_root=git_root)

        assert section.git_root == git_root


class TestDocumentSectionErrorHandling:
    """Test error handling."""

    def test_read_unicode_fallback(self):
        """Test read handles encoding errors gracefully."""
        temp_dir = tempfile.mkdtemp(prefix="scitex_enc_")
        temp_file = Path(temp_dir) / "test.tex"

        # Write with UTF-8
        temp_file.write_text("Valid UTF-8 content\n", encoding="utf-8")

        section = DocumentSection(temp_file)
        content = section.read()

        assert content is not None
        assert "Valid UTF-8 content" in str(content)

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_write_handles_exceptions(self):
        """Test write handles exceptions gracefully."""
        # Use a path that will fail to write
        section = DocumentSection(Path("/root/impossible/path/file.tex"))
        result = section.write("content")

        assert result is False

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/core/_DocumentSection.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-29 06:08:40 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/_DocumentSection.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/writer/dataclasses/_DocumentSection.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# DocumentSection - wrapper for document file with git-backed version control.
# 
# Provides intuitive version control API while leveraging git internally.
# """
# 
# from pathlib import Path
# from typing import Optional
# import subprocess
# 
# from scitex.logging import getLogger
# 
# logger = getLogger(__name__)
# 
# 
# class DocumentSection:
#     """
#     Wrapper for document section file with git-backed version control.
# 
#     Provides simple version control API while leveraging git internally:
#     - Users get intuitive .read(), .write(), .save(), .history(), .diff()
#     - We maintain clean separation from git complexity
#     - Enables advanced users to use git directly when needed
#     """
# 
#     def __init__(self, path: Path, git_root: Optional[Path] = None):
#         """
#         Initialize with file path and optional git root.
# 
#         Args:
#             path: Path to the document file
#             git_root: Path to git repository root (for efficiency)
#         """
#         self.path = path
#         self._git_root = git_root
#         self._cached_git_root = None
# 
#     @property
#     def git_root(self) -> Optional[Path]:
#         """Get cached git root, finding it if needed."""
#         if self._git_root is not None:
#             return self._git_root
#         if self._cached_git_root is None:
#             self._cached_git_root = self._find_git_root()
#         return self._cached_git_root
# 
#     @staticmethod
#     def _find_git_root(start_path: Path = None) -> Optional[Path]:
#         """Find git root by walking up directory tree."""
#         if start_path is None:
#             start_path = Path.cwd()
#         current = start_path.absolute()
#         while current != current.parent:
#             if (current / ".git").exists():
#                 return current
#             current = current.parent
#         return None
# 
#     def read(self):
#         """Read file contents with intelligent fallback strategy."""
#         if not self.path.exists():
#             logger.warning(f"File does not exist: {self.path}")
#             return None
# 
#         try:
#             import scitex.io as stx_io
# 
#             return stx_io.load(str(self.path))
#         except ImportError:
#             logger.debug("scitex.io not available, using plain text reader")
#             return self._read_plain_text()
#         except ValueError as e:
#             logger.warning(
#                 f"scitex.io could not parse {self.path} ({e}), "
#                 "falling back to plain text"
#             )
#             return self._read_plain_text()
#         except Exception as e:
#             logger.error(f"Unexpected error reading {self.path}: {e}", exc_info=True)
#             return None
# 
#     def _read_plain_text(self):
#         """Read file as plain text with proper encoding handling."""
#         try:
#             return self.path.read_text(encoding="utf-8")
#         except UnicodeDecodeError:
#             logger.warning(f"UTF-8 decode failed for {self.path}, trying latin-1")
#             return self.path.read_text(encoding="latin-1")
#         except Exception as e:
#             logger.error(f"Failed to read {self.path} as text: {e}")
#             return None
# 
#     def write(self, content) -> bool:
#         """Write content to file."""
#         try:
#             if isinstance(content, (list, tuple)):
#                 # Join lines if content is a list
#                 text = "\n".join(str(line) for line in content)
#             else:
#                 text = str(content)
#             self.path.write_text(text)
#             return True
#         except Exception as e:
#             logger.error(f"Failed to write {self.path}: {e}")
#             return False
# 
#     def history(self) -> list:
#         """Get version history (uses git log internally)."""
#         if not self.git_root:
#             logger.debug(f"No git repository for {self.path}")
#             return []
# 
#         try:
#             rel_path = self.path.relative_to(self.git_root)
# 
#             result = subprocess.run(
#                 ["git", "log", "--oneline", str(rel_path)],
#                 cwd=self.git_root,
#                 capture_output=True,
#                 text=True,
#                 timeout=5,
#             )
# 
#             if result.returncode != 0:
#                 logger.debug(f"Git log failed: {result.stderr}")
#                 return []
# 
#             return result.stdout.strip().split("\n") if result.stdout.strip() else []
#         except subprocess.TimeoutExpired:
#             logger.warning(f"Git log timed out for {self.path}")
#             return []
#         except Exception as e:
#             logger.error(f"Error getting history for {self.path}: {e}")
#             return []
# 
#     def diff(self, ref: str = "HEAD") -> str:
#         """Get diff against git reference (default: HEAD)."""
#         if not self.git_root:
#             logger.debug(f"No git repository for {self.path}")
#             return ""
# 
#         try:
#             rel_path = self.path.relative_to(self.git_root)
# 
#             result = subprocess.run(
#                 ["git", "diff", ref, str(rel_path)],
#                 cwd=self.git_root,
#                 capture_output=True,
#                 text=True,
#                 timeout=5,
#             )
# 
#             return result.stdout if result.returncode == 0 else ""
#         except subprocess.TimeoutExpired:
#             logger.warning(f"Git diff timed out for {self.path}")
#             return ""
#         except Exception as e:
#             logger.error(f"Error getting diff for {self.path}: {e}")
#             return ""
# 
#     def diff_between(self, ref1: str, ref2: str) -> str:
#         """
#         Compare two arbitrary git references.
# 
#         Args:
#             ref1: First git reference (commit, branch, tag, or human-readable spec)
#             ref2: Second git reference (commit, branch, tag, or human-readable spec)
# 
#         Returns:
#             Diff output string, or "" if error or no differences.
# 
#         Examples:
#             section.diff_between("HEAD~2", "HEAD")
#             section.diff_between("v1.0", "v2.0")
#             section.diff_between("main", "develop")
#             section.diff_between("2 days ago", "now")
#         """
#         if not self.git_root:
#             logger.debug(f"No git repository for {self.path}")
#             return ""
# 
#         try:
#             # Resolve human-readable refs to commit hashes
#             resolved_ref1 = self._resolve_ref(ref1)
#             resolved_ref2 = self._resolve_ref(ref2)
# 
#             if not resolved_ref1 or not resolved_ref2:
#                 logger.error(f"Failed to resolve references: {ref1} or {ref2}")
#                 return ""
# 
#             rel_path = self.path.relative_to(self.git_root)
# 
#             # Use git diff ref1..ref2 file (three-dot shows what changed on ref2 since ref1 diverged)
#             result = subprocess.run(
#                 [
#                     "git",
#                     "diff",
#                     f"{resolved_ref1}..{resolved_ref2}",
#                     str(rel_path),
#                 ],
#                 cwd=self.git_root,
#                 capture_output=True,
#                 text=True,
#                 timeout=5,
#             )
# 
#             return result.stdout if result.returncode == 0 else ""
#         except subprocess.TimeoutExpired:
#             logger.warning(f"Git diff timed out for {self.path}")
#             return ""
#         except Exception as e:
#             logger.error(f"Error getting diff_between for {self.path}: {e}")
#             return ""
# 
#     def _resolve_ref(self, spec: str) -> Optional[str]:
#         """
#         Resolve human-readable reference specification to git reference.
# 
#         Handles:
#         - Standard git refs: HEAD, HEAD~N, branch, tag, commit hash
#         - Relative time: "N days ago", "N weeks ago", "N hours ago", "now"
#         - Absolute dates: "2025-10-28", "2025-10-28 14:30"
# 
#         Args:
#             spec: Reference specification
# 
#         Returns:
#             Git reference (commit hash or ref name), or None if invalid.
#         """
#         if not self.git_root:
#             return None
# 
#         spec = spec.strip()
# 
#         # Direct git reference (HEAD, branch, tag, hash)
#         if self._is_valid_git_ref(spec):
#             return spec
# 
#         # Handle "now" as HEAD
#         if spec.lower() == "now":
#             return "HEAD"
# 
#         # Handle "today" as start of day
#         if spec.lower() == "today":
#             from datetime import datetime, timedelta
# 
#             today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
#             return self._find_commit_at_timestamp(today)
# 
#         # Handle relative time like "2 days ago", "1 week ago", "24 hours ago"
#         time_ref = self._parse_relative_time(spec)
#         if time_ref:
#             return self._find_commit_at_timestamp(time_ref)
# 
#         # Handle absolute date like "2025-10-28" or "2025-10-28 14:30"
#         date_ref = self._parse_absolute_date(spec)
#         if date_ref:
#             return self._find_commit_at_timestamp(date_ref)
# 
#         logger.warning(f"Could not resolve reference: {spec}")
#         return None
# 
#     def _is_valid_git_ref(self, ref: str) -> bool:
#         """Check if reference exists in git repository."""
#         if not self.git_root:
#             return False
# 
#         try:
#             result = subprocess.run(
#                 ["git", "rev-parse", "--verify", ref],
#                 cwd=self.git_root,
#                 capture_output=True,
#                 timeout=2,
#             )
#             return result.returncode == 0
#         except Exception:
#             return False
# 
#     def _parse_relative_time(self, spec: str):
#         """
#         Parse relative time specification like "2 days ago".
# 
#         Returns:
#             datetime object or None if not a valid time spec.
#         """
#         import re
#         from datetime import datetime, timedelta
# 
#         # Pattern: "N <unit> ago"
#         match = re.match(r"(\d+)\s*(day|week|hour|minute)s?\s*ago", spec, re.IGNORECASE)
#         if not match:
#             return None
# 
#         amount = int(match.group(1))
#         unit = match.group(2).lower()
# 
#         now = datetime.now()
#         if unit == "day":
#             return now - timedelta(days=amount)
#         elif unit == "week":
#             return now - timedelta(weeks=amount)
#         elif unit == "hour":
#             return now - timedelta(hours=amount)
#         elif unit == "minute":
#             return now - timedelta(minutes=amount)
# 
#         return None
# 
#     def _parse_absolute_date(self, spec: str):
#         """
#         Parse absolute date specification like "2025-10-28" or "2025-10-28 14:30".
# 
#         Returns:
#             datetime object or None if not a valid date spec.
#         """
#         from datetime import datetime
# 
#         # Try YYYY-MM-DD HH:MM format
#         try:
#             return datetime.strptime(spec, "%Y-%m-%d %H:%M")
#         except ValueError:
#             pass
# 
#         # Try YYYY-MM-DD format
#         try:
#             return datetime.strptime(spec, "%Y-%m-%d")
#         except ValueError:
#             pass
# 
#         return None
# 
#     def _find_commit_at_timestamp(self, target_datetime) -> Optional[str]:
#         """
#         Find commit closest to (before) given timestamp.
# 
#         Args:
#             target_datetime: datetime object
# 
#         Returns:
#             Commit hash or None if not found.
#         """
#         if not self.git_root:
#             return None
# 
#         try:
#             # Format timestamp for git
#             timestamp_str = target_datetime.strftime("%Y-%m-%d %H:%M:%S")
# 
#             # Find commit at or before this timestamp
#             result = subprocess.run(
#                 [
#                     "git",
#                     "log",
#                     "--format=%H",
#                     "--before=" + timestamp_str,
#                     "-1",  # Get only the most recent one
#                 ],
#                 cwd=self.git_root,
#                 capture_output=True,
#                 text=True,
#                 timeout=5,
#             )
# 
#             if result.returncode == 0 and result.stdout.strip():
#                 return result.stdout.strip()
#             else:
#                 logger.warning(f"No commit found before {timestamp_str}")
#                 return None
#         except subprocess.TimeoutExpired:
#             logger.warning(f"Git log timed out looking for commit at {target_datetime}")
#             return None
#         except Exception as e:
#             logger.error(f"Error finding commit at timestamp: {e}")
#             return None
# 
#     def commit(self, message: str) -> bool:
#         """Commit this file to project's git repo with retry logic."""
#         from scitex.git import git_retry
# 
#         if not self.git_root:
#             logger.warning(f"No git repository found for {self.path}")
#             return False
# 
#         def _do_commit():
#             rel_path = self.path.relative_to(self.git_root)
#             subprocess.run(
#                 ["git", "add", str(rel_path)],
#                 cwd=self.git_root,
#                 check=True,
#                 timeout=5,
#             )
#             subprocess.run(
#                 ["git", "commit", "-m", message],
#                 cwd=self.git_root,
#                 check=True,
#                 timeout=5,
#             )
# 
#         try:
#             git_retry(_do_commit)
#             logger.info(f"Committed {self.path}: {message}")
#             return True
#         except TimeoutError as e:
#             logger.error(f"Git lock timeout for {self.path}: {e}")
#             return False
#         except Exception as e:
#             logger.error(f"Failed to commit {self.path}: {e}")
#             return False
# 
#     def checkout(self, ref: str = "HEAD") -> bool:
#         """Checkout file from git reference."""
#         if not self.git_root:
#             logger.warning(f"No git repository found for {self.path}")
#             return False
# 
#         try:
#             rel_path = self.path.relative_to(self.git_root)
# 
#             result = subprocess.run(
#                 ["git", "checkout", ref, str(rel_path)],
#                 cwd=self.git_root,
#                 capture_output=True,
#                 timeout=5,
#             )
# 
#             if result.returncode == 0:
#                 logger.info(f"Checked out {self.path} from {ref}")
#                 return True
#             else:
#                 logger.error(f"Git checkout failed: {result.stderr.decode()}")
#                 return False
#         except subprocess.TimeoutExpired:
#             logger.error(f"Git checkout timed out for {self.path}")
#             return False
#         except Exception as e:
#             logger.error(f"Error checking out {self.path}: {e}")
#             return False
# 
#     def __repr__(self) -> str:
#         """String representation."""
#         return f"DocumentSection({self.path.name})"
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
#     section = DocumentSection(Path(args.file))
# 
#     if args.action == "read":
#         content = section.read()
#         print(content if content else "File not found or empty")
# 
#     elif args.action == "history":
#         history = section.history()
#         print(f"History ({len(history)} commits):")
#         for entry in history:
#             print(f"  {entry}")
# 
#     elif args.action == "diff":
#         diff = section.diff(args.ref)
#         print(diff if diff else "No differences")
# 
#     return 0
# 
# 
# def parse_args():
#     import argparse
# 
#     parser = argparse.ArgumentParser(
#         description="Demonstrate DocumentSection version control"
#     )
#     parser.add_argument(
#         "--file",
#         "-f",
#         type=str,
#         required=True,
#         help="Path to document section file",
#     )
#     parser.add_argument(
#         "--action",
#         "-a",
#         type=str,
#         choices=["read", "history", "diff"],
#         default="read",
#         help="Action to perform (default: read)",
#     )
#     parser.add_argument(
#         "--ref",
#         "-r",
#         type=str,
#         default="HEAD",
#         help="Git reference for diff (default: HEAD)",
#     )
# 
#     return parser.parse_args()
# 
# 
# if __name__ == "__main__":
#     run_session()
# 
# 
# __all__ = ["DocumentSection"]
# 
# # python -m scitex.writer.dataclasses.core._DocumentSection --file ./01_manuscript/contents/introduction.tex --action history
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/core/_DocumentSection.py
# --------------------------------------------------------------------------------
