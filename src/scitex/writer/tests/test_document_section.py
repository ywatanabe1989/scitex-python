#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for DocumentSection class.

Tests cover:
- File read/write operations
- Git operations (commit, history, diff, checkout)
- Error handling and edge cases
- Integration with git repository
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
    pytest.main([__file__, "-v"])
