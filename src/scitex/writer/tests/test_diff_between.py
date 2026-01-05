#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for DocumentSection.diff_between() and ref resolution.

Tests cover:
- Comparing two arbitrary git references
- Human-readable reference resolution
- Relative time specifications (N days ago, etc.)
- Absolute date specifications
- Timestamp-based commit finding
"""

import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scitex.writer.dataclasses.core._DocumentSection import DocumentSection


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
