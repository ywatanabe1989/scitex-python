#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.contents._ManuscriptContents."""

from pathlib import Path

import pytest

from scitex.writer.dataclasses.contents._ManuscriptContents import ManuscriptContents
from scitex.writer.dataclasses.core._DocumentSection import DocumentSection


class TestManuscriptContentsCreation:
    """Tests for ManuscriptContents instantiation."""

    def test_creates_with_root_path(self, tmp_path):
        """Verify ManuscriptContents creates with root path."""
        contents = ManuscriptContents(root=tmp_path)
        assert contents.root == tmp_path

    def test_git_root_optional(self, tmp_path):
        """Verify git_root defaults to None."""
        contents = ManuscriptContents(root=tmp_path)
        assert contents.git_root is None

    def test_git_root_can_be_set(self, tmp_path):
        """Verify git_root can be explicitly set."""
        git_root = tmp_path / "project"
        contents = ManuscriptContents(root=tmp_path, git_root=git_root)
        assert contents.git_root == git_root


class TestManuscriptContentsPostInit:
    """Tests for ManuscriptContents __post_init__ initialization."""

    def test_abstract_initialized(self, tmp_path):
        """Verify abstract DocumentSection is initialized."""
        contents = ManuscriptContents(root=tmp_path)
        assert isinstance(contents.abstract, DocumentSection)
        assert contents.abstract.path == tmp_path / "abstract.tex"

    def test_introduction_initialized(self, tmp_path):
        """Verify introduction DocumentSection is initialized."""
        contents = ManuscriptContents(root=tmp_path)
        assert isinstance(contents.introduction, DocumentSection)
        assert contents.introduction.path == tmp_path / "introduction.tex"

    def test_methods_initialized(self, tmp_path):
        """Verify methods DocumentSection is initialized."""
        contents = ManuscriptContents(root=tmp_path)
        assert isinstance(contents.methods, DocumentSection)
        assert contents.methods.path == tmp_path / "methods.tex"

    def test_results_initialized(self, tmp_path):
        """Verify results DocumentSection is initialized."""
        contents = ManuscriptContents(root=tmp_path)
        assert isinstance(contents.results, DocumentSection)
        assert contents.results.path == tmp_path / "results.tex"

    def test_discussion_initialized(self, tmp_path):
        """Verify discussion DocumentSection is initialized."""
        contents = ManuscriptContents(root=tmp_path)
        assert isinstance(contents.discussion, DocumentSection)
        assert contents.discussion.path == tmp_path / "discussion.tex"

    def test_metadata_sections_initialized(self, tmp_path):
        """Verify metadata sections are initialized."""
        contents = ManuscriptContents(root=tmp_path)
        assert contents.title.path == tmp_path / "title.tex"
        assert contents.authors.path == tmp_path / "authors.tex"
        assert contents.keywords.path == tmp_path / "keywords.tex"
        assert contents.journal_name.path == tmp_path / "journal_name.tex"

    def test_optional_sections_initialized(self, tmp_path):
        """Verify optional sections are initialized."""
        contents = ManuscriptContents(root=tmp_path)
        assert contents.graphical_abstract.path == tmp_path / "graphical_abstract.tex"
        assert contents.highlights.path == tmp_path / "highlights.tex"
        assert contents.data_availability.path == tmp_path / "data_availability.tex"
        assert contents.additional_info.path == tmp_path / "additional_info.tex"
        assert contents.wordcount.path == tmp_path / "wordcount.tex"

    def test_directory_paths_initialized(self, tmp_path):
        """Verify directory paths are initialized."""
        contents = ManuscriptContents(root=tmp_path)
        assert contents.figures == tmp_path / "figures"
        assert contents.tables == tmp_path / "tables"
        assert contents.latex_styles == tmp_path / "latex_styles"

    def test_bibliography_initialized(self, tmp_path):
        """Verify bibliography DocumentSection is initialized."""
        contents = ManuscriptContents(root=tmp_path)
        assert isinstance(contents.bibliography, DocumentSection)
        assert contents.bibliography.path == tmp_path / "bibliography.bib"

    def test_git_root_passed_to_sections(self, tmp_path):
        """Verify git_root is passed to DocumentSection instances."""
        git_root = tmp_path / "project"
        contents = ManuscriptContents(root=tmp_path, git_root=git_root)
        assert contents.abstract.git_root == git_root
        assert contents.introduction.git_root == git_root


class TestManuscriptContentsVerifyStructure:
    """Tests for ManuscriptContents verify_structure method."""

    def test_verify_fails_when_no_files_exist(self, tmp_path):
        """Verify returns False when required files are missing."""
        contents = ManuscriptContents(root=tmp_path)
        is_valid, missing = contents.verify_structure()

        assert is_valid is False
        assert len(missing) == 5

    def test_verify_fails_with_partial_files(self, tmp_path):
        """Verify returns False when only some required files exist."""
        (tmp_path / "abstract.tex").touch()
        (tmp_path / "introduction.tex").touch()

        contents = ManuscriptContents(root=tmp_path)
        is_valid, missing = contents.verify_structure()

        assert is_valid is False
        assert len(missing) == 3

    def test_verify_passes_with_all_required_files(self, tmp_path):
        """Verify returns True when all required files exist."""
        (tmp_path / "abstract.tex").touch()
        (tmp_path / "introduction.tex").touch()
        (tmp_path / "methods.tex").touch()
        (tmp_path / "results.tex").touch()
        (tmp_path / "discussion.tex").touch()

        contents = ManuscriptContents(root=tmp_path)
        is_valid, missing = contents.verify_structure()

        assert is_valid is True
        assert len(missing) == 0

    def test_verify_missing_list_contains_filenames(self, tmp_path):
        """Verify missing list contains descriptive entries."""
        contents = ManuscriptContents(root=tmp_path)
        _, missing = contents.verify_structure()

        assert any("abstract.tex" in m for m in missing)
        assert any("introduction.tex" in m for m in missing)
        assert any("methods.tex" in m for m in missing)
        assert any("results.tex" in m for m in missing)
        assert any("discussion.tex" in m for m in missing)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
