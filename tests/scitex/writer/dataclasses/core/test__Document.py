#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.core._Document."""

from pathlib import Path

import pytest

from scitex.writer.dataclasses.core._Document import Document
from scitex.writer.dataclasses.core._DocumentSection import DocumentSection


class TestDocumentCreation:
    """Tests for Document instantiation."""

    def test_creates_with_doc_dir(self, tmp_path):
        """Verify Document creates with doc_dir."""
        doc = Document(doc_dir=tmp_path)
        assert doc.dir == tmp_path

    def test_git_root_optional(self, tmp_path):
        """Verify git_root defaults to None."""
        doc = Document(doc_dir=tmp_path)
        assert doc.git_root is None

    def test_git_root_can_be_set(self, tmp_path):
        """Verify git_root can be explicitly set."""
        git_root = tmp_path / "project"
        doc = Document(doc_dir=tmp_path, git_root=git_root)
        assert doc.git_root == git_root


class TestDocumentGetattr:
    """Tests for Document __getattr__ dynamic attribute lookup."""

    def test_getattr_returns_document_section(self, tmp_path):
        """Verify attribute access returns DocumentSection."""
        doc = Document(doc_dir=tmp_path)
        section = doc.introduction

        assert isinstance(section, DocumentSection)

    def test_getattr_builds_correct_path(self, tmp_path):
        """Verify attribute access builds correct path."""
        doc = Document(doc_dir=tmp_path)
        section = doc.introduction

        expected_path = tmp_path / "contents" / "introduction.tex"
        assert section.path == expected_path

    def test_getattr_works_for_arbitrary_names(self, tmp_path):
        """Verify attribute access works for any name."""
        doc = Document(doc_dir=tmp_path)

        assert doc.methods.path == tmp_path / "contents" / "methods.tex"
        assert doc.results.path == tmp_path / "contents" / "results.tex"
        assert doc.discussion.path == tmp_path / "contents" / "discussion.tex"
        assert doc.custom_section.path == tmp_path / "contents" / "custom_section.tex"

    def test_getattr_passes_git_root(self, tmp_path):
        """Verify git_root is passed to DocumentSection."""
        git_root = tmp_path / "project"
        doc = Document(doc_dir=tmp_path, git_root=git_root)
        section = doc.introduction

        assert section.git_root == git_root

    def test_getattr_private_attribute_raises(self, tmp_path):
        """Verify accessing private attributes raises AttributeError."""
        doc = Document(doc_dir=tmp_path)

        with pytest.raises(AttributeError, match="has no attribute '_private'"):
            _ = doc._private

    def test_getattr_dunder_attribute_raises(self, tmp_path):
        """Verify accessing dunder attributes raises AttributeError."""
        doc = Document(doc_dir=tmp_path)

        # Python's name mangling transforms __attr to _ClassName__attr
        # So we use getattr to test the actual behavior
        with pytest.raises(AttributeError):
            getattr(doc, "__something")


class TestDocumentRepr:
    """Tests for Document __repr__ method."""

    def test_repr_contains_class_name(self, tmp_path):
        """Verify repr contains class name."""
        doc = Document(doc_dir=tmp_path)
        repr_str = repr(doc)

        assert "Document" in repr_str

    def test_repr_contains_dir_name(self, tmp_path):
        """Verify repr contains directory name."""
        doc = Document(doc_dir=tmp_path / "01_manuscript")
        repr_str = repr(doc)

        assert "01_manuscript" in repr_str


class TestDocumentWithContents:
    """Tests for Document with actual contents directory."""

    def test_can_read_existing_file(self, tmp_path):
        """Verify Document can read existing file through section."""
        contents_dir = tmp_path / "contents"
        contents_dir.mkdir()

        intro_file = contents_dir / "introduction.tex"
        intro_file.write_text("\\section{Introduction}\nThis is the intro.")

        doc = Document(doc_dir=tmp_path)
        content = doc.introduction.read()

        assert content is not None
        assert "Introduction" in str(content)

    def test_section_read_nonexistent_returns_none(self, tmp_path):
        """Verify reading nonexistent section returns None."""
        doc = Document(doc_dir=tmp_path)
        content = doc.nonexistent.read()

        assert content is None

    def test_can_write_to_section(self, tmp_path):
        """Verify Document can write to section."""
        contents_dir = tmp_path / "contents"
        contents_dir.mkdir()

        doc = Document(doc_dir=tmp_path)
        result = doc.methods.write("\\section{Methods}\nNew content.")

        assert result is True
        assert (contents_dir / "methods.tex").exists()
        assert "Methods" in (contents_dir / "methods.tex").read_text()


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
