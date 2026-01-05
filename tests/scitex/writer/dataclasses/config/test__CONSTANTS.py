#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.config._CONSTANTS."""

import pytest

from scitex.writer.dataclasses.config._CONSTANTS import (
    DOC_TYPE_DIRS,
    DOC_TYPE_FLAGS,
    DOC_TYPE_PDFS,
)


class TestDocTypeDirs:
    """Tests for DOC_TYPE_DIRS constant."""

    def test_contains_manuscript(self):
        """Verify manuscript mapping exists."""
        assert "manuscript" in DOC_TYPE_DIRS
        assert DOC_TYPE_DIRS["manuscript"] == "01_manuscript"

    def test_contains_supplementary(self):
        """Verify supplementary mapping exists."""
        assert "supplementary" in DOC_TYPE_DIRS
        assert DOC_TYPE_DIRS["supplementary"] == "02_supplementary"

    def test_contains_revision(self):
        """Verify revision mapping exists."""
        assert "revision" in DOC_TYPE_DIRS
        assert DOC_TYPE_DIRS["revision"] == "03_revision"

    def test_has_exactly_three_entries(self):
        """Verify DOC_TYPE_DIRS has exactly 3 document types."""
        assert len(DOC_TYPE_DIRS) == 3


class TestDocTypeFlags:
    """Tests for DOC_TYPE_FLAGS constant."""

    def test_manuscript_flag(self):
        """Verify manuscript uses -m flag."""
        assert DOC_TYPE_FLAGS["manuscript"] == "-m"

    def test_supplementary_flag(self):
        """Verify supplementary uses -s flag."""
        assert DOC_TYPE_FLAGS["supplementary"] == "-s"

    def test_revision_flag(self):
        """Verify revision uses -r flag."""
        assert DOC_TYPE_FLAGS["revision"] == "-r"

    def test_all_flags_are_unique(self):
        """Verify all flags are unique to avoid conflicts."""
        flags = list(DOC_TYPE_FLAGS.values())
        assert len(flags) == len(set(flags))


class TestDocTypePdfs:
    """Tests for DOC_TYPE_PDFS constant."""

    def test_manuscript_pdf(self):
        """Verify manuscript PDF filename."""
        assert DOC_TYPE_PDFS["manuscript"] == "manuscript.pdf"

    def test_supplementary_pdf(self):
        """Verify supplementary PDF filename."""
        assert DOC_TYPE_PDFS["supplementary"] == "supplementary.pdf"

    def test_revision_pdf(self):
        """Verify revision PDF filename."""
        assert DOC_TYPE_PDFS["revision"] == "revision.pdf"

    def test_all_pdfs_end_with_extension(self):
        """Verify all PDF filenames have .pdf extension."""
        for pdf in DOC_TYPE_PDFS.values():
            assert pdf.endswith(".pdf")


class TestConstantsConsistency:
    """Tests for consistency across constants."""

    def test_all_constants_have_same_keys(self):
        """Verify all constants use same document type keys."""
        assert set(DOC_TYPE_DIRS.keys()) == set(DOC_TYPE_FLAGS.keys())
        assert set(DOC_TYPE_DIRS.keys()) == set(DOC_TYPE_PDFS.keys())


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
