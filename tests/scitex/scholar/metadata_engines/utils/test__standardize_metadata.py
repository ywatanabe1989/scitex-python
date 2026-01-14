#!/usr/bin/env python3
"""Tests for standardize_metadata utility functions."""

import pytest

from scitex.scholar.metadata_engines.utils._standardize_metadata import (
    BASE_STRUCTURE,
    standardize_metadata,
    to_bibtex_entry,
)


class TestBaseStructure:
    """Tests for BASE_STRUCTURE constant."""

    def test_has_id_section(self):
        """BASE_STRUCTURE should have id section."""
        assert "id" in BASE_STRUCTURE

    def test_has_basic_section(self):
        """BASE_STRUCTURE should have basic section."""
        assert "basic" in BASE_STRUCTURE

    def test_has_citation_count_section(self):
        """BASE_STRUCTURE should have citation_count section."""
        assert "citation_count" in BASE_STRUCTURE

    def test_has_publication_section(self):
        """BASE_STRUCTURE should have publication section."""
        assert "publication" in BASE_STRUCTURE

    def test_has_url_section(self):
        """BASE_STRUCTURE should have url section."""
        assert "url" in BASE_STRUCTURE

    def test_has_path_section(self):
        """BASE_STRUCTURE should have path section."""
        assert "path" in BASE_STRUCTURE

    def test_has_system_section(self):
        """BASE_STRUCTURE should have system section."""
        assert "system" in BASE_STRUCTURE

    def test_id_section_has_doi(self):
        """id section should have doi field."""
        assert "doi" in BASE_STRUCTURE["id"]
        assert "doi_engines" in BASE_STRUCTURE["id"]

    def test_basic_section_has_title(self):
        """basic section should have title field."""
        assert "title" in BASE_STRUCTURE["basic"]
        assert "title_engines" in BASE_STRUCTURE["basic"]

    def test_citation_count_has_yearly_counts(self):
        """citation_count section should have yearly count fields."""
        assert "total" in BASE_STRUCTURE["citation_count"]
        assert "2023" in BASE_STRUCTURE["citation_count"]
        assert "2024" in BASE_STRUCTURE["citation_count"]


class TestStandardizeMetadata:
    """Tests for standardize_metadata function."""

    def test_returns_dict(self):
        """standardize_metadata should return a dictionary."""
        result = standardize_metadata({})
        assert isinstance(result, dict)

    def test_empty_input_returns_base_structure(self):
        """Empty input should return base structure with null values."""
        result = standardize_metadata({})
        assert "id" in result
        assert "basic" in result
        assert result["id"]["doi"] is None

    def test_preserves_existing_values(self):
        """Should preserve existing metadata values."""
        metadata = {
            "id": {"doi": "10.1038/nature12373"},
            "basic": {"title": "Test Paper"},
        }
        result = standardize_metadata(metadata)
        assert result["id"]["doi"] == "10.1038/nature12373"
        assert result["basic"]["title"] == "Test Paper"

    def test_adds_missing_sections(self):
        """Should add missing sections with default values."""
        metadata = {"id": {"doi": "10.1038/test"}}
        result = standardize_metadata(metadata)
        assert "basic" in result
        assert "publication" in result
        assert result["basic"]["title"] is None

    def test_adds_missing_fields_within_section(self):
        """Should add missing fields within existing sections."""
        metadata = {"id": {"doi": "10.1038/test"}}
        result = standardize_metadata(metadata)
        assert "pmid" in result["id"]
        assert "arxiv_id" in result["id"]

    def test_does_not_modify_original(self):
        """Should not modify the original metadata dict."""
        metadata = {"id": {"doi": "10.1038/test"}}
        original_doi = metadata["id"]["doi"]
        standardize_metadata(metadata)
        assert metadata["id"]["doi"] == original_doi

    def test_handles_full_metadata(self):
        """Should handle complete metadata structure."""
        metadata = {
            "id": {"doi": "10.1038/test", "pmid": "12345678"},
            "basic": {
                "title": "Test Paper",
                "authors": ["Smith, John", "Doe, Jane"],
                "year": 2023,
            },
            "publication": {"journal": "Nature", "volume": "600"},
        }
        result = standardize_metadata(metadata)
        assert result["id"]["doi"] == "10.1038/test"
        assert result["id"]["pmid"] == "12345678"
        assert result["basic"]["title"] == "Test Paper"
        assert result["publication"]["journal"] == "Nature"


class TestToBibtexEntry:
    """Tests for to_bibtex_entry function."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample standardized metadata."""
        return standardize_metadata(
            {
                "id": {"doi": "10.1038/nature12373"},
                "basic": {
                    "title": "Sample Paper Title",
                    "authors": ["Smith, John", "Doe, Jane"],
                    "year": 2023,
                    "abstract": "This is the abstract.",
                },
                "publication": {"journal": "Nature"},
            }
        )

    def test_returns_string(self, sample_metadata):
        """to_bibtex_entry should return a string."""
        result = to_bibtex_entry(sample_metadata)
        assert isinstance(result, str)

    def test_contains_entry_type(self, sample_metadata):
        """BibTeX entry should contain entry type."""
        result = to_bibtex_entry(sample_metadata)
        assert result.startswith("@article{")

    def test_uses_custom_key(self, sample_metadata):
        """Should use custom key when provided."""
        result = to_bibtex_entry(sample_metadata, key="custom2023")
        assert "@article{custom2023," in result

    def test_generates_key_from_author_year(self, sample_metadata):
        """Should generate key from first author and year."""
        result = to_bibtex_entry(sample_metadata)
        # "Smith, John" -> split by space, take last -> "John" -> "john-2023"
        assert "john-2023" in result.lower()

    def test_includes_title(self, sample_metadata):
        """BibTeX entry should include title."""
        result = to_bibtex_entry(sample_metadata)
        assert "title = {Sample Paper Title}" in result

    def test_includes_authors(self, sample_metadata):
        """BibTeX entry should include authors."""
        result = to_bibtex_entry(sample_metadata)
        assert "author = {Smith, John and Doe, Jane}" in result

    def test_includes_year(self, sample_metadata):
        """BibTeX entry should include year."""
        result = to_bibtex_entry(sample_metadata)
        assert "year = {2023}" in result

    def test_includes_journal(self, sample_metadata):
        """BibTeX entry should include journal."""
        result = to_bibtex_entry(sample_metadata)
        assert "journal = {Nature}" in result

    def test_includes_doi(self, sample_metadata):
        """BibTeX entry should include DOI."""
        result = to_bibtex_entry(sample_metadata)
        assert "doi = {10.1038/nature12373}" in result

    def test_includes_abstract(self, sample_metadata):
        """BibTeX entry should include abstract."""
        result = to_bibtex_entry(sample_metadata)
        assert "abstract = {This is the abstract.}" in result

    def test_ends_with_closing_brace(self, sample_metadata):
        """BibTeX entry should end with closing brace."""
        result = to_bibtex_entry(sample_metadata)
        assert result.strip().endswith("}")

    def test_arxiv_entry_is_misc(self):
        """ArXiv papers should have @misc entry type."""
        metadata = standardize_metadata(
            {
                "id": {"arxiv_id": "2301.12345"},
                "basic": {
                    "title": "ArXiv Paper",
                    "authors": ["Author, Test"],
                    "year": 2023,
                },
                "publication": {},
            }
        )
        result = to_bibtex_entry(metadata)
        assert result.startswith("@misc{")

    def test_no_journal_is_misc(self):
        """Papers without journal should have @misc entry type."""
        metadata = standardize_metadata(
            {
                "id": {},
                "basic": {
                    "title": "Test Paper",
                    "authors": ["Author, Test"],
                    "year": 2023,
                },
                "publication": {},
            }
        )
        result = to_bibtex_entry(metadata)
        assert result.startswith("@misc{")

    def test_escapes_braces_in_title(self):
        """Should escape braces in title."""
        metadata = standardize_metadata(
            {
                "id": {},
                "basic": {
                    "title": "Title with {braces}",
                    "authors": ["Test, Author"],
                    "year": 2023,
                },
                "publication": {},
            }
        )
        result = to_bibtex_entry(metadata)
        assert r"\{braces\}" in result

    def test_handles_no_authors(self):
        """Should handle metadata with no authors."""
        metadata = standardize_metadata(
            {
                "id": {},
                "basic": {"title": "Test Paper", "authors": None, "year": 2023},
                "publication": {},
            }
        )
        result = to_bibtex_entry(metadata)
        assert "unknown-2023" in result.lower()
        assert "author" not in result.lower()

    def test_handles_no_year(self):
        """Should handle metadata with no year."""
        metadata = standardize_metadata(
            {
                "id": {},
                "basic": {
                    "title": "Test Paper",
                    "authors": ["Test, Author"],
                    "year": None,
                },
                "publication": {},
            }
        )
        result = to_bibtex_entry(metadata)
        assert "author-0000" in result.lower()


class TestStandardizeMetadataEdgeCases:
    """Edge case tests for standardize_metadata."""

    def test_handles_unknown_section(self):
        """Should ignore unknown sections."""
        metadata = {"unknown_section": {"field": "value"}}
        result = standardize_metadata(metadata)
        # Unknown section should be ignored
        assert "id" in result
        assert "basic" in result

    def test_handles_nested_dict_update(self):
        """Should properly update nested dictionaries."""
        metadata = {
            "citation_count": {"total": 150, "2023": 50},
        }
        result = standardize_metadata(metadata)
        assert result["citation_count"]["total"] == 150
        assert result["citation_count"]["2023"] == 50
        # Other years should still exist with None
        assert result["citation_count"]["2022"] is None

    def test_preserves_list_values(self):
        """Should preserve list values in metadata."""
        metadata = {
            "url": {"pdfs": ["http://example.com/paper.pdf"]},
        }
        result = standardize_metadata(metadata)
        assert result["url"]["pdfs"] == ["http://example.com/paper.pdf"]


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
