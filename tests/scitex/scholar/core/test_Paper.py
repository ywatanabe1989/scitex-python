#!/usr/bin/env python3
# Timestamp: "2026-01-05"
# File: tests/scitex/scholar/core/test_Paper.py
# ----------------------------------------

"""
Comprehensive tests for the Paper class and related metadata structures.

Tests cover:
- Paper creation and initialization
- Metadata field validation
- Type coercion and range validation
- DOI/URL automatic synchronization
- Serialization/deserialization
"""

import pytest
from pydantic import ValidationError

from scitex.scholar.core.Paper import (
    AccessMetadata,
    BasicMetadata,
    CitationCountMetadata,
    ContainerMetadata,
    IDMetadata,
    Paper,
    PaperMetadataStructure,
    PathMetadata,
    PublicationMetadata,
    SystemMetadata,
    URLMetadata,
)


class TestIDMetadata:
    """Tests for IDMetadata model."""

    def test_create_empty(self):
        """Creating empty IDMetadata should work."""
        id_meta = IDMetadata()
        assert id_meta.doi is None
        assert id_meta.arxiv_id is None
        assert id_meta.pmid is None
        assert id_meta.doi_engines == []

    def test_set_doi(self):
        """Setting DOI should work."""
        id_meta = IDMetadata()
        id_meta.doi = "10.1234/test"
        assert id_meta.doi == "10.1234/test"

    def test_set_with_engines(self):
        """Setting DOI with engines should track source."""
        id_meta = IDMetadata(doi="10.1234/test", doi_engines=["CrossRef", "PubMed"])
        assert id_meta.doi == "10.1234/test"
        assert id_meta.doi_engines == ["CrossRef", "PubMed"]


class TestBasicMetadata:
    """Tests for BasicMetadata model."""

    def test_create_empty(self):
        """Creating empty BasicMetadata should work."""
        basic = BasicMetadata()
        assert basic.title is None
        assert basic.authors is None
        assert basic.year is None

    def test_set_basic_fields(self):
        """Setting basic fields should work."""
        basic = BasicMetadata(
            title="Test Paper",
            authors=["Smith, John", "Doe, Jane"],
            year=2023,
            abstract="This is a test abstract.",
            keywords=["test", "paper"],
        )
        assert basic.title == "Test Paper"
        assert len(basic.authors) == 2
        assert basic.year == 2023
        assert "test" in basic.keywords

    def test_year_validation_valid_range(self):
        """Year within valid range (1900-2100) should work."""
        basic = BasicMetadata(year=2023)
        assert basic.year == 2023

        basic = BasicMetadata(year=1900)
        assert basic.year == 1900

        basic = BasicMetadata(year=2100)
        assert basic.year == 2100

    def test_year_validation_too_old(self):
        """Year before 1900 should raise ValidationError."""
        with pytest.raises(ValidationError):
            BasicMetadata(year=1800)

    def test_year_validation_too_future(self):
        """Year after 2100 should raise ValidationError."""
        with pytest.raises(ValidationError):
            BasicMetadata(year=2200)

    def test_year_type_coercion(self):
        """String year should be coerced to int."""
        basic = BasicMetadata(year="2023")
        assert basic.year == 2023
        assert isinstance(basic.year, int)


class TestCitationCountMetadata:
    """Tests for CitationCountMetadata model."""

    def test_create_empty(self):
        """Creating empty CitationCountMetadata should work."""
        citations = CitationCountMetadata()
        assert citations.total is None
        assert citations.y2024 is None

    def test_set_citation_counts(self):
        """Setting citation counts should work."""
        citations = CitationCountMetadata(total=1000, y2024=150, y2023=200)
        assert citations.total == 1000
        assert citations.y2024 == 150
        assert citations.y2023 == 200

    def test_negative_citations_invalid(self):
        """Negative citation counts should raise ValidationError."""
        with pytest.raises(ValidationError):
            CitationCountMetadata(total=-10)

    def test_year_alias_mapping(self):
        """Numeric year keys should map to y-prefixed attributes."""
        # Test via model_validate for alias support
        citations = CitationCountMetadata.model_validate(
            {"2024": 150, "2023": 200, "total": 1000}
        )
        assert citations.y2024 == 150
        assert citations.y2023 == 200

    def test_serialization_with_aliases(self):
        """model_dump should use aliases for year fields."""
        citations = CitationCountMetadata(total=100, y2024=50)
        data = citations.model_dump()
        assert "2024" in data
        assert data["2024"] == 50


class TestPublicationMetadata:
    """Tests for PublicationMetadata model."""

    def test_create_empty(self):
        """Creating empty PublicationMetadata should work."""
        pub = PublicationMetadata()
        assert pub.journal is None
        assert pub.impact_factor is None

    def test_set_publication_fields(self):
        """Setting publication fields should work."""
        pub = PublicationMetadata(
            journal="Nature",
            short_journal="Nat",
            impact_factor=50.5,
            volume="123",
            pages="1-10",
        )
        assert pub.journal == "Nature"
        assert pub.impact_factor == 50.5
        assert pub.volume == "123"

    def test_negative_impact_factor_invalid(self):
        """Negative impact factor should raise ValidationError."""
        with pytest.raises(ValidationError):
            PublicationMetadata(impact_factor=-5.0)


class TestURLMetadata:
    """Tests for URLMetadata model."""

    def test_create_empty(self):
        """Creating empty URLMetadata should work."""
        url = URLMetadata()
        assert url.doi is None
        assert url.pdfs == []

    def test_set_pdf_urls(self):
        """Setting PDF URLs should work."""
        url = URLMetadata(
            doi="https://doi.org/10.1234/test",
            pdfs=[{"url": "https://example.com/paper.pdf", "source": "publisher"}],
        )
        assert url.doi == "https://doi.org/10.1234/test"
        assert len(url.pdfs) == 1


class TestPaperMetadataStructure:
    """Tests for PaperMetadataStructure model."""

    def test_create_empty(self):
        """Creating empty structure should initialize all sections."""
        meta = PaperMetadataStructure()
        assert meta.id is not None
        assert meta.basic is not None
        assert meta.citation_count is not None
        assert meta.publication is not None
        assert meta.url is not None
        assert meta.path is not None
        assert meta.access is not None
        assert meta.system is not None

    def test_doi_url_sync_from_doi(self):
        """Setting DOI should auto-generate DOI URL."""
        meta = PaperMetadataStructure()
        meta.id.doi = "10.1234/test"
        meta.id.doi_engines = ["CrossRef"]

        # Trigger sync by creating new instance
        meta = PaperMetadataStructure.model_validate(meta.model_dump())

        assert meta.url.doi == "https://doi.org/10.1234/test"

    def test_doi_url_sync_from_url(self):
        """Setting DOI URL should extract DOI."""
        meta = PaperMetadataStructure()
        meta.url.doi = "https://doi.org/10.1234/test"
        meta.url.doi_engines = ["manual"]

        # Trigger sync
        meta = PaperMetadataStructure.model_validate(meta.model_dump())

        assert meta.id.doi == "10.1234/test"

    def test_arxiv_sync_from_id(self):
        """Setting arXiv ID should auto-generate arXiv URL."""
        meta = PaperMetadataStructure()
        meta.id.arxiv_id = "2301.12345"
        meta.id.arxiv_id_engines = ["arXiv"]

        # Trigger sync
        meta = PaperMetadataStructure.model_validate(meta.model_dump())

        assert meta.url.arxiv == "https://arxiv.org/abs/2301.12345"

    def test_set_doi_method(self):
        """set_doi method should sync both ID and URL."""
        meta = PaperMetadataStructure()
        meta.set_doi("10.1234/test")

        assert meta.id.doi == "10.1234/test"
        assert meta.url.doi == "https://doi.org/10.1234/test"

    def test_set_doi_url_method(self):
        """set_doi_url method should sync both ID and URL."""
        meta = PaperMetadataStructure()
        meta.set_doi_url("https://doi.org/10.1234/test")

        assert meta.url.doi == "https://doi.org/10.1234/test"
        assert meta.id.doi == "10.1234/test"


class TestContainerMetadata:
    """Tests for ContainerMetadata model."""

    def test_create_empty(self):
        """Creating empty ContainerMetadata should work."""
        container = ContainerMetadata()
        assert container.scitex_id is None
        assert container.projects == []

    def test_set_container_fields(self):
        """Setting container fields should work."""
        container = ContainerMetadata(
            scitex_id="ABC12345",
            library_id="LIB001",
            projects=["project1", "project2"],
            readable_name="Smith-2023-Nature",
        )
        assert container.scitex_id == "ABC12345"
        assert len(container.projects) == 2
        assert container.readable_name == "Smith-2023-Nature"

    def test_negative_pdf_size_invalid(self):
        """Negative PDF size should raise ValidationError."""
        with pytest.raises(ValidationError):
            ContainerMetadata(pdf_size_bytes=-100)


class TestPaper:
    """Tests for Paper model."""

    def test_create_empty(self):
        """Creating empty Paper should work."""
        paper = Paper()
        assert paper.metadata is not None
        assert paper.container is not None

    def test_set_paper_metadata(self):
        """Setting paper metadata should work."""
        paper = Paper()
        paper.metadata.basic.title = "Test Paper"
        paper.metadata.basic.year = 2023
        paper.metadata.basic.authors = ["Smith, John"]

        assert paper.metadata.basic.title == "Test Paper"
        assert paper.metadata.basic.year == 2023

    def test_from_dict(self):
        """Creating Paper from dict should work."""
        data = {
            "metadata": {
                "basic": {
                    "title": "Test Paper",
                    "year": 2023,
                    "authors": ["Smith, John"],
                },
                "id": {"doi": "10.1234/test"},
            },
            "container": {"scitex_id": "ABC12345"},
        }

        paper = Paper.from_dict(data)

        assert paper.metadata.basic.title == "Test Paper"
        assert paper.metadata.basic.year == 2023
        assert paper.metadata.id.doi == "10.1234/test"
        assert paper.container.scitex_id == "ABC12345"

    def test_to_dict(self):
        """Converting Paper to dict should work."""
        paper = Paper()
        paper.metadata.basic.title = "Test Paper"
        paper.metadata.basic.year = 2023
        paper.container.scitex_id = "ABC12345"

        data = paper.to_dict()

        assert data["metadata"]["basic"]["title"] == "Test Paper"
        assert data["metadata"]["basic"]["year"] == 2023
        assert data["container"]["scitex_id"] == "ABC12345"

    def test_roundtrip_serialization(self):
        """Paper should survive serialization roundtrip."""
        paper = Paper()
        paper.metadata.basic.title = "Roundtrip Test"
        paper.metadata.basic.year = 2023
        paper.metadata.id.doi = "10.1234/roundtrip"
        paper.metadata.citation_count.total = 500
        paper.metadata.publication.journal = "Nature"
        paper.container.projects = ["test_project"]

        # Serialize and deserialize
        data = paper.to_dict()
        paper2 = Paper.from_dict(data)

        assert paper2.metadata.basic.title == paper.metadata.basic.title
        assert paper2.metadata.basic.year == paper.metadata.basic.year
        assert paper2.metadata.id.doi == paper.metadata.id.doi
        assert (
            paper2.metadata.citation_count.total == paper.metadata.citation_count.total
        )
        assert paper2.metadata.publication.journal == paper.metadata.publication.journal
        assert paper2.container.projects == paper.container.projects

    def test_model_dump_uses_aliases(self):
        """model_dump should use aliases for year fields in citations."""
        paper = Paper()
        paper.metadata.citation_count.y2024 = 100
        paper.metadata.citation_count.y2023 = 80

        data = paper.model_dump()

        # Citation counts should use numeric aliases
        assert "2024" in data["metadata"]["citation_count"]
        assert data["metadata"]["citation_count"]["2024"] == 100


class TestAccessMetadata:
    """Tests for AccessMetadata model."""

    def test_create_empty(self):
        """Creating empty AccessMetadata should work."""
        access = AccessMetadata()
        assert access.is_open_access is None
        assert access.oa_status is None

    def test_set_open_access_fields(self):
        """Setting open access fields should work."""
        access = AccessMetadata(
            is_open_access=True,
            oa_status="gold",
            oa_url="https://example.com/paper.pdf",
            license="CC-BY",
        )
        assert access.is_open_access is True
        assert access.oa_status == "gold"
        assert access.license == "CC-BY"


class TestSystemMetadata:
    """Tests for SystemMetadata model."""

    def test_create_empty(self):
        """Creating empty SystemMetadata should work."""
        system = SystemMetadata()
        assert system.searched_by_arXiv is None
        assert system.searched_by_CrossRef is None

    def test_set_search_flags(self):
        """Setting search flags should work."""
        system = SystemMetadata(
            searched_by_arXiv=True, searched_by_PubMed=True, searched_by_CrossRef=False
        )
        assert system.searched_by_arXiv is True
        assert system.searched_by_PubMed is True
        assert system.searched_by_CrossRef is False


class TestPathMetadata:
    """Tests for PathMetadata model."""

    def test_create_empty(self):
        """Creating empty PathMetadata should work."""
        path = PathMetadata()
        assert path.pdfs == []
        assert path.supplementary_files == []

    def test_set_paths(self):
        """Setting file paths should work."""
        path = PathMetadata(
            pdfs=["/path/to/paper.pdf"],
            supplementary_files=["/path/to/supp1.pdf", "/path/to/supp2.xlsx"],
        )
        assert len(path.pdfs) == 1
        assert len(path.supplementary_files) == 2


class TestValidationEdgeCases:
    """Tests for validation edge cases."""

    def test_year_none_valid(self):
        """None year should be valid."""
        basic = BasicMetadata(year=None)
        assert basic.year is None

    def test_impact_factor_none_valid(self):
        """None impact factor should be valid."""
        pub = PublicationMetadata(impact_factor=None)
        assert pub.impact_factor is None

    def test_citation_count_zero_valid(self):
        """Zero citation count should be valid."""
        citations = CitationCountMetadata(total=0)
        assert citations.total == 0

    def test_empty_authors_list_valid(self):
        """Empty authors list should be valid."""
        basic = BasicMetadata(authors=[])
        assert basic.authors == []

    def test_validate_assignment_enabled(self):
        """Validation should occur on assignment too."""
        basic = BasicMetadata()
        basic.year = 2023  # Valid
        assert basic.year == 2023

        with pytest.raises(ValidationError):
            basic.year = 1800  # Invalid


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
