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

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/core/Paper.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2025-10-07 10:47:02 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/scholar/core/Paper.py
# # ----------------------------------------
# from __future__ import annotations
# 
# import os
# 
# __FILE__ = "./src/scitex/scholar/core/Paper.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Type-safe metadata structures for Scholar papers with runtime validation.
# 
# This module uses Pydantic for:
# - Runtime type validation
# - Automatic type coercion
# - JSON key aliasing (e.g., "2025" -> y2025)
# - Clean serialization/deserialization
# """
# 
# from typing import Any, Dict, List, Optional
# 
# from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
# 
# 
# class IDMetadata(BaseModel):
#     """Identification metadata with source tracking."""
# 
#     model_config = ConfigDict(populate_by_name=True, validate_assignment=True)
# 
#     doi: Optional[str] = None
#     doi_engines: List[str] = Field(default_factory=list)
# 
#     arxiv_id: Optional[str] = None
#     arxiv_id_engines: List[str] = Field(default_factory=list)
# 
#     pmid: Optional[str] = None
#     pmid_engines: List[str] = Field(default_factory=list)
# 
#     corpus_id: Optional[str] = None
#     corpus_id_engines: List[str] = Field(default_factory=list)
# 
#     semantic_id: Optional[str] = None
#     semantic_id_engines: List[str] = Field(default_factory=list)
# 
#     ieee_id: Optional[str] = None
#     ieee_id_engines: List[str] = Field(default_factory=list)
# 
#     scholar_id: Optional[str] = None
#     scholar_id_engines: List[str] = Field(default_factory=list)
# 
# 
# class BasicMetadata(BaseModel):
#     """Basic bibliographic metadata with source tracking."""
# 
#     title: Optional[str] = None
#     title_engines: List[str] = Field(default_factory=list)
# 
#     authors: Optional[List[str]] = None
#     authors_engines: List[str] = Field(default_factory=list)
# 
#     year: Optional[int] = None
#     year_engines: List[str] = Field(default_factory=list)
# 
#     abstract: Optional[str] = None
#     abstract_engines: List[str] = Field(default_factory=list)
# 
#     keywords: Optional[List[str]] = None
#     keywords_engines: List[str] = Field(default_factory=list)
# 
#     type: Optional[str] = None  # article, conference, preprint, etc.
#     type_engines: List[str] = Field(default_factory=list)
# 
#     @field_validator("year")
#     @classmethod
#     def validate_year(cls, v):
#         """Validate year is reasonable."""
#         if v is not None and (v < 1900 or v > 2100):
#             raise ValueError(f"Year {v} is outside reasonable range (1900-2100)")
#         return v
# 
#     model_config = ConfigDict(populate_by_name=True, validate_assignment=True)
# 
# 
# class CitationCountMetadata(BaseModel):
#     """Citation count metadata with yearly breakdown and source tracking."""
# 
#     total: Optional[int] = None
#     total_engines: List[str] = Field(default_factory=list)
# 
#     # Yearly counts - use Field(alias=...) to map JSON "2025" to Python y2025
#     y2025: Optional[int] = Field(None, alias="2025")
#     y2025_engines: List[str] = Field(default_factory=list, alias="2025_engines")
# 
#     y2024: Optional[int] = Field(None, alias="2024")
#     y2024_engines: List[str] = Field(default_factory=list, alias="2024_engines")
# 
#     y2023: Optional[int] = Field(None, alias="2023")
#     y2023_engines: List[str] = Field(default_factory=list, alias="2023_engines")
# 
#     y2022: Optional[int] = Field(None, alias="2022")
#     y2022_engines: List[str] = Field(default_factory=list, alias="2022_engines")
# 
#     y2021: Optional[int] = Field(None, alias="2021")
#     y2021_engines: List[str] = Field(default_factory=list, alias="2021_engines")
# 
#     y2020: Optional[int] = Field(None, alias="2020")
#     y2020_engines: List[str] = Field(default_factory=list, alias="2020_engines")
# 
#     y2019: Optional[int] = Field(None, alias="2019")
#     y2019_engines: List[str] = Field(default_factory=list, alias="2019_engines")
# 
#     y2018: Optional[int] = Field(None, alias="2018")
#     y2018_engines: List[str] = Field(default_factory=list, alias="2018_engines")
# 
#     y2017: Optional[int] = Field(None, alias="2017")
#     y2017_engines: List[str] = Field(default_factory=list, alias="2017_engines")
# 
#     y2016: Optional[int] = Field(None, alias="2016")
#     y2016_engines: List[str] = Field(default_factory=list, alias="2016_engines")
# 
#     y2015: Optional[int] = Field(None, alias="2015")
#     y2015_engines: List[str] = Field(default_factory=list, alias="2015_engines")
# 
#     @field_validator(
#         "total",
#         "y2025",
#         "y2024",
#         "y2023",
#         "y2022",
#         "y2021",
#         "y2020",
#         "y2019",
#         "y2018",
#         "y2017",
#         "y2016",
#         "y2015",
#     )
#     @classmethod
#     def validate_citation_counts(cls, v):
#         """Validate citation counts are non-negative."""
#         if v is not None and v < 0:
#             raise ValueError(f"Citation count cannot be negative: {v}")
#         return v
# 
#     model_config = ConfigDict(populate_by_name=True, validate_assignment=True)
# 
#     def model_dump(self, **kwargs) -> Dict[str, Any]:
#         """Custom serialization to use aliases in output."""
#         # Remove by_alias from kwargs if present to avoid duplicate
#         kwargs.pop("by_alias", None)
#         data = super().model_dump(by_alias=True, **kwargs)
#         return data
# 
# 
# class PublicationMetadata(BaseModel):
#     """Publication venue metadata with source tracking."""
# 
#     journal: Optional[str] = None
#     journal_engines: List[str] = Field(default_factory=list)
# 
#     short_journal: Optional[str] = None
#     short_journal_engines: List[str] = Field(default_factory=list)
# 
#     impact_factor: Optional[float] = None
#     impact_factor_engines: List[str] = Field(default_factory=list)
# 
#     issn: Optional[str] = None
#     issn_engines: List[str] = Field(default_factory=list)
# 
#     volume: Optional[str] = None
#     volume_engines: List[str] = Field(default_factory=list)
# 
#     issue: Optional[str] = None
#     issue_engines: List[str] = Field(default_factory=list)
# 
#     first_page: Optional[str] = None
#     first_page_engines: List[str] = Field(default_factory=list)
# 
#     last_page: Optional[str] = None
#     last_page_engines: List[str] = Field(default_factory=list)
# 
#     pages: Optional[str] = None
#     pages_engines: List[str] = Field(default_factory=list)
# 
#     publisher: Optional[str] = None
#     publisher_engines: List[str] = Field(default_factory=list)
# 
#     @field_validator("impact_factor")
#     @classmethod
#     def validate_impact_factor(cls, v):
#         """Validate impact factor is non-negative."""
#         if v is not None and v < 0:
#             raise ValueError(f"Impact factor cannot be negative: {v}")
#         return v
# 
#     model_config = ConfigDict(populate_by_name=True, validate_assignment=True)
# 
# 
# class URLMetadata(BaseModel):
#     """URL metadata with source tracking."""
# 
#     doi: Optional[str] = None
#     doi_engines: List[str] = Field(default_factory=list)
# 
#     publisher: Optional[str] = None
#     publisher_engines: List[str] = Field(default_factory=list)
# 
#     arxiv: Optional[str] = None
#     arxiv_engines: List[str] = Field(default_factory=list)
# 
#     corpus_id: Optional[str] = None
#     corpus_id_engines: List[str] = Field(default_factory=list)
# 
#     openurl_query: Optional[str] = None
#     openurl_engines: List[str] = Field(default_factory=list)
# 
#     openurl_resolved: List[str] = Field(default_factory=list)
#     openurl_resolved_engines: List[str] = Field(default_factory=list)
# 
#     pdfs: List[Dict[str, str]] = Field(default_factory=list)
#     pdfs_engines: List[str] = Field(default_factory=list)
# 
#     supplementary_files: List[str] = Field(default_factory=list)
#     supplementary_files_engines: List[str] = Field(default_factory=list)
# 
#     additional_files: List[str] = Field(default_factory=list)
#     additional_files_engines: List[str] = Field(default_factory=list)
# 
#     model_config = ConfigDict(populate_by_name=True, validate_assignment=True)
# 
# 
# class PathMetadata(BaseModel):
#     """Local file path metadata with source tracking."""
# 
#     pdfs: List[str] = Field(default_factory=list)
#     pdfs_engines: List[str] = Field(default_factory=list)
# 
#     supplementary_files: List[str] = Field(default_factory=list)
#     supplementary_files_engines: List[str] = Field(default_factory=list)
# 
#     additional_files: List[str] = Field(default_factory=list)
#     additional_files_engines: List[str] = Field(default_factory=list)
# 
#     model_config = ConfigDict(populate_by_name=True, validate_assignment=True)
# 
# 
# class AccessMetadata(BaseModel):
#     """Open access and licensing metadata with source tracking.
# 
#     Tracks whether a paper is open access and provides URLs for OA versions.
#     Also includes license information when available.
#     """
# 
#     is_open_access: Optional[bool] = None
#     is_open_access_engines: List[str] = Field(default_factory=list)
# 
#     oa_status: Optional[str] = None  # gold, green, bronze, hybrid, closed
#     oa_status_engines: List[str] = Field(default_factory=list)
# 
#     oa_url: Optional[str] = None  # URL to open access version
#     oa_url_engines: List[str] = Field(default_factory=list)
# 
#     license: Optional[str] = None  # CC-BY, CC-BY-NC, etc.
#     license_engines: List[str] = Field(default_factory=list)
# 
#     license_url: Optional[str] = None
#     license_url_engines: List[str] = Field(default_factory=list)
# 
#     # For paywalled journals - opt-in for local/personal users
#     paywall_bypass_attempted: Optional[bool] = None
#     paywall_bypass_success: Optional[bool] = None
# 
#     model_config = ConfigDict(populate_by_name=True, validate_assignment=True)
# 
# 
# class SystemMetadata(BaseModel):
#     """System tracking metadata (which engines were used to search)."""
# 
#     searched_by_arXiv: Optional[bool] = None
#     searched_by_CrossRef: Optional[bool] = None
#     searched_by_CrossRefLocal: Optional[bool] = None
#     searched_by_OpenAlex: Optional[bool] = None
#     searched_by_PubMed: Optional[bool] = None
#     searched_by_Semantic_Scholar: Optional[bool] = None
#     searched_by_URL: Optional[bool] = None
# 
#     model_config = ConfigDict(populate_by_name=True, validate_assignment=True)
# 
# 
# class PaperMetadataStructure(BaseModel):
#     """Complete paper metadata structure with nested typed sections."""
# 
#     id: IDMetadata = Field(default_factory=IDMetadata)
#     basic: BasicMetadata = Field(default_factory=BasicMetadata)
#     citation_count: CitationCountMetadata = Field(default_factory=CitationCountMetadata)
#     publication: PublicationMetadata = Field(default_factory=PublicationMetadata)
#     url: URLMetadata = Field(default_factory=URLMetadata)
#     path: PathMetadata = Field(default_factory=PathMetadata)
#     access: AccessMetadata = Field(default_factory=AccessMetadata)
#     system: SystemMetadata = Field(default_factory=SystemMetadata)
# 
#     model_config = ConfigDict(populate_by_name=True, validate_assignment=True)
# 
#     @model_validator(mode="after")
#     def sync_ids_and_urls(self):
#         """Automatically sync ID and URL fields with source tracking.
# 
#         Generates URLs from IDs and vice versa for:
#         - DOI ↔ url.doi
#         - arXiv ID ↔ url.arxiv
#         - Corpus ID ↔ url.corpus_id
#         """
#         # DOI sync
#         if self.id.doi and not self.url.doi:
#             self.url.doi = f"https://doi.org/{self.id.doi}"
#             if (
#                 self.id.doi_engines
#                 and "PaperMetadataStructure" not in self.url.doi_engines
#             ):
#                 self.url.doi_engines = (
#                     self.id.doi_engines.copy() if self.id.doi_engines else []
#                 )
#                 if "PaperMetadataStructure" not in self.url.doi_engines:
#                     self.url.doi_engines.append("PaperMetadataStructure")
#         elif self.url.doi and not self.id.doi:
#             url = self.url.doi
#             if "doi.org/" in url:
#                 self.id.doi = url.split("doi.org/")[-1]
#                 if (
#                     self.url.doi_engines
#                     and "PaperMetadataStructure" not in self.id.doi_engines
#                 ):
#                     self.id.doi_engines = (
#                         self.url.doi_engines.copy() if self.url.doi_engines else []
#                     )
#                     if "PaperMetadataStructure" not in self.id.doi_engines:
#                         self.id.doi_engines.append("PaperMetadataStructure")
#         elif self.id.doi and self.url.doi:
#             if not self.url.doi.startswith("https://"):
#                 if self.url.doi.startswith("http://"):
#                     self.url.doi = "https://" + self.url.doi[7:]
#                 else:
#                     self.url.doi = f"https://doi.org/{self.id.doi}"
# 
#         # arXiv sync
#         if self.id.arxiv_id and not self.url.arxiv:
#             self.url.arxiv = f"https://arxiv.org/abs/{self.id.arxiv_id}"
#             if (
#                 self.id.arxiv_id_engines
#                 and "PaperMetadataStructure" not in self.url.arxiv_engines
#             ):
#                 self.url.arxiv_engines = (
#                     self.id.arxiv_id_engines.copy() if self.id.arxiv_id_engines else []
#                 )
#                 if "PaperMetadataStructure" not in self.url.arxiv_engines:
#                     self.url.arxiv_engines.append("PaperMetadataStructure")
#         elif self.url.arxiv and not self.id.arxiv_id:
#             url = self.url.arxiv
#             if "arxiv.org/abs/" in url:
#                 self.id.arxiv_id = (
#                     url.split("arxiv.org/abs/")[-1].split("?")[0].split("#")[0]
#                 )
#                 if (
#                     self.url.arxiv_engines
#                     and "PaperMetadataStructure" not in self.id.arxiv_id_engines
#                 ):
#                     self.id.arxiv_id_engines = (
#                         self.url.arxiv_engines.copy() if self.url.arxiv_engines else []
#                     )
#                     if "PaperMetadataStructure" not in self.id.arxiv_id_engines:
#                         self.id.arxiv_id_engines.append("PaperMetadataStructure")
# 
#         # Corpus ID sync
#         if self.id.corpus_id and not self.url.corpus_id:
#             corpus_id_clean = str(self.id.corpus_id).replace("CorpusId:", "")
#             self.url.corpus_id = (
#                 f"https://www.semanticscholar.org/paper/{corpus_id_clean}"
#             )
#             if (
#                 self.id.corpus_id_engines
#                 and "PaperMetadataStructure" not in self.url.corpus_id_engines
#             ):
#                 self.url.corpus_id_engines = (
#                     self.id.corpus_id_engines.copy()
#                     if self.id.corpus_id_engines
#                     else []
#                 )
#                 if "PaperMetadataStructure" not in self.url.corpus_id_engines:
#                     self.url.corpus_id_engines.append("PaperMetadataStructure")
#         elif self.url.corpus_id and not self.id.corpus_id:
#             url = self.url.corpus_id
#             if "semanticscholar.org/paper/" in url:
#                 self.id.corpus_id = (
#                     url.split("semanticscholar.org/paper/")[-1]
#                     .split("?")[0]
#                     .split("#")[0]
#                 )
#                 if (
#                     self.url.corpus_id_engines
#                     and "PaperMetadataStructure" not in self.id.corpus_id_engines
#                 ):
#                     self.id.corpus_id_engines = (
#                         self.url.corpus_id_engines.copy()
#                         if self.url.corpus_id_engines
#                         else []
#                     )
#                     if "PaperMetadataStructure" not in self.id.corpus_id_engines:
#                         self.id.corpus_id_engines.append("PaperMetadataStructure")
# 
#         return self
# 
#     def set_doi(self, doi: str):
#         """Set DOI and automatically sync URL.
# 
#         Use this method instead of direct assignment for automatic sync.
#         """
#         self.id.doi = doi
#         if doi:
#             self.url.doi = f"https://doi.org/{doi}"
# 
#     def set_doi_url(self, url: str):
#         """Set DOI URL and automatically extract/sync DOI.
# 
#         Use this method instead of direct assignment for automatic sync.
#         """
#         self.url.doi = url
#         if url and "doi.org/" in url:
#             self.id.doi = url.split("doi.org/")[-1]
# 
#     def model_dump(self, **kwargs) -> Dict[str, Any]:
#         """Custom serialization to ensure nested models use aliases."""
#         # Remove by_alias from kwargs if present to avoid duplicate
#         kwargs.pop("by_alias", None)
#         return {
#             "id": self.id.model_dump(by_alias=True, **kwargs),
#             "basic": self.basic.model_dump(by_alias=True, **kwargs),
#             "citation_count": self.citation_count.model_dump(by_alias=True, **kwargs),
#             "publication": self.publication.model_dump(by_alias=True, **kwargs),
#             "url": self.url.model_dump(by_alias=True, **kwargs),
#             "path": self.path.model_dump(by_alias=True, **kwargs),
#             "access": self.access.model_dump(by_alias=True, **kwargs),
#             "system": self.system.model_dump(by_alias=True, **kwargs),
#         }
# 
# 
# class ContainerMetadata(BaseModel):
#     """Container metadata for system tracking."""
# 
#     scitex_id: Optional[str] = None
#     library_id: Optional[str] = None
#     created_at: Optional[str] = None
#     created_by: Optional[str] = None
#     updated_at: Optional[str] = None
#     projects: List[str] = Field(default_factory=list)
#     master_storage_path: Optional[str] = None
#     readable_name: Optional[str] = None
#     metadata_file: Optional[str] = None
#     pdf_downloaded_at: Optional[str] = None
#     pdf_size_bytes: Optional[int] = None
# 
#     @field_validator("pdf_size_bytes")
#     @classmethod
#     def validate_pdf_size(cls, v):
#         """Validate PDF size is non-negative."""
#         if v is not None and v < 0:
#             raise ValueError(f"PDF size cannot be negative: {v}")
#         return v
# 
#     model_config = ConfigDict(populate_by_name=True, validate_assignment=True)
# 
# 
# class Paper(BaseModel):
#     """Complete paper with metadata and container."""
# 
#     metadata: PaperMetadataStructure = Field(default_factory=PaperMetadataStructure)
#     container: ContainerMetadata = Field(default_factory=ContainerMetadata)
# 
#     model_config = ConfigDict(populate_by_name=True, validate_assignment=True)
# 
#     def model_dump(self, **kwargs) -> Dict[str, Any]:
#         """Custom serialization to ensure all nested models use aliases."""
#         # Remove by_alias from kwargs if present to avoid duplicate
#         kwargs.pop("by_alias", None)
#         return {
#             "metadata": self.metadata.model_dump(by_alias=True, **kwargs),
#             "container": self.container.model_dump(by_alias=True, **kwargs),
#         }
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> Paper:
#         """Create from dictionary (for loading from JSON).
# 
#         Uses Pydantic's model_validate which handles:
#         - Type validation
#         - Type coercion (e.g., "2024" -> 2024)
#         - Field aliases (e.g., "2025" -> y2025)
#         """
#         return cls.model_validate(data)
# 
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary for JSON serialization.
# 
#         Alias for model_dump() for backward compatibility.
#         """
#         return self.model_dump()
# 
#     def detect_open_access(
#         self,
#         use_unpaywall: bool = False,
#         update_metadata: bool = True,
#     ) -> OAResult:
#         """
#         Detect open access status for this paper.
# 
#         Uses identifiers (DOI, arXiv ID, PMCID) and known OA sources
#         to determine if the paper is freely available.
# 
#         Args:
#             use_unpaywall: If True, query Unpaywall API for uncertain cases
#             update_metadata: If True, update self.metadata.access with results
# 
#         Returns
#         -------
#             OAResult with detection results
#         """
#         from .open_access import check_oa_status
# 
#         result = check_oa_status(
#             doi=self.metadata.id.doi,
#             arxiv_id=self.metadata.id.arxiv_id,
#             pmcid=None,  # Not currently in IDMetadata
#             source=None,  # Source tracking not in Paper
#             journal=self.metadata.publication.journal,
#             is_open_access_flag=self.metadata.access.is_open_access,
#             use_unpaywall=use_unpaywall,
#         )
# 
#         if update_metadata:
#             self.metadata.access.is_open_access = result.is_open_access
#             self.metadata.access.is_open_access_engines.append(
#                 f"detect_oa:{result.source}"
#             )
#             if result.status:
#                 self.metadata.access.oa_status = result.status.value
#                 self.metadata.access.oa_status_engines.append(
#                     f"detect_oa:{result.source}"
#                 )
#             if result.oa_url:
#                 self.metadata.access.oa_url = result.oa_url
#                 self.metadata.access.oa_url_engines.append(f"detect_oa:{result.source}")
#             if result.license:
#                 self.metadata.access.license = result.license
#                 self.metadata.access.license_engines.append(
#                     f"detect_oa:{result.source}"
#                 )
# 
#         return result
# 
#     @property
#     def is_open_access(self) -> bool:
#         """Check if paper is open access (quick check without API calls)."""
#         if self.metadata.access.is_open_access is not None:
#             return self.metadata.access.is_open_access
# 
#         # Quick detection from identifiers
#         from .open_access import detect_oa_from_identifiers
# 
#         result = detect_oa_from_identifiers(
#             doi=self.metadata.id.doi,
#             arxiv_id=self.metadata.id.arxiv_id,
#             journal=self.metadata.publication.journal,
#         )
#         return result.is_open_access
# 
# 
# if __name__ == "__main__":
#     import json
# 
#     print("=" * 80)
#     print("Paper Class - Pydantic Type-Safe Metadata with Runtime Validation")
#     print("=" * 80)
# 
#     # 1. Create empty paper
#     print("\n1. Create empty Paper:")
#     paper = Paper()
#     print(f"   Empty paper created: {type(paper).__name__}")
# 
#     # 2. Set basic metadata
#     print("\n2. Set basic metadata:")
#     paper.metadata.basic.title = "Attention Is All You Need"
#     paper.metadata.basic.authors = [
#         "Vaswani, Ashish",
#         "Shazeer, Noam",
#         "Parmar, Niki",
#     ]
#     paper.metadata.basic.year = 2017
#     paper.metadata.basic.abstract = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks."
#     paper.metadata.basic.keywords = [
#         "transformer",
#         "attention",
#         "neural networks",
#     ]
#     print(f"   Title: {paper.metadata.basic.title}")
#     print(f"   Authors: {paper.metadata.basic.authors[:2]}...")
#     print(f"   Year: {paper.metadata.basic.year}")
# 
#     # 3. Set DOI (auto-syncs URL)
#     print("\n3. Set DOI (auto-syncs DOI URL):")
#     paper.metadata.set_doi("10.48550/arXiv.1706.03762")
#     print(f"   DOI: {paper.metadata.id.doi}")
#     print(f"   DOI URL (auto-synced): {paper.metadata.url.doi}")
# 
#     # 4. Set publication details
#     print("\n4. Set publication details:")
#     paper.metadata.publication.journal = "NeurIPS"
#     paper.metadata.publication.volume = "30"
#     paper.metadata.publication.impact_factor = 12.345
#     print(f"   Journal: {paper.metadata.publication.journal}")
#     print(f"   Volume: {paper.metadata.publication.volume}")
#     print(f"   Impact Factor: {paper.metadata.publication.impact_factor}")
# 
#     # 5. Set citation counts with year breakdown
#     print("\n5. Set citation counts:")
#     paper.metadata.citation_count.total = 85432
#     paper.metadata.citation_count.y2024 = 15234
#     paper.metadata.citation_count.y2023 = 18765
#     print(f"   Total citations: {paper.metadata.citation_count.total}")
#     print(f"   2024 citations: {paper.metadata.citation_count.y2024}")
#     print(f"   2023 citations: {paper.metadata.citation_count.y2023}")
# 
#     # 6. Set container metadata
#     print("\n6. Set container metadata:")
#     paper.container.projects = ["transformers_research", "nlp_2024"]
#     paper.container.library_id = "ABC12345"
#     paper.container.readable_name = "Vaswani-2017-NeurIPS"
#     print(f"   Projects: {paper.container.projects}")
#     print(f"   Library ID: {paper.container.library_id}")
#     print(f"   Readable name: {paper.container.readable_name}")
# 
#     # 7. Demonstrate type validation
#     print("\n7. Type validation (validate_assignment=True):")
#     print("   ✓ Automatic type coercion: year='2017' -> 2017 (int)")
#     paper.metadata.basic.year = "2017"  # String coerced to int
#     print(
#         f"     Result: {paper.metadata.basic.year} (type: {type(paper.metadata.basic.year).__name__})"
#     )
# 
#     print("   ✓ Range validation: year must be 1900-2100")
#     try:
#         paper.metadata.basic.year = 1800  # Too old
#         print("     ERROR: Should have raised ValidationError")
#     except Exception as e:
#         print(f"     Correctly rejected: {type(e).__name__}")
# 
#     print("   ✓ Non-negative validation: citations cannot be negative")
#     try:
#         paper.metadata.citation_count.total = -100
#         print("     ERROR: Should have raised ValidationError")
#     except Exception as e:
#         print(f"     Correctly rejected: {type(e).__name__}")
# 
#     # Reset to valid value
#     paper.metadata.basic.year = 2017
#     paper.metadata.citation_count.total = 85432
# 
#     # 8. Serialize to JSON (with aliases)
#     print("\n8. Serialize to JSON with field aliases:")
#     paper_dict = paper.to_dict()
#     print("   Year fields use numeric keys in JSON:")
#     print(f"     '2024': {paper_dict['metadata']['citation_count'].get('2024')}")
#     print(f"     '2023': {paper_dict['metadata']['citation_count'].get('2023')}")
# 
#     # 9. Create from dictionary
#     print("\n9. Load from dictionary (from_dict):")
#     sample_data = {
#         "metadata": {
#             "basic": {
#                 "title": "BERT: Pre-training of Deep Bidirectional Transformers",
#                 "year": 2019,
#             },
#             "id": {"doi": "10.18653/v1/N19-1423"},
#             "citation_count": {
#                 "2024": 5678,  # Numeric key maps to y2024
#                 "total": 45000,
#             },
#         }
#     }
# 
#     paper2 = Paper.from_dict(sample_data)
#     print(f"   Title: {paper2.metadata.basic.title}")
#     print(f"   Year: {paper2.metadata.basic.year}")
#     print(f"   DOI: {paper2.metadata.id.doi}")
#     print(f"   DOI URL (auto-synced): {paper2.metadata.url.doi}")
#     print(f"   2024 citations: {paper2.metadata.citation_count.y2024}")
# 
#     # 10. Show JSON structure
#     print("\n10. Full JSON structure (first 500 chars):")
#     json_str = json.dumps(paper.to_dict(), indent=2)
#     print(f"   {json_str[:500]}...")
# 
#     print("\n" + "=" * 80)
#     print("✅ Paper class demonstration complete!")
#     print("=" * 80)
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/core/Paper.py
# --------------------------------------------------------------------------------
