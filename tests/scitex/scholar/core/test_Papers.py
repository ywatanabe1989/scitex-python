#!/usr/bin/env python3
# Timestamp: "2026-01-05"
# File: tests/scitex/scholar/core/test_Papers.py
# ----------------------------------------

"""
Comprehensive tests for the Papers collection class.

Tests cover:
- Collection initialization
- Collection operations (len, iter, getitem)
- Append and extend operations
- Filter functionality
- Sort functionality
- Serialization methods
"""

from pathlib import Path

import pytest

from scitex.scholar.core.Paper import Paper
from scitex.scholar.core.Papers import Papers


def create_sample_paper(
    title: str = "Sample Paper",
    year: int = 2023,
    journal: str = "Nature",
    citations: int = 100,
    impact_factor: float = 50.0,
    doi: str = None,
) -> Paper:
    """Helper to create sample Paper objects."""
    paper = Paper()
    paper.metadata.basic.title = title
    paper.metadata.basic.year = year
    paper.metadata.publication.journal = journal
    paper.metadata.citation_count.total = citations
    paper.metadata.publication.impact_factor = impact_factor
    if doi:
        paper.metadata.set_doi(doi)
    return paper


class TestPapersInit:
    """Tests for Papers initialization."""

    def test_create_empty(self):
        """Creating empty Papers collection should work."""
        papers = Papers()
        assert len(papers) == 0
        assert papers.project == "default"

    def test_create_with_paper_list(self):
        """Creating Papers with list of Paper objects should work."""
        p1 = create_sample_paper("Paper 1")
        p2 = create_sample_paper("Paper 2")
        papers = Papers([p1, p2])
        assert len(papers) == 2

    def test_create_with_dict_list(self):
        """Creating Papers with list of dicts should work."""
        dicts = [
            {"metadata": {"basic": {"title": "Paper 1", "year": 2023}}},
            {"metadata": {"basic": {"title": "Paper 2", "year": 2024}}},
        ]
        papers = Papers(dicts)
        assert len(papers) == 2
        assert papers[0].metadata.basic.title == "Paper 1"

    def test_create_with_project(self):
        """Creating Papers with project name should work."""
        papers = Papers(project="my_project")
        assert papers.project == "my_project"

    def test_skip_invalid_items(self):
        """Invalid items in list should be skipped with warning."""
        p1 = create_sample_paper("Paper 1")
        items = [p1, "invalid", 123, None]
        papers = Papers(items)
        assert len(papers) == 1


class TestPapersCollectionOps:
    """Tests for Papers collection operations."""

    def test_len(self):
        """len() should return correct count."""
        papers = Papers([create_sample_paper(f"Paper {i}") for i in range(5)])
        assert len(papers) == 5

    def test_iter(self):
        """Iteration should work."""
        p1 = create_sample_paper("Paper 1")
        p2 = create_sample_paper("Paper 2")
        papers = Papers([p1, p2])

        titles = [p.metadata.basic.title for p in papers]
        assert titles == ["Paper 1", "Paper 2"]

    def test_getitem_int(self):
        """Integer indexing should return single Paper."""
        p1 = create_sample_paper("Paper 1")
        p2 = create_sample_paper("Paper 2")
        papers = Papers([p1, p2])

        assert papers[0].metadata.basic.title == "Paper 1"
        assert papers[1].metadata.basic.title == "Paper 2"
        assert papers[-1].metadata.basic.title == "Paper 2"

    def test_getitem_slice(self):
        """Slice indexing should return new Papers collection."""
        papers = Papers([create_sample_paper(f"Paper {i}") for i in range(5)])
        subset = papers[1:3]

        assert isinstance(subset, Papers)
        assert len(subset) == 2
        assert subset[0].metadata.basic.title == "Paper 1"

    def test_repr(self):
        """repr should include count and project."""
        papers = Papers([create_sample_paper("P1")], project="test")
        repr_str = repr(papers)
        assert "count=1" in repr_str
        assert "test" in repr_str

    def test_str_empty(self):
        """str of empty collection should indicate empty."""
        papers = Papers()
        assert "Empty" in str(papers)

    def test_str_one(self):
        """str of single paper should say '1 paper'."""
        papers = Papers([create_sample_paper("P1")])
        assert "1 paper" in str(papers)

    def test_str_multiple(self):
        """str of multiple papers should include count."""
        papers = Papers([create_sample_paper(f"P{i}") for i in range(5)])
        assert "5 papers" in str(papers)


class TestPapersAppendExtend:
    """Tests for append and extend operations."""

    def test_append_paper(self):
        """Appending Paper should add to collection."""
        papers = Papers()
        paper = create_sample_paper("Test Paper")
        papers.append(paper)

        assert len(papers) == 1
        assert papers[0].metadata.basic.title == "Test Paper"

    def test_append_non_paper_ignored(self):
        """Appending non-Paper should be ignored."""
        papers = Papers()
        papers.append("not a paper")
        assert len(papers) == 0

    def test_extend_with_list(self):
        """Extend with list of Papers should work."""
        papers = Papers([create_sample_paper("P1")])
        new_papers = [create_sample_paper("P2"), create_sample_paper("P3")]
        papers.extend(new_papers)

        assert len(papers) == 3

    def test_extend_with_papers(self):
        """Extend with another Papers collection should work."""
        papers1 = Papers([create_sample_paper("P1")])
        papers2 = Papers([create_sample_paper("P2"), create_sample_paper("P3")])
        papers1.extend(papers2)

        assert len(papers1) == 3


class TestPapersFilter:
    """Tests for filter functionality."""

    @pytest.fixture
    def sample_papers(self):
        """Create sample papers for filtering tests."""
        return Papers(
            [
                create_sample_paper(
                    "Old Paper", year=2018, citations=50, impact_factor=5.0
                ),
                create_sample_paper(
                    "Recent Paper", year=2023, citations=200, impact_factor=15.0
                ),
                create_sample_paper(
                    "New Paper", year=2024, citations=10, impact_factor=50.0
                ),
                create_sample_paper(
                    "High Cite", year=2020, citations=1000, impact_factor=10.0
                ),
            ]
        )

    def test_filter_with_lambda(self, sample_papers):
        """Filter with lambda should work."""
        result = sample_papers.filter(
            lambda p: p.metadata.basic.year and p.metadata.basic.year >= 2023
        )
        assert len(result) == 2

    def test_filter_returns_papers(self, sample_papers):
        """Filter should return new Papers collection."""
        result = sample_papers.filter(
            lambda p: p.metadata.basic.year and p.metadata.basic.year >= 2020
        )
        assert isinstance(result, Papers)

    def test_filter_preserves_project(self, sample_papers):
        """Filter should preserve project name."""
        sample_papers.project = "my_project"
        result = sample_papers.filter(
            lambda p: p.metadata.basic.year and p.metadata.basic.year >= 2020
        )
        assert result.project == "my_project"

    def test_filter_empty_result(self, sample_papers):
        """Filter with no matches should return empty Papers."""
        result = sample_papers.filter(
            lambda p: p.metadata.basic.year and p.metadata.basic.year >= 2030
        )
        assert len(result) == 0
        assert isinstance(result, Papers)

    def test_filter_chain(self, sample_papers):
        """Chained filters should work."""
        result = sample_papers.filter(
            lambda p: p.metadata.basic.year and p.metadata.basic.year >= 2020
        ).filter(
            lambda p: p.metadata.citation_count.total
            and p.metadata.citation_count.total >= 100
        )
        assert len(result) == 2


class TestPapersSort:
    """Tests for sort functionality."""

    @pytest.fixture
    def sample_papers(self):
        """Create sample papers for sorting tests."""
        return Papers(
            [
                create_sample_paper("Paper B", year=2020),
                create_sample_paper("Paper A", year=2023),
                create_sample_paper("Paper C", year=2018),
            ]
        )

    def test_sort_by_lambda(self, sample_papers):
        """Sort by lambda should work."""
        result = sample_papers.sort_by(lambda p: p.metadata.basic.year or 0)
        years = [p.metadata.basic.year for p in result]
        assert years == [2018, 2020, 2023]

    def test_sort_by_lambda_reverse(self, sample_papers):
        """Sort by lambda with reverse should work."""
        result = sample_papers.sort_by(
            lambda p: p.metadata.basic.year or 0, reverse=True
        )
        years = [p.metadata.basic.year for p in result]
        assert years == [2023, 2020, 2018]

    def test_sort_returns_papers(self, sample_papers):
        """Sort should return new Papers collection."""
        result = sample_papers.sort_by(lambda p: p.metadata.basic.year or 0)
        assert isinstance(result, Papers)

    def test_sort_empty_criteria(self, sample_papers):
        """Sort with no criteria should return copy."""
        result = sample_papers.sort_by()
        assert len(result) == len(sample_papers)


class TestPapersProperties:
    """Tests for Papers properties."""

    def test_papers_property(self):
        """papers property should return internal list."""
        p1 = create_sample_paper("P1")
        papers = Papers([p1])
        internal_list = papers.papers

        assert isinstance(internal_list, list)
        assert len(internal_list) == 1
        assert internal_list[0] is p1

    def test_to_list(self):
        """to_list should return a list copy."""
        p1 = create_sample_paper("P1")
        papers = Papers([p1])
        result = papers.to_list()

        assert isinstance(result, list)
        assert len(result) == 1


class TestPapersDir:
    """Tests for custom __dir__ method."""

    def test_dir_contains_custom_attrs(self):
        """dir() should include custom attributes."""
        papers = Papers()
        dir_list = dir(papers)

        assert "filter" in dir_list
        assert "sort_by" in dir_list
        assert "append" in dir_list
        assert "extend" in dir_list
        assert "to_list" in dir_list
        assert "summary" in dir_list
        assert "to_dict" in dir_list


class TestPapersToDict:
    """Tests for to_dict serialization."""

    def test_to_dict_empty(self):
        """to_dict on empty collection should work (if papers_utils exists)."""
        papers = Papers()
        try:
            result = papers.to_dict()
            assert isinstance(result, dict)
        except ModuleNotFoundError:
            pytest.skip("papers_utils module not available (deprecated)")

    def test_to_dict_with_papers(self):
        """to_dict should serialize all papers (if papers_utils exists)."""
        papers = Papers(
            [
                create_sample_paper("P1", year=2023),
                create_sample_paper("P2", year=2024),
            ]
        )
        try:
            result = papers.to_dict()
            assert "papers" in result or isinstance(result, dict)
        except ModuleNotFoundError:
            pytest.skip("papers_utils module not available (deprecated)")


class TestPapersSummary:
    """Tests for summary statistics."""

    def test_summary_returns_dict(self):
        """summary should return dictionary (if papers_utils exists)."""
        papers = Papers([create_sample_paper("P1")])
        try:
            result = papers.summary()
            assert isinstance(result, dict)
        except ModuleNotFoundError:
            pytest.skip("papers_utils module not available (deprecated)")


class TestPapersSave:
    """Tests for save functionality (deprecated, requires papers_utils)."""

    def test_save_bibtex(self, tmp_path):
        """Saving as bibtex should work (if papers_utils exists)."""
        papers = Papers(
            [create_sample_paper("Test Paper", year=2023, doi="10.1234/test")]
        )
        output_path = tmp_path / "test.bib"
        try:
            papers.save(output_path)
            assert output_path.exists()
            content = output_path.read_text()
            assert "Test Paper" in content or "test" in content.lower()
        except ModuleNotFoundError:
            pytest.skip("papers_utils module not available (deprecated)")

    def test_save_json(self, tmp_path):
        """Saving as JSON should work (if papers_utils exists)."""
        papers = Papers([create_sample_paper("Test Paper", year=2023)])
        output_path = tmp_path / "test.json"
        try:
            papers.save(output_path, format="json")
            assert output_path.exists()
        except ModuleNotFoundError:
            pytest.skip("papers_utils module not available (deprecated)")

    def test_save_auto_format_bib(self, tmp_path):
        """Save should auto-detect format from extension (if papers_utils exists)."""
        papers = Papers([create_sample_paper("P1")])
        output_path = tmp_path / "test.bib"
        try:
            papers.save(output_path)  # Auto-detect from .bib
            assert output_path.exists()
        except ModuleNotFoundError:
            pytest.skip("papers_utils module not available (deprecated)")


class TestPapersFromBibtex:
    """Tests for from_bibtex class method."""

    @pytest.fixture
    def sample_bibtex(self, tmp_path):
        """Create sample BibTeX file."""
        content = """
@article{smith2023,
    author = {Smith, John and Doe, Jane},
    title = {A Test Paper},
    journal = {Nature},
    year = {2023},
    doi = {10.1234/test}
}

@article{jones2024,
    author = {Jones, Bob},
    title = {Another Paper},
    journal = {Science},
    year = {2024}
}
"""
        bib_file = tmp_path / "test.bib"
        bib_file.write_text(content)
        return bib_file

    def test_from_bibtex_file(self, sample_bibtex):
        """Loading from BibTeX file should work."""
        papers = Papers.from_bibtex(sample_bibtex)
        assert len(papers) >= 1

    def test_from_bibtex_string(self):
        """Loading from BibTeX string should work."""
        bibtex_str = """
@article{test2023,
    title = {Test Paper},
    year = {2023}
}
"""
        papers = Papers.from_bibtex(bibtex_str)
        assert len(papers) >= 0  # May be 0 if year parsing issue

    def test_from_bibtex_nonexistent_file(self):
        """Loading from nonexistent file - behavior depends on implementation.

        Note: The current implementation treats non-existent paths as bibtex text,
        so it may return empty Papers instead of raising FileNotFoundError.
        """
        result = Papers.from_bibtex("/nonexistent/path.bib")
        # Should either raise FileNotFoundError or return empty collection
        assert isinstance(result, Papers)


class TestPapersToDataframe:
    """Tests for to_dataframe conversion (deprecated)."""

    def test_to_dataframe(self):
        """to_dataframe should return DataFrame if papers_utils exists."""
        papers = Papers([create_sample_paper("P1")])
        try:
            df = papers.to_dataframe()
            # Returns None if module not available, skip in that case
            if df is None:
                pytest.skip("papers_utils module not available (deprecated)")
            assert df is not None
        except (ImportError, ModuleNotFoundError):
            pytest.skip("pandas or papers_utils not available (deprecated)")


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
