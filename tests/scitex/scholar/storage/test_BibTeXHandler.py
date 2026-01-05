#!/usr/bin/env python3
# Timestamp: "2026-01-05"
# File: tests/scitex/scholar/storage/test_BibTeXHandler.py
# ----------------------------------------

"""
Comprehensive tests for BibTeXHandler class.

Tests cover:
- BibTeX parsing from files and strings
- Paper to BibTeX conversion
- BibTeX to Paper conversion
- Title normalization and duplicate detection
- Metadata merging
"""

from pathlib import Path

import pytest

from scitex.scholar.core.Paper import Paper
from scitex.scholar.storage.BibTeXHandler import BibTeXHandler


class TestBibTeXHandlerInit:
    """Tests for BibTeXHandler initialization."""

    def test_create_default(self):
        """Creating BibTeXHandler with defaults should work."""
        handler = BibTeXHandler()
        assert handler.name == "BibTeXHandler"
        assert handler.project is None
        assert handler.config is None

    def test_create_with_project(self):
        """Creating BibTeXHandler with project should work."""
        handler = BibTeXHandler(project="my_project")
        assert handler.project == "my_project"


class TestBibTeXHandlerExtractPrimitive:
    """Tests for _extract_primitive method."""

    def test_extract_none(self):
        """None should return None."""
        handler = BibTeXHandler()
        assert handler._extract_primitive(None) is None

    def test_extract_string(self):
        """String should return as-is."""
        handler = BibTeXHandler()
        assert handler._extract_primitive("test") == "test"

    def test_extract_int(self):
        """Integer should return as-is."""
        handler = BibTeXHandler()
        assert handler._extract_primitive(42) == 42

    def test_extract_dict(self):
        """Dict should return as-is."""
        handler = BibTeXHandler()
        data = {"key": "value"}
        assert handler._extract_primitive(data) == data


class TestBibTeXHandlerPapersFromBibtex:
    """Tests for papers_from_bibtex method."""

    @pytest.fixture
    def sample_bibtex_file(self, tmp_path):
        """Create sample BibTeX file."""
        content = """
@article{smith2023test,
    author = {Smith, John and Doe, Jane},
    title = {A Test Paper Title},
    journal = {Nature},
    year = {2023},
    doi = {10.1234/test.2023}
}

@article{jones2024example,
    author = {Jones, Bob},
    title = {Another Example Paper},
    journal = {Science},
    year = {2024},
    abstract = {This is an abstract.}
}
"""
        bib_file = tmp_path / "test.bib"
        bib_file.write_text(content)
        return bib_file

    def test_papers_from_bibtex_file(self, sample_bibtex_file):
        """Loading papers from BibTeX file should work."""
        handler = BibTeXHandler()
        papers = handler.papers_from_bibtex(sample_bibtex_file)

        # Should create papers from valid entries
        assert isinstance(papers, list)
        # Note: actual count depends on scitex.io.load behavior

    def test_papers_from_bibtex_string(self):
        """Loading papers from BibTeX string should work."""
        handler = BibTeXHandler()
        bibtex_str = """
@article{test2023,
    author = {Test, Author},
    title = {Test Paper},
    year = {2023}
}
"""
        papers = handler.papers_from_bibtex(bibtex_str)
        assert isinstance(papers, list)

    def test_papers_from_bibtex_nonexistent_file(self):
        """Loading from nonexistent file should raise error."""
        handler = BibTeXHandler()
        with pytest.raises(ValueError, match="not found"):
            handler.papers_from_bibtex("/nonexistent/path.bib")


class TestBibTeXHandlerPaperFromEntry:
    """Tests for paper_from_bibtex_entry method."""

    def test_convert_basic_entry(self):
        """Converting basic BibTeX entry should work."""
        handler = BibTeXHandler(project="test_project")
        entry = {
            "entry_type": "article",
            "key": "smith2023",
            "fields": {
                "title": "Test Paper",
                "author": "Smith, John and Doe, Jane",
                "year": "2023",
                "journal": "Nature",
                "doi": "10.1234/test",
            },
        }

        paper = handler.paper_from_bibtex_entry(entry)

        assert paper is not None
        assert paper.metadata.basic.title == "Test Paper"
        assert paper.metadata.basic.year == 2023
        assert paper.metadata.id.doi == "10.1234/test"
        assert paper.metadata.publication.journal == "Nature"
        assert len(paper.metadata.basic.authors) == 2
        assert "Smith, John" in paper.metadata.basic.authors

    def test_convert_entry_no_title(self):
        """Entry without title should return None."""
        handler = BibTeXHandler()
        entry = {
            "entry_type": "article",
            "key": "noname2023",
            "fields": {"author": "Someone", "year": "2023"},
        }

        paper = handler.paper_from_bibtex_entry(entry)
        assert paper is None

    def test_convert_entry_with_abstract(self):
        """Entry with abstract should be parsed."""
        handler = BibTeXHandler()
        entry = {
            "entry_type": "article",
            "key": "test2023",
            "fields": {
                "title": "Test Paper",
                "abstract": "This is the abstract text.",
                "year": "2023",
            },
        }

        paper = handler.paper_from_bibtex_entry(entry)

        assert paper is not None
        assert paper.metadata.basic.abstract == "This is the abstract text."

    def test_convert_entry_with_arxiv(self):
        """Entry with arXiv ID should be parsed."""
        handler = BibTeXHandler()
        entry = {
            "entry_type": "article",
            "key": "arxiv2023",
            "fields": {"title": "ArXiv Paper", "eprint": "2301.12345", "year": "2023"},
        }

        paper = handler.paper_from_bibtex_entry(entry)

        assert paper is not None
        assert paper.metadata.id.arxiv_id == "2301.12345"

    def test_convert_entry_with_keywords(self):
        """Entry with keywords should be parsed."""
        handler = BibTeXHandler()
        entry = {
            "entry_type": "article",
            "key": "test2023",
            "fields": {
                "title": "Test Paper",
                "keywords": "machine learning, deep learning, AI",
                "year": "2023",
            },
        }

        paper = handler.paper_from_bibtex_entry(entry)

        assert paper is not None
        assert len(paper.metadata.basic.keywords) == 3
        assert "machine learning" in paper.metadata.basic.keywords

    def test_convert_entry_with_corpus_id(self):
        """Entry with Semantic Scholar corpus ID should be parsed."""
        handler = BibTeXHandler()
        entry = {
            "entry_type": "article",
            "key": "test2023",
            "fields": {
                "title": "Test Paper",
                "url": "https://www.semanticscholar.org/paper/CorpusId:12345678",
                "year": "2023",
            },
        }

        paper = handler.paper_from_bibtex_entry(entry)

        assert paper is not None
        assert paper.metadata.id.corpus_id == "12345678"

    def test_convert_entry_project_association(self):
        """Entry should be associated with handler's project."""
        handler = BibTeXHandler(project="my_project")
        entry = {
            "entry_type": "article",
            "key": "test2023",
            "fields": {"title": "Test Paper", "year": "2023"},
        }

        paper = handler.paper_from_bibtex_entry(entry)

        assert paper is not None
        assert "my_project" in paper.container.projects


class TestBibTeXHandlerPaperToBibtex:
    """Tests for paper_to_bibtex_entry method."""

    def test_convert_paper_to_entry(self):
        """Converting Paper to BibTeX entry should work."""
        handler = BibTeXHandler()

        paper = Paper()
        paper.metadata.basic.title = "Test Paper"
        paper.metadata.basic.year = 2023
        paper.metadata.basic.authors = ["Smith, John", "Doe, Jane"]
        paper.metadata.id.doi = "10.1234/test"
        paper.metadata.publication.journal = "Nature"

        entry = handler.paper_to_bibtex_entry(paper)

        assert entry["entry_type"] == "article"
        # Key is created from first author's last name and year
        assert "2023" in entry["key"]
        assert entry["fields"]["title"] == "Test Paper"
        assert entry["fields"]["year"] == "2023"
        assert entry["fields"]["doi"] == "10.1234/test"
        assert entry["fields"]["journal"] == "Nature"
        assert "Smith, John" in entry["fields"]["author"]

    def test_convert_paper_without_journal(self):
        """Paper without journal should use misc entry type."""
        handler = BibTeXHandler()

        paper = Paper()
        paper.metadata.basic.title = "Test Paper"
        paper.metadata.basic.year = 2023
        paper.metadata.basic.authors = ["Author, Test"]

        entry = handler.paper_to_bibtex_entry(paper)

        assert entry["entry_type"] == "misc"

    def test_convert_paper_with_citation_count(self):
        """Paper with citations should include in entry."""
        handler = BibTeXHandler()

        paper = Paper()
        paper.metadata.basic.title = "Cited Paper"
        paper.metadata.basic.year = 2023
        paper.metadata.citation_count.total = 500

        entry = handler.paper_to_bibtex_entry(paper)

        assert "citation_count" in entry["fields"]
        assert entry["fields"]["citation_count"] == "500"

    def test_convert_paper_with_impact_factor(self):
        """Paper with impact factor should include in entry."""
        handler = BibTeXHandler()

        paper = Paper()
        paper.metadata.basic.title = "IF Paper"
        paper.metadata.basic.year = 2023
        paper.metadata.publication.impact_factor = 15.5

        entry = handler.paper_to_bibtex_entry(paper)

        assert "journal_impact_factor" in entry["fields"]
        assert entry["fields"]["journal_impact_factor"] == "15.5"


class TestBibTeXHandlerPapersToBibtex:
    """Tests for papers_to_bibtex method."""

    def test_papers_to_bibtex_string(self):
        """Converting papers to BibTeX string should work."""
        handler = BibTeXHandler()

        paper1 = Paper()
        paper1.metadata.basic.title = "Paper 1"
        paper1.metadata.basic.year = 2023
        paper1.metadata.basic.authors = ["Author1"]

        paper2 = Paper()
        paper2.metadata.basic.title = "Paper 2"
        paper2.metadata.basic.year = 2024
        paper2.metadata.basic.authors = ["Author2"]

        content = handler.papers_to_bibtex([paper1, paper2])

        assert "@" in content
        assert "Paper 1" in content
        assert "Paper 2" in content
        assert "2023" in content
        assert "2024" in content

    def test_papers_to_bibtex_file(self, tmp_path):
        """Converting papers to BibTeX file should work."""
        handler = BibTeXHandler()

        paper = Paper()
        paper.metadata.basic.title = "Test Paper"
        paper.metadata.basic.year = 2023
        paper.metadata.basic.authors = ["Author"]

        output_path = tmp_path / "output.bib"
        handler.papers_to_bibtex([paper], output_path=output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "Test Paper" in content


class TestBibTeXHandlerNormalizeTitle:
    """Tests for _normalize_title method."""

    def test_normalize_basic(self):
        """Basic title normalization should work."""
        handler = BibTeXHandler()

        title = "A Test Paper Title"
        normalized = handler._normalize_title(title)

        assert normalized == "a test paper title"

    def test_normalize_punctuation(self):
        """Punctuation should be removed."""
        handler = BibTeXHandler()

        title = "Machine Learning: A Review!"
        normalized = handler._normalize_title(title)

        assert ":" not in normalized
        assert "!" not in normalized

    def test_normalize_whitespace(self):
        """Extra whitespace should be collapsed."""
        handler = BibTeXHandler()

        title = "Too   Much    Whitespace"
        normalized = handler._normalize_title(title)

        assert "  " not in normalized

    def test_normalize_empty(self):
        """Empty title should return empty string."""
        handler = BibTeXHandler()

        assert handler._normalize_title("") == ""
        assert handler._normalize_title(None) == ""


class TestBibTeXHandlerAreSamePaper:
    """Tests for _are_same_paper method."""

    def test_same_doi(self):
        """Papers with same DOI should be same."""
        handler = BibTeXHandler()

        paper1 = Paper()
        paper1.metadata.id.doi = "10.1234/test"
        paper1.metadata.basic.title = "Paper 1"

        paper2 = Paper()
        paper2.metadata.id.doi = "10.1234/test"
        paper2.metadata.basic.title = "Paper 2 Different Title"

        assert handler._are_same_paper(paper1, paper2) is True

    def test_different_doi(self):
        """Papers with different DOIs should be different."""
        handler = BibTeXHandler()

        paper1 = Paper()
        paper1.metadata.id.doi = "10.1234/test1"
        paper1.metadata.basic.title = "Same Title"

        paper2 = Paper()
        paper2.metadata.id.doi = "10.1234/test2"
        paper2.metadata.basic.title = "Same Title"

        assert handler._are_same_paper(paper1, paper2) is False

    def test_same_title_same_year(self):
        """Papers with same title and year should be same (no DOI)."""
        handler = BibTeXHandler()

        paper1 = Paper()
        paper1.metadata.basic.title = "Test Paper"
        paper1.metadata.basic.year = 2023

        paper2 = Paper()
        paper2.metadata.basic.title = "Test Paper"
        paper2.metadata.basic.year = 2023

        assert handler._are_same_paper(paper1, paper2) is True

    def test_same_title_adjacent_years(self):
        """Papers with same title and adjacent years should be same."""
        handler = BibTeXHandler()

        paper1 = Paper()
        paper1.metadata.basic.title = "Test Paper"
        paper1.metadata.basic.year = 2023

        paper2 = Paper()
        paper2.metadata.basic.title = "Test Paper"
        paper2.metadata.basic.year = 2024  # Online vs print

        assert handler._are_same_paper(paper1, paper2) is True


class TestBibTeXHandlerMergeMetadata:
    """Tests for _merge_paper_metadata method."""

    def test_merge_fills_missing(self):
        """Merging should fill missing fields from donor."""
        handler = BibTeXHandler()

        paper1 = Paper()
        paper1.metadata.basic.title = "Test Paper"
        paper1.metadata.basic.year = 2023
        paper1.metadata.id.doi = "10.1234/test"
        # Missing abstract

        paper2 = Paper()
        paper2.metadata.basic.title = "Test Paper"
        paper2.metadata.basic.year = 2023
        paper2.metadata.basic.abstract = "This is the abstract."
        # Missing DOI

        merged = handler._merge_paper_metadata(paper1, paper2)

        assert merged.metadata.id.doi == "10.1234/test"
        assert merged.metadata.basic.abstract == "This is the abstract."

    def test_merge_keeps_higher_citations(self):
        """Merging should keep higher citation count."""
        handler = BibTeXHandler()

        paper1 = Paper()
        paper1.metadata.basic.title = "Test Paper"
        paper1.metadata.citation_count.total = 100

        paper2 = Paper()
        paper2.metadata.basic.title = "Test Paper"
        paper2.metadata.citation_count.total = 200

        merged = handler._merge_paper_metadata(paper1, paper2)

        assert merged.metadata.citation_count.total == 200


class TestBibTeXHandlerDeduplicate:
    """Tests for _deduplicate_papers method."""

    def test_deduplicate_by_doi(self):
        """Duplicates by DOI should be merged."""
        handler = BibTeXHandler()

        paper1 = Paper()
        paper1.metadata.basic.title = "Test Paper"
        paper1.metadata.id.doi = "10.1234/test"

        paper2 = Paper()
        paper2.metadata.basic.title = "Test Paper Copy"
        paper2.metadata.id.doi = "10.1234/test"

        stats = {"duplicates_found": 0, "duplicates_merged": 0}
        unique = handler._deduplicate_papers([paper1, paper2], stats=stats)

        assert len(unique) == 1
        assert stats["duplicates_found"] == 1

    def test_deduplicate_by_title(self):
        """Duplicates by title/year should be merged."""
        handler = BibTeXHandler()

        paper1 = Paper()
        paper1.metadata.basic.title = "Test Paper"
        paper1.metadata.basic.year = 2023

        paper2 = Paper()
        paper2.metadata.basic.title = "Test Paper"
        paper2.metadata.basic.year = 2023

        stats = {}
        unique = handler._deduplicate_papers([paper1, paper2], stats=stats)

        assert len(unique) == 1

    def test_deduplicate_keeps_unique(self):
        """Unique papers should all be kept."""
        handler = BibTeXHandler()

        paper1 = Paper()
        paper1.metadata.basic.title = "Paper 1"
        paper1.metadata.basic.year = 2023

        paper2 = Paper()
        paper2.metadata.basic.title = "Paper 2"
        paper2.metadata.basic.year = 2023

        paper3 = Paper()
        paper3.metadata.basic.title = "Paper 3"
        paper3.metadata.basic.year = 2024

        stats = {"duplicates_found": 0, "duplicates_merged": 0}
        unique = handler._deduplicate_papers([paper1, paper2, paper3], stats=stats)

        assert len(unique) == 3
        assert stats["duplicates_found"] == 0


class TestBibTeXHandlerRoundtrip:
    """Tests for BibTeX roundtrip conversion."""

    def test_paper_to_bibtex_to_paper(self, tmp_path):
        """Paper -> BibTeX -> Paper should preserve data."""
        handler = BibTeXHandler(project="test")

        # Create original paper
        original = Paper()
        original.metadata.basic.title = "Roundtrip Test Paper"
        original.metadata.basic.year = 2023
        original.metadata.basic.authors = ["Smith, John", "Doe, Jane"]
        original.metadata.id.doi = "10.1234/roundtrip"
        original.metadata.publication.journal = "Nature"
        original.metadata.basic.abstract = "Test abstract for roundtrip."

        # Convert to BibTeX
        entry = handler.paper_to_bibtex_entry(original)

        # Convert back to Paper
        restored = handler.paper_from_bibtex_entry(entry)

        # Check key fields are preserved
        assert restored.metadata.basic.title == original.metadata.basic.title
        assert restored.metadata.basic.year == original.metadata.basic.year
        assert restored.metadata.id.doi == original.metadata.id.doi
        assert (
            restored.metadata.publication.journal
            == original.metadata.publication.journal
        )


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
