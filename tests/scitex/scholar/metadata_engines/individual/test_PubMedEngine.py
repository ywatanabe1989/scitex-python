#!/usr/bin/env python3
"""Tests for PubMedEngine - PubMed metadata retrieval engine."""

import json
from unittest.mock import MagicMock, patch

import pytest

from scitex.scholar.metadata_engines.individual import PubMedEngine


class TestPubMedEngineInit:
    """Tests for PubMedEngine initialization."""

    def test_init_default_email(self):
        """Should initialize with default email."""
        engine = PubMedEngine()
        assert engine.email == "research@example.com"

    def test_init_custom_email(self):
        """Should accept custom email."""
        engine = PubMedEngine(email="custom@test.com")
        assert engine.email == "custom@test.com"


class TestPubMedEngineProperties:
    """Tests for PubMedEngine properties."""

    def test_name_property(self):
        """Name property should return 'PubMed'."""
        engine = PubMedEngine()
        assert engine.name == "PubMed"


class TestPubMedEngineSearch:
    """Tests for search method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return PubMedEngine()

    def test_search_by_pmid_calls_correct_method(self, engine):
        """Should call _search_by_pmid when PMID provided."""
        with patch.object(engine, "_search_by_pmid") as mock_method:
            mock_method.return_value = {"id": {"pmid": "12345678"}}
            engine.search(pmid="12345678")
            mock_method.assert_called_once_with("12345678", "dict")

    def test_search_by_doi_calls_correct_method(self, engine):
        """Should call _search_by_doi when DOI provided."""
        with patch.object(engine, "_search_by_doi") as mock_method:
            mock_method.return_value = {"id": {"doi": "10.1038/test"}}
            engine.search(doi="10.1038/test")
            mock_method.assert_called_once_with("10.1038/test", "dict")

    def test_search_by_title_calls_correct_method(self, engine):
        """Should call _search_by_metadata when title provided."""
        with patch.object(engine, "_search_by_metadata") as mock_method:
            mock_method.return_value = {"basic": {"title": "Test Paper"}}
            engine.search(title="Test Paper")
            mock_method.assert_called_once()

    def test_search_pmid_takes_priority(self, engine):
        """PMID should take priority over DOI and title."""
        with patch.object(engine, "_search_by_pmid") as mock_pmid:
            with patch.object(engine, "_search_by_doi") as mock_doi:
                mock_pmid.return_value = {"id": {"pmid": "12345678"}}
                engine.search(pmid="12345678", doi="10.1038/test", title="Test")
                mock_pmid.assert_called_once()
                mock_doi.assert_not_called()


class TestPubMedEngineSearchByPMID:
    """Tests for _search_by_pmid method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return PubMedEngine()

    def test_successful_pmid_search(self, engine):
        """Should return metadata for valid PMID."""
        xml_response = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <ArticleTitle>Test Paper Title</ArticleTitle>
                        <Journal>
                            <Title>Nature</Title>
                            <ISOAbbreviation>Nat</ISOAbbreviation>
                            <ISSN>0028-0836</ISSN>
                            <JournalIssue>
                                <Volume>500</Volume>
                                <Issue>7463</Issue>
                            </JournalIssue>
                        </Journal>
                        <AuthorList>
                            <Author>
                                <ForeName>John</ForeName>
                                <LastName>Doe</LastName>
                            </Author>
                        </AuthorList>
                        <Abstract>
                            <AbstractText>This is the abstract.</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
                <PubmedData>
                    <ArticleIdList>
                        <ArticleId IdType="doi">10.1038/nature12373</ArticleId>
                    </ArticleIdList>
                </PubmedData>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        mock_response = MagicMock()
        mock_response.text = xml_response
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_pmid("12345678", "dict")
        assert result["id"]["pmid"] == "12345678"
        assert result["id"]["doi"] == "10.1038/nature12373"
        assert result["basic"]["title"] == "Test Paper Title"
        assert result["publication"]["journal"] == "Nature"

    def test_extracts_authors(self, engine):
        """Should extract author names correctly."""
        xml_response = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <AuthorList>
                            <Author>
                                <ForeName>John</ForeName>
                                <LastName>Doe</LastName>
                            </Author>
                            <Author>
                                <ForeName>Jane</ForeName>
                                <LastName>Smith</LastName>
                            </Author>
                            <Author>
                                <LastName>Anonymous</LastName>
                            </Author>
                        </AuthorList>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        mock_response = MagicMock()
        mock_response.text = xml_response
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_pmid("12345678", "dict")
        assert "John Doe" in result["basic"]["authors"]
        assert "Jane Smith" in result["basic"]["authors"]
        assert "Anonymous" in result["basic"]["authors"]

    def test_return_as_json(self, engine):
        """Should return JSON string when return_as='json'."""
        xml_response = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <ArticleTitle>Test</ArticleTitle>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        mock_response = MagicMock()
        mock_response.text = xml_response
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_pmid("12345678", "json")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "id" in parsed


class TestPubMedEngineSearchByDOI:
    """Tests for _search_by_doi method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return PubMedEngine()

    def test_cleans_doi_url(self, engine):
        """Should remove https://doi.org/ prefix from DOI."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        engine._search_by_doi("https://doi.org/10.1038/test", "dict")
        call_params = mock_session.get.call_args[1]["params"]
        assert "https://doi.org/" not in call_params["term"]

    def test_finds_pmid_from_doi(self, engine):
        """Should find PMID from DOI and get metadata."""
        # First call returns PMID
        search_response = MagicMock()
        search_response.json.return_value = {"esearchresult": {"idlist": ["12345678"]}}
        search_response.raise_for_status = MagicMock()

        # Second call returns XML
        xml_response = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <ArticleTitle>Test Paper</ArticleTitle>
                    </Article>
                </MedlineCitation>
                <PubmedData>
                    <ArticleIdList>
                        <ArticleId IdType="doi">10.1038/test</ArticleId>
                    </ArticleIdList>
                </PubmedData>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        fetch_response = MagicMock()
        fetch_response.text = xml_response
        fetch_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.side_effect = [search_response, fetch_response]
        engine._session = mock_session

        result = engine._search_by_doi("10.1038/test", "dict")
        assert result["id"]["pmid"] == "12345678"
        assert result["basic"]["title"] == "Test Paper"

    def test_returns_minimal_when_not_found(self, engine):
        """Should return minimal metadata when DOI not found."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_doi("10.1038/test", "dict")
        assert result["id"]["doi"] == "10.1038/test"


class TestPubMedEngineSearchByMetadata:
    """Tests for _search_by_metadata method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return PubMedEngine()

    def test_builds_correct_query(self, engine):
        """Should build correct query from title and year."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        engine._search_by_metadata(title="Test Paper", year=2023)
        call_params = mock_session.get.call_args[1]["params"]
        assert "Test Paper[Title]" in call_params["term"]
        assert "2023[pdat]" in call_params["term"]

    def test_matches_title(self, engine):
        """Should match when title matches result."""
        # First call returns PMIDs
        search_response = MagicMock()
        search_response.json.return_value = {"esearchresult": {"idlist": ["12345678"]}}
        search_response.raise_for_status = MagicMock()

        # Second call returns XML
        xml_response = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <ArticleTitle>Deep Learning Paper</ArticleTitle>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        fetch_response = MagicMock()
        fetch_response.text = xml_response
        fetch_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.side_effect = [search_response, fetch_response]
        engine._session = mock_session

        result = engine._search_by_metadata(title="Deep Learning Paper")
        assert result is not None
        assert result["basic"]["title"] == "Deep Learning Paper"


class TestPubMedEngineExtractMetadata:
    """Tests for metadata extraction from XML."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return PubMedEngine()

    def test_extracts_all_fields(self, engine):
        """Should extract all available fields."""
        xml_response = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <ArticleTitle>Test Paper Title.</ArticleTitle>
                        <Journal>
                            <Title>Nature Communications</Title>
                            <ISOAbbreviation>Nat Commun</ISOAbbreviation>
                            <ISSN>2041-1723</ISSN>
                            <JournalIssue>
                                <Volume>10</Volume>
                                <Issue>1</Issue>
                                <PubDate>
                                    <Year>2023</Year>
                                </PubDate>
                            </JournalIssue>
                        </Journal>
                        <AuthorList>
                            <Author>
                                <ForeName>John</ForeName>
                                <LastName>Doe</LastName>
                            </Author>
                        </AuthorList>
                        <Abstract>
                            <AbstractText>This is the abstract.</AbstractText>
                        </Abstract>
                    </Article>
                    <MeshHeadingList>
                        <MeshHeading>
                            <DescriptorName>Neural Networks</DescriptorName>
                        </MeshHeading>
                    </MeshHeadingList>
                </MedlineCitation>
                <PubmedData>
                    <ArticleIdList>
                        <ArticleId IdType="doi">10.1038/s41467-023-12345</ArticleId>
                    </ArticleIdList>
                </PubmedData>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        mock_response = MagicMock()
        mock_response.text = xml_response
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_pmid("12345678", "dict")

        # Title should have trailing period removed
        assert result["basic"]["title"] == "Test Paper Title"
        assert result["basic"]["year"] == 2023
        assert result["basic"]["abstract"] == "This is the abstract."
        assert "John Doe" in result["basic"]["authors"]
        assert result["publication"]["journal"] == "Nature Communications"
        assert result["publication"]["short_journal"] == "Nat Commun"
        assert result["publication"]["issn"] == "2041-1723"
        assert result["publication"]["volume"] == "10"
        assert result["publication"]["issue"] == "1"
        assert result["id"]["doi"] == "10.1038/s41467-023-12345"

    def test_tracks_engine_source(self, engine):
        """Should track PubMed as source engine."""
        xml_response = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <ArticleTitle>Test</ArticleTitle>
                    </Article>
                </MedlineCitation>
                <PubmedData>
                    <ArticleIdList>
                        <ArticleId IdType="doi">10.1038/test</ArticleId>
                    </ArticleIdList>
                </PubmedData>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        mock_response = MagicMock()
        mock_response.text = xml_response
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_pmid("12345678", "dict")
        assert result["id"]["pmid_engines"] == ["PubMed"]
        assert result["basic"]["title_engines"] == ["PubMed"]
        assert result["system"]["searched_by_PubMed"] is True


class TestPubMedEngineEdgeCases:
    """Edge case tests for PubMedEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return PubMedEngine()

    def test_handles_missing_fields(self, engine):
        """Should handle missing fields gracefully."""
        xml_response = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        mock_response = MagicMock()
        mock_response.text = xml_response
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_pmid("12345678", "dict")
        assert result["id"]["pmid"] == "12345678"
        assert result["basic"]["title"] is None
        assert result["id"]["doi"] is None

    def test_handles_unicode_content(self, engine):
        """Should handle unicode in content."""
        xml_response = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <ArticleTitle>Analyse des donnees medicales</ArticleTitle>
                        <AuthorList>
                            <Author>
                                <ForeName>Jean-Pierre</ForeName>
                                <LastName>Muller</LastName>
                            </Author>
                        </AuthorList>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        mock_response = MagicMock()
        mock_response.text = xml_response
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_pmid("12345678", "dict")
        assert "Analyse" in result["basic"]["title"]
        assert "Jean-Pierre Muller" in result["basic"]["authors"]

    def test_builds_doi_url(self, engine):
        """Should build DOI URL correctly."""
        xml_response = """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <ArticleTitle>Test</ArticleTitle>
                    </Article>
                </MedlineCitation>
                <PubmedData>
                    <ArticleIdList>
                        <ArticleId IdType="doi">10.1038/test</ArticleId>
                    </ArticleIdList>
                </PubmedData>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        mock_response = MagicMock()
        mock_response.text = xml_response
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_pmid("12345678", "dict")
        assert result["url"]["doi"] == "https://doi.org/10.1038/test"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
