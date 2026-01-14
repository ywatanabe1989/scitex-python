#!/usr/bin/env python3
# Timestamp: "2026-01-14"
# File: tests/scitex/scholar/pipelines/test_ScholarPipelineSingle.py
"""
Comprehensive tests for ScholarPipelineSingle and its step mixins.

Tests cover:
- Pipeline initialization
- DOI normalization (step 1)
- Paper ID generation (step 3)
- Metadata merging helpers
- Content extraction (step 8)
- Project linking (step 9)
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scitex.scholar.core import Paper
from scitex.scholar.pipelines import ScholarPipelineSingle
from scitex.scholar.storage import PaperIO


class TestPipelineInit:
    """Tests for ScholarPipelineSingle initialization."""

    def test_init_default_values(self):
        """Pipeline should initialize with default values."""
        pipeline = ScholarPipelineSingle()

        assert pipeline.name == "ScholarPipelineSingle"
        assert pipeline.browser_mode == "interactive"
        assert pipeline.chrome_profile == "system"

    def test_init_custom_values(self):
        """Pipeline should accept custom browser settings."""
        pipeline = ScholarPipelineSingle(
            browser_mode="stealth", chrome_profile="custom_profile"
        )

        assert pipeline.browser_mode == "stealth"
        assert pipeline.chrome_profile == "custom_profile"


class TestStep01NormalizeAsDoi:
    """Tests for _step_01_normalize_as_doi."""

    @pytest.fixture
    def pipeline(self):
        return ScholarPipelineSingle()

    def test_valid_doi_returns_doi(self, pipeline):
        """Valid DOI should return stripped DOI string."""
        result = pipeline._step_01_normalize_as_doi("10.1038/nature12373")
        assert result == "10.1038/nature12373"

    def test_doi_with_whitespace(self, pipeline):
        """DOI with whitespace should be stripped."""
        result = pipeline._step_01_normalize_as_doi("  10.1038/nature12373  ")
        assert result == "10.1038/nature12373"

    def test_title_returns_none(self, pipeline):
        """Non-DOI input (title) should return None."""
        result = pipeline._step_01_normalize_as_doi(
            "Hippocampal ripples down-regulate synapses"
        )
        assert result is None

    def test_almost_doi_returns_none(self, pipeline):
        """String not starting with '10.' should return None."""
        result = pipeline._step_01_normalize_as_doi("doi:10.1038/nature12373")
        assert result is None

    def test_numeric_non_doi(self, pipeline):
        """Numeric string not starting with '10.' returns None."""
        result = pipeline._step_01_normalize_as_doi("12345/test")
        assert result is None


class TestStep03AddPaperId:
    """Tests for _step_03_add_paper_id and _generate_paper_id."""

    @pytest.fixture
    def pipeline(self):
        return ScholarPipelineSingle()

    def test_generate_paper_id_format(self, pipeline):
        """Paper ID should be 8-character uppercase hex."""
        paper_id = pipeline._generate_paper_id("10.1038/nature12373")

        assert len(paper_id) == 8
        # Should be uppercase hex (all chars in 0-9A-F)
        assert all(c in "0123456789ABCDEF" for c in paper_id)
        # Any letters should be uppercase
        assert paper_id == paper_id.upper()

    def test_generate_paper_id_deterministic(self, pipeline):
        """Same DOI should always produce same ID."""
        doi = "10.1038/nature12373"
        id1 = pipeline._generate_paper_id(doi)
        id2 = pipeline._generate_paper_id(doi)

        assert id1 == id2

    def test_generate_paper_id_unique(self, pipeline):
        """Different DOIs should produce different IDs."""
        id1 = pipeline._generate_paper_id("10.1038/nature12373")
        id2 = pipeline._generate_paper_id("10.1038/nature12374")

        assert id1 != id2

    def test_step_03_adds_library_id(self, pipeline):
        """Step 3 should add library_id to paper container."""
        paper = Paper()
        paper.metadata.id.doi = "10.1038/nature12373"

        result = pipeline._step_03_add_paper_id(paper)

        assert result.container.library_id is not None
        assert len(result.container.library_id) == 8


class TestMergeMetadataIntoPaper:
    """Tests for _merge_metadata_into_paper helper."""

    @pytest.fixture
    def pipeline(self):
        return ScholarPipelineSingle()

    @pytest.fixture
    def paper(self):
        paper = Paper()
        paper.container.library_id = "TEST1234"
        return paper

    def test_merge_basic_metadata(self, pipeline, paper):
        """Basic metadata (title, authors, year) should be merged."""
        metadata_dict = {
            "basic": {
                "title": "Test Paper Title",
                "title_engines": ["CrossRef"],
                "authors": ["Smith, John", "Doe, Jane"],
                "authors_engines": ["CrossRef"],
                "year": 2023,
                "year_engines": ["CrossRef"],
            }
        }

        pipeline._merge_metadata_into_paper(paper, metadata_dict)

        assert paper.metadata.basic.title == "Test Paper Title"
        assert paper.metadata.basic.authors == ["Smith, John", "Doe, Jane"]
        assert paper.metadata.basic.year == 2023

    def test_merge_id_metadata(self, pipeline, paper):
        """ID metadata (DOI, PMID, etc.) should be merged."""
        metadata_dict = {
            "id": {
                "doi": "10.1038/test",
                "doi_engines": ["CrossRef"],
                "pmid": "12345678",
                "pmid_engines": ["PubMed"],
            }
        }

        pipeline._merge_metadata_into_paper(paper, metadata_dict)

        assert paper.metadata.id.doi == "10.1038/test"
        assert paper.metadata.id.pmid == "12345678"

    def test_merge_publication_metadata(self, pipeline, paper):
        """Publication metadata should be merged."""
        metadata_dict = {
            "publication": {
                "journal": "Nature",
                "journal_engines": ["CrossRef"],
                "volume": "123",
                "volume_engines": ["CrossRef"],
                "impact_factor": 42.5,
                "impact_factor_engines": ["JCR"],
            }
        }

        pipeline._merge_metadata_into_paper(paper, metadata_dict)

        assert paper.metadata.publication.journal == "Nature"
        assert paper.metadata.publication.volume == "123"
        assert paper.metadata.publication.impact_factor == 42.5

    def test_merge_citation_count(self, pipeline, paper):
        """Citation count should be merged with type conversion."""
        metadata_dict = {
            "citation_count": {
                "total": 150,
                "total_engines": ["SemanticScholar"],
            }
        }

        pipeline._merge_metadata_into_paper(paper, metadata_dict)

        assert paper.metadata.citation_count.total == 150

    def test_merge_year_type_conversion(self, pipeline, paper):
        """Year should be converted to int if string."""
        metadata_dict = {
            "basic": {
                "year": "2023",
                "year_engines": ["CrossRef"],
            }
        }

        pipeline._merge_metadata_into_paper(paper, metadata_dict)

        assert paper.metadata.basic.year == 2023
        assert isinstance(paper.metadata.basic.year, int)

    def test_merge_empty_dict(self, pipeline, paper):
        """Empty metadata dict should not raise error."""
        pipeline._merge_metadata_into_paper(paper, {})
        # Should not raise


class TestStep08ExtractContent:
    """Tests for _step_08_extract_content."""

    @pytest.fixture
    def pipeline(self):
        return ScholarPipelineSingle()

    @pytest.fixture
    def paper_with_io(self, tmp_path):
        """Create paper with PaperIO and mock PDF."""
        paper = Paper()
        paper.container.library_id = "TEST1234"
        paper.metadata.basic.authors = ["Smith"]
        paper.metadata.basic.year = 2023
        paper.metadata.publication.journal = "Nature"

        io = PaperIO(paper, base_dir=tmp_path)

        # Create fake PDF
        pdf_path = io.get_pdf_path()
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        return paper, io

    def test_extract_skips_when_no_pdf(self, pipeline, tmp_path):
        """Should skip extraction when no PDF exists."""
        paper = Paper()
        paper.container.library_id = "TEST1234"
        io = PaperIO(paper, base_dir=tmp_path)

        # No PDF created
        pipeline._step_08_extract_content(io, force=False)

        assert not io.has_content()

    def test_extract_skips_when_content_exists(self, pipeline, paper_with_io):
        """Should skip extraction when content already exists."""
        paper, io = paper_with_io

        # Create existing content
        io.save_text("Existing content")

        with patch("scitex.io.load") as mock_load:
            pipeline._step_08_extract_content(io, force=False)
            mock_load.assert_not_called()

    def test_extract_forces_when_flag_set(self, pipeline, paper_with_io):
        """Should extract when force=True even if content exists."""
        paper, io = paper_with_io

        # Create existing content
        io.save_text("Old content")

        mock_content = MagicMock()
        mock_content.text = "New extracted content"
        mock_content.tables = {}
        mock_content.stats = {"num_tables": 0, "num_images": 0}

        with patch("scitex.io.load", return_value=mock_content):
            pipeline._step_08_extract_content(io, force=True)

        assert io.load_text() == "New extracted content"

    def test_extract_saves_text_and_tables(self, pipeline, paper_with_io):
        """Should save text and tables from extraction."""
        paper, io = paper_with_io

        mock_content = MagicMock()
        mock_content.text = "Extracted paper text"
        mock_content.tables = {1: [MagicMock(columns=["A"], shape=(2, 1))]}
        mock_content.stats = {"num_tables": 1, "num_images": 3}

        # Mock DataFrame to_dict
        mock_content.tables[1][0].to_dict.return_value = [{"A": 1}, {"A": 2}]

        with patch("scitex.io.load", return_value=mock_content):
            pipeline._step_08_extract_content(io, force=False)

        assert io.has_content()
        assert io.load_text() == "Extracted paper text"
        assert io.has_tables()


class TestStep09LinkToProject:
    """Tests for _step_09_link_to_project and _link_to_project."""

    @pytest.fixture
    def pipeline(self):
        return ScholarPipelineSingle()

    @pytest.fixture
    def paper_with_io(self, tmp_path):
        """Create paper with PaperIO."""
        paper = Paper()
        paper.container.library_id = "TEST1234"
        paper.metadata.basic.authors = ["Smith, John"]
        paper.metadata.basic.year = 2023
        paper.metadata.publication.journal = "Nature"
        paper.metadata.publication.impact_factor = 42.0
        paper.metadata.citation_count.total = 100

        io = PaperIO(paper, base_dir=tmp_path / "MASTER")
        return paper, io, tmp_path

    def test_link_returns_none_when_no_project(self, pipeline, paper_with_io):
        """Should return None when project is None."""
        paper, io, _ = paper_with_io

        result = pipeline._step_09_link_to_project(paper, io, project=None)

        assert result is None

    def test_link_creates_symlink(self, pipeline, paper_with_io):
        """Should create symlink in project directory."""
        paper, io, tmp_path = paper_with_io

        # Create a mock PDF so n_pdfs > 0
        (io.paper_dir / "test.pdf").write_bytes(b"%PDF")

        # Create project directory
        project_dir = tmp_path / "projects" / "test_project"
        project_dir.mkdir(parents=True)

        with patch("scitex.scholar.ScholarConfig") as mock_config:
            mock_path_manager = MagicMock()
            mock_path_manager.get_library_project_dir.return_value = project_dir
            mock_path_manager.get_library_project_entry_dirname.return_value = (
                "PDF-01_CC-000100_IF-042_2023_Smith_Nature"
            )
            mock_config.return_value.path_manager = mock_path_manager

            result = pipeline._link_to_project(paper, "test_project", io)

        assert result is not None
        assert result.is_symlink()


class TestEnrichImpactFactor:
    """Tests for _enrich_impact_factor helper."""

    @pytest.fixture
    def pipeline(self):
        return ScholarPipelineSingle()

    def test_skips_when_if_present(self, pipeline):
        """Should skip when impact factor already set."""
        paper = Paper()
        paper.metadata.publication.impact_factor = 42.0

        with patch(
            "scitex.scholar.impact_factor.ImpactFactorEngine"
        ) as mock_engine_class:
            pipeline._enrich_impact_factor(paper)
            mock_engine_class.assert_not_called()

    def test_skips_when_no_journal(self, pipeline):
        """Should skip when no journal name available."""
        paper = Paper()
        # No journal set

        with patch(
            "scitex.scholar.impact_factor.ImpactFactorEngine"
        ) as mock_engine_class:
            pipeline._enrich_impact_factor(paper)
            mock_engine_class.assert_not_called()

    def test_enriches_from_engine(self, pipeline):
        """Should enrich IF from ImpactFactorEngine."""
        paper = Paper()
        paper.metadata.publication.journal = "Nature"

        mock_engine = MagicMock()
        mock_engine.get_metrics.return_value = {
            "impact_factor": 42.778,
            "source": "JCR",
        }

        with patch(
            "scitex.scholar.impact_factor.ImpactFactorEngine",
            return_value=mock_engine,
        ):
            pipeline._enrich_impact_factor(paper)

        assert paper.metadata.publication.impact_factor == 42.778


class TestStep02CreatePaper:
    """Tests for _step_02_create_paper (async)."""

    @pytest.fixture
    def pipeline(self):
        return ScholarPipelineSingle()

    @pytest.mark.asyncio
    async def test_create_paper_with_doi(self, pipeline):
        """Should create paper directly when DOI provided."""
        paper = await pipeline._step_02_create_paper(
            doi="10.1038/nature12373", doi_or_title="10.1038/nature12373"
        )

        assert paper.metadata.id.doi == "10.1038/nature12373"
        assert "user_input" in paper.metadata.id.doi_engines

    @pytest.mark.asyncio
    async def test_create_paper_from_title(self, pipeline):
        """Should resolve DOI from title using ScholarEngine."""
        mock_engine = AsyncMock()
        mock_engine.search_async.return_value = {
            "id": {"doi": "10.1038/resolved", "doi_engines": ["CrossRef"]},
            "basic": {"title": "Test Paper"},
        }

        with patch(
            "scitex.scholar.metadata_engines.ScholarEngine", return_value=mock_engine
        ):
            paper = await pipeline._step_02_create_paper(
                doi=None, doi_or_title="Test Paper Title"
            )

        assert paper.metadata.id.doi == "10.1038/resolved"

    @pytest.mark.asyncio
    async def test_create_paper_raises_when_no_doi(self, pipeline):
        """Should raise ValueError when DOI cannot be resolved."""
        mock_engine = AsyncMock()
        mock_engine.search_async.return_value = None

        with patch(
            "scitex.scholar.metadata_engines.ScholarEngine", return_value=mock_engine
        ):
            with pytest.raises(ValueError, match="No DOI found"):
                await pipeline._step_02_create_paper(
                    doi=None, doi_or_title="Unknown Paper"
                )


class TestStep10LogFinalStatus:
    """Tests for _step_10_log_final_status."""

    @pytest.fixture
    def pipeline(self):
        return ScholarPipelineSingle()

    def test_logs_file_status(self, pipeline, tmp_path):
        """Should log status of all files."""
        paper = Paper()
        paper.container.library_id = "TEST1234"
        paper.metadata.basic.authors = ["Smith"]
        paper.metadata.basic.year = 2023
        paper.metadata.publication.journal = "Nature"
        io = PaperIO(paper, base_dir=tmp_path)

        # Create some files
        io.save_metadata()
        io.save_text("Test content")

        # Should not raise
        pipeline._step_10_log_final_status(io)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/pipelines/ScholarPipelineSingle.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2026-01-14 (ywatanabe)"
# # File: src/scitex/scholar/pipelines/ScholarPipelineSingle.py
# """
# Single paper acquisition pipeline orchestrator.
# 
# Functionalities:
#   - Orchestrates full paper acquisition pipeline from query to storage
#   - Single command: query (DOI/title) + project -> complete paper in library
#   - Coordinates all workers: metadata, URLs, download, extraction, storage
# 
# IO:
#   - output-files:
#     - library/MASTER/{paper_id}/metadata.json
#     - library/MASTER/{paper_id}/main.pdf
#     - library/MASTER/{paper_id}/content.txt
#     - library/MASTER/{paper_id}/tables.json
#     - library/MASTER/{paper_id}/images/
#     - library/{project}/{paper_id} -> ../MASTER/{paper_id}
# """
# 
# from __future__ import annotations
# 
# import argparse
# import asyncio
# from typing import Optional
# 
# from scitex import logging
# from scitex.scholar.storage import PaperIO
# 
# from ._single_steps import PipelineHelpersMixin, PipelineStepsMixin
# 
# logger = logging.getLogger(__name__)
# 
# 
# class ScholarPipelineSingle(PipelineStepsMixin, PipelineHelpersMixin):
#     """Orchestrates full paper acquisition pipeline."""
# 
#     def __init__(
#         self, browser_mode: str = "interactive", chrome_profile: str = "system"
#     ):
#         self.name = self.__class__.__name__
#         self.browser_mode = browser_mode
#         self.chrome_profile = chrome_profile
# 
#     async def process_single_paper(
#         self,
#         doi_or_title: str,
#         project: Optional[str] = None,
#         force: bool = False,
#     ):
#         """Process single paper from query (DOI or Title) to complete storage.
# 
#         Pipeline:
#         1. Normalize as DOI
#         2. Create Paper object (resolve DOI from title if needed)
#         3. Add paper ID (8-digit hash)
#         4. Resolve metadata (ScholarEngine)
#         5. Setup browser
#         6. Find PDF URLs
#         7. Download PDF
#         8. Extract content (text, tables, images)
#         9. Link to project (if specified)
#         10. Log final status
# 
#         Args:
#             doi_or_title: DOI or title string
#             project: Optional project name for symlinking
#             force: If True, ignore existing files and force fresh processing
# 
#         Returns
#         -------
#             Tuple of (Complete Paper object, symlink_path)
#         """
#         # Step 1-3: Initialize
#         doi = self._step_01_normalize_as_doi(doi_or_title)
#         paper = await self._step_02_create_paper(doi, doi_or_title)
#         paper = self._step_03_add_paper_id(paper)
# 
#         io = PaperIO(paper)
#         logger.info(f"{self.name}: Paper directory: {io.paper_dir}")
# 
#         with logger.to(io.paper_dir / "logs" / "pipeline.log"):
#             # Step 4: Metadata
#             paper = await self._step_04_resolve_metadata(paper, io, force)
# 
#             # Steps 5-7: Browser and PDF
#             browser_manager, context, auth_gateway = await self._step_05_setup_browser(
#                 paper, io
#             )
#             if context:
#                 await self._step_06_find_pdf_urls(
#                     paper, io, context, auth_gateway, force
#                 )
#                 await self._step_07_download_pdf(
#                     paper, io, context, auth_gateway, force
#                 )
#             if browser_manager:
#                 await browser_manager.close()
# 
#             # Step 8: Content extraction
#             self._step_08_extract_content(io, force)
# 
#             # Step 9-10: Finalize
#             symlink_path = self._step_09_link_to_project(paper, io, project)
#             self._step_10_log_final_status(io)
# 
#             return paper, symlink_path
# 
# 
# def main(args):
#     """Run single paper pipeline."""
#     pipeline = ScholarPipelineSingle(
#         browser_mode=args.browser_mode, chrome_profile=args.chrome_profile
#     )
#     paper, symlink_path = asyncio.run(
#         pipeline.process_single_paper(
#             doi_or_title=args.doi_or_title,
#             project=args.project,
#             force=args.force,
#         )
#     )
#     return 0
# 
# 
# def parse_args() -> argparse.Namespace:
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Orchestrate full paper acquisition pipeline"
#     )
#     parser.add_argument(
#         "--doi-or-title", type=str, required=True, help="DOI or paper title"
#     )
#     parser.add_argument(
#         "--project", type=str, default=None, help="Project name for symlinking"
#     )
#     parser.add_argument(
#         "--browser-mode",
#         type=str,
#         choices=["stealth", "interactive"],
#         default="stealth",
#         help="Browser mode (default: stealth)",
#     )
#     parser.add_argument(
#         "--chrome-profile",
#         type=str,
#         required=True,
#         help="Chrome profile name (default: system)",
#     )
#     parser.add_argument(
#         "--force",
#         "-f",
#         action="store_true",
#         default=False,
#         help="Force fresh processing",
#     )
#     return parser.parse_args()
# 
# 
# def run_main() -> None:
#     """Initialize scitex framework, run main function, and cleanup."""
#     import sys
# 
#     import matplotlib.pyplot as plt
# 
#     import scitex as stx
# 
#     args = parse_args()
#     CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
#         sys, plt, args=args, file=__file__, sdir_suffix=None, verbose=False, agg=True
#     )
#     exit_status = main(args)
#     stx.session.close(
#         CONFIG, verbose=False, notify=False, message="", exit_status=exit_status
#     )
# 
# 
# if __name__ == "__main__":
#     run_main()
# 
# # Usage:
# # python -m scitex.scholar.pipelines.ScholarPipelineSingle \
# #     --doi-or-title "10.1038/nature12373" \
# #     --project test \
# #     --chrome-profile system \
# #     --browser-mode stealth
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/pipelines/ScholarPipelineSingle.py
# --------------------------------------------------------------------------------
