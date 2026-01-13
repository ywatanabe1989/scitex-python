#!/usr/bin/env python3
# Timestamp: "2026-01-14"
# File: tests/scitex/scholar/pdf_download/test_ScholarPDFDownloader.py
"""
Comprehensive tests for ScholarPDFDownloader.

Tests cover:
- Initialization and configuration
- Download preference settings
- Batch download functionality
- Open access download handling
- URL validation
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scitex.scholar.pdf_download import ScholarPDFDownloader


class TestScholarPDFDownloaderInit:
    """Tests for ScholarPDFDownloader initialization."""

    def test_init_with_context(self):
        """Downloader should initialize with browser context."""
        mock_context = MagicMock()

        with patch(
            "scitex.scholar.pdf_download.ScholarPDFDownloader.ScholarConfig"
        ) as mock_config:
            mock_config.return_value.get_library_downloads_dir.return_value = Path(
                "/tmp/downloads"
            )
            mock_config.return_value.resolve.return_value = True

            downloader = ScholarPDFDownloader(context=mock_context)

        assert downloader.name == "ScholarPDFDownloader"
        assert downloader.context == mock_context

    def test_init_with_custom_config(self):
        """Downloader should accept custom config."""
        mock_context = MagicMock()
        mock_config = MagicMock()
        mock_config.get_library_downloads_dir.return_value = Path("/custom/downloads")
        mock_config.resolve.return_value = True

        downloader = ScholarPDFDownloader(context=mock_context, config=mock_config)

        assert downloader.config == mock_config

    def test_init_loads_access_preferences(self):
        """Downloader should load access preferences from config."""
        mock_context = MagicMock()

        with patch(
            "scitex.scholar.pdf_download.ScholarPDFDownloader.ScholarConfig"
        ) as mock_config:
            mock_config.return_value.get_library_downloads_dir.return_value = Path(
                "/tmp"
            )
            mock_config.return_value.resolve.side_effect = lambda key, **kwargs: {
                "prefer_open_access": True,
                "enable_paywall_access": False,
                "track_paywall_attempts": True,
            }.get(key, kwargs.get("default"))

            downloader = ScholarPDFDownloader(context=mock_context)

        assert downloader.prefer_open_access is True
        assert downloader.enable_paywall_access is False
        assert downloader.track_paywall_attempts is True


class TestScholarPDFDownloaderDownloadFromUrls:
    """Tests for download_from_urls batch method."""

    @pytest.fixture
    def downloader(self, tmp_path):
        """Create downloader with mocked dependencies."""
        mock_context = MagicMock()

        with patch(
            "scitex.scholar.pdf_download.ScholarPDFDownloader.ScholarConfig"
        ) as mock_config:
            mock_config.return_value.get_library_downloads_dir.return_value = tmp_path
            mock_config.return_value.resolve.return_value = True

            return ScholarPDFDownloader(context=mock_context)

    @pytest.mark.asyncio
    async def test_download_from_urls_empty_list(self, downloader):
        """Should return empty list for no URLs."""
        result = await downloader.download_from_urls([])

        assert result == []

    @pytest.mark.asyncio
    async def test_download_from_urls_calls_download_from_url(self, downloader):
        """Should call download_from_url for each URL."""
        downloader.download_from_url = AsyncMock(return_value=Path("/tmp/test.pdf"))

        result = await downloader.download_from_urls(
            ["https://example.com/paper1.pdf", "https://example.com/paper2.pdf"]
        )

        assert downloader.download_from_url.call_count == 2


class TestScholarPDFDownloaderDownloadOpenAccess:
    """Tests for download_open_access method."""

    @pytest.fixture
    def downloader(self, tmp_path):
        """Create downloader with mocked dependencies."""
        mock_context = MagicMock()

        with patch(
            "scitex.scholar.pdf_download.ScholarPDFDownloader.ScholarConfig"
        ) as mock_config:
            mock_config.return_value.get_library_downloads_dir.return_value = tmp_path
            mock_config.return_value.resolve.return_value = True

            return ScholarPDFDownloader(context=mock_context)

    @pytest.mark.asyncio
    async def test_download_open_access_no_url(self, downloader, tmp_path):
        """Should return None when no OA URL provided."""
        result = await downloader.download_open_access(
            oa_url=None,
            output_path=tmp_path / "test.pdf",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_download_open_access_empty_url(self, downloader, tmp_path):
        """Should return None when empty OA URL provided."""
        result = await downloader.download_open_access(
            oa_url="",
            output_path=tmp_path / "test.pdf",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_download_open_access_adds_pdf_extension(self, downloader, tmp_path):
        """Should add .pdf extension if missing."""
        with patch(
            "scitex.scholar.pdf_download.ScholarPDFDownloader.try_download_open_access_async",
            new_callable=AsyncMock,
        ) as mock_download:
            mock_download.return_value = None

            await downloader.download_open_access(
                oa_url="https://arxiv.org/pdf/2308.09312",
                output_path=tmp_path / "paper",  # No .pdf extension
            )

            # Check that output_path was modified to include .pdf
            call_args = mock_download.call_args
            assert str(call_args.kwargs["output_path"]).endswith(".pdf")


class TestScholarPDFDownloaderDownloadFromUrl:
    """Tests for download_from_url method."""

    @pytest.fixture
    def downloader(self, tmp_path):
        """Create downloader with mocked dependencies."""
        mock_context = MagicMock()
        mock_context.add_init_script = AsyncMock()

        with patch(
            "scitex.scholar.pdf_download.ScholarPDFDownloader.ScholarConfig"
        ) as mock_config:
            mock_config.return_value.get_library_downloads_dir.return_value = tmp_path
            mock_config.return_value.resolve.return_value = True

            return ScholarPDFDownloader(context=mock_context)

    @pytest.mark.asyncio
    async def test_download_from_url_no_url(self, downloader, tmp_path):
        """Should return None when no URL provided."""
        result = await downloader.download_from_url(
            pdf_url=None,
            output_path=tmp_path / "test.pdf",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_download_from_url_empty_url(self, downloader, tmp_path):
        """Should return None when empty URL provided."""
        result = await downloader.download_from_url(
            pdf_url="",
            output_path=tmp_path / "test.pdf",
        )

        assert result is None


class TestScholarPDFDownloaderDownloadSmart:
    """Tests for download_smart method."""

    @pytest.fixture
    def downloader(self, tmp_path):
        """Create downloader with mocked dependencies."""
        mock_context = MagicMock()

        with patch(
            "scitex.scholar.pdf_download.ScholarPDFDownloader.ScholarConfig"
        ) as mock_config:
            mock_config.return_value.get_library_downloads_dir.return_value = tmp_path
            mock_config.return_value.resolve.side_effect = lambda key, **kwargs: {
                "prefer_open_access": True,
                "enable_paywall_access": False,
                "track_paywall_attempts": True,
            }.get(key, kwargs.get("default"))

            return ScholarPDFDownloader(context=mock_context)

    @pytest.mark.asyncio
    async def test_download_smart_tries_oa_first(self, downloader, tmp_path):
        """Should try Open Access URL first when prefer_open_access=True."""
        mock_paper = MagicMock()
        mock_paper.metadata.access.is_open_access = True
        mock_paper.metadata.access.oa_url = "https://arxiv.org/pdf/test.pdf"
        mock_paper.metadata.url.pdfs = []
        mock_paper.metadata.id.doi = "10.1234/test"

        downloader.download_open_access = AsyncMock(return_value=tmp_path / "test.pdf")

        result = await downloader.download_smart(
            paper=mock_paper,
            output_path=tmp_path / "output.pdf",
        )

        downloader.download_open_access.assert_called_once()
        assert result == tmp_path / "test.pdf"

    @pytest.mark.asyncio
    async def test_download_smart_tries_pdf_urls_after_oa_fails(
        self, downloader, tmp_path
    ):
        """Should try PDF URLs when OA fails."""
        mock_paper = MagicMock()
        mock_paper.metadata.access.is_open_access = False
        mock_paper.metadata.access.oa_url = "https://arxiv.org/pdf/test.pdf"
        mock_paper.metadata.url.pdfs = [{"url": "https://example.com/paper.pdf"}]
        mock_paper.metadata.id.doi = "10.1234/test"

        downloader.download_open_access = AsyncMock(return_value=None)
        downloader.download_from_url = AsyncMock(return_value=tmp_path / "paper.pdf")

        result = await downloader.download_smart(
            paper=mock_paper,
            output_path=tmp_path / "output.pdf",
        )

        downloader.download_from_url.assert_called_once()


class TestScholarPDFDownloaderPathHandling:
    """Tests for path handling utilities."""

    @pytest.fixture
    def downloader(self, tmp_path):
        """Create downloader with mocked dependencies."""
        mock_context = MagicMock()

        with patch(
            "scitex.scholar.pdf_download.ScholarPDFDownloader.ScholarConfig"
        ) as mock_config:
            mock_config.return_value.get_library_downloads_dir.return_value = tmp_path
            mock_config.return_value.resolve.return_value = True

            return ScholarPDFDownloader(context=mock_context)

    def test_output_dir_from_config(self, downloader, tmp_path):
        """Output directory should come from config."""
        assert downloader.output_dir == tmp_path


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
