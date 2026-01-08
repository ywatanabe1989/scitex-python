#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: tests/scitex/dev/cv/test_title_card.py
"""Tests for scitex.dev.cv title card functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scitex.dev.cv import create_closing, create_opening, create_title_card


class TestCreateTitleCard:
    """Tests for create_title_card function."""

    def test_create_title_card_basic(self, tmp_path: Path) -> None:
        """Test basic title card creation."""
        output = tmp_path / "title.png"
        result = create_title_card(
            title="Test Title",
            output_path=str(output),
        )
        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0

    def test_create_title_card_with_subtitle(self, tmp_path: Path) -> None:
        """Test title card with subtitle."""
        output = tmp_path / "title_sub.png"
        result = create_title_card(
            title="Main Title",
            subtitle="Subtitle Text",
            output_path=str(output),
        )
        assert output.exists()

    def test_create_title_card_with_timestamp(self, tmp_path: Path) -> None:
        """Test title card with custom timestamp."""
        output = tmp_path / "title_ts.png"
        result = create_title_card(
            title="Test",
            timestamp="2026-01-08 12:00",
            output_path=str(output),
        )
        assert output.exists()

    def test_create_title_card_custom_dimensions(self, tmp_path: Path) -> None:
        """Test title card with custom dimensions."""
        output = tmp_path / "title_custom.png"
        result = create_title_card(
            title="Custom Size",
            output_path=str(output),
            width=1280,
            height=720,
        )
        assert output.exists()

    def test_create_title_card_solid_background(self, tmp_path: Path) -> None:
        """Test title card with solid background."""
        output = tmp_path / "title_solid.png"
        result = create_title_card(
            title="Solid BG",
            output_path=str(output),
            background="solid",
        )
        assert output.exists()

    def test_create_title_card_hex_background(self, tmp_path: Path) -> None:
        """Test title card with hex color background."""
        output = tmp_path / "title_hex.png"
        result = create_title_card(
            title="Hex BG",
            output_path=str(output),
            background="#ff0000",
        )
        assert output.exists()

    def test_create_title_card_auto_path(self) -> None:
        """Test title card with auto-generated path."""
        result = create_title_card(title="Auto Path")
        assert result.exists()
        assert result.suffix == ".png"
        # Cleanup
        result.unlink(missing_ok=True)

    def test_create_title_card_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that parent directories are created."""
        output = tmp_path / "nested" / "dirs" / "title.png"
        result = create_title_card(
            title="Nested",
            output_path=str(output),
        )
        assert output.exists()

    def test_create_title_card_custom_font_sizes(self, tmp_path: Path) -> None:
        """Test title card with custom font sizes."""
        output = tmp_path / "title_fonts.png"
        result = create_title_card(
            title="Big Title",
            subtitle="Small Subtitle",
            output_path=str(output),
            title_fontsize=96,
            subtitle_fontsize=24,
            timestamp_fontsize=12,
        )
        assert output.exists()


class TestCreateOpening:
    """Tests for create_opening function."""

    def test_create_opening_basic(self, tmp_path: Path) -> None:
        """Test basic opening card creation."""
        output = tmp_path / "opening.png"
        result = create_opening(
            title="Demo Video",
            output_path=str(output),
        )
        assert result == output
        assert output.exists()

    def test_create_opening_with_subtitle(self, tmp_path: Path) -> None:
        """Test opening with custom subtitle."""
        output = tmp_path / "opening_sub.png"
        result = create_opening(
            title="Demo",
            subtitle="Custom Subtitle",
            output_path=str(output),
        )
        assert output.exists()

    def test_create_opening_with_version(self, tmp_path: Path) -> None:
        """Test opening with version string."""
        output = tmp_path / "opening_ver.png"
        result = create_opening(
            title="Feature Demo",
            product="SciTeX",
            version="v2.1.0",
            output_path=str(output),
        )
        assert output.exists()

    def test_create_opening_custom_product(self, tmp_path: Path) -> None:
        """Test opening with custom product name."""
        output = tmp_path / "opening_prod.png"
        result = create_opening(
            title="Demo",
            product="FigRecipe",
            output_path=str(output),
        )
        assert output.exists()

    def test_create_opening_auto_path(self) -> None:
        """Test opening with auto-generated path."""
        result = create_opening(title="Auto")
        assert result.exists()
        assert "opening" in result.name
        # Cleanup
        result.unlink(missing_ok=True)

    def test_create_opening_with_timestamp(self, tmp_path: Path) -> None:
        """Test opening with custom timestamp."""
        output = tmp_path / "opening_ts.png"
        result = create_opening(
            title="Demo",
            timestamp="2026-01-01 00:00",
            output_path=str(output),
        )
        assert output.exists()


class TestCreateClosing:
    """Tests for create_closing function."""

    def test_create_closing_basic(self, tmp_path: Path) -> None:
        """Test basic closing card creation."""
        output = tmp_path / "closing.png"
        result = create_closing(output_path=str(output))
        assert result == output
        assert output.exists()

    def test_create_closing_custom_product(self, tmp_path: Path) -> None:
        """Test closing with custom product name."""
        output = tmp_path / "closing_prod.png"
        result = create_closing(
            product="CustomProduct",
            output_path=str(output),
        )
        assert output.exists()

    def test_create_closing_custom_tagline(self, tmp_path: Path) -> None:
        """Test closing with custom tagline."""
        output = tmp_path / "closing_tag.png"
        result = create_closing(
            tagline="Custom Tagline",
            output_path=str(output),
        )
        assert output.exists()

    def test_create_closing_custom_url(self, tmp_path: Path) -> None:
        """Test closing with custom URL."""
        output = tmp_path / "closing_url.png"
        result = create_closing(
            url="https://example.com",
            output_path=str(output),
        )
        assert output.exists()

    def test_create_closing_custom_dimensions(self, tmp_path: Path) -> None:
        """Test closing with custom dimensions."""
        output = tmp_path / "closing_dim.png"
        result = create_closing(
            output_path=str(output),
            width=1280,
            height=720,
        )
        assert output.exists()

    def test_create_closing_auto_path(self) -> None:
        """Test closing with auto-generated path."""
        result = create_closing()
        assert result.exists()
        assert "closing" in result.name
        # Cleanup
        result.unlink(missing_ok=True)


# EOF
