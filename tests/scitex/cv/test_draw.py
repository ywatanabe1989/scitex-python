#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: tests/scitex/cv/test_draw.py
"""Tests for scitex.cv drawing functions."""

from __future__ import annotations

import numpy as np

from scitex.cv import arrow, circle, line, polylines, rectangle, text


class TestRectangle:
    """Tests for rectangle function."""

    def test_rectangle_basic(self) -> None:
        """Test basic rectangle drawing."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = rectangle(img, (10, 10), (50, 50), color=(0, 255, 0))
        assert result is img  # Modified in place
        assert img[10, 10:50, 1].sum() > 0  # Green pixels on edge

    def test_rectangle_filled(self) -> None:
        """Test filled rectangle."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        rectangle(img, (10, 10), (50, 50), color=(255, 0, 0), filled=True)
        assert img[30, 30, 0] == 255  # Inside is blue (BGR)

    def test_rectangle_thickness(self) -> None:
        """Test rectangle with custom thickness."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        rectangle(img, (10, 10), (90, 90), color=(255, 255, 255), thickness=5)
        assert img.sum() > 0

    def test_rectangle_returns_image(self) -> None:
        """Test that rectangle returns the image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = rectangle(img, (10, 10), (50, 50))
        assert result.shape == img.shape


class TestCircle:
    """Tests for circle function."""

    def test_circle_basic(self) -> None:
        """Test basic circle drawing."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = circle(img, (50, 50), 20, color=(0, 0, 255))
        assert result is img
        assert img.sum() > 0

    def test_circle_filled(self) -> None:
        """Test filled circle."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        circle(img, (50, 50), 20, color=(0, 255, 0), filled=True)
        assert img[50, 50, 1] == 255  # Center is green

    def test_circle_thickness(self) -> None:
        """Test circle with custom thickness."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        circle(img, (50, 50), 30, thickness=3)
        assert img.sum() > 0


class TestLine:
    """Tests for line function."""

    def test_line_basic(self) -> None:
        """Test basic line drawing."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = line(img, (0, 0), (99, 99), color=(255, 255, 255))
        assert result is img
        assert img.sum() > 0

    def test_line_horizontal(self) -> None:
        """Test horizontal line."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        line(img, (10, 50), (90, 50), color=(255, 0, 0))
        assert img[50, 50, 0] == 255

    def test_line_vertical(self) -> None:
        """Test vertical line."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        line(img, (50, 10), (50, 90), color=(0, 255, 0))
        assert img[50, 50, 1] == 255

    def test_line_thickness(self) -> None:
        """Test line with custom thickness."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        line(img, (0, 50), (99, 50), thickness=5)
        # Thick line should have pixels above and below center
        assert img[48:53, 50, :].sum() > 0


class TestText:
    """Tests for text function."""

    def test_text_basic(self) -> None:
        """Test basic text drawing."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = text(img, "Hello", (10, 50), color=(255, 255, 255))
        assert result is img
        assert img.sum() > 0

    def test_text_scale(self) -> None:
        """Test text with different scales."""
        img1 = np.zeros((100, 200, 3), dtype=np.uint8)
        img2 = np.zeros((100, 200, 3), dtype=np.uint8)

        text(img1, "A", (50, 50), scale=1.0)
        text(img2, "A", (50, 50), scale=2.0)

        # Larger text should have more pixels
        assert img2.sum() > img1.sum()

    def test_text_fonts(self) -> None:
        """Test different font types."""
        fonts = ["simplex", "plain", "duplex", "complex", "triplex"]
        for font in fonts:
            img = np.zeros((100, 200, 3), dtype=np.uint8)
            text(img, "Test", (10, 50), font=font)
            assert img.sum() > 0

    def test_text_color(self) -> None:
        """Test text with specific color."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        text(img, "Red", (10, 50), color=(0, 0, 255))  # BGR
        # Red channel should have values
        assert img[:, :, 2].sum() > 0


class TestPolylines:
    """Tests for polylines function."""

    def test_polylines_closed(self) -> None:
        """Test closed polyline (polygon)."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        points = np.array([[10, 10], [50, 10], [50, 50], [10, 50]])
        result = polylines(img, points, closed=True)
        assert result is img
        assert img.sum() > 0

    def test_polylines_open(self) -> None:
        """Test open polyline."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        points = np.array([[10, 10], [50, 30], [90, 10]])
        polylines(img, points, closed=False)
        assert img.sum() > 0

    def test_polylines_3d_input(self) -> None:
        """Test polylines with (N, 1, 2) shaped input."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        points = np.array([[[10, 10]], [[50, 10]], [[50, 50]]])
        polylines(img, points, closed=True)
        assert img.sum() > 0

    def test_polylines_color(self) -> None:
        """Test polylines with custom color."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        points = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
        polylines(img, points, color=(255, 0, 255))
        # Magenta = Blue + Red
        assert img[:, :, 0].sum() > 0  # Blue
        assert img[:, :, 2].sum() > 0  # Red


class TestArrow:
    """Tests for arrow function."""

    def test_arrow_basic(self) -> None:
        """Test basic arrow drawing."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = arrow(img, (10, 50), (90, 50), color=(0, 255, 0))
        assert result is img
        assert img.sum() > 0

    def test_arrow_tip_length(self) -> None:
        """Test arrow with custom tip length."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        arrow(img, (10, 50), (90, 50), tip_length=0.3)
        assert img.sum() > 0

    def test_arrow_diagonal(self) -> None:
        """Test diagonal arrow."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        arrow(img, (10, 10), (90, 90))
        assert img.sum() > 0

    def test_arrow_thickness(self) -> None:
        """Test arrow with custom thickness."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)

        arrow(img1, (10, 50), (90, 50), thickness=1)
        arrow(img2, (10, 50), (90, 50), thickness=5)

        assert img2.sum() > img1.sum()


# EOF
