#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: tests/scitex/cv/test_io.py
"""Tests for scitex.cv I/O functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from scitex.cv import load, save, to_bgr, to_gray, to_rgb


class TestLoad:
    """Tests for load function."""

    def test_load_color_image(self, tmp_path: Path) -> None:
        """Test loading a color image."""
        # Create test image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)

        # Load and verify
        loaded = load(path, color=True)
        assert loaded.shape == (100, 100, 3)
        assert loaded.dtype == np.uint8

    def test_load_grayscale(self, tmp_path: Path) -> None:
        """Test loading as grayscale."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)

        loaded = load(path, color=False)
        assert len(loaded.shape) == 2
        assert loaded.shape == (100, 100)

    def test_load_with_alpha(self, tmp_path: Path) -> None:
        """Test loading image with alpha channel."""
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)

        loaded = load(path, alpha=True)
        assert loaded.shape == (100, 100, 4)

    def test_load_nonexistent_raises(self) -> None:
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load("/nonexistent/path/image.png")

    def test_load_with_pathlib(self, tmp_path: Path) -> None:
        """Test loading with Path object."""
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)

        loaded = load(path)
        assert loaded.shape == (50, 50, 3)


class TestSave:
    """Tests for save function."""

    def test_save_png(self, tmp_path: Path) -> None:
        """Test saving PNG image."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        path = tmp_path / "output.png"

        result = save(img, path)
        assert result == path
        assert path.exists()

    def test_save_jpeg(self, tmp_path: Path) -> None:
        """Test saving JPEG image."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        path = tmp_path / "output.jpg"

        result = save(img, path, quality=90)
        assert result == path
        assert path.exists()

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that save creates parent directories."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        path = tmp_path / "subdir" / "nested" / "output.png"

        result = save(img, path)
        assert result == path
        assert path.exists()

    def test_save_grayscale(self, tmp_path: Path) -> None:
        """Test saving grayscale image."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        path = tmp_path / "gray.png"

        result = save(img, path)
        assert path.exists()


class TestColorConversions:
    """Tests for color conversion functions."""

    def test_to_rgb_from_bgr(self) -> None:
        """Test BGR to RGB conversion."""
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # Blue channel

        rgb = to_rgb(bgr)
        assert rgb[:, :, 2].mean() == 255  # Blue now in R position? No, B->R
        assert rgb[:, :, 0].mean() == 0

    def test_to_rgb_from_gray(self) -> None:
        """Test grayscale to RGB conversion."""
        gray = np.ones((10, 10), dtype=np.uint8) * 128
        rgb = to_rgb(gray)
        assert rgb.shape == (10, 10, 3)
        assert np.all(rgb[:, :, 0] == 128)

    def test_to_rgb_from_bgra(self) -> None:
        """Test BGRA to RGB conversion."""
        bgra = np.zeros((10, 10, 4), dtype=np.uint8)
        bgra[:, :, 0] = 255  # Blue
        bgra[:, :, 3] = 128  # Alpha

        rgb = to_rgb(bgra)
        assert rgb.shape == (10, 10, 3)

    def test_to_bgr_from_rgb(self) -> None:
        """Test RGB to BGR conversion."""
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255  # Red channel

        bgr = to_bgr(rgb)
        assert bgr[:, :, 2].mean() == 255  # Red now in B position

    def test_to_bgr_from_gray(self) -> None:
        """Test grayscale to BGR conversion."""
        gray = np.ones((10, 10), dtype=np.uint8) * 100
        bgr = to_bgr(gray)
        assert bgr.shape == (10, 10, 3)

    def test_to_gray_from_color(self) -> None:
        """Test color to grayscale conversion."""
        color = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        gray = to_gray(color)
        assert len(gray.shape) == 2
        assert gray.shape == (10, 10)

    def test_to_gray_from_gray(self) -> None:
        """Test grayscale passthrough."""
        gray_in = np.ones((10, 10), dtype=np.uint8) * 50
        gray_out = to_gray(gray_in)
        assert np.array_equal(gray_in, gray_out)

    def test_to_gray_from_bgra(self) -> None:
        """Test BGRA to grayscale conversion."""
        bgra = np.random.randint(0, 255, (10, 10, 4), dtype=np.uint8)
        gray = to_gray(bgra)
        assert len(gray.shape) == 2


# EOF
