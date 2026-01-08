#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: tests/scitex/cv/test_transform.py
"""Tests for scitex.cv transform functions."""

from __future__ import annotations

import numpy as np
import pytest

from scitex.cv import crop, flip, pad, resize, rotate


class TestResize:
    """Tests for resize function."""

    def test_resize_by_scale(self) -> None:
        """Test resizing by scale factor."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized = resize(img, scale=0.5)
        assert resized.shape == (50, 50, 3)

    def test_resize_by_size(self) -> None:
        """Test resizing by target size."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized = resize(img, size=(200, 150))  # (width, height)
        assert resized.shape == (150, 200, 3)

    def test_resize_scale_up(self) -> None:
        """Test scaling up image."""
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        resized = resize(img, scale=2.0)
        assert resized.shape == (100, 100, 3)

    def test_resize_interpolation_nearest(self) -> None:
        """Test resize with nearest neighbor interpolation."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized = resize(img, scale=0.5, interpolation="nearest")
        assert resized.shape == (50, 50, 3)

    def test_resize_interpolation_cubic(self) -> None:
        """Test resize with cubic interpolation."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized = resize(img, scale=2.0, interpolation="cubic")
        assert resized.shape == (200, 200, 3)

    def test_resize_no_params_raises(self) -> None:
        """Test that resize without size or scale raises ValueError."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            resize(img)

    def test_resize_grayscale(self) -> None:
        """Test resizing grayscale image."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        resized = resize(img, scale=0.5)
        assert resized.shape == (50, 50)


class TestRotate:
    """Tests for rotate function."""

    def test_rotate_90_degrees(self) -> None:
        """Test 90 degree rotation."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[0:10, :, 0] = 255  # Red bar at top

        rotated = rotate(img, 90)
        assert rotated.shape == (100, 100, 3)

    def test_rotate_45_degrees(self) -> None:
        """Test 45 degree rotation."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        rotated = rotate(img, 45)
        assert rotated.shape == (100, 100, 3)

    def test_rotate_with_scale(self) -> None:
        """Test rotation with scale factor."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        rotated = rotate(img, 30, scale=0.5)
        assert rotated.shape == (100, 100, 3)

    def test_rotate_with_custom_center(self) -> None:
        """Test rotation with custom center point."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        rotated = rotate(img, 45, center=(25, 25))
        assert rotated.shape == (100, 100, 3)

    def test_rotate_grayscale(self) -> None:
        """Test rotating grayscale image."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        rotated = rotate(img, 90)
        assert len(rotated.shape) == 2


class TestFlip:
    """Tests for flip function."""

    def test_flip_horizontal(self) -> None:
        """Test horizontal flip."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, 0, 0] = 255  # Left column red

        flipped = flip(img, "horizontal")
        assert flipped[:, -1, 0].mean() == 255  # Now right column

    def test_flip_vertical(self) -> None:
        """Test vertical flip."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[0, :, 0] = 255  # Top row red

        flipped = flip(img, "vertical")
        assert flipped[-1, :, 0].mean() == 255  # Now bottom row

    def test_flip_both(self) -> None:
        """Test flip in both directions."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[0, 0, 0] = 255  # Top-left pixel

        flipped = flip(img, "both")
        assert flipped[-1, -1, 0] == 255  # Now bottom-right

    def test_flip_grayscale(self) -> None:
        """Test flipping grayscale image."""
        img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        flipped = flip(img, "horizontal")
        assert flipped.shape == (10, 10)


class TestCrop:
    """Tests for crop function."""

    def test_crop_basic(self) -> None:
        """Test basic cropping."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cropped = crop(img, x=10, y=20, width=50, height=30)
        assert cropped.shape == (30, 50, 3)

    def test_crop_corner(self) -> None:
        """Test cropping from corner."""
        img = np.arange(100).reshape(10, 10).astype(np.uint8)
        cropped = crop(img, x=0, y=0, width=5, height=5)
        assert cropped.shape == (5, 5)
        assert cropped[0, 0] == 0

    def test_crop_preserves_values(self) -> None:
        """Test that crop preserves pixel values."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:50, 10:60, 1] = 128  # Green region

        cropped = crop(img, x=10, y=20, width=50, height=30)
        assert cropped[:, :, 1].mean() == 128

    def test_crop_grayscale(self) -> None:
        """Test cropping grayscale image."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        cropped = crop(img, x=10, y=10, width=20, height=20)
        assert cropped.shape == (20, 20)


class TestPad:
    """Tests for pad function."""

    def test_pad_constant(self) -> None:
        """Test constant padding."""
        img = np.ones((10, 10, 3), dtype=np.uint8) * 128
        padded = pad(img, top=5, bottom=5, left=5, right=5, color=0)
        assert padded.shape == (20, 20, 3)
        assert padded[0, 0, 0] == 0  # Padding is black

    def test_pad_asymmetric(self) -> None:
        """Test asymmetric padding."""
        img = np.ones((10, 10, 3), dtype=np.uint8)
        padded = pad(img, top=10, bottom=5, left=3, right=7)
        assert padded.shape == (25, 20, 3)

    def test_pad_reflect(self) -> None:
        """Test reflect padding."""
        img = np.arange(25).reshape(5, 5).astype(np.uint8)
        padded = pad(img, top=2, bottom=2, left=2, right=2, mode="reflect")
        assert padded.shape == (9, 9)

    def test_pad_replicate(self) -> None:
        """Test replicate padding."""
        img = np.zeros((5, 5), dtype=np.uint8)
        img[0, :] = 255  # Top row white

        padded = pad(img, top=2, bottom=0, left=0, right=0, mode="replicate")
        assert padded.shape == (7, 5)
        assert padded[0, 0] == 255  # Replicated top value

    def test_pad_color_tuple(self) -> None:
        """Test padding with color tuple."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        padded = pad(img, top=5, bottom=5, left=5, right=5, color=(255, 0, 0))
        assert padded.shape == (20, 20, 3)

    def test_pad_grayscale(self) -> None:
        """Test padding grayscale image."""
        img = np.ones((10, 10), dtype=np.uint8) * 100
        padded = pad(img, top=5, bottom=5, left=5, right=5, color=50)
        assert padded.shape == (20, 20)
        assert padded[0, 0] == 50


# EOF
