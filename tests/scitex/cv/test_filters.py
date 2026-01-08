#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: tests/scitex/cv/test_filters.py
"""Tests for scitex.cv filter functions."""

from __future__ import annotations

import numpy as np
import pytest

from scitex.cv import blur, denoise, edge_detect, sharpen, threshold


class TestBlur:
    """Tests for blur function."""

    def test_blur_gaussian(self) -> None:
        """Test Gaussian blur."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        blurred = blur(img, ksize=5, method="gaussian")
        assert blurred.shape == img.shape
        assert blurred.dtype == np.uint8

    def test_blur_median(self) -> None:
        """Test median blur."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        blurred = blur(img, ksize=5, method="median")
        assert blurred.shape == img.shape

    def test_blur_box(self) -> None:
        """Test box blur."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        blurred = blur(img, ksize=5, method="box")
        assert blurred.shape == img.shape

    def test_blur_bilateral(self) -> None:
        """Test bilateral blur."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        blurred = blur(img, ksize=5, method="bilateral")
        assert blurred.shape == img.shape

    def test_blur_even_ksize_corrected(self) -> None:
        """Test that even kernel size is corrected to odd."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        blurred = blur(img, ksize=4)  # Even, should be corrected to 5
        assert blurred.shape == img.shape

    def test_blur_unknown_method_raises(self) -> None:
        """Test that unknown blur method raises ValueError."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            blur(img, method="unknown")

    def test_blur_grayscale(self) -> None:
        """Test blurring grayscale image."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        blurred = blur(img, ksize=5)
        assert blurred.shape == (100, 100)


class TestSharpen:
    """Tests for sharpen function."""

    def test_sharpen_default(self) -> None:
        """Test default sharpening."""
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        sharpened = sharpen(img)
        assert sharpened.shape == img.shape

    def test_sharpen_strength(self) -> None:
        """Test sharpening with different strengths."""
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        sharpened_low = sharpen(img, strength=0.5)
        sharpened_high = sharpen(img, strength=2.0)
        assert sharpened_low.shape == img.shape
        assert sharpened_high.shape == img.shape

    def test_sharpen_grayscale(self) -> None:
        """Test sharpening grayscale image."""
        img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        sharpened = sharpen(img)
        assert sharpened.shape == (100, 100)


class TestEdgeDetect:
    """Tests for edge_detect function."""

    def test_edge_detect_canny(self) -> None:
        """Test Canny edge detection."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60, :] = 255  # White square

        edges = edge_detect(img, method="canny")
        assert len(edges.shape) == 2
        assert edges.dtype == np.uint8

    def test_edge_detect_sobel(self) -> None:
        """Test Sobel edge detection."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60, :] = 255

        edges = edge_detect(img, method="sobel")
        assert len(edges.shape) == 2

    def test_edge_detect_laplacian(self) -> None:
        """Test Laplacian edge detection."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60, :] = 255

        edges = edge_detect(img, method="laplacian")
        assert len(edges.shape) == 2

    def test_edge_detect_custom_thresholds(self) -> None:
        """Test Canny with custom thresholds."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        edges = edge_detect(img, method="canny", low=100, high=200)
        assert len(edges.shape) == 2

    def test_edge_detect_unknown_method_raises(self) -> None:
        """Test that unknown method raises ValueError."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            edge_detect(img, method="unknown")

    def test_edge_detect_grayscale_input(self) -> None:
        """Test edge detection on grayscale input."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[40:60, 40:60] = 255

        edges = edge_detect(img, method="canny")
        assert len(edges.shape) == 2


class TestThreshold:
    """Tests for threshold function."""

    def test_threshold_binary(self) -> None:
        """Test binary thresholding."""
        img = np.arange(256).reshape(16, 16).astype(np.uint8)
        result = threshold(img, thresh=127, method="binary")
        assert len(result.shape) == 2
        assert set(np.unique(result)).issubset({0, 255})

    def test_threshold_binary_inv(self) -> None:
        """Test inverse binary thresholding."""
        img = np.arange(256).reshape(16, 16).astype(np.uint8)
        result = threshold(img, thresh=127, method="binary_inv")
        assert len(result.shape) == 2

    def test_threshold_trunc(self) -> None:
        """Test truncate thresholding."""
        img = np.arange(256).reshape(16, 16).astype(np.uint8)
        result = threshold(img, thresh=127, method="trunc")
        assert result.max() <= 127

    def test_threshold_tozero(self) -> None:
        """Test to-zero thresholding."""
        img = np.arange(256).reshape(16, 16).astype(np.uint8)
        result = threshold(img, thresh=127, method="tozero")
        assert len(result.shape) == 2

    def test_threshold_otsu(self) -> None:
        """Test Otsu thresholding."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = threshold(img, method="otsu")
        assert len(result.shape) == 2

    def test_threshold_adaptive_mean(self) -> None:
        """Test adaptive mean thresholding."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = threshold(img, method="adaptive_mean")
        assert len(result.shape) == 2

    def test_threshold_adaptive_gaussian(self) -> None:
        """Test adaptive Gaussian thresholding."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = threshold(img, method="adaptive_gaussian")
        assert len(result.shape) == 2

    def test_threshold_color_image(self) -> None:
        """Test thresholding color image (converts to gray)."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = threshold(img, thresh=127)
        assert len(result.shape) == 2


class TestDenoise:
    """Tests for denoise function."""

    def test_denoise_fastNl_color(self) -> None:
        """Test fast non-local means denoising on color image."""
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        denoised = denoise(img, strength=10, method="fastNl")
        assert denoised.shape == img.shape

    def test_denoise_fastNl_gray(self) -> None:
        """Test fast non-local means denoising on grayscale."""
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        denoised = denoise(img, strength=10, method="fastNl")
        assert denoised.shape == img.shape

    def test_denoise_bilateral(self) -> None:
        """Test bilateral denoising."""
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        denoised = denoise(img, strength=10, method="bilateral")
        assert denoised.shape == img.shape

    def test_denoise_unknown_method_raises(self) -> None:
        """Test that unknown method raises ValueError."""
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            denoise(img, method="unknown")


# EOF
