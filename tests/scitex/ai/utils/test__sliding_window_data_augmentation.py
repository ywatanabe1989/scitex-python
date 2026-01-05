#!/usr/bin/env python3
# Time-stamp: "2026-01-04 00:00:00 (ywatanabe)"
# File: ./tests/scitex/ai/utils/test__sliding_window_data_augmentation.py

"""Tests for scitex.ai.utils._sliding_window_data_augmentation module.

This module provides sliding window data augmentation for time series data,
commonly used in machine learning for creating training samples from continuous
signals.
"""

import random
from unittest.mock import patch

import numpy as np
import pytest

# Conditionally import torch
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from scitex.ai.utils import sliding_window_data_augmentation


class TestSlidingWindowDataAugmentationBasic:
    """Basic functionality tests for sliding_window_data_augmentation."""

    def test_basic_1d_array(self):
        """Test with basic 1D array."""
        x = np.arange(100)
        window_size = 30

        result = sliding_window_data_augmentation(x, window_size)

        # Check output shape
        assert result.shape == (window_size,)
        # Check that values are from original array
        assert np.all(np.isin(result, x))
        # Check consecutive values
        assert np.all(np.diff(result) == 1)

    def test_2d_array(self):
        """Test with 2D array (channels x time)."""
        x = np.random.rand(5, 1000)  # 5 channels, 1000 time points
        window_size = 200

        result = sliding_window_data_augmentation(x, window_size)

        # Check output shape
        assert result.shape == (5, window_size)
        # Check that all channels are included
        assert result.shape[0] == x.shape[0]

    def test_3d_array(self):
        """Test with 3D array (batch x channels x time)."""
        x = np.random.rand(10, 5, 500)
        window_size = 100

        result = sliding_window_data_augmentation(x, window_size)

        # Check output shape
        assert result.shape == (10, 5, window_size)
        # Check that batch and channel dimensions are preserved
        assert result.shape[:-1] == x.shape[:-1]

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_torch_tensor(self):
        """Test with PyTorch tensor."""
        x = torch.randn(3, 4, 1000)
        window_size = 256

        result = sliding_window_data_augmentation(x, window_size)

        # Check output type and shape
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 4, window_size)

    def test_window_size_equals_array_size(self):
        """Test when window size equals array size."""
        x = np.arange(50)
        window_size = 50

        result = sliding_window_data_augmentation(x, window_size)

        # Should return the entire array
        np.testing.assert_array_equal(result, x)

    def test_deterministic_with_fixed_seed(self):
        """Test deterministic behavior with fixed random seed."""
        x = np.arange(100)
        window_size = 30

        # Fix random seed
        with patch("random.randint", return_value=10):
            result1 = sliding_window_data_augmentation(x, window_size)
            result2 = sliding_window_data_augmentation(x, window_size)

        # Should produce same results
        np.testing.assert_array_equal(result1, result2)
        # Check specific window
        np.testing.assert_array_equal(result1, x[10:40])

    def test_random_window_positions(self):
        """Test that different windows are selected randomly."""
        x = np.arange(1000)
        window_size = 100

        # Get multiple windows
        windows = []
        for _ in range(10):
            window = sliding_window_data_augmentation(x, window_size)
            windows.append(window[0])  # Store first element

        # Should have different starting positions
        unique_starts = len(set(windows))
        assert unique_starts > 1  # Very unlikely to get same position 10 times

    def test_zero_window_size(self):
        """Test with zero window size."""
        x = np.arange(100)
        window_size = 0

        result = sliding_window_data_augmentation(x, window_size)

        # Should return empty array with same number of dimensions
        assert result.shape == (0,)

    def test_float_array(self):
        """Test with float array."""
        x = np.random.randn(200).astype(np.float32)
        window_size = 50

        result = sliding_window_data_augmentation(x, window_size)

        assert result.dtype == np.float32
        assert result.shape == (window_size,)

    def test_preserves_data_continuity(self):
        """Test that data continuity is preserved."""
        # Create array with specific pattern
        x = np.sin(np.linspace(0, 10 * np.pi, 1000))
        window_size = 200

        result = sliding_window_data_augmentation(x, window_size)

        # Check that the window contains continuous data
        # by verifying the sine wave pattern is preserved
        diffs = np.diff(result)
        # Adjacent differences should be smooth (no jumps)
        assert np.all(np.abs(diffs) < 0.1)  # Small threshold for continuity

    @pytest.mark.parametrize(
        "shape,window_size",
        [
            ((100,), 50),
            ((10, 100), 30),
            ((5, 10, 200), 100),
            ((2, 3, 4, 500), 250),
        ],
    )
    def test_various_shapes(self, shape, window_size):
        """Test with various array shapes."""
        x = np.random.rand(*shape)

        result = sliding_window_data_augmentation(x, window_size)

        # Check that all dimensions except last are preserved
        assert result.shape[:-1] == x.shape[:-1]
        assert result.shape[-1] == window_size

    def test_boundary_cases(self):
        """Test boundary cases for window selection."""
        x = np.arange(100)
        window_size = 20

        # Mock to test boundary positions
        # Test start position
        with patch("random.randint", return_value=0):
            result = sliding_window_data_augmentation(x, window_size)
            np.testing.assert_array_equal(result, x[:20])

        # Test end position
        with patch("random.randint", return_value=80):
            result = sliding_window_data_augmentation(x, window_size)
            np.testing.assert_array_equal(result, x[80:100])


class TestSlidingWindowDataAugmentationIntegration:
    """Integration tests with ML workflows."""

    def test_training_data_generation(self):
        """Test generating training data for ML model."""
        # Simulate multi-channel time series
        n_samples = 10000
        n_channels = 32
        data = np.random.randn(n_channels, n_samples)

        window_size = 256
        n_augmentations = 100

        # Generate augmented training set
        training_samples = []
        for _ in range(n_augmentations):
            sample = sliding_window_data_augmentation(data, window_size)
            training_samples.append(sample)

        training_data = np.array(training_samples)
        assert training_data.shape == (n_augmentations, n_channels, window_size)

        # Verify diversity - check that we're using different parts of the signal
        first_samples = training_data[:, 0, 0]  # First value of first channel
        unique_count = len(set(first_samples))
        # Should have at least some variety (not all identical)
        assert unique_count > 1

    def test_multi_scale_augmentation(self):
        """Test augmentation with multiple window sizes."""
        x = np.random.randn(8, 10000)
        window_sizes = [100, 200, 500, 1000]

        multi_scale_samples = {}
        for size in window_sizes:
            samples = []
            for _ in range(10):
                sample = sliding_window_data_augmentation(x, size)
                samples.append(sample)
            multi_scale_samples[size] = np.array(samples)

        # Verify different scales
        for size in window_sizes:
            assert multi_scale_samples[size].shape == (10, 8, size)

    def test_augmentation_with_labels(self):
        """Test augmentation preserving label correspondence."""
        # Simulate labeled segments
        n_samples = 10000
        n_channels = 16
        data = np.random.randn(n_channels, n_samples)

        # Create labels (e.g., 0-3 for different states)
        labels = np.repeat([0, 1, 2, 3], n_samples // 4)

        window_size = 200

        # Augment with label preservation
        augmented_data = []
        augmented_labels = []

        for _ in range(50):
            # Get random start position
            start = random.randint(0, n_samples - window_size)

            # Extract window and corresponding labels
            window_data = data[:, start : start + window_size]
            window_labels = labels[start : start + window_size]

            # If window spans multiple labels, take majority
            unique, counts = np.unique(window_labels, return_counts=True)
            majority_label = unique[np.argmax(counts)]

            augmented_data.append(window_data)
            augmented_labels.append(majority_label)

        augmented_data = np.array(augmented_data)
        augmented_labels = np.array(augmented_labels)

        assert augmented_data.shape == (50, n_channels, window_size)
        assert augmented_labels.shape == (50,)
        assert np.all(np.isin(augmented_labels, [0, 1, 2, 3]))


class TestSlidingWindowDataAugmentationDocumentation:
    """Test documentation and usage examples."""

    def test_function_signature(self):
        """Test function signature."""
        import inspect

        # Check signature
        sig = inspect.signature(sliding_window_data_augmentation)
        params = list(sig.parameters.keys())
        assert params == ["x", "window_size_pts"]

    def test_example_eeg_processing(self):
        """Example: EEG signal processing."""
        # Simulate 32-channel EEG at 256 Hz for 10 seconds
        fs = 256  # Hz
        duration = 10  # seconds
        n_channels = 32
        n_samples = fs * duration

        # Generate synthetic EEG
        eeg = np.random.randn(n_channels, n_samples)

        # Add some structure (alpha rhythm at 10 Hz)
        t = np.linspace(0, duration, n_samples)
        for ch in range(n_channels):
            eeg[ch] += 0.5 * np.sin(2 * np.pi * 10 * t)

        # Augment with 1-second windows
        window_samples = int(1.0 * fs)  # 1 second

        augmented = sliding_window_data_augmentation(eeg, window_samples)
        assert augmented.shape == (n_channels, window_samples)

    def test_example_audio_processing(self):
        """Example: Audio signal processing."""
        # Simulate stereo audio at 44.1 kHz
        fs = 44100  # Hz
        duration = 5  # seconds
        n_channels = 2  # Stereo
        n_samples = fs * duration

        # Generate synthetic audio
        audio = np.random.randn(n_channels, n_samples) * 0.1

        # Add some tones
        t = np.linspace(0, duration, n_samples)
        audio[0] += 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
        audio[1] += 0.3 * np.sin(2 * np.pi * 554.37 * t)  # C#5 note

        # Augment with 0.5-second windows
        window_samples = int(0.5 * fs)

        augmented = sliding_window_data_augmentation(audio, window_samples)
        assert augmented.shape == (n_channels, window_samples)

    def test_example_sensor_data(self):
        """Example: Multi-sensor time series."""
        # Simulate IoT sensor data
        n_sensors = 5
        n_samples = 100000  # 100k samples

        # Different sensors with different characteristics
        sensor_data = np.zeros((n_sensors, n_samples))

        # Temperature sensor (slow variation)
        sensor_data[0] = 20 + 5 * np.sin(np.linspace(0, 10, n_samples))

        # Vibration sensor (high frequency)
        sensor_data[1] = np.random.randn(n_samples) * 0.1

        # Pressure sensor (step changes)
        sensor_data[2] = np.repeat([1, 1.5, 1.2, 0.8], n_samples // 4)

        # Humidity sensor (drift)
        sensor_data[3] = 50 + np.linspace(0, 10, n_samples)

        # Binary sensor (on/off)
        sensor_data[4] = np.random.randint(0, 2, n_samples)

        # Augment with various window sizes
        for window_size in [100, 500, 1000]:
            augmented = sliding_window_data_augmentation(sensor_data, window_size)
            assert augmented.shape == (n_sensors, window_size)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/utils/_sliding_window_data_augmentation.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-01-24 13:56:36 (ywatanabe)"
#
# import random
#
#
# def sliding_window_data_augmentation(x, window_size_pts):
#     start = random.randint(0, x.shape[-1] - window_size_pts)
#     end = start + window_size_pts
#     return x[..., start:end]
#
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/utils/_sliding_window_data_augmentation.py
# --------------------------------------------------------------------------------
