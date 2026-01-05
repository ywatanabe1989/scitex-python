#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-01 15:40:00 (ywatanabe)"
# File: ./tests/scitex/ai/utils/test__merge_labels.py

"""Tests for scitex.ai.utils._merge_labels module."""

import pytest
pytest.importorskip("zarr")
import numpy as np
from scitex.ai.utils import merge_labels


class TestMergeLabels:
    """Test suite for merge_labels function."""

    def test_merge_two_labels_basic(self):
        """Test merging two label arrays."""
        y1 = np.array([0, 1, 0, 1])
        y2 = np.array([0, 0, 1, 1])
        
        result = merge_labels(y1, y2)
        
        # Check that result contains the expected merged labels
        expected = np.array(['0-0', '1-0', '0-1', '1-1'])
        np.testing.assert_array_equal(result, expected)

    def test_merge_three_labels(self):
        """Test merging three label arrays."""
        y1 = np.array([0, 1, 2])
        y2 = np.array([3, 4, 5])
        y3 = np.array([6, 7, 8])
        
        result = merge_labels(y1, y2, y3)
        
        expected = np.array(['0-3-6', '1-4-7', '2-5-8'])
        np.testing.assert_array_equal(result, expected)

    def test_merge_labels_with_to_int_true(self):
        """Test merging labels with integer conversion."""
        y1 = np.array([0, 1, 0, 1, 0])
        y2 = np.array([0, 0, 1, 1, 0])
        
        result = merge_labels(y1, y2, to_int=True)
        
        # Should create unique integer labels for each combination
        assert result.dtype in [np.int32, np.int64, int]
        # Should have 4 unique combinations: 0-0, 1-0, 0-1, 1-1
        unique_labels = np.unique(result)
        assert len(unique_labels) == 4  # 0-0, 1-0, 0-1, 1-1

    def test_single_label_array_returns_as_is(self):
        """Test that single label array is returned unchanged."""
        y = np.array([1, 2, 3, 4])
        
        result = merge_labels(y)
        
        # Should return the same array
        np.testing.assert_array_equal(result, y)

    def test_empty_arrays(self):
        """Test merging empty arrays."""
        y1 = np.array([], dtype=int)
        y2 = np.array([], dtype=int)
        
        result = merge_labels(y1, y2)
        
        # Should return empty array
        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_mismatched_lengths_behavior(self):
        """Test that mismatched array lengths use zip behavior."""
        y1 = np.array([0, 1, 2])
        y2 = np.array([0, 1])  # Different length
        
        result = merge_labels(y1, y2)
        
        # zip will only iterate up to the shorter length
        # So this should only merge first 2 elements
        assert len(result) == 2
        expected = np.array(['0-0', '1-1'])
        np.testing.assert_array_equal(result, expected)

    def test_string_labels(self):
        """Test merging string labels."""
        y1 = ['cat', 'dog', 'cat']
        y2 = ['A', 'B', 'A']
        
        result = merge_labels(y1, y2)
        
        expected = np.array(['cat-A', 'dog-B', 'cat-A'])
        np.testing.assert_array_equal(result, expected)

    def test_string_labels_to_int(self):
        """Test merging string labels with integer conversion."""
        y1 = ['cat', 'dog', 'cat', 'dog']
        y2 = ['A', 'B', 'A', 'B']
        
        result = merge_labels(y1, y2, to_int=True)
        
        # Should have 2 unique combinations: cat-A, dog-B
        assert len(np.unique(result)) == 2
        assert result.dtype in [np.int32, np.int64, int]

    def test_mixed_type_labels(self):
        """Test merging labels of mixed types."""
        y1 = np.array([0, 1, 2])
        y2 = np.array(['A', 'B', 'C'])
        
        result = merge_labels(y1, y2)
        
        expected = np.array(['0-A', '1-B', '2-C'])
        np.testing.assert_array_equal(result, expected)

    def test_many_labels(self):
        """Test merging many label arrays."""
        n_samples = 100
        n_label_arrays = 5
        
        # Create multiple label arrays
        label_arrays = [np.random.randint(0, 3, size=n_samples) for _ in range(n_label_arrays)]
        
        result = merge_labels(*label_arrays)
        
        assert len(result) == n_samples
        # Each result should have n_label_arrays parts
        assert all('-' in str(label) for label in result)
        assert all(len(str(label).split('-')) == n_label_arrays for label in result)

    def test_to_int_preserves_order(self):
        """Test that to_int conversion maps duplicate combinations to same integer."""
        y1 = np.array([0, 0, 1, 1, 2, 2])
        y2 = np.array([0, 0, 0, 0, 0, 0])
        
        result = merge_labels(y1, y2, to_int=True)
        
        # Same combinations should get same integer
        assert result[0] == result[1]  # Both are 0-0
        assert result[2] == result[3]  # Both are 1-0
        assert result[4] == result[5]  # Both are 2-0

    def test_numpy_array_output(self):
        """Test that output is always numpy array."""
        y1 = [0, 1, 2]  # List input
        y2 = [3, 4, 5]  # List input
        
        result = merge_labels(y1, y2)
        
        assert isinstance(result, np.ndarray)

    def test_deterministic_int_mapping(self):
        """Test that integer mapping is deterministic."""
        y1 = np.array([1, 0, 1, 0])
        y2 = np.array([1, 1, 0, 0])
        
        # Run twice
        result1 = merge_labels(y1, y2, to_int=True)
        result2 = merge_labels(y1, y2, to_int=True)
        
        # Should get same results
        np.testing.assert_array_equal(result1, result2)

    def test_numeric_string_labels(self):
        """Test that numeric values are converted to strings properly."""
        y1 = np.array([10, 20, 30])
        y2 = np.array([100, 200, 300])
        
        result = merge_labels(y1, y2)
        
        expected = np.array(['10-100', '20-200', '30-300'])
        np.testing.assert_array_equal(result, expected)

    def test_float_labels(self):
        """Test merging float labels."""
        y1 = np.array([0.5, 1.5, 2.5])
        y2 = np.array([1.1, 2.2, 3.3])
        
        result = merge_labels(y1, y2)
        
        # Check that result contains float values as strings
        assert '0.5-1.1' in result
        assert '1.5-2.2' in result
        assert '2.5-3.3' in result

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
