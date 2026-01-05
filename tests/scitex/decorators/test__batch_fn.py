#!/usr/bin/env python3
# Time-stamp: "2026-01-04 21:10:00 (ywatanabe)"
# File: ./tests/scitex/decorators/test__batch_fn.py

"""Test batch_fn decorator functionality.

The batch_fn decorator is designed for memory-efficient processing of large datasets.
It splits input data into batches, processes each batch independently, and combines
results by stacking. This is useful when data doesn't fit in memory all at once.

Key behaviors:
- Splits data along axis 0 (first dimension)
- Processes each batch independently
- Combines results via vstack (concatenation along axis 0)
- Suitable for row-wise operations, NOT global aggregations
"""

import numpy as np
import pytest

# Required for scitex.decorators module
pytest.importorskip("tqdm")

# Optional dependencies
torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

from scitex.decorators import batch_fn, numpy_fn, torch_fn


class TestBatchFn:
    """Test batch_fn decorator"""

    def test_basic_functionality(self):
        """Test basic batch processing preserves data through batching"""

        @batch_fn
        def double_rows(x, batch_size=4):
            # Returns 2D array - each row doubled
            return x * 2

        # Use 12 rows with batch_size=3 for even batches (4 batches of 3)
        data = np.random.randn(12, 5)
        result = double_rows(data, batch_size=3)
        expected = data * 2

        np.testing.assert_allclose(result, expected)

    def test_default_batch_size(self):
        """Test default batch_size=4"""
        call_count = 0

        @batch_fn
        def count_calls(x, batch_size=4):
            nonlocal call_count
            call_count += 1
            return x  # Identity - returns batch as-is

        data = np.arange(10).reshape(-1, 1)  # 2D for vstack compatibility
        result = count_calls(data)  # No batch_size specified, uses default 4

        # With 10 elements and batch_size=4, should have 3 calls (4+4+2)
        assert call_count == 3
        np.testing.assert_array_equal(result.flatten(), data.flatten())

    def test_batch_size_larger_than_data(self):
        """Test batch_size larger than data processes all at once"""
        call_count = 0

        @batch_fn
        def count_calls(x, batch_size=4):
            nonlocal call_count
            call_count += 1
            return x

        data = np.arange(5).reshape(-1, 1)
        result = count_calls(data, batch_size=10)

        # Data smaller than batch_size, should only call once
        assert call_count == 1
        np.testing.assert_array_equal(result.flatten(), data.flatten())

    def test_batch_processing_produces_per_batch_results(self):
        """Test that batch processing produces independent per-batch results"""

        @batch_fn
        def batch_mean(x, batch_size=4):
            # Return mean of each batch (scalar per batch)
            return np.mean(x)

        data = np.arange(12).reshape(-1).astype(float)
        result = batch_mean(data, batch_size=4)

        # With 12 elements split into 3 batches of 4:
        # batch1: [0,1,2,3] mean = 1.5
        # batch2: [4,5,6,7] mean = 5.5
        # batch3: [8,9,10,11] mean = 9.5
        expected = np.array([1.5, 5.5, 9.5])
        np.testing.assert_allclose(result, expected)

    def test_tuple_results(self):
        """Test handling of tuple results with 2D outputs"""

        @batch_fn
        def transform_data(x, batch_size=4):
            # Returns tuple of 2D arrays
            return x * 2, x + 1

        # Use 12 rows with batch_size=3 for even batches
        data = np.random.randn(12, 5)
        doubled, incremented = transform_data(data, batch_size=3)

        np.testing.assert_allclose(doubled, data * 2)
        np.testing.assert_allclose(incremented, data + 1)

    def test_mixed_tuple_results(self):
        """Test handling of tuples with mixed types (array + non-stackable)"""

        @batch_fn
        def describe_rows(x, batch_size=4):
            # Return 2D array and a list
            transformed = x * 2
            labels = ["doubled"]  # Non-stackable (list), same for all batches
            return transformed, labels

        # Use 12 rows with batch_size=3 for even batches
        data = np.random.randn(12, 5)
        transformed, labels = describe_rows(data, batch_size=3)

        np.testing.assert_allclose(transformed, data * 2)
        # Non-tensor elements use first batch value
        assert labels == ["doubled"]

    def test_torch_tensor_2d_results(self):
        """Test handling of torch tensor with 2D results"""

        @batch_fn
        @torch_fn
        def double_tensor(x, batch_size=4):
            # Returns 2D tensor - preserves shape
            return x * 2

        # Use 12 rows with batch_size=3 for even batches
        data = torch.randn(12, 5)
        result = double_tensor(data, batch_size=3)
        expected = data * 2

        assert torch.allclose(result, expected)

    def test_torch_tensor_multidim_results(self):
        """Test handling of torch tensor multidimensional results"""

        @batch_fn
        @torch_fn
        def reduce_to_2d(x, batch_size=4):
            # Each input row (N, 5) -> reduced row (N, 2)
            return x[:, :2]  # Keep first 2 columns

        # Use 12 rows with batch_size=3 for even batches
        data = torch.randn(12, 5)
        result = reduce_to_2d(data, batch_size=3)
        expected = data[:, :2]

        assert torch.allclose(result, expected)

    def test_parameter_compatibility(self):
        """Test that batch_size is only passed to functions that accept it"""

        @batch_fn
        def no_batch_param(x):
            # This function doesn't accept batch_size
            return x * 2  # Row-wise operation

        # Use 12 rows with batch_size=3 for even batches
        data = np.random.randn(12, 3)
        # Should work without error
        result = no_batch_param(data, batch_size=3)
        expected = data * 2

        np.testing.assert_allclose(result, expected)

    def test_with_kwargs(self):
        """Test batch processing with additional kwargs"""

        @batch_fn
        def scale_rows(x, scale=1.0, batch_size=4):
            return x * scale

        # Use 12 rows with batch_size=3 for even batches
        data = np.random.randn(12, 5)
        scale = 2.5

        result = scale_rows(data, scale=scale, batch_size=3)
        expected = data * scale

        np.testing.assert_allclose(result, expected)

    def test_empty_input(self):
        """Test handling of empty input"""

        @batch_fn
        def process(x, batch_size=4):
            return x * 2

        data = np.array([])
        result = process(data)

        assert len(result) == 0

    def test_uneven_batches(self):
        """Test handling of uneven batch sizes"""
        batch_sizes_seen = []

        @batch_fn
        def track_batch_size(x, batch_size=4):
            batch_sizes_seen.append(len(x))
            return x.reshape(-1, 1)  # Ensure 2D for vstack

        data = np.arange(10)  # 10 elements
        result = track_batch_size(data, batch_size=4)

        # Should see batches of size 4, 4, 2
        assert batch_sizes_seen == [4, 4, 2]
        np.testing.assert_array_equal(result.flatten(), data)

    def test_2d_array_processing(self):
        """Test that 2D arrays are processed correctly row-by-row"""

        @batch_fn
        def normalize_rows(x, batch_size=4):
            # Normalize each row to have mean 0
            return x - x.mean(axis=1, keepdims=True)

        # Use 12 rows with batch_size=3 for even batches
        data = np.random.randn(12, 5)
        result = normalize_rows(data, batch_size=3)
        expected = data - data.mean(axis=1, keepdims=True)

        np.testing.assert_allclose(result, expected)


class TestBatchFnWithOtherDecorators:
    """Test batch_fn combined with other decorators"""

    def test_with_torch_fn(self):
        """Test batch_fn with torch_fn for 2D operations"""

        @batch_fn
        @torch_fn
        def torch_scale(x, batch_size=4):
            # Returns 2D tensor - preserves shape
            return x * 3

        # Use 12 rows with batch_size=3 for even batches
        data = np.random.randn(12, 5)
        result = torch_scale(data, batch_size=3)
        expected = data * 3

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_with_numpy_fn(self):
        """Test batch_fn with numpy_fn for 2D operations"""

        @batch_fn
        @numpy_fn
        def numpy_scale(x, batch_size=4):
            # Returns 2D array - preserves shape
            return x * 2.5

        # Use 12 rows with batch_size=3 for even batches
        data = torch.randn(12, 5)
        result = numpy_scale(data, batch_size=3)
        expected = data.numpy() * 2.5

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_nested_decorator_context(self):
        """Test nested decorator context handling"""

        @batch_fn
        @torch_fn
        def nested_func(x, batch_size=4):
            # Returns 2D tensor - preserves shape
            return x + 1

        # Use 12 rows with batch_size=3 for even batches
        data = torch.randn(12, 5)
        result = nested_func(data, batch_size=3)
        expected = data + 1
        assert torch.allclose(result, expected)


class TestBatchFnEdgeCases:
    """Test edge cases for batch_fn"""

    def test_single_row(self):
        """Test with single row input"""

        @batch_fn
        def process(x, batch_size=4):
            return x * 2

        data = np.array([[1, 2, 3]])  # Single row
        result = process(data, batch_size=4)

        np.testing.assert_array_equal(result, data * 2)

    def test_exact_batch_boundary(self):
        """Test when data size is exact multiple of batch_size"""
        call_count = 0

        @batch_fn
        def count_calls(x, batch_size=4):
            nonlocal call_count
            call_count += 1
            return x

        data = np.arange(12).reshape(-1, 1)  # Exactly 3 batches of 4
        result = count_calls(data, batch_size=4)

        assert call_count == 3
        np.testing.assert_array_equal(result.flatten(), data.flatten())

    def test_preserves_dtype(self):
        """Test that dtype is preserved through batching"""

        @batch_fn
        def identity(x, batch_size=4):
            return x

        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        result = identity(data, batch_size=2)

        assert result.dtype == np.float32

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
