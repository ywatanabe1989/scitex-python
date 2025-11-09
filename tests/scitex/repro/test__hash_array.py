#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-11-09"
# File: ./tests/scitex/repro/test__hash_array.py

"""Tests for hash_array function."""

import pytest
import numpy as np
from scitex.repro import hash_array


class TestHashArrayBasic:
    """Basic hash_array functionality tests."""

    def test_hash_array_exists(self):
        """Test hash_array function exists."""
        assert callable(hash_array)

    def test_hash_simple_array(self):
        """Test hashing simple array."""
        arr = np.array([1, 2, 3, 4, 5])
        hash_val = hash_array(arr)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 16

    def test_hash_2d_array(self):
        """Test hashing 2D array."""
        arr = np.array([[1, 2], [3, 4]])
        hash_val = hash_array(arr)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 16

    def test_hash_float_array(self):
        """Test hashing float array."""
        arr = np.array([1.1, 2.2, 3.3])
        hash_val = hash_array(arr)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 16


class TestHashArrayDeterminism:
    """Test deterministic behavior of hash_array."""

    def test_same_data_same_hash(self):
        """Test same data produces same hash."""
        arr = np.array([1, 2, 3, 4, 5])
        hash1 = hash_array(arr)
        hash2 = hash_array(arr)

        assert hash1 == hash2

    def test_different_data_different_hash(self):
        """Test different data produces different hash."""
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([1, 2, 3, 4, 6])

        hash1 = hash_array(arr1)
        hash2 = hash_array(arr2)

        assert hash1 != hash2

    def test_copied_array_same_hash(self):
        """Test copied array produces same hash."""
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = arr1.copy()

        hash1 = hash_array(arr1)
        hash2 = hash_array(arr2)

        assert hash1 == hash2


class TestHashArrayTypes:
    """Test hash_array with different data types."""

    def test_int_array(self):
        """Test integer array."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        hash_val = hash_array(arr)
        assert len(hash_val) == 16

    def test_float_array(self):
        """Test float array."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        hash_val = hash_array(arr)
        assert len(hash_val) == 16

    def test_complex_array(self):
        """Test complex array."""
        arr = np.array([1+2j, 3+4j], dtype=np.complex128)
        hash_val = hash_array(arr)
        assert len(hash_val) == 16

    def test_bool_array(self):
        """Test boolean array."""
        arr = np.array([True, False, True])
        hash_val = hash_array(arr)
        assert len(hash_val) == 16


class TestHashArrayShapes:
    """Test hash_array with different array shapes."""

    def test_1d_array(self):
        """Test 1D array."""
        arr = np.array([1, 2, 3, 4, 5])
        hash_val = hash_array(arr)
        assert len(hash_val) == 16

    def test_2d_array(self):
        """Test 2D array."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        hash_val = hash_array(arr)
        assert len(hash_val) == 16

    def test_3d_array(self):
        """Test 3D array."""
        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        hash_val = hash_array(arr)
        assert len(hash_val) == 16

    def test_single_element(self):
        """Test single element array."""
        arr = np.array([42])
        hash_val = hash_array(arr)
        assert len(hash_val) == 16

    def test_empty_array(self):
        """Test empty array."""
        arr = np.array([])
        hash_val = hash_array(arr)
        assert len(hash_val) == 16


class TestHashArraySensitivity:
    """Test hash_array sensitivity to changes."""

    def test_order_matters(self):
        """Test that element order affects hash."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([3, 2, 1])

        hash1 = hash_array(arr1)
        hash2 = hash_array(arr2)

        assert hash1 != hash2

    def test_shape_ignored_same_data(self):
        """Test that shape doesn't affect hash if data is same.

        Note: hash_array uses tobytes() which flattens the array,
        so [1,2,3,4] and [[1,2],[3,4]] produce the same hash.
        """
        arr1 = np.array([1, 2, 3, 4])
        arr2 = np.array([[1, 2], [3, 4]])

        hash1 = hash_array(arr1)
        hash2 = hash_array(arr2)

        # Same underlying data = same hash (shape ignored)
        assert hash1 == hash2

    def test_dtype_matters(self):
        """Test that dtype affects hash."""
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([1, 2, 3], dtype=np.int64)

        hash1 = hash_array(arr1)
        hash2 = hash_array(arr2)

        # Different dtypes might give different hashes
        # (byte representation is different)
        assert hash1 != hash2

    def test_small_value_change(self):
        """Test sensitivity to small changes."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0000001])

        hash1 = hash_array(arr1)
        hash2 = hash_array(arr2)

        # Even tiny differences should produce different hashes
        assert hash1 != hash2


class TestHashArrayReproducibility:
    """Test hash_array for reproducibility verification."""

    def test_reproducible_generation(self):
        """Test hash verification with reproducible generation."""
        from scitex.repro import RandomStateManager

        # Generate data
        mgr1 = RandomStateManager(seed=42, verbose=False)
        data1 = mgr1("data").random(100)
        hash1 = hash_array(data1)

        # Reproduce
        mgr2 = RandomStateManager(seed=42, verbose=False)
        data2 = mgr2("data").random(100)
        hash2 = hash_array(data2)

        # Hashes should match
        assert hash1 == hash2

    def test_different_seeds_different_hashes(self):
        """Test different seeds produce different hashes."""
        from scitex.repro import RandomStateManager

        mgr1 = RandomStateManager(seed=42, verbose=False)
        data1 = mgr1("data").random(100)
        hash1 = hash_array(data1)

        mgr2 = RandomStateManager(seed=123, verbose=False)
        data2 = mgr2("data").random(100)
        hash2 = hash_array(data2)

        # Different seeds should give different hashes
        assert hash1 != hash2


class TestHashArrayEdgeCases:
    """Test edge cases."""

    def test_very_large_array(self):
        """Test hashing very large array."""
        arr = np.random.rand(10000)
        hash_val = hash_array(arr)
        assert len(hash_val) == 16

    def test_array_with_nan(self):
        """Test array containing NaN."""
        arr = np.array([1.0, np.nan, 3.0])
        hash_val = hash_array(arr)
        assert len(hash_val) == 16

    def test_array_with_inf(self):
        """Test array containing infinity."""
        arr = np.array([1.0, np.inf, -np.inf])
        hash_val = hash_array(arr)
        assert len(hash_val) == 16

    def test_nan_determinism(self):
        """Test that NaN values produce consistent hashes."""
        arr1 = np.array([1.0, np.nan, 3.0])
        arr2 = np.array([1.0, np.nan, 3.0])

        hash1 = hash_array(arr1)
        hash2 = hash_array(arr2)

        # NaN should hash consistently
        assert hash1 == hash2


class TestHashArrayIntegration:
    """Integration tests with common workflows."""

    def test_data_integrity_check(self):
        """Test using hash for data integrity checks."""
        # Original data
        data = np.array([1, 2, 3, 4, 5])
        original_hash = hash_array(data)

        # Process data
        processed = data * 2

        # Verify original unchanged
        new_hash = hash_array(data)
        assert original_hash == new_hash

        # Processed should be different
        processed_hash = hash_array(processed)
        assert processed_hash != original_hash

    def test_experiment_verification_workflow(self):
        """Test typical experiment verification workflow."""
        from scitex.repro import RandomStateManager, gen_id

        # Run experiment
        exp_id = gen_id(N=6)
        mgr = RandomStateManager(seed=42, verbose=False)
        results = mgr("experiment").random(100)
        results_hash = hash_array(results)

        # Later verification
        mgr2 = RandomStateManager(seed=42, verbose=False)
        verified_results = mgr2("experiment").random(100)
        verified_hash = hash_array(verified_results)

        # Should match
        assert results_hash == verified_hash


if __name__ == "__main__":
    import os
    pytest.main([os.path.abspath(__file__), "-v"])


# EOF
