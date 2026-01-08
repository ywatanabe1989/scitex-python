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

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/repro/_hash_array.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-09-14 02:18:30 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/reproduce/_hash_array.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import hashlib
# 
# import numpy as np
# 
# 
# def hash_array(array_data: np.ndarray) -> str:
#     """Generate hash for array data.
# 
#     Creates a deterministic hash for numpy arrays, useful for
#     verifying data integrity and reproducibility.
# 
#     Parameters
#     ----------
#     array_data : np.ndarray
#         Array to hash
# 
#     Returns
#     -------
#     str
#         16-character hash string
# 
#     Examples
#     --------
#     >>> import numpy as np
#     >>> data = np.array([1, 2, 3, 4, 5])
#     >>> hash1 = hash_array(data)
#     >>> hash2 = hash_array(data)
#     >>> hash1 == hash2
#     True
#     """
#     data_bytes = array_data.tobytes()
#     return hashlib.sha256(data_bytes).hexdigest()[:16]
# 
# 
# # ================================================================================
# # Example Usage
# # ================================================================================
# def parse_args():
#     """Parse command line arguments."""
#     import argparse
# 
#     parser = argparse.ArgumentParser(description="Demonstrate array hashing")
#     parser.add_argument(
#         "--size", type=int, default=100, help="Array size (default: 100)"
#     )
#     parser.add_argument(
#         "--seed", type=int, default=42, help="Random seed (default: 42)"
#     )
#     return parser.parse_args()
# 
# 
# def main(args):
#     """Main execution function.
# 
#     Demonstrates array hashing for reproducibility verification.
#     """
#     print(f"\n{'=' * 60}")
#     print("Array Hashing Demo")
#     print(f"{'=' * 60}")
#     print(f"Array size: {args.size}")
#     print(f"Seed: {args.seed}")
# 
#     # Generate arrays using rng
#     gen = rng("demo")
# 
#     # Create array and hash it
#     print(f"\n{'Hash Generation':-^60}")
#     data1 = gen.random(args.size)
#     hash1 = hash_array(data1)
#     print(f"Array 1 hash: {hash1}")
# 
#     # Same data should produce same hash
#     hash1_again = hash_array(data1)
#     print(f"Array 1 hash (again): {hash1_again}")
#     print(f"Hashes match: {hash1 == hash1_again}")
# 
#     # Different data should produce different hash
#     print(f"\n{'Different Data':-^60}")
#     data2 = gen.random(args.size)
#     hash2 = hash_array(data2)
#     print(f"Array 2 hash: {hash2}")
#     print(f"Hashes differ: {hash1 != hash2}")
# 
#     # Reset generator and create same data
#     print(f"\n{'Reproducibility Check':-^60}")
#     gen_repro = rng("demo")  # Same name = same seed
#     data3 = gen_repro.random(args.size)
#     hash3 = hash_array(data3)
#     print(f"Array 3 hash (reproduced): {hash3}")
#     print(f"Reproduces original: {hash1 == hash3}")
# 
#     print(f"\n{'=' * 60}")
#     print("Demo completed successfully!")
#     print(f"{'=' * 60}\n")
# 
#     return 0
# 
# 
# if __name__ == "__main__":
#     import sys
#     import matplotlib.pyplot as plt
#     import scitex as stx
# 
#     args = parse_args()
# 
#     CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
#         sys,
#         plt,
#         args=args,
#         file=__file__,
#         sdir_suffix="hash_array_demo",
#         verbose=True,
#         agg=True,
#         seed=args.seed,
#     )
# 
#     exit_status = main(args)
# 
#     stx.session.close(
#         CONFIG,
#         verbose=True,
#         notify=False,
#         message="Array hashing demo completed",
#         exit_status=exit_status,
#     )
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/repro/_hash_array.py
# --------------------------------------------------------------------------------
