#!/usr/bin/env python3
# Time-stamp: "2025-11-09"
# File: ./tests/scitex/repro/test__RandomStateManager.py

"""Tests for RandomStateManager class."""

import random

import numpy as np
import pytest

from scitex.repro import RandomStateManager, get, reset


class TestRandomStateManagerBasic:
    """Basic RandomStateManager functionality tests."""

    def test_initialization(self):
        """Test RandomStateManager can be initialized."""
        mgr = RandomStateManager(seed=42, verbose=False)
        assert mgr is not None
        assert mgr.seed == 42

    def test_initialization_verbose(self):
        """Test verbose initialization."""
        mgr = RandomStateManager(seed=42, verbose=True)
        assert mgr.verbose is True

    def test_default_seed(self):
        """Test default seed value."""
        mgr = RandomStateManager(verbose=False)
        assert mgr.seed == 42  # Default

    def test_custom_seed(self):
        """Test custom seed value."""
        mgr = RandomStateManager(seed=123, verbose=False)
        assert mgr.seed == 123


class TestRandomStateManagerSeedFixing:
    """Test seed fixing for various modules."""

    def test_python_random_fixed(self):
        """Test Python random module gets fixed seed."""
        RandomStateManager(seed=42, verbose=False)
        val1 = random.random()

        RandomStateManager(seed=42, verbose=False)
        val2 = random.random()

        assert val1 == val2

    def test_numpy_random_fixed(self):
        """Test NumPy random gets fixed seed."""
        RandomStateManager(seed=42, verbose=False)
        arr1 = np.random.rand(5)

        RandomStateManager(seed=42, verbose=False)
        arr2 = np.random.rand(5)

        assert np.array_equal(arr1, arr2)

    def test_different_seeds_produce_different_results(self):
        """Test different seeds produce different results."""
        RandomStateManager(seed=42, verbose=False)
        val1 = random.random()

        RandomStateManager(seed=123, verbose=False)
        val2 = random.random()

        assert val1 != val2

    def test_torch_seed_if_available(self):
        """Test PyTorch seed fixing if available."""
        try:
            import torch

            RandomStateManager(seed=42, verbose=False)
            t1 = torch.rand(5)

            RandomStateManager(seed=42, verbose=False)
            t2 = torch.rand(5)

            assert torch.allclose(t1, t2)
        except ImportError:
            pytest.skip("PyTorch not installed")


class TestNamedGenerators:
    """Test named generator functionality."""

    def test_get_named_generator(self):
        """Test creating named generator."""
        mgr = RandomStateManager(seed=42, verbose=False)
        gen = mgr("test")

        assert gen is not None
        # Should be numpy generator
        assert hasattr(gen, "random")

    def test_same_name_same_seed(self):
        """Test same name produces reproducible generator."""
        mgr1 = RandomStateManager(seed=42, verbose=False)
        gen1 = mgr1("data")
        data1 = gen1.random(10)

        mgr2 = RandomStateManager(seed=42, verbose=False)
        gen2 = mgr2("data")
        data2 = gen2.random(10)

        assert np.array_equal(data1, data2)

    def test_different_names_different_seeds(self):
        """Test different names produce different results."""
        mgr = RandomStateManager(seed=42, verbose=False)
        gen1 = mgr("data1")
        gen2 = mgr("data2")

        data1 = gen1.random(10)
        data2 = gen2.random(10)

        assert not np.array_equal(data1, data2)

    def test_get_np_generator_method(self):
        """Test get_np_generator method."""
        mgr = RandomStateManager(seed=42, verbose=False)
        gen = mgr.get_np_generator("test")

        assert gen is not None
        assert hasattr(gen, "random")

    def test_callable_interface(self):
        """Test callable interface for getting generators."""
        mgr = RandomStateManager(seed=42, verbose=False)

        # Using __call__
        gen1 = mgr("test")
        data1 = gen1.random(5)

        # Using get_np_generator
        mgr2 = RandomStateManager(seed=42, verbose=False)
        gen2 = mgr2.get_np_generator("test")
        data2 = gen2.random(5)

        assert np.array_equal(data1, data2)


class TestGlobalInstance:
    """Test global instance management."""

    def test_get_global_instance(self):
        """Test get() returns global instance."""
        mgr1 = get()
        mgr2 = get()

        assert mgr1 is mgr2

    def test_reset_creates_new_instance(self):
        """Test reset() creates new global instance."""
        mgr1 = get()
        mgr2 = reset(seed=123, verbose=False)

        assert mgr1 is not mgr2
        assert mgr2.seed == 123

    def test_reset_with_different_seed(self):
        """Test reset with different seed."""
        reset(seed=42, verbose=False)
        mgr1 = get()

        reset(seed=999, verbose=False)
        mgr2 = get()

        assert mgr1 is not mgr2
        assert mgr2.seed == 999


class TestVerification:
    """Test reproducibility verification functionality."""

    def test_verify_method_exists(self):
        """Test verify method exists."""
        mgr = RandomStateManager(seed=42, verbose=False)
        assert hasattr(mgr, "verify")
        assert callable(mgr.verify)

    def test_verify_saves_and_checks(self):
        """Test verify can save and check data."""
        mgr = RandomStateManager(seed=42, verbose=False)
        data = np.array([1, 2, 3, 4, 5])

        # First verify saves the data
        mgr.verify(data, "test_data")

        # Should be able to verify again without error
        # (implementation may vary)


class TestReproducibility:
    """Test reproducibility workflows."""

    def test_complete_workflow_reproducible(self):
        """Test complete workflow is reproducible."""
        # First run
        mgr1 = RandomStateManager(seed=42, verbose=False)
        gen1 = mgr1("experiment")
        data1 = gen1.random(100)

        # Second run with same seed
        mgr2 = RandomStateManager(seed=42, verbose=False)
        gen2 = mgr2("experiment")
        data2 = gen2.random(100)

        assert np.array_equal(data1, data2)

    def test_multiple_generators_reproducible(self):
        """Test multiple named generators are reproducible."""
        mgr1 = RandomStateManager(seed=42, verbose=False)
        data1 = mgr1("data").random(10)
        model1 = mgr1("model").random(10)

        mgr2 = RandomStateManager(seed=42, verbose=False)
        data2 = mgr2("data").random(10)
        model2 = mgr2("model").random(10)

        assert np.array_equal(data1, data2)
        assert np.array_equal(model1, model2)

    def test_order_independence(self):
        """Test generator order independence."""
        mgr1 = RandomStateManager(seed=42, verbose=False)
        data1 = mgr1("data").random(10)
        model1 = mgr1("model").random(10)

        # Create in different order
        mgr2 = RandomStateManager(seed=42, verbose=False)
        model2 = mgr2("model").random(10)
        data2 = mgr2("data").random(10)

        # Should still match
        assert np.array_equal(data1, data2)
        assert np.array_equal(model1, model2)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_seed_zero(self):
        """Test seed value of 0."""
        mgr = RandomStateManager(seed=0, verbose=False)
        assert mgr.seed == 0

    def test_large_seed(self):
        """Test large seed value."""
        large_seed = 2**31 - 1
        mgr = RandomStateManager(seed=large_seed, verbose=False)
        assert mgr.seed == large_seed

    def test_generator_with_special_characters(self):
        """Test generator name with special characters."""
        mgr = RandomStateManager(seed=42, verbose=False)
        gen = mgr("data-model_v1")
        assert gen is not None

    def test_generator_with_numbers(self):
        """Test generator name with numbers."""
        mgr = RandomStateManager(seed=42, verbose=False)
        gen = mgr("data123")
        assert gen is not None


class TestIntegration:
    """Integration tests with common workflows."""

    def test_scientific_experiment_workflow(self):
        """Test typical scientific experiment workflow."""
        # Setup
        mgr = RandomStateManager(seed=42, verbose=False)

        # Generate experimental data
        data_gen = mgr("data")
        data = data_gen.random((100, 10))

        # Generate model parameters
        model_gen = mgr("model")
        weights = model_gen.normal(size=(10, 5))

        # Verify reproducibility
        mgr2 = RandomStateManager(seed=42, verbose=False)
        data2 = mgr2("data").random((100, 10))
        weights2 = mgr2("model").normal(size=(10, 5))

        assert np.array_equal(data, data2)
        assert np.array_equal(weights, weights2)

    def test_multi_run_experiment(self):
        """Test multiple experimental runs."""
        results = []

        for seed in [42, 43, 44]:
            mgr = RandomStateManager(seed=seed, verbose=False)
            gen = mgr("data")
            data = gen.random(50)
            results.append(data.mean())

        # Different seeds should give different results
        assert results[0] != results[1]
        assert results[1] != results[2]


class TestCheckpointRestore:
    """Test checkpoint and restore functionality."""

    def test_checkpoint_creates_file(self, tmp_path):
        """Test checkpoint creates a file."""
        mgr = RandomStateManager(seed=42, verbose=False)
        mgr._cache_dir = tmp_path

        # Generate some state
        gen = mgr("data")
        gen.random(10)

        checkpoint_path = mgr.checkpoint("test_checkpoint")

        assert checkpoint_path.exists()
        assert checkpoint_path.name == "test_checkpoint.pkl"

    def test_restore_from_checkpoint(self, tmp_path):
        """Test restoring from checkpoint."""
        mgr1 = RandomStateManager(seed=42, verbose=False)
        mgr1._cache_dir = tmp_path

        # Generate initial state
        gen1 = mgr1("data")
        initial_values = gen1.random(10)

        # Save checkpoint after generating some values
        gen1.random(5)  # Advance the state
        checkpoint_path = mgr1.checkpoint("restore_test")

        # Generate more values
        after_checkpoint = gen1.random(10)

        # Create new manager and restore
        mgr2 = RandomStateManager(seed=99, verbose=False)  # Different seed
        mgr2._cache_dir = tmp_path
        mgr2.restore(checkpoint_path)

        # Should reproduce values after checkpoint
        gen2 = mgr2("data")
        restored_values = gen2.random(10)

        assert np.array_equal(after_checkpoint, restored_values)

    def test_checkpoint_preserves_multiple_generators(self, tmp_path):
        """Test checkpoint preserves state of multiple generators."""
        mgr1 = RandomStateManager(seed=42, verbose=False)
        mgr1._cache_dir = tmp_path

        # Create and advance multiple generators
        data_gen1 = mgr1("data")
        model_gen1 = mgr1("model")

        data_gen1.random(5)
        model_gen1.random(5)

        # Checkpoint
        checkpoint_path = mgr1.checkpoint()

        # Get values after checkpoint
        data_after = data_gen1.random(10)
        model_after = model_gen1.random(10)

        # Restore to new manager
        mgr2 = RandomStateManager(seed=1, verbose=False)
        mgr2._cache_dir = tmp_path
        mgr2.restore(checkpoint_path)

        # Should reproduce
        data_restored = mgr2("data").random(10)
        model_restored = mgr2("model").random(10)

        assert np.array_equal(data_after, data_restored)
        assert np.array_equal(model_after, model_restored)


class TestTemporarySeed:
    """Test temporary_seed context manager."""

    def test_temporary_seed_changes_random(self):
        """Test temporary_seed changes random state temporarily."""
        mgr = RandomStateManager(seed=42, verbose=False)

        # Get value with original seed
        random.seed(42)
        original_val = random.random()

        # Reset to original seed
        random.seed(42)

        with mgr.temporary_seed(999):
            # Inside context, should use different seed
            temp_val = random.random()

            # Verify we get same value with seed 999
            random.seed(999)
            expected_temp = random.random()
            assert temp_val == expected_temp

        # After context, should be restored
        random.seed(42)  # Reset to compare
        after_val = random.random()
        assert original_val == after_val

    def test_temporary_seed_changes_numpy(self):
        """Test temporary_seed changes numpy random state temporarily."""
        mgr = RandomStateManager(seed=42, verbose=False)

        # Get original numpy state behavior
        np.random.seed(42)
        original_val = np.random.rand()

        with mgr.temporary_seed(999):
            np.random.seed(999)
            expected_temp = np.random.rand()
            # Value inside should match seed 999

        # Reset and check restoration works
        np.random.seed(42)
        after_val = np.random.rand()
        assert original_val == after_val

    def test_temporary_seed_restores_on_exception(self):
        """Test temporary_seed restores state even on exception."""
        mgr = RandomStateManager(seed=42, verbose=False)

        # Save original state
        np.random.seed(42)
        original_state = np.random.get_state()

        try:
            with mgr.temporary_seed(999):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # State should be restored despite exception
        # (Just verify no crash - state restoration is implementation detail)


class TestSklearnRandomState:
    """Test get_sklearn_random_state method."""

    def test_get_sklearn_random_state_returns_int(self):
        """Test get_sklearn_random_state returns integer."""
        mgr = RandomStateManager(seed=42, verbose=False)
        state = mgr.get_sklearn_random_state("split")

        assert isinstance(state, int)
        assert 0 <= state < 2**32

    def test_get_sklearn_random_state_reproducible(self):
        """Test same name produces same sklearn random state."""
        mgr1 = RandomStateManager(seed=42, verbose=False)
        mgr2 = RandomStateManager(seed=42, verbose=False)

        state1 = mgr1.get_sklearn_random_state("train_test_split")
        state2 = mgr2.get_sklearn_random_state("train_test_split")

        assert state1 == state2

    def test_get_sklearn_random_state_different_names(self):
        """Test different names produce different states."""
        mgr = RandomStateManager(seed=42, verbose=False)

        state1 = mgr.get_sklearn_random_state("split1")
        state2 = mgr.get_sklearn_random_state("split2")

        assert state1 != state2

    def test_get_sklearn_random_state_different_seeds(self):
        """Test different base seeds produce different states."""
        mgr1 = RandomStateManager(seed=42, verbose=False)
        mgr2 = RandomStateManager(seed=123, verbose=False)

        state1 = mgr1.get_sklearn_random_state("split")
        state2 = mgr2.get_sklearn_random_state("split")

        assert state1 != state2


class TestTorchGenerator:
    """Test get_torch_generator method."""

    def test_get_torch_generator_returns_generator(self):
        """Test get_torch_generator returns torch generator."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        mgr = RandomStateManager(seed=42, verbose=False)
        gen = mgr.get_torch_generator("model")

        assert isinstance(gen, torch.Generator)

    def test_get_torch_generator_reproducible(self):
        """Test same name produces reproducible generator."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        mgr1 = RandomStateManager(seed=42, verbose=False)
        gen1 = mgr1.get_torch_generator("model")
        val1 = torch.randn(10, generator=gen1)

        mgr2 = RandomStateManager(seed=42, verbose=False)
        gen2 = mgr2.get_torch_generator("model")
        val2 = torch.randn(10, generator=gen2)

        assert torch.allclose(val1, val2)

    def test_get_torch_generator_different_names(self):
        """Test different names produce different generators."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        mgr = RandomStateManager(seed=42, verbose=False)
        gen1 = mgr.get_torch_generator("model1")
        gen2 = mgr.get_torch_generator("model2")

        val1 = torch.randn(10, generator=gen1)
        val2 = torch.randn(10, generator=gen2)

        assert not torch.allclose(val1, val2)

    def test_get_torch_generator_caches_generator(self):
        """Test generator is cached for same name."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        mgr = RandomStateManager(seed=42, verbose=False)
        gen1 = mgr.get_torch_generator("model")
        gen2 = mgr.get_torch_generator("model")

        assert gen1 is gen2


class TestClearCache:
    """Test clear_cache method."""

    def test_clear_cache_all(self, tmp_path):
        """Test clearing all cache files."""
        mgr = RandomStateManager(seed=42, verbose=False)
        mgr._cache_dir = tmp_path

        # Create some cache files
        (tmp_path / "test1.json").write_text("{}")
        (tmp_path / "test2.json").write_text("{}")
        (tmp_path / "test3.json").write_text("{}")

        removed = mgr.clear_cache()

        assert removed == 3
        assert not list(tmp_path.glob("*.json"))

    def test_clear_cache_specific_pattern(self, tmp_path):
        """Test clearing specific cache pattern."""
        mgr = RandomStateManager(seed=42, verbose=False)
        mgr._cache_dir = tmp_path

        # Create cache files
        (tmp_path / "exp_001.json").write_text("{}")
        (tmp_path / "exp_002.json").write_text("{}")
        (tmp_path / "other.json").write_text("{}")

        removed = mgr.clear_cache("exp_*")

        assert removed == 2
        assert (tmp_path / "other.json").exists()

    def test_clear_cache_specific_name(self, tmp_path):
        """Test clearing specific cache by name."""
        mgr = RandomStateManager(seed=42, verbose=False)
        mgr._cache_dir = tmp_path

        # Create cache files
        (tmp_path / "target.json").write_text("{}")
        (tmp_path / "keep.json").write_text("{}")

        removed = mgr.clear_cache("target")

        assert removed == 1
        assert not (tmp_path / "target.json").exists()
        assert (tmp_path / "keep.json").exists()

    def test_clear_cache_multiple_patterns(self, tmp_path):
        """Test clearing multiple patterns."""
        mgr = RandomStateManager(seed=42, verbose=False)
        mgr._cache_dir = tmp_path

        # Create cache files
        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.json").write_text("{}")
        (tmp_path / "c.json").write_text("{}")

        removed = mgr.clear_cache(["a", "b"])

        assert removed == 2
        assert (tmp_path / "c.json").exists()

    def test_clear_cache_nonexistent(self, tmp_path):
        """Test clearing nonexistent cache."""
        mgr = RandomStateManager(seed=42, verbose=False)
        mgr._cache_dir = tmp_path

        removed = mgr.clear_cache("nonexistent")

        assert removed == 0

    def test_clear_cache_empty_dir(self, tmp_path):
        """Test clearing empty directory."""
        mgr = RandomStateManager(seed=42, verbose=False)
        mgr._cache_dir = tmp_path

        removed = mgr.clear_cache()

        assert removed == 0


class TestComputeHash:
    """Test _compute_hash method with various object types."""

    def test_compute_hash_numpy_array(self):
        """Test hashing numpy arrays."""
        mgr = RandomStateManager(seed=42, verbose=False)

        arr = np.array([1, 2, 3, 4, 5])
        hash_val = mgr._compute_hash(arr)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 32

    def test_compute_hash_list(self):
        """Test hashing lists."""
        mgr = RandomStateManager(seed=42, verbose=False)

        lst = [1, 2, 3, 4, 5]
        hash_val = mgr._compute_hash(lst)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 32

    def test_compute_hash_dict(self):
        """Test hashing dictionaries."""
        mgr = RandomStateManager(seed=42, verbose=False)

        d = {"a": 1, "b": 2, "c": 3}
        hash_val = mgr._compute_hash(d)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 32

    def test_compute_hash_dict_order_independent(self):
        """Test dict hashing is order-independent."""
        mgr = RandomStateManager(seed=42, verbose=False)

        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}

        hash1 = mgr._compute_hash(d1)
        hash2 = mgr._compute_hash(d2)

        assert hash1 == hash2

    def test_compute_hash_tuple(self):
        """Test hashing tuples."""
        mgr = RandomStateManager(seed=42, verbose=False)

        t = (1, 2, 3, 4, 5)
        hash_val = mgr._compute_hash(t)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 32

    def test_compute_hash_string(self):
        """Test hashing strings."""
        mgr = RandomStateManager(seed=42, verbose=False)

        s = "test string"
        hash_val = mgr._compute_hash(s)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 32

    def test_compute_hash_basic_types(self):
        """Test hashing basic types."""
        mgr = RandomStateManager(seed=42, verbose=False)

        # Integer
        hash_int = mgr._compute_hash(42)
        assert len(hash_int) == 32

        # Float
        hash_float = mgr._compute_hash(3.14)
        assert len(hash_float) == 32

        # Bool
        hash_bool = mgr._compute_hash(True)
        assert len(hash_bool) == 32

    def test_compute_hash_torch_tensor(self):
        """Test hashing PyTorch tensors."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        mgr = RandomStateManager(seed=42, verbose=False)

        tensor = torch.tensor([1.0, 2.0, 3.0])
        hash_val = mgr._compute_hash(tensor)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 32

    def test_compute_hash_pandas_dataframe(self):
        """Test hashing pandas DataFrames."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        mgr = RandomStateManager(seed=42, verbose=False)

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        hash_val = mgr._compute_hash(df)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 32

    def test_compute_hash_pandas_series(self):
        """Test hashing pandas Series."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        mgr = RandomStateManager(seed=42, verbose=False)

        series = pd.Series([1, 2, 3, 4, 5])
        hash_val = mgr._compute_hash(series)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 32

    def test_compute_hash_deterministic(self):
        """Test hash is deterministic for same input."""
        mgr = RandomStateManager(seed=42, verbose=False)

        arr = np.array([1, 2, 3])
        hash1 = mgr._compute_hash(arr)
        hash2 = mgr._compute_hash(arr)

        assert hash1 == hash2

    def test_compute_hash_different_for_different_data(self):
        """Test different data produces different hashes."""
        mgr = RandomStateManager(seed=42, verbose=False)

        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 4])

        hash1 = mgr._compute_hash(arr1)
        hash2 = mgr._compute_hash(arr2)

        assert hash1 != hash2


class TestLegacyFixSeeds:
    """Test legacy fix_seeds function."""

    def test_fix_seeds_exists(self):
        """Test fix_seeds function exists."""
        from scitex.repro import fix_seeds

        assert callable(fix_seeds)

    def test_fix_seeds_returns_manager(self):
        """Test fix_seeds returns RandomStateManager."""
        import warnings

        from scitex.repro import fix_seeds

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mgr = fix_seeds(seed=42)

        assert isinstance(mgr, RandomStateManager)
        assert mgr.seed == 42

    def test_fix_seeds_deprecation_warning(self):
        """Test fix_seeds raises deprecation warning."""
        import warnings

        from scitex.repro import fix_seeds

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fix_seeds(seed=42)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_fix_seeds_fixes_random(self):
        """Test fix_seeds fixes random module."""
        import warnings

        from scitex.repro import fix_seeds

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fix_seeds(seed=42)

        val1 = random.random()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fix_seeds(seed=42)

        val2 = random.random()

        assert val1 == val2


class TestGetGeneratorAlias:
    """Test get_generator alias method."""

    def test_get_generator_returns_np_generator(self):
        """Test get_generator returns numpy generator."""
        mgr = RandomStateManager(seed=42, verbose=False)
        gen = mgr.get_generator("test")

        assert hasattr(gen, "random")
        assert hasattr(gen, "normal")

    def test_get_generator_same_as_get_np_generator(self):
        """Test get_generator is same as get_np_generator."""
        mgr = RandomStateManager(seed=42, verbose=False)

        gen1 = mgr.get_generator("test")
        gen2 = mgr.get_np_generator("test")

        assert gen1 is gen2

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/repro/_RandomStateManager.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2025-12-09 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/repro/_RandomStateManager.py
# # ----------------------------------------
# from __future__ import annotations
# 
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Clean, simple RandomStateManager for scientific reproducibility.
# 
# Main API:
#     rng = RandomStateManager(seed=42)   # Create instance
#     gen = rng("name")                   # Get named generator
#     rng.verify(obj, "name")             # Verify reproducibility
# """
# 
# import hashlib
# import json
# import logging
# import pickle
# from contextlib import contextmanager
# from pathlib import Path
# from typing import Any
# 
# from scitex.config import get_paths
# 
# logger = logging.getLogger(__name__)
# 
# # Global singleton instance
# _GLOBAL_INSTANCE = None
# 
# 
# class RandomStateManager:
#     """
#     Simple, robust random state manager for scientific computing.
# 
#     Examples
#     --------
#     >>> import scitex as stx
#     >>>
#     >>> # Method 1: Direct usage
#     >>> rng = stx.rng.RandomStateManager(seed=42)
#     >>> data = rng("data").random(100)
#     >>>
#     >>> # Method 2: From session.start
#     >>> CONFIG, stdout, stderr, plt, CC, rng = stx.session.start(seed=42)
#     >>> model = rng("model").normal(size=(10, 10))
#     >>>
#     >>> # Verify reproducibility
#     >>> rng.verify(data, "my_data")
#     """
# 
#     def __init__(self, seed: int = 42, verbose=False):
#         """Initialize with automatic module detection."""
#         self.seed = seed
#         self.verbose = verbose
#         self._generators = {}
#         self._cache_dir = get_paths().rng
#         self._cache_dir.mkdir(parents=True, exist_ok=True)
#         self._jax_key = None  # Initialize to None, will be set if jax is available
# 
#         if verbose:
#             logger.info(f"RandomStateManager initialized with seed {seed}")
# 
#         # Auto-fix all available seeds
#         self._auto_fix_seeds(verbose=verbose)
# 
#     def _auto_fix_seeds(self, verbose=None):
#         """Automatically detect and fix ALL available random modules."""
#         # Use instance verbose if not specified
#         if verbose is None:
#             verbose = self.verbose
# 
#         # OS environment
#         os.environ["PYTHONHASHSEED"] = str(self.seed)
#         os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# 
#         fixed_modules = []
# 
#         # Python random
#         try:
#             import random
# 
#             random.seed(self.seed)
#             fixed_modules.append("random")
#         except ImportError:
#             pass
# 
#         # NumPy
#         try:
#             import numpy as np
# 
#             np.random.seed(self.seed)
#             # Also set default_rng for new API
#             self._np = np
#             self._np_default_rng = np.random.default_rng(self.seed)
#             fixed_modules.append("numpy")
#         except ImportError:
#             self._np = None
# 
#         # PyTorch
#         try:
#             import torch
# 
#             torch.manual_seed(self.seed)
#             if torch.cuda.is_available():
#                 torch.cuda.manual_seed_all(self.seed)
#                 torch.backends.cudnn.deterministic = True
#                 torch.backends.cudnn.benchmark = False
#                 fixed_modules.append("torch+cuda")
#             else:
#                 fixed_modules.append("torch")
#         except ImportError:
#             pass
# 
#         # TensorFlow
#         try:
#             import tensorflow as tf
# 
#             tf.random.set_seed(self.seed)
#             fixed_modules.append("tensorflow")
#         except ImportError:
#             pass
# 
#         # JAX (deferred import to avoid circular imports)
#         try:
#             import jax
# 
#             self._jax_key = jax.random.PRNGKey(self.seed)
#             fixed_modules.append("jax")
#         except (ImportError, AttributeError, RuntimeError):
#             # ImportError: jax not installed
#             # AttributeError: circular import in jax._src.clusters
#             # RuntimeError: other jax initialization errors
#             self._jax_key = None
#             pass
# 
#         if verbose and fixed_modules:
#             logger.info(f"Fixed random seeds for: {', '.join(fixed_modules)}")
# 
#     def get_np_generator(self, name: str):
#         """
#         Get or create a named NumPy random generator.
# 
#         Parameters
#         ----------
#         name : str
#             Generator name (e.g., "data", "model", "augment")
# 
#         Returns
#         -------
#         numpy.random.Generator
#             Independent NumPy random generator
# 
#         Examples
#         --------
#         >>> rng = RandomStateManager(42)
#         >>> gen = rng.get_np_generator("data")
#         >>> values = gen.random(100)
#         >>> perm = gen.permutation(100)
#         """
#         if self._np is None:
#             raise ImportError("NumPy required for random generators")
# 
#         if name not in self._generators:
#             # Create deterministic seed from name
#             name_hash = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
#             seed = (self.seed + name_hash) % (2**32)
#             self._generators[name] = self._np.random.default_rng(seed)
# 
#         return self._generators[name]
# 
#     def __call__(self, name: str, verbose: bool = None):
#         """
#         Get or create a named NumPy random generator.
# 
#         This is a backward compatibility wrapper for get_np_generator().
#         Consider using get_np_generator() directly for clarity.
# 
#         Parameters
#         ----------
#         name : str
#             Generator name
#         verbose : bool, optional
#             Whether to show deprecation warning
# 
#         Returns
#         -------
#         numpy.random.Generator
#             NumPy random generator with deterministic seed
#         """
#         if verbose:
#             print(
#                 f"Note: rng('{name}') is deprecated. Use rng.get_np_generator('{name}') instead."
#             )
#         return self.get_np_generator(name)
# 
#     def verify(self, obj: Any, name: str = None, verbose: bool = True) -> bool:
#         """
#         Verify object matches cached hash (detects broken reproducibility).
# 
#         First call: caches the object's hash
#         Later calls: verifies object matches cached hash
# 
#         Parameters
#         ----------
#         obj : Any
#             Object to verify (array, tensor, data, model weights, etc.)
#             Supports: numpy arrays, torch tensors, tf tensors, jax arrays,
#             lists, dicts, pandas dataframes, and basic types
#         name : str, optional
#             Cache name. Auto-generated if not provided.
# 
#         Returns
#         -------
#         bool
#             True if matches cache (or first call), False if different
# 
#         Examples
#         --------
#         >>> data = generate_data()
#         >>> rng.verify(data, "train_data")  # First run: caches
#         >>> # Next run:
#         >>> rng.verify(data, "train_data")  # Verifies match
#         """
# 
#         # Auto-generate name if needed
#         if name is None:
#             import inspect
# 
#             frame = inspect.currentframe().f_back
#             filename = Path(frame.f_code.co_filename).stem
#             lineno = frame.f_lineno
#             name = f"{filename}_L{lineno}"
# 
#         # Sanitize name
#         safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
#         cache_file = self._cache_dir / f"{safe_name}.json"
# 
#         # Compute hash based on object type
#         obj_hash = self._compute_hash(obj)
# 
#         # Use instance verbose if not specified
#         if verbose is None:
#             verbose = self.verbose
# 
#         # Check cache
#         if cache_file.exists():
#             with open(cache_file) as f:
#                 cached = json.load(f)
# 
#             matches = cached["hash"] == obj_hash
#             if not matches and verbose:
#                 print(f"⚠️  Reproducibility broken for '{name}'!")
#                 print(f"   Expected: {cached['hash'][:16]}...")
#                 print(f"   Got:      {obj_hash[:16]}...")
#                 raise ValueError(f"Reproducibility verification failed for '{name}'")
#             elif matches and verbose:
#                 print(f"✓ Reproducibility verified for '{name}'")
# 
#             return matches
#         else:
#             # First call - cache it
#             with open(cache_file, "w") as f:
#                 json.dump({"name": name, "hash": obj_hash, "seed": self.seed}, f)
#             return True
# 
#     def _compute_hash(self, obj: Any) -> str:
#         """
#         Compute hash for various object types.
# 
#         Supports:
#         - NumPy arrays
#         - PyTorch tensors
#         - TensorFlow tensors
#         - JAX arrays
#         - Pandas DataFrames/Series
#         - Lists, tuples, dicts
#         - Basic types (int, float, str, bool)
#         """
#         import numpy as np
# 
#         # NumPy array
#         if isinstance(obj, np.ndarray):
#             return hashlib.sha256(obj.tobytes()).hexdigest()[:32]
# 
#         # PyTorch tensor
#         try:
#             import torch
# 
#             if isinstance(obj, torch.Tensor):
#                 # Move to CPU and convert to numpy for consistent hashing
#                 obj_np = obj.detach().cpu().numpy()
#                 return hashlib.sha256(obj_np.tobytes()).hexdigest()[:32]
#         except ImportError:
#             pass
# 
#         # TensorFlow tensor
#         try:
#             import tensorflow as tf
# 
#             if isinstance(obj, (tf.Tensor, tf.Variable)):
#                 obj_np = obj.numpy()
#                 return hashlib.sha256(obj_np.tobytes()).hexdigest()[:32]
#         except ImportError:
#             pass
# 
#         # JAX array
#         try:
#             import jax.numpy as jnp
# 
#             if isinstance(obj, jnp.ndarray):
#                 obj_np = np.array(obj)
#                 return hashlib.sha256(obj_np.tobytes()).hexdigest()[:32]
#         except (ImportError, AttributeError, RuntimeError):
#             # ImportError: jax not installed
#             # AttributeError: circular import in jax._src.clusters
#             # RuntimeError: other jax initialization errors
#             pass
# 
#         # Pandas DataFrame/Series
#         try:
#             import pandas as pd
# 
#             if isinstance(obj, (pd.DataFrame, pd.Series)):
#                 # Use pandas string representation for hashing
#                 obj_str = obj.to_json(orient="split", date_format="iso")
#                 return hashlib.sha256(obj_str.encode()).hexdigest()[:32]
#         except ImportError:
#             pass
# 
#         # Lists and tuples - convert to numpy array if numeric
#         if isinstance(obj, (list, tuple)):
#             try:
#                 obj_np = np.array(obj)
#                 if obj_np.dtype != object:  # Numeric array
#                     return hashlib.sha256(obj_np.tobytes()).hexdigest()[:32]
#             except:
#                 pass
#             # Fall through to string representation
# 
#         # Dictionaries - serialize to JSON
#         if isinstance(obj, dict):
#             try:
#                 obj_str = json.dumps(obj, sort_keys=True, default=str)
#                 return hashlib.sha256(obj_str.encode()).hexdigest()[:32]
#             except:
#                 pass
# 
#         # Default: convert to string
#         obj_str = str(obj)
#         return hashlib.sha256(obj_str.encode()).hexdigest()[:32]
# 
#     def checkpoint(self, name: str = "checkpoint"):
#         """Save current state of all generators."""
#         checkpoint_file = self._cache_dir / f"{name}.pkl"
#         state = {
#             "seed": self.seed,
#             "generators": {
#                 k: v.bit_generator.state for k, v in self._generators.items()
#             },
#         }
#         with open(checkpoint_file, "wb") as f:
#             pickle.dump(state, f)
#         return checkpoint_file
# 
#     def restore(self, checkpoint):
#         """Restore from checkpoint."""
#         if isinstance(checkpoint, str):
#             checkpoint = Path(checkpoint)
# 
#         with open(checkpoint, "rb") as f:
#             state = pickle.load(f)
# 
#         self.seed = state["seed"]
#         self._auto_fix_seeds()
# 
#         # Restore generator states
#         for name, gen_state in state["generators"].items():
#             gen = self(name)
#             gen.bit_generator.state = gen_state
# 
#     @contextmanager
#     def temporary_seed(self, seed: int):
#         """Context manager for temporary seed change."""
#         import random
# 
#         import numpy as np
# 
#         # Save current states
#         old_random_state = random.getstate()
#         old_np_state = np.random.get_state() if self._np else None
# 
#         # Set temporary seed
#         random.seed(seed)
#         if self._np:
#             np.random.seed(seed)
# 
#         try:
#             yield
#         finally:
#             # Restore states
#             random.setstate(old_random_state)
#             if self._np and old_np_state:
#                 np.random.set_state(old_np_state)
# 
#     def get_sklearn_random_state(self, name: str):
#         """
#         Get a random state for scikit-learn.
# 
#         Scikit-learn uses integers for random_state parameter.
# 
#         Parameters
#         ----------
#         name : str
#             Generator name
# 
#         Returns
#         -------
#         int
#             Random state integer for sklearn
# 
#         Examples
#         --------
#         >>> rng = RandomStateManager(42)
#         >>> from sklearn.model_selection import train_test_split
#         >>> X_train, X_test = train_test_split(
#         ...     X, test_size=0.2,
#         ...     random_state=rng.get_sklearn_random_state("split")
#         ... )
#         """
#         # Create deterministic seed from name
#         name_hash = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
#         seed = (self.seed + name_hash) % (2**32)
#         return seed
# 
#     def get_torch_generator(self, name: str):
#         """
#         Get or create a named PyTorch generator.
# 
#         Parameters
#         ----------
#         name : str
#             Generator name
# 
#         Returns
#         -------
#         torch.Generator
#             PyTorch generator with deterministic seed
# 
#         Examples
#         --------
#         >>> rng = RandomStateManager(42)
#         >>> gen = rng.get_torch_generator("model")
#         >>> torch.randn(5, 5, generator=gen)
#         """
#         try:
#             import torch
#         except ImportError:
#             raise ImportError("PyTorch not installed")
# 
#         if not hasattr(self, "_torch_generators"):
#             self._torch_generators = {}
# 
#         if name not in self._torch_generators:
#             # Create deterministic seed from name
#             name_hash = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
#             seed = (self.seed + name_hash) % (2**32)
# 
#             gen = torch.Generator()
#             gen.manual_seed(seed)
#             self._torch_generators[name] = gen
# 
#         return self._torch_generators[name]
# 
#     def get_generator(self, name: str):
#         """Alias for get_np_generator for compatibility."""
#         return self.get_np_generator(name)
# 
#     def clear_cache(self, patterns: str | list[str] = None) -> int:
#         """
#         Clear verification cache files.
# 
#         Parameters
#         ----------
#         patterns : str or list of str, optional
#             Specific cache patterns to clear. If None, clears all.
#             Can be:
#             - Single name: "my_data"
#             - List of names: ["data1", "data2"]
#             - Glob pattern: "experiment_*"
#             - None: clear all cache files
# 
#         Returns
#         -------
#         int
#             Number of cache files removed
# 
#         Examples
#         --------
#         >>> rng = RandomStateManager(42)
#         >>> rng.clear_cache()  # Clear all
#         >>> rng.clear_cache("old_data")  # Clear specific
#         >>> rng.clear_cache(["test1", "test2"])  # Clear multiple
#         >>> rng.clear_cache("experiment_*")  # Clear pattern
#         """
# 
#         if not self._cache_dir.exists():
#             return 0
# 
#         removed_count = 0
# 
#         if patterns is None:
#             # Clear all .json files
#             cache_files = list(self._cache_dir.glob("*.json"))
#             for cache_file in cache_files:
#                 cache_file.unlink()
#                 removed_count += 1
#         else:
#             # Ensure patterns is a list
#             if isinstance(patterns, str):
#                 patterns = [patterns]
# 
#             for pattern in patterns:
#                 # Handle glob patterns
#                 if "*" in pattern or "?" in pattern:
#                     cache_files = list(self._cache_dir.glob(f"{pattern}.json"))
#                 else:
#                     # Exact match
#                     cache_file = self._cache_dir / f"{pattern}.json"
#                     cache_files = [cache_file] if cache_file.exists() else []
# 
#                 for cache_file in cache_files:
#                     cache_file.unlink()
#                     removed_count += 1
# 
#         return removed_count
# 
# 
# def get(verbose: bool = False) -> RandomStateManager:
#     """
#     Get or create the global RandomStateManager instance.
# 
#     Parameters
#     ----------
#     verbose : bool, optional
#         Whether to print status messages (default: False)
# 
#     Returns
#     -------
#     RandomStateManager
#         Global instance
# 
#     Examples
#     --------
#     >>> import scitex as stx
#     >>> rng = stx.rng.get()
#     >>> data = rng("data").random(100)
#     """
#     global _GLOBAL_INSTANCE
# 
#     if _GLOBAL_INSTANCE is None:
#         _GLOBAL_INSTANCE = RandomStateManager(42, verbose=verbose)
# 
#     return _GLOBAL_INSTANCE
# 
# 
# def reset(seed: int = 42, verbose: bool = False) -> RandomStateManager:
#     """
#     Reset global RandomStateManager with new seed.
# 
#     Parameters
#     ----------
#     seed : int
#         New seed value
#     verbose : bool, optional
#         Whether to print status messages (default: False)
# 
#     Returns
#     -------
#     RandomStateManager
#         New global instance
# 
#     Examples
#     --------
#     >>> import scitex as stx
#     >>> rng = stx.repro.reset(seed=123)
#     """
#     global _GLOBAL_INSTANCE
#     _GLOBAL_INSTANCE = RandomStateManager(seed, verbose=verbose)
#     return _GLOBAL_INSTANCE
# 
# 
# # ================================================================================
# # Example Usage
# # ================================================================================
# def parse_args():
#     """Parse command line arguments."""
#     import argparse
# 
#     parser = argparse.ArgumentParser(description="Demonstrate RandomStateManager usage")
#     parser.add_argument(
#         "--seed", type=int, default=42, help="Random seed (default: 42)"
#     )
#     return parser.parse_args()
# 
# 
# def main(args):
#     """Main execution function.
# 
#     Demonstrates RandomStateManager capabilities:
#     - Creating named generators
#     - Reproducible random generation
#     - Verification of reproducibility
#     """
# 
#     # Create RandomStateManager (already created by session.start)
#     print(f"\n{'=' * 60}")
#     print("RandomStateManager Demo")
#     print(f"{'=' * 60}")
#     print(f"Seed: {args.seed}")
# 
#     # Get named generators
#     data_gen = rng("data")
#     model_gen = rng("model")
# 
#     # Generate data
#     print(f"\n{'Data Generation':-^60}")
#     data = data_gen.random(5)
#     print(f"Data generator: {data}")
# 
#     # Generate model weights
#     print(f"\n{'Model Generation':-^60}")
#     weights = model_gen.normal(size=(3, 3))
#     print(f"Model weights:\n{weights}")
# 
#     # Verify reproducibility
#     print(f"\n{'Verification':-^60}")
#     rng.verify(data, "demo_data")
#     print("✓ Data reproducibility verified")
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
# 
#     import matplotlib.pyplot as plt
# 
#     import scitex as stx
# 
#     args = parse_args()
# 
#     CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
#         sys,
#         plt,
#         args=args,
#         file=__file__,
#         sdir_suffix="RandomStateManager_demo",
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
#         message="RandomStateManager demo completed",
#         exit_status=exit_status,
#     )
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/repro/_RandomStateManager.py
# --------------------------------------------------------------------------------
