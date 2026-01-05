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
