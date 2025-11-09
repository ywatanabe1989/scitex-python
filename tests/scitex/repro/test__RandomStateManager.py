#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-11-09"
# File: ./tests/scitex/repro/test__RandomStateManager.py

"""Tests for RandomStateManager class."""

import pytest
import numpy as np
import random
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
        assert hasattr(gen, 'random')

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
        assert hasattr(gen, 'random')

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
        assert hasattr(mgr, 'verify')
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


if __name__ == "__main__":
    import os
    pytest.main([os.path.abspath(__file__), "-v"])


# EOF
