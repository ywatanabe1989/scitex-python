#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-11-09"
# File: ./tests/scitex/repro/test__fix_seeds.py

"""
Minimal backward compatibility tests for deprecated fix_seeds function.

NOTE: fix_seeds is deprecated in favor of RandomStateManager.
See test__RandomStateManager.py for comprehensive reproducibility tests.
"""

import pytest
import warnings
import random
import numpy as np
from scitex.repro import fix_seeds, RandomStateManager


class TestFixSeedsDeprecated:
    """Test deprecated fix_seeds function for backward compatibility."""

    def test_fix_seeds_exists(self):
        """Test fix_seeds function still exists."""
        assert callable(fix_seeds)

    def test_fix_seeds_shows_deprecation_warning(self):
        """Test fix_seeds shows deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fix_seeds(seed=42, verbose=False)

            # Should have deprecation warning
            assert len(w) > 0
            assert issubclass(w[0].category, DeprecationWarning)
            assert "fix_seeds is deprecated" in str(w[0].message)
            assert "RandomStateManager" in str(w[0].message)

    def test_fix_seeds_returns_manager(self):
        """Test fix_seeds returns RandomStateManager instance."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fix_seeds(seed=42, verbose=False)

        assert isinstance(result, RandomStateManager)
        assert result.seed == 42

    def test_fix_seeds_actually_fixes_seeds(self):
        """Test fix_seeds actually fixes random seeds."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Fix seeds
            fix_seeds(seed=42, verbose=False)
            val1 = random.random()

            # Fix again with same seed
            fix_seeds(seed=42, verbose=False)
            val2 = random.random()

            # Should produce same value
            assert val1 == val2

    def test_fix_seeds_with_custom_seed(self):
        """Test fix_seeds with custom seed value."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            mgr = fix_seeds(seed=999, verbose=False)
            assert mgr.seed == 999

    def test_fix_seeds_verbose_option(self):
        """Test fix_seeds verbose parameter."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Should not raise error
            mgr = fix_seeds(seed=42, verbose=True)
            assert mgr is not None

    def test_fix_seeds_with_all_params(self):
        """Test fix_seeds with all legacy parameters."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Old signature included these params (now ignored)
            mgr = fix_seeds(
                seed=42,
                os=True,
                random=True,
                np=True,
                torch=True,
                tf=False,
                jax=False,
                verbose=False
            )

            assert isinstance(mgr, RandomStateManager)


class TestFixSeedsMigrationGuide:
    """Examples showing migration from fix_seeds to RandomStateManager."""

    def test_old_way_vs_new_way(self):
        """Show old way (fix_seeds) vs new way (RandomStateManager)."""
        # Old way (deprecated)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fix_seeds(seed=42, verbose=False)
            old_val = random.random()

        # New way (recommended)
        mgr = RandomStateManager(seed=42, verbose=False)
        new_val = random.random()

        # Both should work
        assert isinstance(old_val, float)
        assert isinstance(new_val, float)

    def test_migration_reproducibility(self):
        """Test that migration maintains reproducibility."""
        # Old approach
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fix_seeds(seed=42, verbose=False)

        val_old = random.random()

        # New approach
        RandomStateManager(seed=42, verbose=False)
        val_new = random.random()

        # Should produce same results
        assert val_old == val_new


if __name__ == "__main__":
    import os
    pytest.main([os.path.abspath(__file__), "-v"])


# EOF
