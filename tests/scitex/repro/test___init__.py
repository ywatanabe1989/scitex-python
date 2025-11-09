#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-11-09 (updated)"
# File: ./scitex_repo/tests/scitex/repro/test___init__.py

"""Tests for unified repro module __init__.py."""

import pytest
import scitex.repro
import warnings


def test_repro_module_imports():
    """Test that repro module imports all expected functions and classes."""
    # Check that expected items are available
    expected_items = [
        # ID and timestamp utilities
        'gen_id',
        'gen_ID',  # backward compatibility
        'gen_timestamp',
        'timestamp',  # backward compatibility

        # Hash utilities
        'hash_array',

        # Random state management (moved from scitex.rng)
        'RandomStateManager',
        'get',
        'reset',

        # Legacy (deprecated)
        'fix_seeds',
    ]

    for item_name in expected_items:
        assert hasattr(scitex.repro, item_name), f"Missing {item_name} in scitex.repro"


def test_random_state_manager_available():
    """Test that RandomStateManager is properly imported."""
    assert hasattr(scitex.repro, 'RandomStateManager')
    assert callable(scitex.repro.RandomStateManager)

    # Can create instance
    mgr = scitex.repro.RandomStateManager(seed=42, verbose=False)
    assert mgr is not None


def test_get_and_reset_functions():
    """Test global RandomStateManager get and reset functions."""
    # Test get function
    mgr1 = scitex.repro.get()
    assert mgr1 is not None

    # Get again should return same instance
    mgr2 = scitex.repro.get()
    assert mgr1 is mgr2

    # Reset should create new instance
    mgr3 = scitex.repro.reset(seed=123, verbose=False)
    assert mgr3 is not mgr1


def test_hash_array_available():
    """Test that hash_array is properly imported."""
    import numpy as np

    assert hasattr(scitex.repro, 'hash_array')
    assert callable(scitex.repro.hash_array)

    # Can hash arrays
    arr = np.array([1, 2, 3, 4, 5])
    hash1 = scitex.repro.hash_array(arr)
    assert isinstance(hash1, str)
    assert len(hash1) == 16


def test_gen_id_functionality():
    """Test gen_id works when imported from scitex.repro."""
    # Basic test
    id1 = scitex.repro.gen_id()
    assert isinstance(id1, str)
    assert '_' in id1

    # Custom parameters
    id2 = scitex.repro.gen_id(N=4)
    parts = id2.split('_')
    assert len(parts[1]) == 4


def test_gen_timestamp_functionality():
    """Test gen_timestamp works when imported from scitex.repro."""
    ts = scitex.repro.gen_timestamp()
    assert isinstance(ts, str)
    assert len(ts) == 14  # YYYY-MMDD-HHMM format

    # Also test alias
    ts2 = scitex.repro.timestamp()
    assert isinstance(ts2, str)
    assert len(ts2) == 14


def test_fix_seeds_deprecated():
    """Test fix_seeds is available but deprecated."""
    # Should show deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scitex.repro.fix_seeds(seed=42, verbose=False)

        # Should have deprecation warning
        assert len(w) > 0
        assert issubclass(w[0].category, DeprecationWarning)
        assert "fix_seeds is deprecated" in str(w[0].message)


def test_backward_compatibility_aliases():
    """Test backward compatibility aliases."""
    # gen_ID should be available and same as gen_id
    assert hasattr(scitex.repro, 'gen_ID')
    assert scitex.repro.gen_ID is scitex.repro.gen_id

    # timestamp should be available and same as gen_timestamp
    assert hasattr(scitex.repro, 'timestamp')
    assert scitex.repro.timestamp is scitex.repro.gen_timestamp


def test_module_documentation():
    """Test that imported functions retain documentation."""
    # Check docstrings exist
    assert scitex.repro.gen_id.__doc__ is not None
    assert scitex.repro.gen_timestamp.__doc__ is not None
    assert scitex.repro.hash_array.__doc__ is not None
    assert scitex.repro.RandomStateManager.__doc__ is not None

    # Check they contain expected content
    assert "unique identifier" in scitex.repro.gen_id.__doc__.lower()
    assert "timestamp" in scitex.repro.gen_timestamp.__doc__.lower()
    assert "hash" in scitex.repro.hash_array.__doc__.lower()


def test_random_state_manager_workflow():
    """Test typical RandomStateManager workflow."""
    import numpy as np

    # Create manager
    mgr = scitex.repro.RandomStateManager(seed=42, verbose=False)

    # Get named generator
    gen = mgr("test")
    assert gen is not None

    # Generate data
    data1 = gen.random(10)
    assert len(data1) == 10

    # Same name should give same generator (and thus different data)
    gen2 = mgr("test")
    data2 = gen2.random(10)
    assert not np.array_equal(data1, data2)  # Different because state advanced

    # New manager with same seed + same name = reproducible
    mgr_new = scitex.repro.RandomStateManager(seed=42, verbose=False)
    gen_new = mgr_new("test")
    data3 = gen_new.random(10)
    assert np.array_equal(data1, data3)  # Reproduces original


def test_hash_array_reproducibility():
    """Test hash_array for reproducibility verification."""
    import numpy as np

    # Same data should produce same hash
    data = np.array([1, 2, 3, 4, 5])
    hash1 = scitex.repro.hash_array(data)
    hash2 = scitex.repro.hash_array(data)
    assert hash1 == hash2

    # Different data should produce different hash
    data2 = np.array([1, 2, 3, 4, 6])
    hash3 = scitex.repro.hash_array(data2)
    assert hash1 != hash3


def test_all_in_all():
    """Test that __all__ matches what's actually exported."""
    # Get __all__
    all_items = scitex.repro.__all__

    # Check all items in __all__ are actually available
    for item_name in all_items:
        assert hasattr(scitex.repro, item_name), f"{item_name} in __all__ but not available"


def test_reproducibility_workflow():
    """Test complete reproducibility workflow using the unified module."""
    import numpy as np

    # Create RandomStateManager
    mgr = scitex.repro.RandomStateManager(seed=42, verbose=False)

    # Generate reproducible ID
    exp_id = scitex.repro.gen_id(N=6)
    assert len(exp_id.split('_')[1]) == 6

    # Generate timestamp
    ts = scitex.repro.gen_timestamp()
    assert len(ts) == 14

    # Create reproducible random data
    gen = mgr("experiment")
    data1 = gen.random(10)

    # Hash the data
    hash1 = scitex.repro.hash_array(data1)

    # Reset and reproduce
    mgr2 = scitex.repro.RandomStateManager(seed=42, verbose=False)
    gen2 = mgr2("experiment")
    data2 = gen2.random(10)

    # Should be reproducible
    assert np.array_equal(data1, data2)

    # Hash should match
    hash2 = scitex.repro.hash_array(data2)
    assert hash1 == hash2


def test_deprecated_rng_module_alias():
    """Test that scitex.rng still works but is deprecated."""
    # Should show deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from scitex.rng import RandomStateManager

        # Should have deprecation warning
        assert len(w) > 0
        assert issubclass(w[0].category, DeprecationWarning)
        assert "rng" in str(w[0].message).lower()
        assert "repro" in str(w[0].message).lower()


def test_deprecated_reproduce_module_alias():
    """Test that scitex.reproduce still works but is deprecated."""
    # Should show deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from scitex.reproduce import gen_id

        # Should have deprecation warning
        assert len(w) > 0
        assert issubclass(w[0].category, DeprecationWarning)
        assert "reproduce" in str(w[0].message).lower()
        assert "repro" in str(w[0].message).lower()


if __name__ == "__main__":
    import os
    pytest.main([os.path.abspath(__file__), "-v"])


# EOF
