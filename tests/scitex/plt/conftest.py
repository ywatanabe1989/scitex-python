#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-09 23:30:00"
# File: ./tests/scitex/plt/conftest.py
"""Pytest configuration for scitex.plt tests.

This conftest ensures proper test isolation by reloading scitex.plt
after each test module to prevent state corruption between tests.
"""

import pytest
import sys


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Clean up matplotlib state after each test."""
    import matplotlib.pyplot as plt
    yield
    plt.close('all')


@pytest.fixture(autouse=True)
def ensure_scitex_plt_callable():
    """Ensure scitex.plt.subplots is a function (not corrupted module).

    This fixture runs before each test and verifies that scitex.plt.subplots
    is callable. If it's been corrupted to a module, it forces a reimport.
    """
    import scitex
    import scitex.plt

    # Check if subplots is callable
    if not callable(getattr(scitex.plt, 'subplots', None)):
        # Force reimport
        keys_to_remove = [k for k in sys.modules.keys() if k.startswith('scitex.plt')]
        for key in keys_to_remove:
            del sys.modules[key]
        if 'scitex' in sys.modules:
            del sys.modules['scitex']

        import scitex
        import scitex.plt

    yield
