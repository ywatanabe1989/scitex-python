#!/usr/bin/env python3
"""Shared fixtures for dev.plt tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Provide a numpy random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def mock_plt(monkeypatch):
    """Provide a mock plt module that returns mock figure and axes."""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    return plt
