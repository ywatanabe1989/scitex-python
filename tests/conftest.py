#!/usr/bin/env python3
"""
Pytest configuration for scitex tests.

Path setup is handled by pyproject.toml's pythonpath setting.
"""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def redirect_catboost_info(tmp_path_factory):
    """Redirect catboost training artifacts to temp directory."""
    catboost_dir = tmp_path_factory.mktemp("catboost_info")
    # Set environment variable that catboost respects
    os.environ["CATBOOST_TRAIN_DIR"] = str(catboost_dir)
    yield catboost_dir
    # Cleanup is automatic with tmp_path_factory


def pytest_collect_file(file_path):
    """Only collect test files that actually contain test functions."""
    if str(file_path).endswith(".py") and (
        file_path.name.startswith("test_") or file_path.name.endswith("_test.py")
    ):
        try:
            content = Path(file_path).read_text()
            if "def test_" not in content:
                return None
        except Exception:
            pass
    return None
