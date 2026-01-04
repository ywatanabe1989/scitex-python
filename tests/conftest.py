#!/usr/bin/env python3
"""
Pytest configuration for scitex tests.

Path setup is handled by pyproject.toml's pythonpath setting.
"""

from pathlib import Path


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
