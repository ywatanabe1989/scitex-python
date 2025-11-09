#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-11-09"
# File: ./tests/scitex/session/test___init__.py

"""Tests for session module __init__.py."""

import pytest
import scitex.session


def test_session_module_imports():
    """Test that session module imports all expected items."""
    expected_items = [
        # Session lifecycle
        'start',
        'close',
        'running2finished',

        # Session decorator
        'session',
        'run',

        # Advanced session management
        'SessionManager',
    ]

    for item_name in expected_items:
        assert hasattr(scitex.session, item_name), f"Missing {item_name} in scitex.session"


def test_start_function_exists():
    """Test start function is available."""
    assert callable(scitex.session.start)


def test_close_function_exists():
    """Test close function is available."""
    assert callable(scitex.session.close)


def test_running2finished_function_exists():
    """Test running2finished function is available."""
    assert callable(scitex.session.running2finished)


def test_session_decorator_exists():
    """Test session decorator is available."""
    assert callable(scitex.session.session)


def test_run_function_exists():
    """Test run function is available."""
    assert callable(scitex.session.run)


def test_session_manager_class_exists():
    """Test SessionManager class is available."""
    assert hasattr(scitex.session, 'SessionManager')
    # Can instantiate
    manager = scitex.session.SessionManager()
    assert manager is not None


def test_all_exports():
    """Test __all__ contains expected items."""
    expected_all = [
        'start',
        'close',
        'running2finished',
        'session',
        'run',
        'SessionManager',
    ]

    all_items = scitex.session.__all__

    for item in expected_all:
        assert item in all_items, f"{item} not in __all__"


def test_module_documentation():
    """Test module functions have documentation."""
    # Module __doc__ may be None due to import structure, that's OK
    # Just verify the functions have docs
    assert scitex.session.start.__doc__ is not None
    assert scitex.session.close.__doc__ is not None
    assert scitex.session.session.__doc__ is not None


def test_start_returns_six_values():
    """Test start() returns 6 values (CONFIG, stdout, stderr, plt, CC, rng_manager)."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        sdir = Path(tmpdir) / "test"

        result = scitex.session.start(
            sys=None,
            plt=None,
            sdir=sdir,
            verbose=False
        )

        # Should return tuple of 6 items
        assert isinstance(result, tuple)
        assert len(result) == 6

        CONFIG, stdout, stderr, plt, CC, rng_manager = result

        # Check types
        assert CONFIG is not None
        assert rng_manager is not None


def test_backward_compatibility_five_value_unpack():
    """Test that old code unpacking 5 values raises clear error."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        sdir = Path(tmpdir) / "test"

        # Old style (5 values) should fail
        with pytest.raises(ValueError, match="too many values to unpack"):
            CONFIG, stdout, stderr, plt, CC = scitex.session.start(
                sys=None,
                plt=None,
                sdir=sdir,
                verbose=False
            )


def test_function_signatures_preserved():
    """Test that imported functions preserve their signatures."""
    import inspect

    # Test start signature
    sig = inspect.signature(scitex.session.start)
    params = list(sig.parameters.keys())
    assert 'sys' in params
    assert 'plt' in params
    assert 'seed' in params

    # Test close signature
    sig = inspect.signature(scitex.session.close)
    params = list(sig.parameters.keys())
    assert 'CONFIG' in params

    # Test session decorator signature
    sig = inspect.signature(scitex.session.session)
    params = list(sig.parameters.keys())
    assert 'func' in params
    assert 'verbose' in params


if __name__ == "__main__":
    import os
    pytest.main([os.path.abspath(__file__), "-v"])


# EOF
