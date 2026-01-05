#!/usr/bin/env python3
# Time-stamp: "2026-01-05"
# File: ./tests/scitex/session/test_template.py

"""Tests for session template module.

The template.py file is an example script demonstrating @stx.session decorator usage.
These tests verify basic importability and structure.
"""

import pytest

# Required for scitex.session module
pytest.importorskip("natsort")
pytest.importorskip("h5py")
pytest.importorskip("zarr")


class TestTemplateModule:
    """Tests for session template module."""

    def test_template_importable(self):
        """Test template module can be imported."""
        from scitex.session import template

        assert template is not None

    def test_template_has_main_function(self):
        """Test template has main function defined."""
        from scitex.session import template

        assert hasattr(template, "main")
        assert callable(template.main)

    def test_main_is_session_wrapped(self):
        """Test main function is wrapped with @session decorator."""
        from scitex.session import template

        assert hasattr(template.main, "_is_session_wrapped")
        assert template.main._is_session_wrapped is True

    def test_main_has_original_function(self):
        """Test decorated main has reference to original function."""
        from scitex.session import template

        assert hasattr(template.main, "_func")
        assert callable(template.main._func)


class TestTemplateExecution:
    """Tests for template execution behavior."""

    def test_main_with_direct_call(self):
        """Test main can be called directly with arguments (bypasses session)."""
        from scitex.session import template

        # When called with arguments, it bypasses session management
        # and runs the function directly
        result = template.main(arg1="test", arg2="value")

        # Function returns None (pprint returns None)
        assert result is None


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/session/template.py
# --------------------------------------------------------------------------------
# (Source code reference maintained for sync_tests_with_source.sh)
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/session/template.py
# --------------------------------------------------------------------------------
