#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: tests/scitex/introspect/test__examples.py

"""Tests for scitex.introspect._examples module."""

import pytest


class TestFindExamples:
    """Tests for find_examples function."""

    def test_find_examples_success(self):
        """Test finding examples successfully."""
        from scitex.introspect import find_examples

        result = find_examples("scitex.introspect.q")
        assert result["success"] is True
        assert "examples" in result
        assert "count" in result

    def test_examples_with_search_paths(self):
        """Test finding examples with custom search paths."""
        from scitex.introspect import find_examples

        result = find_examples(
            "scitex.introspect.q",
            search_paths=["tests/scitex/introspect"],
        )
        assert result["success"] is True

    def test_examples_max_results(self):
        """Test max_results limits output."""
        from scitex.introspect import find_examples

        result = find_examples("scitex.introspect.q", max_results=2)
        assert result["success"] is True
        assert len(result["examples"]) <= 2

    def test_examples_has_context(self):
        """Test examples include context."""
        from scitex.introspect import find_examples

        result = find_examples("scitex.introspect.q")
        assert result["success"] is True
        if result["examples"]:
            for ex in result["examples"]:
                assert "file" in ex
                assert "line" in ex
                assert "context" in ex

    def test_examples_no_results(self):
        """Test no examples found returns empty list."""
        from scitex.introspect import find_examples

        result = find_examples("nonexistent_function_xyz123")
        # May return success=False if function not found, or success=True with empty list
        if result["success"]:
            assert result["count"] == 0
            assert result["examples"] == []
        else:
            # Function not found case
            assert "error" in result
