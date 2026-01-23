#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: tests/scitex/introspect/test__call_graph.py

"""Tests for scitex.introspect._call_graph module."""

import pytest


class TestGetCallGraph:
    """Tests for get_call_graph function."""

    def test_get_call_graph_function(self):
        """Test getting call graph for a function."""
        from scitex.introspect import get_call_graph

        result = get_call_graph("scitex.introspect._resolve.resolve_object")
        assert result["success"] is True
        assert "calls" in result
        assert "call_count" in result

    def test_call_graph_module(self):
        """Test getting call graph for a module."""
        from scitex.introspect import get_call_graph

        result = get_call_graph("scitex.introspect._resolve")
        assert result["success"] is True
        assert "graph" in result or "calls" in result

    def test_call_graph_with_timeout(self):
        """Test call graph respects timeout."""
        from scitex.introspect import get_call_graph

        # Short timeout should still work for small modules
        result = get_call_graph("scitex.introspect._resolve", timeout_seconds=30)
        assert "success" in result

    def test_call_graph_internal_only(self):
        """Test internal_only filters external calls."""
        from scitex.introspect import get_call_graph

        result = get_call_graph(
            "scitex.introspect._resolve.resolve_object", internal_only=True
        )
        assert result["success"] is True

    def test_call_graph_includes_external(self):
        """Test including external calls."""
        from scitex.introspect import get_call_graph

        result = get_call_graph(
            "scitex.introspect._resolve.resolve_object", internal_only=False
        )
        assert result["success"] is True
        # Should include calls to importlib, etc.

    def test_call_graph_invalid_path(self):
        """Test invalid path returns error."""
        from scitex.introspect import get_call_graph

        result = get_call_graph("nonexistent.module")
        assert result["success"] is False


class TestGetFunctionCalls:
    """Tests for get_function_calls function."""

    def test_get_function_calls_success(self):
        """Test getting function calls."""
        from scitex.introspect import get_function_calls

        result = get_function_calls("scitex.introspect._resolve.resolve_object")
        assert result["success"] is True
        assert "calls" in result

    def test_function_calls_builtin(self):
        """Test function calls for builtin fails gracefully."""
        from scitex.introspect import get_function_calls

        result = get_function_calls("len")
        # Builtins don't have source
        assert result["success"] is False
