#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-11-09"
# File: ./tests/scitex/session/test__decorator.py

"""Tests for session decorator."""

import pytest
import tempfile
from pathlib import Path
from scitex.session import session


class TestSessionDecorator:
    """Test @session decorator functionality."""

    def test_decorator_exists(self):
        """Test session decorator is importable."""
        assert callable(session)

    def test_decorator_without_args(self):
        """Test decorator can be used without arguments."""
        @session
        def dummy_func():
            return 0

        assert hasattr(dummy_func, '_is_session_wrapped')
        assert dummy_func._is_session_wrapped is True

    def test_decorator_with_args(self):
        """Test decorator can be used with arguments."""
        @session(verbose=False, agg=True)
        def dummy_func():
            return 0

        assert hasattr(dummy_func, '_is_session_wrapped')
        assert dummy_func._is_session_wrapped is True

    def test_decorator_preserves_function_attributes(self):
        """Test decorator preserves function name and docstring."""
        @session
        def my_function():
            """My docstring."""
            return 0

        assert my_function.__name__ == 'my_function'
        assert my_function.__doc__ == """My docstring."""

    def test_decorator_with_parameters(self):
        """Test decorator with function parameters."""
        @session
        def func_with_params(x: int, y: str = "default"):
            """Function with parameters."""
            return 0

        assert hasattr(func_with_params, '_is_session_wrapped')

    def test_decorator_callable_with_args_bypasses_session(self):
        """Test calling decorated function with args bypasses session management."""
        call_count = []

        @session
        def test_func(value: int = 1):
            call_count.append(value)
            return value

        # Call with arguments - should bypass session management
        result = test_func(42)

        assert result == 42
        assert call_count == [42]

    def test_decorator_stores_original_function(self):
        """Test decorator stores reference to original function."""
        def original():
            return 42

        wrapped = session(original)

        assert hasattr(wrapped, '_func')
        assert wrapped._func is original


class TestSessionDecoratorOptions:
    """Test session decorator configuration options."""

    def test_verbose_option(self):
        """Test verbose parameter."""
        @session(verbose=True)
        def dummy():
            return 0

        assert hasattr(dummy, '_is_session_wrapped')

    def test_agg_option(self):
        """Test agg parameter."""
        @session(agg=False)
        def dummy():
            return 0

        assert hasattr(dummy, '_is_session_wrapped')

    def test_notify_option(self):
        """Test notify parameter."""
        @session(notify=True)
        def dummy():
            return 0

        assert hasattr(dummy, '_is_session_wrapped')

    def test_sdir_suffix_option(self):
        """Test sdir_suffix parameter."""
        @session(sdir_suffix="custom_suffix")
        def dummy():
            return 0

        assert hasattr(dummy, '_is_session_wrapped')

    def test_multiple_options(self):
        """Test multiple configuration options."""
        @session(verbose=True, agg=False, notify=True, sdir_suffix="test")
        def dummy():
            return 0

        assert hasattr(dummy, '_is_session_wrapped')


class TestRunFunction:
    """Test session.run() function."""

    def test_run_function_exists(self):
        """Test run function exists."""
        from scitex.session import run
        assert callable(run)


class TestDecoratorIntegration:
    """Integration tests for decorator."""

    def test_decorator_with_return_value(self):
        """Test decorator handles return values."""
        @session
        def return_func(value: int = 5):
            return value * 2

        # Call with args to bypass session
        result = return_func(10)
        assert result == 20

    def test_decorator_with_no_return(self):
        """Test decorator handles functions with no return."""
        @session
        def no_return_func(x: int = 1):
            pass

        # Call with args
        result = no_return_func(5)
        assert result is None


if __name__ == "__main__":
    import os
    pytest.main([os.path.abspath(__file__), "-v"])


# EOF
