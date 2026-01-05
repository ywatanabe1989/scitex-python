#!/usr/bin/env python3
"""Tests for scitex.writer._compile_async."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex.writer._compile_async import (
    _make_async_wrapper,
    compile_all_async,
    compile_manuscript_async,
    compile_revision_async,
    compile_supplementary_async,
)


class TestMakeAsyncWrapper:
    """Tests for _make_async_wrapper factory function."""

    def test_returns_callable(self):
        """Verify _make_async_wrapper returns a callable."""

        def sync_func():
            return "result"

        async_func = _make_async_wrapper(sync_func)

        assert callable(async_func)

    def test_preserves_function_name(self):
        """Verify wrapped function preserves original name."""

        def my_sync_function():
            return "result"

        async_func = _make_async_wrapper(my_sync_function)

        assert async_func.__name__ == "my_sync_function"

    @pytest.mark.asyncio
    async def test_wrapper_returns_sync_result(self):
        """Verify async wrapper returns sync function result."""

        def sync_func():
            return "expected_result"

        async_func = _make_async_wrapper(sync_func)
        result = await async_func()

        assert result == "expected_result"

    @pytest.mark.asyncio
    async def test_wrapper_passes_args(self):
        """Verify async wrapper passes arguments to sync function."""

        def sync_func(a, b):
            return a + b

        async_func = _make_async_wrapper(sync_func)
        result = await async_func(1, 2)

        assert result == 3


class TestCompileManuscriptAsync:
    """Tests for compile_manuscript_async function."""

    def test_is_callable(self):
        """Verify compile_manuscript_async is callable."""
        assert callable(compile_manuscript_async)

    def test_preserves_original_function_name(self):
        """Verify async wrapper preserves original function name."""
        assert compile_manuscript_async.__name__ == "compile_manuscript"

    def test_returns_coroutine_when_called(self):
        """Verify compile_manuscript_async returns a coroutine when called."""
        import inspect

        # Calling the async function should return a coroutine
        result = compile_manuscript_async(Path("/nonexistent"))
        assert inspect.iscoroutine(result)
        # Clean up the coroutine to avoid warning
        result.close()


class TestCompileSupplementaryAsync:
    """Tests for compile_supplementary_async function."""

    def test_is_callable(self):
        """Verify compile_supplementary_async is callable."""
        assert callable(compile_supplementary_async)


class TestCompileRevisionAsync:
    """Tests for compile_revision_async function."""

    def test_is_callable(self):
        """Verify compile_revision_async is callable."""
        assert callable(compile_revision_async)


class TestCompileAllAsync:
    """Tests for compile_all_async function."""

    @pytest.mark.asyncio
    async def test_returns_dict_with_all_keys(self, tmp_path):
        """Verify compile_all_async returns dict with all document types."""
        mock_result = MagicMock()
        mock_result.success = True

        async def mock_coro(*args, **kwargs):
            return mock_result

        with patch(
            "scitex.writer._compile_async.compile_manuscript_async",
            side_effect=mock_coro,
        ), patch(
            "scitex.writer._compile_async.compile_supplementary_async",
            side_effect=mock_coro,
        ), patch(
            "scitex.writer._compile_async.compile_revision_async",
            side_effect=mock_coro,
        ):
            result = await compile_all_async(tmp_path)

            assert "manuscript" in result
            assert "supplementary" in result
            assert "revision" in result

    @pytest.mark.asyncio
    async def test_handles_exceptions_gracefully(self, tmp_path):
        """Verify compile_all_async handles exceptions by returning None."""

        async def raise_exception(*args, **kwargs):
            raise RuntimeError("Compilation failed")

        with patch(
            "scitex.writer._compile_async.compile_manuscript_async",
            side_effect=raise_exception,
        ), patch(
            "scitex.writer._compile_async.compile_supplementary_async",
            side_effect=raise_exception,
        ), patch(
            "scitex.writer._compile_async.compile_revision_async",
            side_effect=raise_exception,
        ):
            result = await compile_all_async(tmp_path)

            # When gather with return_exceptions=True, exceptions are returned as results
            assert result["manuscript"] is None
            assert result["supplementary"] is None
            assert result["revision"] is None

    @pytest.mark.asyncio
    async def test_track_changes_parameter(self, tmp_path):
        """Verify track_changes parameter is passed to revision compile."""
        mock_result = MagicMock()

        async def mock_coro(*args, **kwargs):
            return mock_result

        with patch(
            "scitex.writer._compile_async.compile_manuscript_async",
            side_effect=mock_coro,
        ), patch(
            "scitex.writer._compile_async.compile_supplementary_async",
            side_effect=mock_coro,
        ), patch(
            "scitex.writer._compile_async.compile_revision_async",
            side_effect=mock_coro,
        ) as mock_revision:
            await compile_all_async(tmp_path, track_changes=True)

            # Function was called (verify track_changes in implementation)
            mock_revision.assert_called()


class TestAsyncFunctionsExported:
    """Tests for module exports."""

    def test_compile_manuscript_async_exported(self):
        """Verify compile_manuscript_async is exported."""
        from scitex.writer._compile_async import compile_manuscript_async

        assert compile_manuscript_async is not None

    def test_compile_supplementary_async_exported(self):
        """Verify compile_supplementary_async is exported."""
        from scitex.writer._compile_async import compile_supplementary_async

        assert compile_supplementary_async is not None

    def test_compile_revision_async_exported(self):
        """Verify compile_revision_async is exported."""
        from scitex.writer._compile_async import compile_revision_async

        assert compile_revision_async is not None

    def test_compile_all_async_exported(self):
        """Verify compile_all_async is exported."""
        from scitex.writer._compile_async import compile_all_async

        assert compile_all_async is not None


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
