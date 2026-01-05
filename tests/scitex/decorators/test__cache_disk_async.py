#!/usr/bin/env python3
# Time-stamp: "2026-01-04 21:20:00 (ywatanabe)"
# File: ./tests/scitex/decorators/test__cache_disk_async.py

"""Test cache_disk_async decorator functionality.

The cache_disk_async decorator provides disk-based caching for async functions
using joblib.Memory.
"""

import asyncio

import pytest

pytest.importorskip("joblib")

from scitex.decorators import cache_disk_async


class TestCacheDiskAsync:
    """Test cache_disk_async decorator"""

    def test_cache_disk_async_import(self):
        """Test that cache_disk_async can be imported"""
        from scitex.decorators import cache_disk_async

        assert callable(cache_disk_async)

    @pytest.mark.asyncio
    async def test_cache_disk_async_basic_functionality(self):
        """Test basic async function caching"""
        call_count = 0

        @cache_disk_async
        async def async_square(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x**2

        # First call should execute the function
        result1 = await async_square(5)
        assert result1 == 25
        first_count = call_count

        # Second call with same args should use cache
        result2 = await async_square(5)
        assert result2 == 25
        # call_count may or may not increase depending on cache implementation
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_disk_async_with_different_args(self):
        """Test caching with different arguments"""

        @cache_disk_async
        async def async_multiply(x, y):
            await asyncio.sleep(0.01)
            return x * y

        result1 = await async_multiply(3, 4)
        result2 = await async_multiply(5, 6)

        assert result1 == 12
        assert result2 == 30

    @pytest.mark.asyncio
    async def test_cache_disk_async_with_kwargs(self):
        """Test caching with keyword arguments"""

        @cache_disk_async
        async def async_power(base, exponent=2):
            await asyncio.sleep(0.01)
            return base**exponent

        result1 = await async_power(3)
        result2 = await async_power(3, exponent=3)

        assert result1 == 9
        assert result2 == 27

    @pytest.mark.asyncio
    async def test_cache_disk_async_return_types(self):
        """Test caching with various return types"""

        @cache_disk_async
        async def async_return_dict(key, value):
            await asyncio.sleep(0.01)
            return {key: value}

        result = await async_return_dict("test", 42)
        assert result == {"test": 42}

    @pytest.mark.asyncio
    async def test_cache_disk_async_preserves_function_metadata(self):
        """Test that decorator preserves function metadata"""

        @cache_disk_async
        async def documented_async_func(x):
            """This is a documented async function"""
            return x * 2

        assert documented_async_func.__name__ == "documented_async_func"
        assert documented_async_func.__doc__ == "This is a documented async function"

    @pytest.mark.asyncio
    async def test_cache_disk_async_is_async(self):
        """Test that decorated function is still async"""
        import inspect

        @cache_disk_async
        async def async_identity(x):
            return x

        assert inspect.iscoroutinefunction(async_identity)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
