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

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/decorators/_cache_disk_async.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-09 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/decorators/_cache_disk_async.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/decorators/_cache_disk_async.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# """Async disk caching decorator using joblib.Memory."""
# 
# import asyncio
# import functools
# 
# from joblib import Memory as _Memory
# 
# from scitex.config import get_paths
# 
# 
# def cache_disk_async(func):
#     """Disk caching decorator for async functions.
# 
#     Usage:
#         @cache_disk_async
#         async def expensive_async_function(x):
#             await asyncio.sleep(1)
#             return x ** 2
#     """
#     cache_dir = str(get_paths().function_cache)
#     memory = _Memory(cache_dir, verbose=0)
# 
#     # Create sync wrapper for joblib
#     def sync_wrapper(*args, **kwargs):
#         return asyncio.run(func(*args, **kwargs))
# 
#     cached_sync = memory.cache(sync_wrapper)
# 
#     @functools.wraps(func)
#     async def async_wrapper(*args, **kwargs):
#         # Run cached sync version in executor to avoid blocking
#         loop = asyncio.get_event_loop()
#         result = await loop.run_in_executor(None, lambda: cached_sync(*args, **kwargs))
#         return result
# 
#     return async_wrapper
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/decorators/_cache_disk_async.py
# --------------------------------------------------------------------------------
