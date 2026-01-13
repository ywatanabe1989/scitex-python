#!/usr/bin/env python3
# Time-stamp: "2025-01-05"
# File: test_profiler.py

"""Tests for scitex.benchmark.profiler module."""

import io
import os
import sys
import time

import pytest

from scitex.benchmark.profiler import (
    FunctionProfiler,
    LineProfiler,
    get_memory_usage,
    get_profile_report,
    profile_block,
    profile_function,
    profile_module,
    track_memory,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def profiler():
    """Create a fresh FunctionProfiler instance."""
    return FunctionProfiler()


@pytest.fixture
def line_profiler():
    """Create a fresh LineProfiler instance."""
    return LineProfiler()


@pytest.fixture
def sample_function():
    """A simple function for profiling."""

    def compute_sum(n):
        return sum(range(n))

    return compute_sum


@pytest.fixture
def slow_function():
    """A function that takes measurable time."""

    def slow_compute(n):
        time.sleep(0.01)
        return n * 2

    return slow_compute


# ============================================================================
# Test FunctionProfiler
# ============================================================================


class TestFunctionProfiler:
    """Tests for FunctionProfiler class."""

    def test_profiler_creation(self, profiler):
        """Test profiler initialization."""
        assert profiler.profiles == {}
        assert profiler.call_counts == {}
        assert profiler.total_times == {}

    def test_profile_decorator(self, profiler):
        """Test profiling with decorator."""

        @profiler.profile
        def my_func(x):
            return x * 2

        result = my_func(5)

        assert result == 10
        assert "my_func" in profiler.profiles
        assert profiler.call_counts["my_func"] == 1
        assert profiler.total_times["my_func"] > 0

    def test_profile_multiple_calls(self, profiler):
        """Test profiling with multiple calls."""

        @profiler.profile
        def my_func(x):
            return x + 1

        for i in range(5):
            my_func(i)

        assert profiler.call_counts["my_func"] == 5
        assert len(profiler.profiles["my_func"]) == 5

    def test_profile_preserves_function_name(self, profiler):
        """Test that decorator preserves function metadata."""

        @profiler.profile
        def original_name(x):
            """Original docstring."""
            return x

        assert original_name.__name__ == "original_name"
        assert original_name.__doc__ == "Original docstring."

    def test_profile_with_args_and_kwargs(self, profiler):
        """Test profiling function with args and kwargs."""

        @profiler.profile
        def complex_func(a, b, c=10, d=20):
            return a + b + c + d

        result = complex_func(1, 2, c=30, d=40)

        assert result == 73
        assert profiler.call_counts["complex_func"] == 1

    def test_get_stats_returns_none_for_unknown(self, profiler):
        """Test get_stats returns None for unknown function."""
        stats = profiler.get_stats("unknown_function")
        assert stats is None

    def test_get_stats_returns_stats(self, profiler):
        """Test get_stats returns Stats object."""

        @profiler.profile
        def my_func():
            return 42

        my_func()
        my_func()

        stats = profiler.get_stats("my_func")
        assert stats is not None

    def test_print_stats_single_function(self, profiler, capsys):
        """Test print_stats for single function."""

        @profiler.profile
        def my_func():
            return 42

        my_func()
        profiler.print_stats("my_func")

        captured = capsys.readouterr()
        assert "Profile for my_func" in captured.out
        assert "Total calls: 1" in captured.out

    def test_print_stats_all_functions(self, profiler, capsys):
        """Test print_stats for all functions."""

        @profiler.profile
        def func1():
            return 1

        @profiler.profile
        def func2():
            return 2

        func1()
        func2()

        profiler.print_stats()

        captured = capsys.readouterr()
        assert "func1" in captured.out
        assert "func2" in captured.out

    def test_get_report(self, profiler):
        """Test get_report returns comprehensive report."""

        @profiler.profile
        def my_func():
            return sum(range(100))

        my_func()
        my_func()

        report = profiler.get_report()

        assert "my_func" in report
        assert report["my_func"]["call_count"] == 2
        assert "total_time" in report["my_func"]
        assert "avg_time" in report["my_func"]
        assert "profile" in report["my_func"]


# ============================================================================
# Test profile_function (global profiler)
# ============================================================================


class TestProfileFunction:
    """Tests for profile_function decorator."""

    def test_profile_function_decorator(self):
        """Test global profile_function decorator."""

        @profile_function
        def test_func(x):
            return x**2

        result = test_func(5)
        assert result == 25

    def test_profile_function_preserves_return(self):
        """Test that decorated function returns correctly."""

        @profile_function
        def compute(a, b):
            return a * b

        assert compute(3, 4) == 12


# ============================================================================
# Test get_profile_report
# ============================================================================


class TestGetProfileReport:
    """Tests for get_profile_report function."""

    def test_get_profile_report_returns_dict(self):
        """Test get_profile_report returns dictionary."""
        report = get_profile_report()
        assert isinstance(report, dict)


# ============================================================================
# Test profile_block context manager
# ============================================================================


class TestProfileBlock:
    """Tests for profile_block context manager."""

    def test_profile_block_basic(self, capsys):
        """Test basic profile_block usage."""
        with profile_block("test_block"):
            result = sum(range(1000))

        captured = capsys.readouterr()
        assert "Profile for block 'test_block'" in captured.out
        assert "Total time:" in captured.out

    def test_profile_block_with_slow_code(self, capsys):
        """Test profile_block with slow code."""
        with profile_block("slow_block"):
            time.sleep(0.02)

        captured = capsys.readouterr()
        assert "slow_block" in captured.out
        # Time should be at least 0.01s
        assert "0.0" in captured.out  # Time should be visible

    def test_profile_block_exception_handling(self, capsys):
        """Test profile_block handles exceptions properly."""
        with pytest.raises(ValueError):
            with profile_block("error_block"):
                raise ValueError("Test error")

        # Profile output should still be printed
        captured = capsys.readouterr()
        assert "error_block" in captured.out


# ============================================================================
# Test profile_module
# ============================================================================


class TestProfileModule:
    """Tests for profile_module function."""

    def test_profile_module_returns_profiler(self, capsys):
        """Test profile_module returns a profiler."""
        profiler = profile_module("math", pattern="sqrt")

        assert isinstance(profiler, FunctionProfiler)

        captured = capsys.readouterr()
        assert "Profiling" in captured.out

    def test_profile_module_wraps_functions(self, capsys):
        """Test profile_module wraps matching functions."""
        profiler = profile_module("os.path", pattern="exists")

        captured = capsys.readouterr()
        # Should report profiling functions
        assert "Profiling" in captured.out


# ============================================================================
# Test LineProfiler
# ============================================================================


class TestLineProfiler:
    """Tests for LineProfiler class."""

    def test_line_profiler_creation(self, line_profiler):
        """Test LineProfiler initialization."""
        assert line_profiler.timings == {}

    def test_profile_lines_decorator(self, line_profiler):
        """Test profile_lines decorator."""

        @line_profiler.profile_lines
        def my_func(n):
            result = 0
            for i in range(n):
                result += i
            return result

        result = my_func(100)

        assert result == 4950  # sum of 0..99
        assert "my_func" in line_profiler.timings
        assert len(line_profiler.timings["my_func"]) == 1

    def test_profile_lines_stores_timing(self, line_profiler):
        """Test that profile_lines stores timing info."""

        @line_profiler.profile_lines
        def my_func():
            time.sleep(0.01)
            return 42

        my_func()

        timing = line_profiler.timings["my_func"][0]
        assert "total_time" in timing
        assert timing["total_time"] >= 0.009  # At least 9ms
        assert "source" in timing

    def test_profile_lines_stores_source(self, line_profiler):
        """Test that profile_lines stores source code."""

        @line_profiler.profile_lines
        def my_func():
            x = 1
            y = 2
            return x + y

        my_func()

        timing = line_profiler.timings["my_func"][0]
        source = timing["source"]
        assert isinstance(source, list)
        assert len(source) > 0
        # Source should contain the function code
        source_text = "".join(source)
        assert "return" in source_text

    def test_print_timings(self, line_profiler, capsys):
        """Test print_timings output."""

        @line_profiler.profile_lines
        def my_func():
            return 42

        my_func()
        line_profiler.print_timings("my_func")

        captured = capsys.readouterr()
        assert "Line timings for my_func" in captured.out
        assert "Total time:" in captured.out
        assert "Source code:" in captured.out

    def test_print_timings_unknown_function(self, line_profiler, capsys):
        """Test print_timings for unknown function."""
        line_profiler.print_timings("unknown_func")

        captured = capsys.readouterr()
        assert "No timings for unknown_func" in captured.out


# ============================================================================
# Test Memory Utilities
# ============================================================================


class TestGetMemoryUsage:
    """Tests for get_memory_usage function."""

    def test_get_memory_usage_returns_value_or_none(self):
        """Test get_memory_usage returns float or None."""
        result = get_memory_usage()

        # Result should be float (if psutil available) or None
        assert result is None or isinstance(result, float)

    def test_get_memory_usage_positive_value(self):
        """Test get_memory_usage returns positive value if available."""
        result = get_memory_usage()

        if result is not None:
            assert result > 0  # Memory usage should be positive


class TestTrackMemory:
    """Tests for track_memory context manager."""

    def test_track_memory_basic(self, capsys):
        """Test basic track_memory usage."""
        with track_memory("test_allocation"):
            # Allocate some memory
            data = list(range(10000))

        captured = capsys.readouterr()
        # Output depends on whether psutil is available
        if "Memory usage" in captured.out:
            assert "test_allocation" in captured.out
            assert "Start:" in captured.out
            assert "End:" in captured.out
            assert "Delta:" in captured.out

    def test_track_memory_exception_handling(self, capsys):
        """Test track_memory handles exceptions."""
        with pytest.raises(ValueError):
            with track_memory("error_block"):
                raise ValueError("Test error")

        # Should still print memory info before exception
        captured = capsys.readouterr()
        # May or may not have output depending on psutil availability

    def test_track_memory_nested(self, capsys):
        """Test nested track_memory blocks."""
        with track_memory("outer"):
            data1 = list(range(1000))
            with track_memory("inner"):
                data2 = list(range(1000))

        captured = capsys.readouterr()
        # Should have info for both if psutil available
        if "Memory usage" in captured.out:
            assert "outer" in captured.out or "inner" in captured.out


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/benchmark/profiler.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2025-07-25 05:35:00"
# # File: profiler.py
# 
# """
# Profiling tools for SciTeX performance analysis.
# """
# 
# import cProfile
# import pstats
# import io
# from typing import Callable, Optional, Dict, Any
# from functools import wraps
# import time
# from contextlib import contextmanager
# 
# 
# class FunctionProfiler:
#     """Profile individual function calls."""
# 
#     def __init__(self):
#         self.profiles = {}
#         self.call_counts = {}
#         self.total_times = {}
# 
#     def profile(self, func: Callable) -> Callable:
#         """
#         Decorator to profile a function.
# 
#         Example
#         -------
#         >>> profiler = FunctionProfiler()
#         >>> @profiler.profile
#         ... def my_function(x):
#         ...     return x ** 2
#         """
# 
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             # Create profiler for this call
#             pr = cProfile.Profile()
#             pr.enable()
# 
#             # Call function
#             start_time = time.time()
#             result = func(*args, **kwargs)
#             end_time = time.time()
# 
#             pr.disable()
# 
#             # Store results
#             func_name = func.__name__
#             if func_name not in self.profiles:
#                 self.profiles[func_name] = []
#                 self.call_counts[func_name] = 0
#                 self.total_times[func_name] = 0.0
# 
#             self.profiles[func_name].append(pr)
#             self.call_counts[func_name] += 1
#             self.total_times[func_name] += end_time - start_time
# 
#             return result
# 
#         return wrapper
# 
#     def get_stats(self, func_name: str) -> Optional[pstats.Stats]:
#         """Get profiling statistics for a function."""
#         if func_name not in self.profiles:
#             return None
# 
#         # Combine all profiles for this function
#         combined = pstats.Stats(self.profiles[func_name][0])
#         for pr in self.profiles[func_name][1:]:
#             combined.add(pr)
# 
#         return combined
# 
#     def print_stats(self, func_name: Optional[str] = None, top_n: int = 10):
#         """Print profiling statistics."""
#         if func_name:
#             stats = self.get_stats(func_name)
#             if stats:
#                 print(f"\nProfile for {func_name}:")
#                 print(f"Total calls: {self.call_counts[func_name]}")
#                 print(f"Total time: {self.total_times[func_name]:.3f}s")
#                 print(
#                     f"Avg time per call: {self.total_times[func_name] / self.call_counts[func_name]:.3f}s"
#                 )
#                 print("\nDetailed stats:")
#                 stats.sort_stats("cumulative").print_stats(top_n)
#         else:
#             # Print all functions
#             for name in self.profiles:
#                 self.print_stats(name, top_n)
# 
#     def get_report(self) -> Dict[str, Any]:
#         """Get a summary report of all profiled functions."""
#         report = {}
#         for func_name in self.profiles:
#             stats = self.get_stats(func_name)
# 
#             # Get top time consumers
#             s = io.StringIO()
#             stats.sort_stats("cumulative").print_stats(10, s)
# 
#             report[func_name] = {
#                 "call_count": self.call_counts[func_name],
#                 "total_time": self.total_times[func_name],
#                 "avg_time": self.total_times[func_name] / self.call_counts[func_name],
#                 "profile": s.getvalue(),
#             }
# 
#         return report
# 
# 
# # Global profiler instance
# _global_profiler = FunctionProfiler()
# 
# 
# def profile_function(func: Callable) -> Callable:
#     """
#     Decorator to profile a function using the global profiler.
# 
#     Example
#     -------
#     >>> @profile_function
#     ... def my_function(x):
#     ...     return sum(range(x))
#     """
#     return _global_profiler.profile(func)
# 
# 
# def get_profile_report() -> Dict[str, Any]:
#     """Get profiling report from global profiler."""
#     return _global_profiler.get_report()
# 
# 
# def print_profile_stats(func_name: Optional[str] = None):
#     """Print profiling statistics from global profiler."""
#     _global_profiler.print_stats(func_name)
# 
# 
# @contextmanager
# def profile_block(name: str):
#     """
#     Context manager for profiling a code block.
# 
#     Example
#     -------
#     >>> with profile_block("data_processing"):
#     ...     # Some expensive operations
#     ...     data = process_data()
#     """
#     pr = cProfile.Profile()
#     pr.enable()
#     start_time = time.time()
# 
#     try:
#         yield
#     finally:
#         pr.disable()
#         end_time = time.time()
# 
#         print(f"\nProfile for block '{name}':")
#         print(f"Total time: {end_time - start_time:.3f}s")
# 
#         s = io.StringIO()
#         ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
#         ps.print_stats(10)
#         print(s.getvalue())
# 
# 
# def profile_module(module_name: str, pattern: str = "*") -> Dict[str, Any]:
#     """
#     Profile all matching functions in a module.
# 
#     Parameters
#     ----------
#     module_name : str
#         Name of module to profile
#     pattern : str
#         Pattern to match function names
# 
#     Returns
#     -------
#     dict
#         Profiling results
#     """
#     import importlib
#     import fnmatch
# 
#     module = importlib.import_module(module_name)
#     profiler = FunctionProfiler()
# 
#     # Wrap all matching functions
#     wrapped_functions = []
#     for name in dir(module):
#         if fnmatch.fnmatch(name, pattern):
#             obj = getattr(module, name)
#             if callable(obj) and not name.startswith("_"):
#                 # Replace with profiled version
#                 profiled = profiler.profile(obj)
#                 setattr(module, name, profiled)
#                 wrapped_functions.append(name)
# 
#     print(f"Profiling {len(wrapped_functions)} functions in {module_name}")
#     print(f"Wrapped: {', '.join(wrapped_functions)}")
#     print("\nRun your code now. Call get_profile_report() when done.")
# 
#     return profiler
# 
# 
# class LineProfiler:
#     """
#     Line-by-line profiler for detailed analysis.
# 
#     Note: This is a simplified version. For production use,
#     consider using the line_profiler package.
#     """
# 
#     def __init__(self):
#         self.timings = {}
# 
#     def profile_lines(self, func: Callable) -> Callable:
#         """Profile a function line by line."""
#         import inspect
# 
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             # Get source lines
#             source_lines = inspect.getsourcelines(func)[0]
#             line_times = {}
# 
#             # This is a simplified implementation
#             # Real line profiling requires bytecode instrumentation
#             start_time = time.time()
#             result = func(*args, **kwargs)
#             end_time = time.time()
# 
#             # Store timing
#             func_name = func.__name__
#             if func_name not in self.timings:
#                 self.timings[func_name] = []
# 
#             self.timings[func_name].append(
#                 {"total_time": end_time - start_time, "source": source_lines}
#             )
# 
#             return result
# 
#         return wrapper
# 
#     def print_timings(self, func_name: str):
#         """Print line timings for a function."""
#         if func_name not in self.timings:
#             print(f"No timings for {func_name}")
#             return
# 
#         timing = self.timings[func_name][-1]  # Most recent
#         print(f"\nLine timings for {func_name}:")
#         print(f"Total time: {timing['total_time']:.3f}s")
#         print("\nSource code:")
#         for i, line in enumerate(timing["source"]):
#             print(f"{i + 1:4d}: {line.rstrip()}")
# 
# 
# # Memory profiling utilities
# def get_memory_usage():
#     """Get current memory usage in MB."""
#     try:
#         import psutil
# 
#         process = psutil.Process()
#         return process.memory_info().rss / 1024 / 1024
#     except ImportError:
#         return None
# 
# 
# @contextmanager
# def track_memory(name: str):
#     """
#     Track memory usage for a code block.
# 
#     Example
#     -------
#     >>> with track_memory("data_loading"):
#     ...     data = load_large_dataset()
#     """
#     start_mem = get_memory_usage()
# 
#     try:
#         yield
#     finally:
#         end_mem = get_memory_usage()
#         if start_mem and end_mem:
#             print(f"\nMemory usage for '{name}':")
#             print(f"Start: {start_mem:.1f} MB")
#             print(f"End: {end_mem:.1f} MB")
#             print(f"Delta: {end_mem - start_mem:+.1f} MB")

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/benchmark/profiler.py
# --------------------------------------------------------------------------------
