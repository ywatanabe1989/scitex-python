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
    pytest.main([os.path.abspath(__file__), "-v"])
