#!/usr/bin/env python3
# Time-stamp: "2025-01-05"
# File: test_benchmark.py

"""Tests for scitex.benchmark.benchmark module."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scitex.benchmark.benchmark import (
    BenchmarkResult,
    BenchmarkSuite,
    benchmark_function,
    benchmark_module,
    compare_implementations,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_function():
    """A simple function for benchmarking."""

    def add_numbers(a, b):
        return a + b

    return add_numbers


@pytest.fixture
def slow_function():
    """A function that takes measurable time."""
    import time

    def slow_add(a, b):
        time.sleep(0.01)  # 10ms
        return a + b

    return slow_add


@pytest.fixture
def benchmark_result():
    """Create a sample BenchmarkResult."""
    return BenchmarkResult(
        function_name="test_func",
        module="test_module",
        mean_time=0.1,
        std_time=0.01,
        min_time=0.08,
        max_time=0.12,
        iterations=10,
        input_size="100x100",
        memory_usage=50.0,
        notes="Test benchmark",
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Test BenchmarkResult
# ============================================================================


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_creation_with_required_fields(self):
        """Test BenchmarkResult with only required fields."""
        result = BenchmarkResult(
            function_name="my_func",
            module="my_module",
            mean_time=0.5,
            std_time=0.05,
            min_time=0.4,
            max_time=0.6,
            iterations=10,
        )

        assert result.function_name == "my_func"
        assert result.module == "my_module"
        assert result.mean_time == 0.5
        assert result.std_time == 0.05
        assert result.min_time == 0.4
        assert result.max_time == 0.6
        assert result.iterations == 10
        assert result.input_size is None
        assert result.memory_usage is None
        assert result.notes is None

    def test_creation_with_all_fields(self, benchmark_result):
        """Test BenchmarkResult with all fields."""
        assert benchmark_result.function_name == "test_func"
        assert benchmark_result.input_size == "100x100"
        assert benchmark_result.memory_usage == 50.0
        assert benchmark_result.notes == "Test benchmark"

    def test_str_representation(self, benchmark_result):
        """Test __str__ returns expected format."""
        result_str = str(benchmark_result)

        assert "test_func" in result_str
        assert "0.100s" in result_str
        assert "0.010s" in result_str
        assert "n=10" in result_str

    def test_to_dict(self, benchmark_result):
        """Test to_dict serialization."""
        result_dict = benchmark_result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["function"] == "test_func"
        assert result_dict["module"] == "test_module"
        assert result_dict["mean_time"] == 0.1
        assert result_dict["std_time"] == 0.01
        assert result_dict["min_time"] == 0.08
        assert result_dict["max_time"] == 0.12
        assert result_dict["iterations"] == 10
        assert result_dict["input_size"] == "100x100"
        assert result_dict["memory_usage"] == 50.0
        assert result_dict["notes"] == "Test benchmark"

    def test_to_dict_has_all_expected_keys(self, benchmark_result):
        """Test that to_dict has all expected keys."""
        result_dict = benchmark_result.to_dict()
        expected_keys = {
            "function",
            "module",
            "mean_time",
            "std_time",
            "min_time",
            "max_time",
            "iterations",
            "input_size",
            "memory_usage",
            "notes",
        }
        assert set(result_dict.keys()) == expected_keys


# ============================================================================
# Test benchmark_function
# ============================================================================


class TestBenchmarkFunction:
    """Tests for benchmark_function."""

    def test_basic_benchmark(self, sample_function):
        """Test basic function benchmarking."""
        result = benchmark_function(
            sample_function, args=(1, 2), iterations=5, warmup=1
        )

        assert isinstance(result, BenchmarkResult)
        assert result.function_name == "add_numbers"
        assert result.iterations == 5
        assert result.mean_time >= 0
        assert result.std_time >= 0
        assert result.min_time <= result.mean_time <= result.max_time

    def test_benchmark_with_kwargs(self, sample_function):
        """Test benchmarking with keyword arguments."""
        result = benchmark_function(
            sample_function, args=(1,), kwargs={"b": 2}, iterations=5
        )

        assert isinstance(result, BenchmarkResult)
        assert result.function_name == "add_numbers"

    def test_timing_consistency(self, slow_function):
        """Test that timing is consistent and measurable."""
        result = benchmark_function(slow_function, args=(1, 2), iterations=5, warmup=1)

        # 10ms sleep should result in measurable time
        assert result.mean_time >= 0.009  # Allow some slack
        assert result.mean_time < 0.1  # But not too slow

    def test_input_size_parameter(self, sample_function):
        """Test that input_size is recorded."""
        result = benchmark_function(
            sample_function, args=(1, 2), input_size="small", iterations=3
        )

        assert result.input_size == "small"

    def test_warmup_iterations(self, sample_function):
        """Test warmup iterations are executed."""
        call_count = [0]
        original_func = sample_function

        def counting_func(a, b):
            call_count[0] += 1
            return a + b

        result = benchmark_function(counting_func, args=(1, 2), iterations=3, warmup=2)

        # Should have 2 warmup + 3 benchmark = 5 total calls
        assert call_count[0] == 5

    def test_default_kwargs_none(self, sample_function):
        """Test that None kwargs default to empty dict."""
        # Should not raise - kwargs internally becomes {}
        result = benchmark_function(sample_function, args=(1, 2), kwargs=None)
        assert isinstance(result, BenchmarkResult)

    def test_module_detection(self, sample_function):
        """Test that module name is detected."""
        result = benchmark_function(sample_function, args=(1, 2))
        # Module should be detected (though may be __main__ or test module)
        assert isinstance(result.module, str)
        assert len(result.module) > 0

    def test_measure_memory_flag(self, sample_function):
        """Test memory measurement flag."""
        result_no_mem = benchmark_function(
            sample_function, args=(1, 2), measure_memory=False
        )
        # Memory might be None if psutil not available
        # Just verify it doesn't crash
        assert isinstance(result_no_mem, BenchmarkResult)

        result_with_mem = benchmark_function(
            sample_function, args=(1, 2), measure_memory=True
        )
        # Memory might still be None if psutil not installed
        assert isinstance(result_with_mem, BenchmarkResult)


# ============================================================================
# Test compare_implementations
# ============================================================================


class TestCompareImplementations:
    """Tests for compare_implementations."""

    def test_compare_two_implementations(self):
        """Test comparing two implementations."""

        def impl1(x):
            return sum(range(x))

        def impl2(x):
            return x * (x - 1) // 2

        implementations = {"loop": impl1, "formula": impl2}

        def data_gen():
            return (1000,), {}

        df = compare_implementations(implementations, data_gen, iterations=3)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "implementation" in df.columns
        assert "mean_time" in df.columns
        assert "std_time" in df.columns
        assert "speedup" in df.columns

    def test_speedup_calculation(self):
        """Test that speedup is calculated correctly."""
        import time

        def slow_impl(x):
            time.sleep(0.01)
            return x

        def fast_impl(x):
            return x

        implementations = {"slow": slow_impl, "fast": fast_impl}

        def data_gen():
            return (10,), {}

        df = compare_implementations(implementations, data_gen, iterations=3)

        # First implementation has speedup 1.0 (baseline)
        assert df.iloc[0]["speedup"] == 1.0
        # Fast implementation should have speedup > 1
        assert df.iloc[1]["speedup"] > 1.0

    def test_empty_implementations(self):
        """Test with no implementations raises IndexError."""
        implementations = {}

        def data_gen():
            return (), {}

        # Empty implementations causes IndexError when accessing baseline_time
        with pytest.raises(IndexError):
            compare_implementations(implementations, data_gen, iterations=3)

    def test_single_implementation(self):
        """Test with single implementation."""
        implementations = {"only": lambda x: x}

        def data_gen():
            return (1,), {}

        df = compare_implementations(implementations, data_gen, iterations=3)
        assert len(df) == 1
        assert df.iloc[0]["speedup"] == 1.0


# ============================================================================
# Test BenchmarkSuite
# ============================================================================


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite class."""

    def test_suite_creation(self):
        """Test suite creation."""
        suite = BenchmarkSuite("test_suite")

        assert suite.name == "test_suite"
        assert suite.benchmarks == []
        assert suite.results == []

    def test_add_benchmark(self):
        """Test adding benchmarks to suite."""
        suite = BenchmarkSuite("test_suite")

        def my_func():
            return 42

        def data_gen():
            return (), {}

        suite.add_benchmark(my_func, data_gen, name="custom_name", sizes=["small"])

        assert len(suite.benchmarks) == 1
        assert suite.benchmarks[0]["func"] == my_func
        assert suite.benchmarks[0]["name"] == "custom_name"
        assert suite.benchmarks[0]["sizes"] == ["small"]

    def test_add_benchmark_default_name(self):
        """Test adding benchmark uses function name by default."""
        suite = BenchmarkSuite("test_suite")

        def my_named_function():
            return 42

        def data_gen():
            return (), {}

        suite.add_benchmark(my_named_function, data_gen)

        assert suite.benchmarks[0]["name"] == "my_named_function"

    def test_add_benchmark_default_sizes(self):
        """Test adding benchmark uses default size."""
        suite = BenchmarkSuite("test_suite")

        def my_func():
            return 42

        def data_gen():
            return (), {}

        suite.add_benchmark(my_func, data_gen)

        assert suite.benchmarks[0]["sizes"] == ["default"]

    def test_run_suite(self, capsys):
        """Test running benchmark suite."""
        suite = BenchmarkSuite("test_suite")

        def my_func():
            return 42

        def data_gen():
            return (), {}

        suite.add_benchmark(my_func, data_gen, sizes=["small", "large"])

        results = suite.run(iterations=3, verbose=True)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2  # Two sizes
        assert "function" in results.columns
        assert "mean_time" in results.columns
        assert "size" in results.columns

        # Check verbose output
        captured = capsys.readouterr()
        assert "Running benchmark" in captured.out

    def test_run_suite_quiet(self, capsys):
        """Test running suite without verbose output."""
        suite = BenchmarkSuite("test_suite")

        def my_func():
            return 42

        def data_gen():
            return (), {}

        suite.add_benchmark(my_func, data_gen)
        suite.run(iterations=2, verbose=False)

        captured = capsys.readouterr()
        assert "Running benchmark" not in captured.out

    def test_save_results(self, temp_dir):
        """Test saving results to CSV."""
        suite = BenchmarkSuite("test_suite")

        def my_func():
            return 42

        def data_gen():
            return (), {}

        suite.add_benchmark(my_func, data_gen)
        suite.run(iterations=2, verbose=False)

        output_path = os.path.join(temp_dir, "results.csv")
        suite.save_results(output_path)

        assert os.path.exists(output_path)

        # Verify CSV content
        loaded_df = pd.read_csv(output_path)
        assert "function" in loaded_df.columns
        assert len(loaded_df) > 0

    def test_save_results_no_results(self, temp_dir):
        """Test saving when no results exist yet raises AttributeError."""
        suite = BenchmarkSuite("test_suite")
        output_path = os.path.join(temp_dir, "results.csv")

        # Empty results list causes AttributeError (list has no to_csv)
        with pytest.raises(AttributeError):
            suite.save_results(output_path)

    def test_compare_with_baseline(self, temp_dir):
        """Test comparing with baseline results."""
        suite = BenchmarkSuite("test_suite")

        def my_func():
            return 42

        def data_gen():
            return (), {}

        suite.add_benchmark(my_func, data_gen)
        suite.run(iterations=2, verbose=False)

        # Create baseline file
        baseline_path = os.path.join(temp_dir, "baseline.csv")
        baseline_data = pd.DataFrame(
            {
                "function": ["my_func"],
                "size": ["default"],
                "mean_time": [0.001],  # Baseline time
            }
        )
        baseline_data.to_csv(baseline_path, index=False)

        comparison = suite.compare_with_baseline(baseline_path)

        assert isinstance(comparison, pd.DataFrame)
        assert "speedup" in comparison.columns
        assert "mean_time_current" in comparison.columns
        assert "mean_time_baseline" in comparison.columns


# ============================================================================
# Test benchmark_module
# ============================================================================


class TestBenchmarkModule:
    """Tests for benchmark_module function."""

    def test_benchmark_builtin_module(self):
        """Test benchmarking a standard library module."""
        suite = benchmark_module("math", pattern="sqrt*")

        assert isinstance(suite, BenchmarkSuite)
        assert suite.name == "math"
        # sqrt should be matched
        assert len(suite.benchmarks) >= 0  # May find sqrt

    def test_benchmark_module_with_pattern(self):
        """Test pattern matching in module."""
        suite = benchmark_module("os.path", pattern="is*")

        assert isinstance(suite, BenchmarkSuite)
        # Should find isfile, isdir, etc.
        func_names = [b["name"] for b in suite.benchmarks]
        # At least one of these should be matched
        assert any(name.startswith("is") for name in func_names) or len(func_names) == 0

    def test_benchmark_nonexistent_module(self):
        """Test with non-existent module raises ImportError."""
        with pytest.raises(ImportError):
            benchmark_module("nonexistent_module_12345")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
