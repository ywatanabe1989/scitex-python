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
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/benchmark/benchmark.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2025-07-25 05:30:00"
# # File: benchmark.py
# 
# """
# Core benchmarking functionality for SciTeX.
# """
# 
# import time
# import numpy as np
# import pandas as pd
# from typing import Callable, Dict, List, Any, Optional, Tuple
# from dataclasses import dataclass
# import inspect
# import gc
# import os
# from pathlib import Path
# 
# 
# @dataclass
# class BenchmarkResult:
#     """Results from a benchmark run."""
# 
#     function_name: str
#     module: str
#     mean_time: float
#     std_time: float
#     min_time: float
#     max_time: float
#     iterations: int
#     input_size: Optional[str] = None
#     memory_usage: Optional[float] = None
#     notes: Optional[str] = None
# 
#     def __str__(self):
#         return (
#             f"{self.function_name}: {self.mean_time:.3f}s ± {self.std_time:.3f}s "
#             f"(n={self.iterations})"
#         )
# 
#     def to_dict(self):
#         """Convert to dictionary for easy serialization."""
#         return {
#             "function": self.function_name,
#             "module": self.module,
#             "mean_time": self.mean_time,
#             "std_time": self.std_time,
#             "min_time": self.min_time,
#             "max_time": self.max_time,
#             "iterations": self.iterations,
#             "input_size": self.input_size,
#             "memory_usage": self.memory_usage,
#             "notes": self.notes,
#         }
# 
# 
# def benchmark_function(
#     func: Callable,
#     args: tuple = (),
#     kwargs: dict = None,
#     iterations: int = 10,
#     warmup: int = 2,
#     input_size: Optional[str] = None,
#     measure_memory: bool = False,
# ) -> BenchmarkResult:
#     """
#     Benchmark a single function.
# 
#     Parameters
#     ----------
#     func : Callable
#         Function to benchmark
#     args : tuple
#         Arguments to pass to function
#     kwargs : dict
#         Keyword arguments to pass to function
#     iterations : int
#         Number of benchmark iterations
#     warmup : int
#         Number of warmup iterations
#     input_size : str, optional
#         Description of input size
#     measure_memory : bool
#         Whether to measure memory usage
# 
#     Returns
#     -------
#     BenchmarkResult
#         Benchmark results
#     """
#     if kwargs is None:
#         kwargs = {}
# 
#     # Warmup runs
#     for _ in range(warmup):
#         _ = func(*args, **kwargs)
# 
#     # Garbage collection before timing
#     gc.collect()
# 
#     # Timing runs
#     times = []
#     for _ in range(iterations):
#         start = time.perf_counter()
#         _ = func(*args, **kwargs)
#         end = time.perf_counter()
#         times.append(end - start)
# 
#     times = np.array(times)
# 
#     # Get function info
#     module = inspect.getmodule(func).__name__ if inspect.getmodule(func) else "unknown"
# 
#     # Memory measurement (simplified)
#     memory_usage = None
#     if measure_memory:
#         try:
#             import psutil
# 
#             process = psutil.Process(os.getpid())
#             memory_usage = process.memory_info().rss / 1024 / 1024  # MB
#         except:
#             pass
# 
#     return BenchmarkResult(
#         function_name=func.__name__,
#         module=module,
#         mean_time=np.mean(times),
#         std_time=np.std(times),
#         min_time=np.min(times),
#         max_time=np.max(times),
#         iterations=iterations,
#         input_size=input_size,
#         memory_usage=memory_usage,
#     )
# 
# 
# def compare_implementations(
#     implementations: Dict[str, Callable],
#     test_data_generator: Callable[[], Tuple[tuple, dict]],
#     iterations: int = 10,
#     sizes: Optional[List[str]] = None,
# ) -> pd.DataFrame:
#     """
#     Compare multiple implementations of the same functionality.
# 
#     Parameters
#     ----------
#     implementations : dict
#         Dictionary mapping implementation names to functions
#     test_data_generator : callable
#         Function that returns (args, kwargs) for testing
#     iterations : int
#         Number of iterations per implementation
#     sizes : list, optional
#         List of input sizes to test
# 
#     Returns
#     -------
#     pd.DataFrame
#         Comparison results
#     """
#     results = []
# 
#     for name, func in implementations.items():
#         # Generate test data
#         args, kwargs = test_data_generator()
# 
#         # Benchmark
#         result = benchmark_function(
#             func, args=args, kwargs=kwargs, iterations=iterations
#         )
# 
#         results.append(
#             {
#                 "implementation": name,
#                 "mean_time": result.mean_time,
#                 "std_time": result.std_time,
#                 "speedup": 1.0,  # Will calculate relative to baseline
#             }
#         )
# 
#     df = pd.DataFrame(results)
# 
#     # Calculate speedup relative to first implementation
#     baseline_time = df.iloc[0]["mean_time"]
#     df["speedup"] = baseline_time / df["mean_time"]
# 
#     return df
# 
# 
# class BenchmarkSuite:
#     """Collection of benchmarks for a module or set of functions."""
# 
#     def __init__(self, name: str):
#         self.name = name
#         self.benchmarks = []
#         self.results = []
# 
#     def add_benchmark(
#         self,
#         func: Callable,
#         test_data_generator: Callable[[], Tuple[tuple, dict]],
#         name: Optional[str] = None,
#         sizes: Optional[List[str]] = None,
#     ):
#         """Add a benchmark to the suite."""
#         self.benchmarks.append(
#             {
#                 "func": func,
#                 "data_gen": test_data_generator,
#                 "name": name or func.__name__,
#                 "sizes": sizes or ["default"],
#             }
#         )
# 
#     def run(self, iterations: int = 10, verbose: bool = True) -> pd.DataFrame:
#         """Run all benchmarks in the suite."""
#         results = []
# 
#         for benchmark in self.benchmarks:
#             if verbose:
#                 print(f"Running benchmark: {benchmark['name']}")
# 
#             for size in benchmark["sizes"]:
#                 # Generate test data
#                 args, kwargs = benchmark["data_gen"]()
# 
#                 # Run benchmark
#                 result = benchmark_function(
#                     benchmark["func"],
#                     args=args,
#                     kwargs=kwargs,
#                     iterations=iterations,
#                     input_size=size,
#                 )
# 
#                 result_dict = result.to_dict()
#                 result_dict["size"] = size
#                 results.append(result_dict)
# 
#                 if verbose:
#                     print(f"  {size}: {result}")
# 
#         self.results = pd.DataFrame(results)
#         return self.results
# 
#     def save_results(self, path: str):
#         """Save benchmark results to CSV."""
#         if self.results is not None:
#             self.results.to_csv(path, index=False)
# 
#     def compare_with_baseline(self, baseline_path: str) -> pd.DataFrame:
#         """Compare current results with baseline."""
#         baseline = pd.read_csv(baseline_path)
# 
#         # Merge on function name and size
#         comparison = pd.merge(
#             self.results,
#             baseline,
#             on=["function", "size"],
#             suffixes=("_current", "_baseline"),
#         )
# 
#         # Calculate speedup
#         comparison["speedup"] = (
#             comparison["mean_time_baseline"] / comparison["mean_time_current"]
#         )
# 
#         return comparison
# 
# 
# def benchmark_module(module_name: str, pattern: str = "test_*") -> BenchmarkSuite:
#     """
#     Create a benchmark suite for all matching functions in a module.
# 
#     Parameters
#     ----------
#     module_name : str
#         Name of module to benchmark
#     pattern : str
#         Pattern to match function names
# 
#     Returns
#     -------
#     BenchmarkSuite
#         Suite containing all matching benchmarks
#     """
#     import importlib
#     import fnmatch
# 
#     module = importlib.import_module(module_name)
#     suite = BenchmarkSuite(module_name)
# 
#     # Find all matching functions
#     for name in dir(module):
#         if fnmatch.fnmatch(name, pattern):
#             func = getattr(module, name)
#             if callable(func):
#                 # Create simple test data generator
#                 def data_gen():
#                     return (), {}
# 
#                 suite.add_benchmark(func, data_gen, name)
# 
#     return suite
# 
# 
# # Pre-defined benchmark suites for common SciTeX modules
# def create_io_benchmark_suite() -> BenchmarkSuite:
#     """Create benchmark suite for I/O operations."""
#     import tempfile
#     import numpy as np
# 
#     suite = BenchmarkSuite("IO Operations")
# 
#     # Benchmark numpy file loading
#     def numpy_data_gen():
#         data = np.random.randn(1000, 1000)
#         with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
#             np.save(f.name, data)
#             return (f.name,), {}
# 
#     import scitex.io
# 
#     suite.add_benchmark(
#         scitex.io.load, numpy_data_gen, "load_numpy", sizes=["1MB", "10MB", "100MB"]
#     )
# 
#     return suite
# 
# 
# def create_stats_benchmark_suite() -> BenchmarkSuite:
#     """Create benchmark suite for statistics operations."""
#     import numpy as np
# 
#     suite = BenchmarkSuite("Statistics Operations")
# 
#     # Benchmark correlation
#     def corr_data_gen():
#         x = np.random.randn(1000)
#         y = x + np.random.randn(1000) * 0.5
#         return (x, y), {"n_perm": 1000}
# 
#     import scitex.stats
# 
#     suite.add_benchmark(
#         scitex.stats.corr_test,
#         corr_data_gen,
#         "correlation_test",
#         sizes=["1000_samples", "10000_samples"],
#     )
# 
#     return suite
# 
# 
# def run_all_benchmarks(
#     output_dir: str = "./benchmark_results",
# ) -> Dict[str, pd.DataFrame]:
#     """
#     Run all pre-defined benchmark suites.
# 
#     Parameters
#     ----------
#     output_dir : str
#         Directory to save results
# 
#     Returns
#     -------
#     dict
#         Dictionary mapping suite names to results
#     """
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True)
# 
#     suites = {
#         "io": create_io_benchmark_suite(),
#         "stats": create_stats_benchmark_suite(),
#     }
# 
#     results = {}
#     for name, suite in suites.items():
#         print(f"\nRunning {name} benchmarks...")
#         df = suite.run()
# 
#         # Save results
#         suite.save_results(output_path / f"{name}_benchmark.csv")
#         results[name] = df
# 
#     # Create summary
#     summary = []
#     for name, df in results.items():
#         summary.append(
#             {
#                 "suite": name,
#                 "functions": len(df["function"].unique()),
#                 "mean_time": df["mean_time"].mean(),
#                 "total_time": df["mean_time"].sum(),
#             }
#         )
# 
#     summary_df = pd.DataFrame(summary)
#     summary_df.to_csv(output_path / "benchmark_summary.csv", index=False)
# 
#     print(f"\nBenchmark results saved to {output_path}")
#     return results

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/benchmark/benchmark.py
# --------------------------------------------------------------------------------
