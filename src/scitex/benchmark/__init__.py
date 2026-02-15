#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-07-25 05:25:00"
# File: __init__.py

"""
SciTeX Performance Benchmarking Suite

This module provides tools for benchmarking and monitoring the performance
of SciTeX functions.
"""

from .benchmark import (
    BenchmarkResult,
    BenchmarkSuite,
    benchmark_function,
    benchmark_module,
    compare_implementations,
    run_all_benchmarks,
)
from .monitor import PerformanceMonitor, get_performance_stats, track_performance
from .profiler import get_profile_report, profile_function, profile_module

__all__ = [
    # Benchmarking
    "benchmark_function",
    "benchmark_module",
    "BenchmarkResult",
    "BenchmarkSuite",
    "run_all_benchmarks",
    "compare_implementations",
    # Profiling
    "profile_function",
    "profile_module",
    "get_profile_report",
    # Monitoring
    "PerformanceMonitor",
    "track_performance",
    "get_performance_stats",
]
