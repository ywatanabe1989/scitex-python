#!/usr/bin/env python3
# Time-stamp: "2025-01-05"
# File: test_monitor.py

"""Tests for scitex.benchmark.monitor module."""

import json
import os
import tempfile
import threading
import time
import warnings

import pytest

from scitex.benchmark.monitor import (
    PerformanceMetric,
    PerformanceMonitor,
    add_performance_alert_handler,
    get_performance_stats,
    set_performance_alerts,
    track_performance,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def monitor():
    """Create a fresh PerformanceMonitor instance."""
    return PerformanceMonitor(max_history=100)


@pytest.fixture
def started_monitor():
    """Create a started PerformanceMonitor instance."""
    mon = PerformanceMonitor(max_history=100)
    mon.start()
    yield mon
    mon.stop()


@pytest.fixture
def sample_metric():
    """Create a sample PerformanceMetric."""
    return PerformanceMetric(
        timestamp=time.time(),
        function="test_function",
        duration=0.5,
        memory_delta=10.0,
        args_size=100,
        result_size=50,
        exception=None,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Test PerformanceMetric
# ============================================================================


class TestPerformanceMetric:
    """Tests for PerformanceMetric dataclass."""

    def test_creation_with_all_fields(self, sample_metric):
        """Test PerformanceMetric with all fields."""
        assert sample_metric.function == "test_function"
        assert sample_metric.duration == 0.5
        assert sample_metric.memory_delta == 10.0
        assert sample_metric.args_size == 100
        assert sample_metric.result_size == 50
        assert sample_metric.exception is None

    def test_creation_with_required_fields_only(self):
        """Test PerformanceMetric with only required fields."""
        metric = PerformanceMetric(
            timestamp=1234567890.0,
            function="my_func",
            duration=0.1,
        )

        assert metric.timestamp == 1234567890.0
        assert metric.function == "my_func"
        assert metric.duration == 0.1
        assert metric.memory_delta is None
        assert metric.args_size is None
        assert metric.result_size is None
        assert metric.exception is None

    def test_creation_with_exception(self):
        """Test PerformanceMetric with exception."""
        metric = PerformanceMetric(
            timestamp=time.time(),
            function="error_func",
            duration=0.01,
            exception="ValueError: test error",
        )

        assert metric.exception == "ValueError: test error"


# ============================================================================
# Test PerformanceMonitor
# ============================================================================


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    def test_monitor_creation(self, monitor):
        """Test monitor initialization."""
        assert monitor.max_history == 100
        assert len(monitor.metrics) == 0
        assert monitor.is_monitoring is False
        # New instances start with empty callbacks (default handler only on global)
        assert isinstance(monitor.alert_callbacks, list)

    def test_start_stop(self, monitor):
        """Test start and stop monitoring."""
        assert monitor.is_monitoring is False

        monitor.start()
        assert monitor.is_monitoring is True

        monitor.stop()
        assert monitor.is_monitoring is False

    def test_record_metric_when_monitoring(self, started_monitor, sample_metric):
        """Test recording metrics when monitoring is active."""
        started_monitor.record_metric(sample_metric)

        assert len(started_monitor.metrics) == 1
        assert started_monitor.function_stats["test_function"]["count"] == 1

    def test_record_metric_when_not_monitoring(self, monitor, sample_metric):
        """Test that metrics are not recorded when monitoring is off."""
        monitor.record_metric(sample_metric)

        assert len(monitor.metrics) == 0

    def test_function_stats_updated(self, started_monitor):
        """Test that function stats are updated correctly."""
        for i in range(5):
            metric = PerformanceMetric(
                timestamp=time.time(),
                function="my_func",
                duration=0.1 * (i + 1),
            )
            started_monitor.record_metric(metric)

        stats = started_monitor.function_stats["my_func"]
        assert stats["count"] == 5
        assert stats["min_time"] == 0.1
        assert stats["max_time"] == 0.5
        assert abs(stats["total_time"] - 1.5) < 0.001

    def test_error_tracking(self, started_monitor):
        """Test that errors are tracked."""
        # Record normal metric
        started_monitor.record_metric(
            PerformanceMetric(timestamp=time.time(), function="my_func", duration=0.1)
        )

        # Record error metric
        started_monitor.record_metric(
            PerformanceMetric(
                timestamp=time.time(),
                function="my_func",
                duration=0.1,
                exception="Error!",
            )
        )

        stats = started_monitor.function_stats["my_func"]
        assert stats["count"] == 2
        assert stats["errors"] == 1

    def test_max_history_limit(self):
        """Test that max_history limits stored metrics."""
        monitor = PerformanceMonitor(max_history=5)
        monitor.start()

        for i in range(10):
            monitor.record_metric(
                PerformanceMetric(timestamp=time.time(), function="func", duration=0.01)
            )

        assert len(monitor.metrics) == 5
        monitor.stop()

    def test_get_stats_all(self, started_monitor):
        """Test get_stats for all functions."""
        started_monitor.record_metric(
            PerformanceMetric(timestamp=time.time(), function="func1", duration=0.1)
        )
        started_monitor.record_metric(
            PerformanceMetric(timestamp=time.time(), function="func2", duration=0.2)
        )

        stats = started_monitor.get_stats()

        assert "func1" in stats
        assert "func2" in stats
        assert stats["func1"]["avg_time"] == 0.1
        assert stats["func2"]["avg_time"] == 0.2

    def test_get_stats_single_function(self, started_monitor):
        """Test get_stats for single function."""
        started_monitor.record_metric(
            PerformanceMetric(timestamp=time.time(), function="my_func", duration=0.1)
        )
        started_monitor.record_metric(
            PerformanceMetric(timestamp=time.time(), function="my_func", duration=0.3)
        )

        stats = started_monitor.get_stats("my_func")

        assert stats["function"] == "my_func"
        assert stats["count"] == 2
        assert stats["avg_time"] == 0.2
        assert stats["min_time"] == 0.1
        assert stats["max_time"] == 0.3

    def test_get_stats_unknown_function(self, started_monitor):
        """Test get_stats for unknown function."""
        stats = started_monitor.get_stats("unknown_func")
        assert stats == {}

    def test_get_recent_metrics(self, started_monitor):
        """Test get_recent_metrics."""
        for i in range(10):
            started_monitor.record_metric(
                PerformanceMetric(
                    timestamp=time.time() + i, function=f"func_{i}", duration=0.01
                )
            )

        recent = started_monitor.get_recent_metrics(5)

        assert len(recent) == 5
        # Should be the last 5
        assert recent[0].function == "func_5"
        assert recent[-1].function == "func_9"

    def test_clear_metrics(self, started_monitor, sample_metric):
        """Test clearing metrics."""
        started_monitor.record_metric(sample_metric)
        assert len(started_monitor.metrics) == 1

        started_monitor.clear()

        assert len(started_monitor.metrics) == 0
        assert len(started_monitor.function_stats) == 0

    def test_save_metrics(self, started_monitor, sample_metric, temp_dir):
        """Test saving metrics to file."""
        started_monitor.record_metric(sample_metric)

        output_path = os.path.join(temp_dir, "metrics.json")
        started_monitor.save_metrics(output_path)

        assert os.path.exists(output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "metrics" in data
        assert "stats" in data
        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["function"] == "test_function"

    def test_load_metrics(self, monitor, temp_dir):
        """Test loading metrics from file."""
        # Create test data file
        data = {
            "metrics": [
                {
                    "timestamp": 123456.0,
                    "function": "loaded_func",
                    "duration": 0.5,
                    "memory_delta": None,
                    "args_size": None,
                    "result_size": None,
                    "exception": None,
                }
            ],
            "stats": {"loaded_func": {"count": 1, "total_time": 0.5}},
        }

        input_path = os.path.join(temp_dir, "metrics.json")
        with open(input_path, "w") as f:
            json.dump(data, f)

        monitor.load_metrics(input_path)

        assert len(monitor.metrics) == 1
        assert monitor.metrics[0].function == "loaded_func"

    def test_thread_safety(self, started_monitor):
        """Test thread safety of recording metrics."""
        num_threads = 5
        metrics_per_thread = 100

        def record_metrics():
            for i in range(metrics_per_thread):
                started_monitor.record_metric(
                    PerformanceMetric(
                        timestamp=time.time(),
                        function="thread_func",
                        duration=0.001,
                    )
                )

        threads = [threading.Thread(target=record_metrics) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = started_monitor.function_stats["thread_func"]
        assert stats["count"] == num_threads * metrics_per_thread


# ============================================================================
# Test Alerts
# ============================================================================


class TestAlerts:
    """Tests for performance alerts."""

    def test_slow_function_alert(self, started_monitor):
        """Test slow function alert."""
        alerts_received = []

        def alert_handler(alert):
            alerts_received.append(alert)

        started_monitor.alert_callbacks = [alert_handler]
        started_monitor.alerts["slow_function"] = 0.1

        # Record slow metric
        started_monitor.record_metric(
            PerformanceMetric(timestamp=time.time(), function="slow_func", duration=0.5)
        )

        assert len(alerts_received) == 1
        assert alerts_received[0]["type"] == "slow_function"
        assert alerts_received[0]["function"] == "slow_func"

    def test_memory_spike_alert(self, started_monitor):
        """Test memory spike alert."""
        alerts_received = []

        def alert_handler(alert):
            alerts_received.append(alert)

        started_monitor.alert_callbacks = [alert_handler]
        started_monitor.alerts["memory_spike"] = 50

        # Record metric with large memory delta
        started_monitor.record_metric(
            PerformanceMetric(
                timestamp=time.time(),
                function="mem_func",
                duration=0.1,
                memory_delta=100,
            )
        )

        assert len(alerts_received) == 1
        assert alerts_received[0]["type"] == "memory_spike"

    def test_no_alert_below_threshold(self, started_monitor):
        """Test no alert when below threshold."""
        alerts_received = []

        def alert_handler(alert):
            alerts_received.append(alert)

        started_monitor.alert_callbacks = [alert_handler]
        started_monitor.alerts["slow_function"] = 1.0

        # Record fast metric
        started_monitor.record_metric(
            PerformanceMetric(timestamp=time.time(), function="fast_func", duration=0.1)
        )

        # No slow function alert expected
        slow_alerts = [a for a in alerts_received if a["type"] == "slow_function"]
        assert len(slow_alerts) == 0

    def test_add_alert_callback(self, monitor):
        """Test adding alert callbacks."""
        initial_count = len(monitor.alert_callbacks)

        def my_handler(alert):
            pass

        monitor.add_alert_callback(my_handler)

        assert len(monitor.alert_callbacks) == initial_count + 1


# ============================================================================
# Test track_performance decorator
# ============================================================================


class TestTrackPerformance:
    """Tests for track_performance decorator."""

    def test_track_performance_basic(self):
        """Test basic track_performance usage."""

        @track_performance
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_track_performance_preserves_function(self):
        """Test that decorator preserves function metadata."""

        @track_performance
        def original_name(x):
            """Original docstring."""
            return x

        assert original_name.__name__ == "original_name"
        assert original_name.__doc__ == "Original docstring."

    def test_track_performance_with_exception(self):
        """Test track_performance with exception."""

        @track_performance
        def error_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            error_func()


# ============================================================================
# Test Module-Level Functions
# ============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_performance_stats(self):
        """Test get_performance_stats function."""
        stats = get_performance_stats()
        assert isinstance(stats, dict)

    def test_get_performance_stats_with_function(self):
        """Test get_performance_stats with function name."""
        stats = get_performance_stats("unknown_func")
        # Should return empty dict for unknown function
        assert isinstance(stats, dict)

    def test_set_performance_alerts(self):
        """Test set_performance_alerts function."""
        # Should not raise
        set_performance_alerts(slow_function=2.0, memory_spike=200)

    def test_add_performance_alert_handler(self):
        """Test add_performance_alert_handler function."""

        def my_handler(alert):
            pass

        # Should not raise
        add_performance_alert_handler(my_handler)


# ============================================================================
# Test Default Alert Handler
# ============================================================================


class TestDefaultAlertHandler:
    """Tests for default alert handler."""

    def test_default_handler_slow_function_warning(self):
        """Test default handler issues warning for slow function."""
        from scitex.benchmark.monitor import _default_alert_handler

        # Create monitor with default handler registered
        monitor = PerformanceMonitor(max_history=100)
        monitor.add_alert_callback(_default_alert_handler)
        monitor.start()
        monitor.alerts["slow_function"] = 0.01

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                monitor.record_metric(
                    PerformanceMetric(
                        timestamp=time.time(), function="slow_func", duration=0.1
                    )
                )

                # Check for warning
                slow_warnings = [
                    warning for warning in w if "Slow function" in str(warning.message)
                ]
                assert len(slow_warnings) >= 1
        finally:
            monitor.stop()


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
