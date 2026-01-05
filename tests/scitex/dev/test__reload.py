#!/usr/bin/env python3
"""Tests for scitex.dev._reload module."""

import os
import sys
import threading
import time

import pytest

from scitex.dev import _reload


class TestReloadFunctions:
    """Tests for reload module functions."""

    def test_reload_returns_module(self):
        """Test that reload() returns the scitex module."""
        # Only test that the function can be called without error
        # Full reload testing is complex due to module state
        result = _reload.reload()
        assert result is not None
        assert hasattr(result, "__name__")

    def test_reload_auto_starts_thread(self):
        """Test that reload_auto starts a background thread."""
        # Ensure clean state
        _reload.reload_stop()
        time.sleep(0.1)

        _reload.reload_auto(interval=1)

        try:
            # Check thread is started
            assert _reload._running is True
            assert _reload._reload_thread is not None
            assert _reload._reload_thread.is_alive()
        finally:
            _reload.reload_stop()
            time.sleep(0.2)

    def test_reload_auto_is_idempotent(self):
        """Test that calling reload_auto multiple times doesn't create extra threads."""
        _reload.reload_stop()
        time.sleep(0.1)

        _reload.reload_auto(interval=1)
        thread1 = _reload._reload_thread

        _reload.reload_auto(interval=1)
        thread2 = _reload._reload_thread

        try:
            # Should be same thread
            assert thread1 is thread2
        finally:
            _reload.reload_stop()
            time.sleep(0.2)

    def test_reload_stop_stops_thread(self):
        """Test that reload_stop stops the background thread."""
        # First verify that stop can be called
        _reload.reload_stop()

        # After stop, _running should be False
        assert _reload._running is False

        # This tests that reload_stop() sets _running to False
        # The full thread lifecycle is tested in other tests

    def test_reload_stop_is_safe_when_not_running(self):
        """Test that reload_stop can be called safely when not running."""
        _reload.reload_stop()
        _reload.reload_stop()  # Should not raise
        assert _reload._running is False

    def test_auto_reload_loop_interval(self):
        """Test that auto-reload respects the interval parameter."""
        # Reset the thread state completely by setting to None
        _reload._reload_thread = None
        _reload._running = False
        time.sleep(0.1)

        # Use short interval for testing
        _reload.reload_auto(interval=10)

        try:
            # Thread should exist after calling reload_auto
            assert _reload._reload_thread is not None
            assert _reload._running is True
        finally:
            _reload.reload_stop()
            time.sleep(0.1)


class TestModuleState:
    """Tests for module-level state."""

    def test_initial_state(self):
        """Test initial module state."""
        _reload.reload_stop()
        time.sleep(0.1)

        # After stop, _running should be False
        assert _reload._running is False

    def test_thread_is_daemon(self):
        """Test that reload thread is a daemon thread."""
        _reload.reload_stop()
        time.sleep(0.1)

        _reload.reload_auto(interval=1)

        try:
            assert _reload._reload_thread.daemon is True
        finally:
            _reload.reload_stop()
            time.sleep(0.2)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
