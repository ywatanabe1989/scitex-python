#!/usr/bin/env python3
"""Tests for scitex.capture.session module.

Tests Session context manager for automatic capture start/stop:
- Session initialization with parameters
- Context manager protocol (__enter__/__exit__)
- session() factory function
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class TestSessionInit:
    """Test Session class initialization."""

    def test_default_initialization(self):
        """Test Session initializes with default parameters."""
        from scitex.capture.session import Session

        sess = Session()

        assert sess.output_dir == "~/.scitex/capture/"
        assert sess.interval == 1.0
        assert sess.jpeg is True
        assert sess.quality == 60
        assert sess.on_capture is None
        assert sess.on_error is None
        assert sess.verbose is True
        assert sess.monitor_id == 0
        assert sess.capture_all is False
        assert sess.worker is None

    def test_custom_initialization(self):
        """Test Session initializes with custom parameters."""
        from scitex.capture.session import Session

        on_capture = lambda x: None
        on_error = lambda x: None

        sess = Session(
            output_dir="/custom/path",
            interval=2.5,
            jpeg=False,
            quality=90,
            on_capture=on_capture,
            on_error=on_error,
            verbose=False,
            monitor_id=1,
            capture_all=True,
        )

        assert sess.output_dir == "/custom/path"
        assert sess.interval == 2.5
        assert sess.jpeg is False
        assert sess.quality == 90
        assert sess.on_capture is on_capture
        assert sess.on_error is on_error
        assert sess.verbose is False
        assert sess.monitor_id == 1
        assert sess.capture_all is True


class TestSessionContextManager:
    """Test Session context manager protocol."""

    def test_enter_starts_monitoring(self):
        """Test __enter__ starts capture monitoring."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.session import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                sess = Session(output_dir=tmpdir, verbose=False)

                result = sess.__enter__()

                assert result is sess
                assert sess.worker is not None
                assert sess.worker.running is True

                sess.__exit__(None, None, None)
                assert sess.worker.running is False

    def test_exit_stops_monitoring(self):
        """Test __exit__ stops capture monitoring."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.session import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                sess = Session(output_dir=tmpdir, verbose=False)
                sess.__enter__()

                worker = sess.worker
                assert worker.running is True

                result = sess.__exit__(None, None, None)

                assert result is False  # Don't suppress exceptions
                assert worker.running is False

    def test_exit_returns_false(self):
        """Test __exit__ returns False (doesn't suppress exceptions)."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.session import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                sess = Session(output_dir=tmpdir, verbose=False)
                sess.__enter__()

                # Test with exception info
                result = sess.__exit__(ValueError, ValueError("test"), None)
                assert result is False

    def test_context_manager_with_statement(self):
        """Test Session works with 'with' statement."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.session import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                worker_ref = None

                with Session(output_dir=tmpdir, verbose=False) as sess:
                    worker_ref = sess.worker
                    assert sess.worker.running is True

                # After exiting context, worker should be stopped
                assert worker_ref.running is False

    def test_context_manager_passes_parameters(self):
        """Test Session passes parameters to start_monitor."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.session import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                with Session(
                    output_dir=tmpdir,
                    interval=0.5,
                    jpeg=False,
                    quality=80,
                    verbose=False,
                    monitor_id=2,
                    capture_all=True,
                ) as sess:
                    worker = sess.worker

                    assert worker.interval_sec == 0.5
                    assert worker.use_jpeg is False
                    assert worker.jpeg_quality == 80
                    assert worker.verbose is False
                    assert worker.monitor == 2
                    assert worker.capture_all is True


class TestSessionCallbacks:
    """Test Session callback functionality."""

    def test_on_capture_callback_passed(self):
        """Test on_capture callback is passed to worker."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.session import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            captures = []
            callback = lambda p: captures.append(p)

            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                with Session(
                    output_dir=tmpdir, on_capture=callback, verbose=False
                ) as sess:
                    assert sess.worker.on_capture is callback

    def test_on_error_callback_passed(self):
        """Test on_error callback is passed to worker."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.session import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            errors = []
            callback = lambda e: errors.append(e)

            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                with Session(
                    output_dir=tmpdir, on_error=callback, verbose=False
                ) as sess:
                    assert sess.worker.on_error is callback


class TestSessionExceptionHandling:
    """Test Session behavior with exceptions."""

    def test_exception_in_context_still_stops_worker(self):
        """Test worker stops even when exception occurs in context."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.session import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                sess = Session(output_dir=tmpdir, verbose=False)
                worker_ref = None

                try:
                    with sess:
                        worker_ref = sess.worker
                        raise ValueError("Test exception")
                except ValueError:
                    pass

                # Worker should still be stopped
                assert worker_ref.running is False


class TestSessionFactoryFunction:
    """Test session() factory function."""

    def test_session_returns_session_instance(self):
        """Test session() returns a Session instance."""
        from scitex.capture.session import Session, session

        result = session()
        assert isinstance(result, Session)

    def test_session_passes_kwargs(self):
        """Test session() passes kwargs to Session.__init__."""
        from scitex.capture.session import session

        with tempfile.TemporaryDirectory() as tmpdir:
            sess = session(
                output_dir=tmpdir,
                interval=2.0,
                jpeg=False,
                quality=75,
                verbose=False,
                monitor_id=3,
                capture_all=True,
            )

            assert sess.output_dir == tmpdir
            assert sess.interval == 2.0
            assert sess.jpeg is False
            assert sess.quality == 75
            assert sess.verbose is False
            assert sess.monitor_id == 3
            assert sess.capture_all is True

    def test_session_factory_works_as_context_manager(self):
        """Test session() result works as context manager."""
        from scitex.capture.capture import ScreenshotWorker
        from scitex.capture.session import session

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ScreenshotWorker, "_take_screenshot", return_value=None):
                with session(output_dir=tmpdir, verbose=False) as sess:
                    assert sess.worker is not None
                    assert sess.worker.running is True

                assert sess.worker.running is False


class TestModuleExports:
    """Test module exports."""

    def test_session_class_importable(self):
        """Test Session class can be imported."""
        from scitex.capture.session import Session

        assert Session is not None

    def test_session_function_importable(self):
        """Test session function can be imported."""
        from scitex.capture.session import session

        assert callable(session)

    def test_session_from_package_init(self):
        """Test session is accessible from package init."""
        from scitex.capture import session

        assert callable(session)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
