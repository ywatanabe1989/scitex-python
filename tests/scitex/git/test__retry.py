#!/usr/bin/env python3

"""Tests for git retry logic with exponential backoff."""

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("git")

from scitex.git._retry import git_retry


class TestGitRetry:
    """Tests for git_retry function."""

    def test_success_on_first_attempt(self):
        """Operation succeeds on first try."""
        mock_operation = MagicMock(return_value="success")

        result = git_retry(mock_operation)

        assert result == "success"
        assert mock_operation.call_count == 1

    def test_success_after_retries(self):
        """Operation succeeds after a few lock errors."""
        call_count = 0

        def operation_with_locks():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                error = subprocess.CalledProcessError(128, "git")
                error.stderr = (
                    b"fatal: Unable to create '.git/index.lock': File exists."
                )
                raise error
            return "success"

        result = git_retry(
            operation_with_locks,
            max_retries=5,
            initial_delay=0.01,
            max_delay=0.05,
        )

        assert result == "success"
        assert call_count == 3

    def test_raises_on_last_attempt_with_lock(self):
        """Raises CalledProcessError on last retry attempt (current behavior)."""

        def always_locked():
            error = subprocess.CalledProcessError(128, "git")
            error.stderr = b"fatal: Unable to create '.git/index.lock': File exists."
            raise error

        with pytest.raises(subprocess.CalledProcessError):
            git_retry(
                always_locked,
                max_retries=3,
                initial_delay=0.01,
                max_delay=0.05,
            )

    def test_non_lock_error_not_retried(self):
        """Non-lock CalledProcessError is raised immediately."""
        call_count = 0

        def non_lock_error():
            nonlocal call_count
            call_count += 1
            error = subprocess.CalledProcessError(128, "git")
            error.stderr = b"fatal: not a git repository"
            raise error

        with pytest.raises(subprocess.CalledProcessError):
            git_retry(
                non_lock_error,
                max_retries=5,
                initial_delay=0.01,
            )

        assert call_count == 1

    def test_non_subprocess_error_not_retried(self):
        """Non-subprocess exceptions are raised immediately."""
        call_count = 0

        def other_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Some other error")

        with pytest.raises(ValueError):
            git_retry(
                other_error,
                max_retries=5,
                initial_delay=0.01,
            )

        assert call_count == 1

    def test_lock_error_with_string_stderr(self):
        """Handle lock error with string stderr instead of bytes."""
        call_count = 0

        def lock_with_string_stderr():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                error = subprocess.CalledProcessError(128, "git")
                error.stderr = "fatal: Unable to create '.git/index.lock': File exists."
                raise error
            return "success"

        result = git_retry(
            lock_with_string_stderr,
            max_retries=3,
            initial_delay=0.01,
        )

        assert result == "success"
        assert call_count == 2

    def test_lock_error_with_none_stderr(self):
        """Handle CalledProcessError with None stderr - should not retry."""
        call_count = 0

        def error_with_none_stderr():
            nonlocal call_count
            call_count += 1
            error = subprocess.CalledProcessError(128, "git")
            error.stderr = None
            raise error

        with pytest.raises(subprocess.CalledProcessError):
            git_retry(
                error_with_none_stderr,
                max_retries=3,
                initial_delay=0.01,
            )

        assert call_count == 1

    def test_exponential_backoff(self):
        """Verify exponential backoff is applied."""
        call_times = []

        def record_time_and_fail():
            call_times.append(time.time())
            error = subprocess.CalledProcessError(128, "git")
            error.stderr = b"index.lock"
            raise error

        try:
            git_retry(
                record_time_and_fail,
                max_retries=4,
                initial_delay=0.05,
                backoff_factor=2.0,
                max_delay=1.0,
            )
        except subprocess.CalledProcessError:
            pass

        assert len(call_times) == 4

        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        delay3 = call_times[3] - call_times[2]

        assert delay1 >= 0.04
        assert delay2 >= 0.08
        assert delay3 >= 0.16

    def test_max_delay_cap(self):
        """Verify delay is capped at max_delay."""
        call_times = []

        def record_time_and_fail():
            call_times.append(time.time())
            error = subprocess.CalledProcessError(128, "git")
            error.stderr = b"index.lock"
            raise error

        try:
            git_retry(
                record_time_and_fail,
                max_retries=5,
                initial_delay=0.1,
                backoff_factor=10.0,
                max_delay=0.15,
            )
        except subprocess.CalledProcessError:
            pass

        assert len(call_times) == 5

        for i in range(1, len(call_times)):
            delay = call_times[i] - call_times[i - 1]
            assert delay <= 0.25

    def test_returns_operation_result(self):
        """Verify the operation's return value is passed through."""

        def return_dict():
            return {"status": "ok", "data": [1, 2, 3]}

        result = git_retry(return_dict)

        assert result == {"status": "ok", "data": [1, 2, 3]}


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
