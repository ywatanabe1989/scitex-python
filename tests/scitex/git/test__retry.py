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

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_retry.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-29 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/git/retry.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/git/retry.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Git retry logic with exponential backoff.
# 
# Handles git index.lock conflicts when multiple processes access git.
# Shared across all scitex modules.
# """
# 
# import time
# import subprocess
# from typing import Callable, TypeVar
# 
# from scitex.logging import getLogger
# 
# logger = getLogger(__name__)
# 
# T = TypeVar("T")
# 
# 
# def git_retry(
#     operation: Callable[[], T],
#     max_retries: int = 5,
#     initial_delay: float = 0.1,
#     max_delay: float = 2.0,
#     backoff_factor: float = 2.0,
# ) -> T:
#     """
#     Retry git operations with exponential backoff.
# 
#     Handles git index.lock conflicts when multiple processes access git.
# 
#     Parameters
#     ----------
#     operation : Callable
#         Function to retry
#     max_retries : int
#         Maximum number of retry attempts
#     initial_delay : float
#         Initial delay in seconds
#     max_delay : float
#         Maximum delay between retries
#     backoff_factor : float
#         Exponential backoff multiplier
# 
#     Returns
#     -------
#     T
#         Result of operation
# 
#     Raises
#     ------
#     TimeoutError
#         If all retries exhausted due to lock
#     Exception
#         Original exception if not a lock error
# 
#     Examples
#     --------
#     >>> def commit_file():
#     ...     subprocess.run(["git", "commit", "-m", "msg"], check=True)
#     >>> git_retry(commit_file)
#     """
#     delay = initial_delay
#     last_exception = None
# 
#     for attempt in range(max_retries):
#         try:
#             return operation()
#         except subprocess.CalledProcessError as e:
#             # Check if it's a lock error
#             stderr = (
#                 e.stderr
#                 if isinstance(e.stderr, str)
#                 else (e.stderr.decode("utf-8", errors="ignore") if e.stderr else "")
#             )
# 
#             if "index.lock" in stderr and attempt < max_retries - 1:
#                 logger.debug(
#                     f"Git lock detected, retrying in {delay:.2f}s "
#                     f"(attempt {attempt + 1}/{max_retries})"
#                 )
#                 time.sleep(delay)
#                 delay = min(delay * backoff_factor, max_delay)
#                 last_exception = e
#                 continue
# 
#             # Not a lock error, or retries exhausted
#             raise
#         except Exception:
#             # Non-git errors: don't retry
#             raise
# 
#     # All retries exhausted
#     if last_exception:
#         raise TimeoutError(
#             f"Could not acquire git lock after {max_retries} attempts"
#         ) from last_exception
# 
# 
# __all__ = ["git_retry"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_retry.py
# --------------------------------------------------------------------------------
