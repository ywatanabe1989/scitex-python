#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/git_utils.py

"""
Git utilities for Writer.

Provides git lock handling and retry logic for concurrent access.
"""

import time
import subprocess
from typing import Callable, TypeVar
from scitex.logging import getLogger

logger = getLogger(__name__)

T = TypeVar('T')


def git_retry(
    operation: Callable[[], T],
    max_retries: int = 5,
    initial_delay: float = 0.1,
    max_delay: float = 2.0,
    backoff_factor: float = 2.0
) -> T:
    """
    Retry git operations with exponential backoff.

    Handles git index.lock conflicts when multiple processes access git.

    Args:
        operation: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Exponential backoff multiplier

    Returns:
        Result of operation

    Raises:
        TimeoutError: If all retries exhausted due to lock
        Exception: Original exception if not a lock error

    Example:
        >>> def commit_file():
        ...     subprocess.run(["git", "commit", "-m", "msg"], check=True)
        >>> git_retry(commit_file)
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return operation()
        except subprocess.CalledProcessError as e:
            # Check if it's a lock error
            stderr = e.stderr if isinstance(e.stderr, str) else (
                e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
            )

            if 'index.lock' in stderr and attempt < max_retries - 1:
                logger.debug(
                    f"Git lock detected, retrying in {delay:.2f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
                last_exception = e
                continue

            # Not a lock error, or retries exhausted
            raise
        except Exception:
            # Non-git errors: don't retry
            raise

    # All retries exhausted
    if last_exception:
        raise TimeoutError(
            f"Could not acquire git lock after {max_retries} attempts"
        ) from last_exception


__all__ = ['git_retry']

# EOF
