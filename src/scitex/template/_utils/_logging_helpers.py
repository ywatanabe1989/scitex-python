#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging helpers for clean, hierarchical output during template operations.

Provides utilities for grouped operations with visual hierarchy.
"""

from contextlib import contextmanager
from typing import Optional
from scitex.logging import getLogger

logger = getLogger(__name__)


class LogContext:
    """Context manager for grouped logging operations."""

    def __init__(self, title: str, emoji: str = ""):
        """
        Initialize logging context.

        Args:
            title: Title of the operation
            emoji: Optional emoji prefix
        """
        self.title = title
        self.emoji = emoji
        self.indent = "   "

    def __enter__(self):
        """Start the operation."""
        prefix = f"{self.emoji} " if self.emoji else ""
        logger.info(f"{prefix}{self.title}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the operation."""
        if exc_type is None:
            # Success - no message needed, individual steps already logged
            pass
        return False  # Don't suppress exceptions

    def step(self, message: str, success: bool = True):
        """
        Log a step within the context.

        Args:
            message: Step message
            success: Whether step succeeded
        """
        symbol = "✓" if success else "✗"
        logger.info(f"{self.indent}{symbol} {message}")

    def substep(self, message: str):
        """
        Log a sub-step with extra indentation.

        Args:
            message: Sub-step message
        """
        logger.info(f"{self.indent}  {message}")


@contextmanager
def log_group(title: str, emoji: str = ""):
    """
    Context manager for grouped logging operations.

    Example:
        with log_group("Setting up project", "📦") as ctx:
            ctx.step("Created directory")
            ctx.step("Copied files")

    Args:
        title: Title of the operation
        emoji: Optional emoji prefix

    Yields:
        LogContext instance for logging steps
    """
    ctx = LogContext(title, emoji)
    with ctx:
        yield ctx


def log_step(message: str, success: bool = True, indent: int = 1):
    """
    Log a single step with indentation.

    Args:
        message: Step message
        success: Whether step succeeded
        indent: Indentation level (spaces = indent * 3)
    """
    symbol = "✓" if success else "✗"
    prefix = "   " * indent
    logger.info(f"{prefix}{symbol} {message}")


def log_final(message: str, success: bool = True):
    """
    Log final result message.

    Args:
        message: Final message
        success: Whether operation succeeded
    """
    if success:
        logger.success(f"✓ {message}")
    else:
        logger.error(f"✗ {message}")


__all__ = [
    "LogContext",
    "log_group",
    "log_step",
    "log_final",
]

# EOF
