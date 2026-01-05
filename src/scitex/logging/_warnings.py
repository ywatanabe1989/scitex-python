#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-21"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_warnings.py

"""Warning system for SciTeX, mimicking Python's warnings module.

Usage:
    import scitex.logging as logging
    from scitex.logging import UnitWarning

    # Emit a warning
    logging.warn("Missing units on axis label", UnitWarning)

    # Filter warnings (like warnings.filterwarnings)
    logging.filterwarnings("ignore", category=UnitWarning)
    logging.filterwarnings("error", category=UnitWarning)  # Raise as exception
    logging.filterwarnings("always", category=UnitWarning)  # Always show
"""

import logging as _logging
from typing import Dict, Optional, Type, Set

# =============================================================================
# Warning Categories (similar to Python's warning classes)
# =============================================================================


class SciTeXWarning(UserWarning):
    """Base warning class for all SciTeX warnings."""

    pass


class UnitWarning(SciTeXWarning):
    """Warning for axis label unit issues (educational for SI conventions).

    Raised when:
    - Axis labels are missing units
    - Units use parentheses instead of brackets (SI prefers [])
    - Units use division instead of negative exponents (m/s vs m·s⁻¹)
    """

    pass


class StyleWarning(SciTeXWarning):
    """Warning for style/formatting issues."""

    pass


class SciTeXDeprecationWarning(SciTeXWarning):
    """Warning for deprecated SciTeX features."""

    pass


class PerformanceWarning(SciTeXWarning):
    """Warning for performance issues."""

    pass


class DataLossWarning(SciTeXWarning):
    """Warning for potential data loss."""

    pass


# =============================================================================
# Warning Filter Registry
# =============================================================================

# Actions: "ignore", "error", "always", "default", "once", "module"
_filters: Dict[Type[SciTeXWarning], str] = {}
_seen_warnings: Set[str] = set()  # For "once" action


def filterwarnings(
    action: str,
    category: Type[SciTeXWarning] = SciTeXWarning,
    message: Optional[str] = None,
) -> None:
    """Control warning behavior (like warnings.filterwarnings).

    Parameters
    ----------
    action : str
        One of:
        - "ignore": Never show this warning
        - "error": Raise as exception
        - "always": Always show
        - "default": Show first occurrence per location
        - "once": Show only once total
        - "module": Show once per module
    category : type
        Warning category (default: SciTeXWarning = all)
    message : str, optional
        Regex pattern to match warning message (not implemented yet)

    Examples
    --------
    >>> import scitex.logging as logging
    >>> from scitex.logging import UnitWarning
    >>> logging.filterwarnings("ignore", category=UnitWarning)
    """
    valid_actions = {"ignore", "error", "always", "default", "once", "module"}
    if action not in valid_actions:
        raise ValueError(f"Invalid action '{action}'. Must be one of: {valid_actions}")

    _filters[category] = action


def resetwarnings() -> None:
    """Reset all warning filters to default behavior."""
    global _filters, _seen_warnings
    _filters = {}
    _seen_warnings = set()


def _get_action(category: Type[SciTeXWarning]) -> str:
    """Get the action for a warning category, checking inheritance."""
    # Check exact match first
    if category in _filters:
        return _filters[category]

    # Check parent classes
    for filter_cat, action in _filters.items():
        if issubclass(category, filter_cat):
            return action

    # Default action
    return "default"


# =============================================================================
# Warning Emission
# =============================================================================


def warn(
    message: str,
    category: Type[SciTeXWarning] = SciTeXWarning,
    stacklevel: int = 2,
) -> None:
    """Emit a warning (like warnings.warn but integrated with scitex.logging).

    Parameters
    ----------
    message : str
        Warning message
    category : type
        Warning category (default: SciTeXWarning)
    stacklevel : int
        Stack level for source location (default: 2 = caller)

    Examples
    --------
    >>> import scitex.logging as logging
    >>> from scitex.logging import UnitWarning
    >>> logging.warn("X axis has no units", UnitWarning)
    """
    import inspect

    action = _get_action(category)

    # Handle action
    if action == "ignore":
        return

    if action == "error":
        raise category(message)

    # Get source location for "once", "module", "default" actions
    frame = inspect.currentframe()
    for _ in range(stacklevel):
        if frame is not None:
            frame = frame.f_back

    location = ""
    if frame is not None:
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        location = f"{filename}:{lineno}"

    # Check if already seen
    warn_key = f"{category.__name__}:{message}:{location}"

    if action == "once":
        if warn_key in _seen_warnings:
            return
        _seen_warnings.add(warn_key)
    elif action == "default":
        # Show first per location
        loc_key = f"{category.__name__}:{location}"
        if loc_key in _seen_warnings:
            return
        _seen_warnings.add(loc_key)
    elif action == "module":
        # Show once per module
        if frame is not None:
            module_key = f"{category.__name__}:{frame.f_code.co_filename}"
            if module_key in _seen_warnings:
                return
            _seen_warnings.add(module_key)

    # Emit via scitex.logging
    logger = _logging.getLogger("scitex.warnings")
    category_name = category.__name__

    # Format: "UnitWarning: message"
    logger.warning(f"{category_name}: {message}")


# =============================================================================
# Convenience Warning Functions
# =============================================================================


def warn_deprecated(
    old_name: str, new_name: str, version: Optional[str] = None
) -> None:
    """Issue a deprecation warning."""
    message = f"{old_name} is deprecated. Use {new_name} instead."
    if version:
        message += f" Will be removed in version {version}."
    warn(message, SciTeXDeprecationWarning, stacklevel=3)


def warn_performance(operation: str, suggestion: str) -> None:
    """Issue a performance warning."""
    message = f"Performance warning in {operation}: {suggestion}"
    warn(message, PerformanceWarning, stacklevel=3)


def warn_data_loss(operation: str, detail: str) -> None:
    """Issue a data loss warning."""
    message = f"Potential data loss in {operation}: {detail}"
    warn(message, DataLossWarning, stacklevel=3)


__all__ = [
    # Warning categories
    "SciTeXWarning",
    "UnitWarning",
    "StyleWarning",
    "SciTeXDeprecationWarning",
    "PerformanceWarning",
    "DataLossWarning",
    # Functions
    "warn",
    "filterwarnings",
    "resetwarnings",
    # Convenience functions
    "warn_deprecated",
    "warn_performance",
    "warn_data_loss",
]

# EOF
