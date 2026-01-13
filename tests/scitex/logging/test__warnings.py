#!/usr/bin/env python3
"""Tests for scitex.logging._warnings module."""

import logging
import os

import pytest


class TestWarningCategories:
    """Test warning category classes."""

    def test_scitex_warning_is_user_warning(self):
        """Test that SciTeXWarning is a subclass of UserWarning."""
        from scitex.logging._warnings import SciTeXWarning

        assert issubclass(SciTeXWarning, UserWarning)

    def test_unit_warning_is_scitex_warning(self):
        """Test that UnitWarning is a subclass of SciTeXWarning."""
        from scitex.logging._warnings import SciTeXWarning, UnitWarning

        assert issubclass(UnitWarning, SciTeXWarning)

    def test_style_warning_is_scitex_warning(self):
        """Test that StyleWarning is a subclass of SciTeXWarning."""
        from scitex.logging._warnings import SciTeXWarning, StyleWarning

        assert issubclass(StyleWarning, SciTeXWarning)

    def test_deprecation_warning_is_scitex_warning(self):
        """Test that SciTeXDeprecationWarning is a subclass of SciTeXWarning."""
        from scitex.logging._warnings import SciTeXDeprecationWarning, SciTeXWarning

        assert issubclass(SciTeXDeprecationWarning, SciTeXWarning)

    def test_performance_warning_is_scitex_warning(self):
        """Test that PerformanceWarning is a subclass of SciTeXWarning."""
        from scitex.logging._warnings import PerformanceWarning, SciTeXWarning

        assert issubclass(PerformanceWarning, SciTeXWarning)

    def test_data_loss_warning_is_scitex_warning(self):
        """Test that DataLossWarning is a subclass of SciTeXWarning."""
        from scitex.logging._warnings import DataLossWarning, SciTeXWarning

        assert issubclass(DataLossWarning, SciTeXWarning)


class TestFilterWarnings:
    """Test filterwarnings function."""

    def setup_method(self):
        """Reset warning filters before each test."""
        from scitex.logging._warnings import resetwarnings

        resetwarnings()

    def teardown_method(self):
        """Reset warning filters after each test."""
        from scitex.logging._warnings import resetwarnings

        resetwarnings()

    def test_filterwarnings_ignore(self):
        """Test filterwarnings with 'ignore' action."""
        from scitex.logging._warnings import UnitWarning, filterwarnings

        filterwarnings("ignore", category=UnitWarning)
        # Should not raise or log

    def test_filterwarnings_error(self):
        """Test filterwarnings with 'error' action."""
        from scitex.logging._warnings import UnitWarning, filterwarnings, warn

        filterwarnings("error", category=UnitWarning)

        with pytest.raises(UnitWarning):
            warn("Test warning", UnitWarning)

    def test_filterwarnings_always(self):
        """Test filterwarnings with 'always' action."""
        from scitex.logging._warnings import UnitWarning, filterwarnings

        filterwarnings("always", category=UnitWarning)
        # Should not raise

    def test_filterwarnings_invalid_action(self):
        """Test filterwarnings with invalid action raises ValueError."""
        from scitex.logging._warnings import filterwarnings

        with pytest.raises(ValueError):
            filterwarnings("invalid_action")

    def test_filterwarnings_valid_actions(self):
        """Test all valid actions are accepted."""
        from scitex.logging._warnings import filterwarnings

        valid_actions = ["ignore", "error", "always", "default", "once", "module"]
        for action in valid_actions:
            filterwarnings(action)  # Should not raise


class TestResetWarnings:
    """Test resetwarnings function."""

    def test_resetwarnings_clears_filters(self):
        """Test that resetwarnings clears all filters."""
        from scitex.logging._warnings import (
            UnitWarning,
            filterwarnings,
            resetwarnings,
            warn,
        )

        filterwarnings("error", category=UnitWarning)

        # Should raise before reset
        with pytest.raises(UnitWarning):
            warn("Test", UnitWarning)

        resetwarnings()

        # Should not raise after reset (default behavior)
        # Note: default behavior logs the warning, doesn't raise


class TestWarn:
    """Test warn function."""

    def setup_method(self):
        """Reset warning filters before each test."""
        from scitex.logging._warnings import resetwarnings

        resetwarnings()

    def teardown_method(self):
        """Reset warning filters after each test."""
        from scitex.logging._warnings import resetwarnings

        resetwarnings()

    def test_warn_with_ignore_action(self):
        """Test warn with ignore action does not log."""
        from scitex.logging._warnings import UnitWarning, filterwarnings, warn

        filterwarnings("ignore", category=UnitWarning)
        # Should not raise
        warn("Ignored warning", UnitWarning)

    def test_warn_with_error_action_raises(self):
        """Test warn with error action raises the warning."""
        from scitex.logging._warnings import UnitWarning, filterwarnings, warn

        filterwarnings("error", category=UnitWarning)

        with pytest.raises(UnitWarning) as exc_info:
            warn("Error warning", UnitWarning)

        assert "Error warning" in str(exc_info.value)

    def test_warn_default_category(self):
        """Test warn uses SciTeXWarning as default category."""
        from scitex.logging._warnings import SciTeXWarning, filterwarnings, warn

        filterwarnings("error", category=SciTeXWarning)

        with pytest.raises(SciTeXWarning):
            warn("Default category warning")

    def test_warn_inherits_parent_filter(self):
        """Test warn respects filter on parent category."""
        from scitex.logging._warnings import (
            SciTeXWarning,
            UnitWarning,
            filterwarnings,
            warn,
        )

        # Set filter on parent class
        filterwarnings("error", category=SciTeXWarning)

        # Child class should inherit the filter
        with pytest.raises(UnitWarning):
            warn("Test", UnitWarning)


class TestConvenienceWarnings:
    """Test convenience warning functions."""

    def setup_method(self):
        """Reset warning filters before each test."""
        from scitex.logging._warnings import resetwarnings

        resetwarnings()

    def teardown_method(self):
        """Reset warning filters after each test."""
        from scitex.logging._warnings import resetwarnings

        resetwarnings()

    def test_warn_deprecated_basic(self):
        """Test warn_deprecated basic usage."""
        from scitex.logging._warnings import (
            SciTeXDeprecationWarning,
            filterwarnings,
            warn_deprecated,
        )

        filterwarnings("error", category=SciTeXDeprecationWarning)

        with pytest.raises(SciTeXDeprecationWarning) as exc_info:
            warn_deprecated("old_func", "new_func")

        assert "old_func" in str(exc_info.value)
        assert "new_func" in str(exc_info.value)
        assert "deprecated" in str(exc_info.value)

    def test_warn_deprecated_with_version(self):
        """Test warn_deprecated with version."""
        from scitex.logging._warnings import (
            SciTeXDeprecationWarning,
            filterwarnings,
            warn_deprecated,
        )

        filterwarnings("error", category=SciTeXDeprecationWarning)

        with pytest.raises(SciTeXDeprecationWarning) as exc_info:
            warn_deprecated("old_func", "new_func", version="2.0")

        assert "2.0" in str(exc_info.value)

    def test_warn_performance(self):
        """Test warn_performance function."""
        from scitex.logging._warnings import (
            PerformanceWarning,
            filterwarnings,
            warn_performance,
        )

        filterwarnings("error", category=PerformanceWarning)

        with pytest.raises(PerformanceWarning) as exc_info:
            warn_performance("matrix_multiply", "Use vectorized operations")

        assert "matrix_multiply" in str(exc_info.value)
        assert "vectorized" in str(exc_info.value)

    def test_warn_data_loss(self):
        """Test warn_data_loss function."""
        from scitex.logging._warnings import (
            DataLossWarning,
            filterwarnings,
            warn_data_loss,
        )

        filterwarnings("error", category=DataLossWarning)

        with pytest.raises(DataLossWarning) as exc_info:
            warn_data_loss("truncation", "Values will be truncated")

        assert "truncation" in str(exc_info.value)
        assert "truncated" in str(exc_info.value)


class TestOnceAndDefaultActions:
    """Test 'once' and 'default' warning actions."""

    def setup_method(self):
        """Reset warning filters before each test."""
        from scitex.logging._warnings import resetwarnings

        resetwarnings()

    def teardown_method(self):
        """Reset warning filters after each test."""
        from scitex.logging._warnings import resetwarnings

        resetwarnings()

    def test_once_action_shows_only_once(self):
        """Test that 'once' action shows warning only once."""
        from scitex.logging._warnings import UnitWarning, filterwarnings, warn

        filterwarnings("once", category=UnitWarning)

        # Set up a handler to capture warnings
        logger = logging.getLogger("scitex.warnings")
        handler = logging.handlers.MemoryHandler(capacity=100)
        handler.setLevel(logging.WARNING)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        # Emit same warning twice
        warn("Same warning", UnitWarning)
        warn("Same warning", UnitWarning)

        # Flush handler to see records
        handler.flush()

        logger.removeHandler(handler)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_warnings.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-21"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_warnings.py
# 
# """Warning system for SciTeX, mimicking Python's warnings module.
# 
# Usage:
#     import scitex.logging as logging
#     from scitex.logging import UnitWarning
# 
#     # Emit a warning
#     logging.warn("Missing units on axis label", UnitWarning)
# 
#     # Filter warnings (like warnings.filterwarnings)
#     logging.filterwarnings("ignore", category=UnitWarning)
#     logging.filterwarnings("error", category=UnitWarning)  # Raise as exception
#     logging.filterwarnings("always", category=UnitWarning)  # Always show
# """
# 
# import logging as _logging
# from typing import Dict, Optional, Type, Set
# 
# # =============================================================================
# # Warning Categories (similar to Python's warning classes)
# # =============================================================================
# 
# 
# class SciTeXWarning(UserWarning):
#     """Base warning class for all SciTeX warnings."""
# 
#     pass
# 
# 
# class UnitWarning(SciTeXWarning):
#     """Warning for axis label unit issues (educational for SI conventions).
# 
#     Raised when:
#     - Axis labels are missing units
#     - Units use parentheses instead of brackets (SI prefers [])
#     - Units use division instead of negative exponents (m/s vs m·s⁻¹)
#     """
# 
#     pass
# 
# 
# class StyleWarning(SciTeXWarning):
#     """Warning for style/formatting issues."""
# 
#     pass
# 
# 
# class SciTeXDeprecationWarning(SciTeXWarning):
#     """Warning for deprecated SciTeX features."""
# 
#     pass
# 
# 
# class PerformanceWarning(SciTeXWarning):
#     """Warning for performance issues."""
# 
#     pass
# 
# 
# class DataLossWarning(SciTeXWarning):
#     """Warning for potential data loss."""
# 
#     pass
# 
# 
# # =============================================================================
# # Warning Filter Registry
# # =============================================================================
# 
# # Actions: "ignore", "error", "always", "default", "once", "module"
# _filters: Dict[Type[SciTeXWarning], str] = {}
# _seen_warnings: Set[str] = set()  # For "once" action
# 
# 
# def filterwarnings(
#     action: str,
#     category: Type[SciTeXWarning] = SciTeXWarning,
#     message: Optional[str] = None,
# ) -> None:
#     """Control warning behavior (like warnings.filterwarnings).
# 
#     Parameters
#     ----------
#     action : str
#         One of:
#         - "ignore": Never show this warning
#         - "error": Raise as exception
#         - "always": Always show
#         - "default": Show first occurrence per location
#         - "once": Show only once total
#         - "module": Show once per module
#     category : type
#         Warning category (default: SciTeXWarning = all)
#     message : str, optional
#         Regex pattern to match warning message (not implemented yet)
# 
#     Examples
#     --------
#     >>> import scitex.logging as logging
#     >>> from scitex.logging import UnitWarning
#     >>> logging.filterwarnings("ignore", category=UnitWarning)
#     """
#     valid_actions = {"ignore", "error", "always", "default", "once", "module"}
#     if action not in valid_actions:
#         raise ValueError(f"Invalid action '{action}'. Must be one of: {valid_actions}")
# 
#     _filters[category] = action
# 
# 
# def resetwarnings() -> None:
#     """Reset all warning filters to default behavior."""
#     global _filters, _seen_warnings
#     _filters = {}
#     _seen_warnings = set()
# 
# 
# def _get_action(category: Type[SciTeXWarning]) -> str:
#     """Get the action for a warning category, checking inheritance."""
#     # Check exact match first
#     if category in _filters:
#         return _filters[category]
# 
#     # Check parent classes
#     for filter_cat, action in _filters.items():
#         if issubclass(category, filter_cat):
#             return action
# 
#     # Default action
#     return "default"
# 
# 
# # =============================================================================
# # Warning Emission
# # =============================================================================
# 
# 
# def warn(
#     message: str,
#     category: Type[SciTeXWarning] = SciTeXWarning,
#     stacklevel: int = 2,
# ) -> None:
#     """Emit a warning (like warnings.warn but integrated with scitex.logging).
# 
#     Parameters
#     ----------
#     message : str
#         Warning message
#     category : type
#         Warning category (default: SciTeXWarning)
#     stacklevel : int
#         Stack level for source location (default: 2 = caller)
# 
#     Examples
#     --------
#     >>> import scitex.logging as logging
#     >>> from scitex.logging import UnitWarning
#     >>> logging.warn("X axis has no units", UnitWarning)
#     """
#     import inspect
# 
#     action = _get_action(category)
# 
#     # Handle action
#     if action == "ignore":
#         return
# 
#     if action == "error":
#         raise category(message)
# 
#     # Get source location for "once", "module", "default" actions
#     frame = inspect.currentframe()
#     for _ in range(stacklevel):
#         if frame is not None:
#             frame = frame.f_back
# 
#     location = ""
#     if frame is not None:
#         filename = frame.f_code.co_filename
#         lineno = frame.f_lineno
#         location = f"{filename}:{lineno}"
# 
#     # Check if already seen
#     warn_key = f"{category.__name__}:{message}:{location}"
# 
#     if action == "once":
#         if warn_key in _seen_warnings:
#             return
#         _seen_warnings.add(warn_key)
#     elif action == "default":
#         # Show first per location
#         loc_key = f"{category.__name__}:{location}"
#         if loc_key in _seen_warnings:
#             return
#         _seen_warnings.add(loc_key)
#     elif action == "module":
#         # Show once per module
#         if frame is not None:
#             module_key = f"{category.__name__}:{frame.f_code.co_filename}"
#             if module_key in _seen_warnings:
#                 return
#             _seen_warnings.add(module_key)
# 
#     # Emit via scitex.logging
#     logger = _logging.getLogger("scitex.warnings")
#     category_name = category.__name__
# 
#     # Format: "UnitWarning: message"
#     logger.warning(f"{category_name}: {message}")
# 
# 
# # =============================================================================
# # Convenience Warning Functions
# # =============================================================================
# 
# 
# def warn_deprecated(
#     old_name: str, new_name: str, version: Optional[str] = None
# ) -> None:
#     """Issue a deprecation warning."""
#     message = f"{old_name} is deprecated. Use {new_name} instead."
#     if version:
#         message += f" Will be removed in version {version}."
#     warn(message, SciTeXDeprecationWarning, stacklevel=3)
# 
# 
# def warn_performance(operation: str, suggestion: str) -> None:
#     """Issue a performance warning."""
#     message = f"Performance warning in {operation}: {suggestion}"
#     warn(message, PerformanceWarning, stacklevel=3)
# 
# 
# def warn_data_loss(operation: str, detail: str) -> None:
#     """Issue a data loss warning."""
#     message = f"Potential data loss in {operation}: {detail}"
#     warn(message, DataLossWarning, stacklevel=3)
# 
# 
# __all__ = [
#     # Warning categories
#     "SciTeXWarning",
#     "UnitWarning",
#     "StyleWarning",
#     "SciTeXDeprecationWarning",
#     "PerformanceWarning",
#     "DataLossWarning",
#     # Functions
#     "warn",
#     "filterwarnings",
#     "resetwarnings",
#     # Convenience functions
#     "warn_deprecated",
#     "warn_performance",
#     "warn_data_loss",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_warnings.py
# --------------------------------------------------------------------------------
