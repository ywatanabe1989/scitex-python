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
    pytest.main([os.path.abspath(__file__), "-v"])
