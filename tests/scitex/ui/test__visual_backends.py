#!/usr/bin/env python3
# Timestamp: "2026-01-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/ui/test__visual_backends.py

"""Tests for visual notification backends (matplotlib, playwright)."""

import pytest

from scitex.ui._backends import (
    BACKENDS,
    MatplotlibBackend,
    NotifyLevel,
    NotifyResult,
    PlaywrightBackend,
    get_backend,
)


class TestMatplotlibBackend:
    """Tests for MatplotlibBackend."""

    def test_in_registry(self):
        assert "matplotlib" in BACKENDS
        assert BACKENDS["matplotlib"] == MatplotlibBackend

    def test_get_backend(self):
        backend = get_backend("matplotlib")
        assert isinstance(backend, MatplotlibBackend)
        assert backend.name == "matplotlib"

    def test_init_default_timeout(self):
        backend = MatplotlibBackend()
        assert backend.timeout == 5.0

    def test_init_custom_timeout(self):
        backend = MatplotlibBackend(timeout=10.0)
        assert backend.timeout == 10.0

    def test_is_available(self):
        backend = MatplotlibBackend()
        result = backend.is_available()
        assert isinstance(result, bool)
        # matplotlib should be available in test environment
        assert result is True

    @pytest.mark.asyncio
    async def test_send_returns_notify_result(self):
        """send() should return NotifyResult even without display."""
        backend = MatplotlibBackend(timeout=0.1)  # Short timeout for test
        if backend.is_available():
            # This may fail in headless environment, but should return result
            result = await backend.send(
                "Test message",
                title="Test Title",
                level=NotifyLevel.INFO,
            )
            assert isinstance(result, NotifyResult)
            assert result.backend == "matplotlib"


class TestPlaywrightBackend:
    """Tests for PlaywrightBackend."""

    def test_in_registry(self):
        assert "playwright" in BACKENDS
        assert BACKENDS["playwright"] == PlaywrightBackend

    def test_get_backend(self):
        backend = get_backend("playwright")
        assert isinstance(backend, PlaywrightBackend)
        assert backend.name == "playwright"

    def test_init_default_timeout(self):
        backend = PlaywrightBackend()
        assert backend.timeout == 5.0

    def test_init_custom_timeout(self):
        backend = PlaywrightBackend(timeout=3.0)
        assert backend.timeout == 3.0

    def test_is_available(self):
        backend = PlaywrightBackend()
        result = backend.is_available()
        assert isinstance(result, bool)
        # playwright should be available if installed
        # Just verify it returns a bool, not the actual availability

    @pytest.mark.asyncio
    async def test_send_returns_notify_result(self):
        """send() should return NotifyResult."""
        backend = PlaywrightBackend(timeout=0.1)
        if backend.is_available():
            # May fail without browser, but should return result
            result = await backend.send(
                "Test message",
                title="Test Title",
                level=NotifyLevel.WARNING,
            )
            assert isinstance(result, NotifyResult)
            assert result.backend == "playwright"


class TestVisualBackendLevels:
    """Tests for notification level handling in visual backends."""

    @pytest.fixture
    def matplotlib_backend(self):
        return MatplotlibBackend(timeout=0.1)

    @pytest.fixture
    def playwright_backend(self):
        return PlaywrightBackend(timeout=0.1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "level",
        [
            NotifyLevel.INFO,
            NotifyLevel.WARNING,
            NotifyLevel.ERROR,
            NotifyLevel.CRITICAL,
        ],
    )
    async def test_matplotlib_handles_all_levels(self, matplotlib_backend, level):
        """MatplotlibBackend should handle all notification levels."""
        if matplotlib_backend.is_available():
            result = await matplotlib_backend.send("Test", level=level)
            assert isinstance(result, NotifyResult)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "level",
        [
            NotifyLevel.INFO,
            NotifyLevel.WARNING,
            NotifyLevel.ERROR,
            NotifyLevel.CRITICAL,
        ],
    )
    async def test_playwright_handles_all_levels(self, playwright_backend, level):
        """PlaywrightBackend should handle all notification levels."""
        if playwright_backend.is_available():
            result = await playwright_backend.send("Test", level=level)
            assert isinstance(result, NotifyResult)


if __name__ == "__main__":
    pytest.main([__file__])

# EOF
