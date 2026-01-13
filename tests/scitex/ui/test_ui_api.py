#!/usr/bin/env python3
# Timestamp: "2026-01-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/ui/test_ui_api.py

"""Tests for scitex.ui public API."""

import pytest


class TestPublicAPI:
    """Tests for the public API of scitex.ui module."""

    def test_import_alert(self):
        from scitex.ui import alert

        assert callable(alert)

    def test_import_alert_async(self):
        from scitex.ui import alert_async

        assert callable(alert_async)

    def test_import_available_backends(self):
        from scitex.ui import available_backends

        assert callable(available_backends)

    def test_all_exports(self):
        import scitex.ui as ui

        assert "alert" in ui.__all__
        assert "alert_async" in ui.__all__
        assert "available_backends" in ui.__all__

    def test_minimal_api(self):
        """Verify only minimal API is exposed in __all__."""
        import scitex.ui as ui

        # Should only have 3 public items
        assert len(ui.__all__) == 3


class TestAvailableBackends:
    """Tests for available_backends function."""

    def test_returns_list(self):
        from scitex.ui import available_backends

        result = available_backends()
        assert isinstance(result, list)

    def test_audio_available(self):
        from scitex.ui import available_backends

        backends = available_backends()
        # Audio should typically be available
        assert "audio" in backends


class TestAlertFunction:
    """Tests for alert function."""

    def test_alert_returns_bool(self):
        from scitex.ui import alert

        # Use a backend that won't actually send (webhook with no URL)
        result = alert("Test", backend="webhook")
        assert isinstance(result, bool)

    def test_alert_with_invalid_backend(self):
        from scitex.ui import alert

        # Should return False when backend is invalid and fallback disabled
        result = alert("Test", backend="nonexistent_backend", fallback=False)
        assert result is False

    def test_alert_with_level_string(self):
        from scitex.ui import alert

        # Should accept string levels
        result = alert("Test", backend="webhook", level="critical")
        assert isinstance(result, bool)

    def test_alert_with_invalid_level(self):
        from scitex.ui import alert

        # Invalid level should default to INFO, not raise
        result = alert("Test", backend="webhook", level="invalid_level")
        assert isinstance(result, bool)


@pytest.mark.asyncio
class TestAlertAsyncFunction:
    """Async tests for alert_async function."""

    async def test_alert_async_returns_bool(self):
        from scitex.ui import alert_async

        result = await alert_async("Test", backend="webhook")
        assert isinstance(result, bool)

    async def test_alert_async_with_title(self):
        from scitex.ui import alert_async

        result = await alert_async(
            "Test message", title="Test Title", backend="webhook"
        )
        assert isinstance(result, bool)

    async def test_alert_async_multiple_backends(self):
        from scitex.ui import alert_async

        # Test with list of backends
        result = await alert_async("Test", backend=["webhook", "desktop"])
        assert isinstance(result, bool)


class TestCompatModule:
    """Tests for backward compatibility module."""

    def test_compat_notify_exists(self):
        from scitex.compat import notify

        assert callable(notify)

    def test_compat_notify_warns(self):
        from scitex.compat import notify

        with pytest.warns(DeprecationWarning, match="deprecated"):
            notify("Test", backend="webhook")


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
