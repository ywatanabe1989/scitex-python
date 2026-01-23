#!/usr/bin/env python3
# Timestamp: "2026-01-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/ui/test__backends.py

"""Tests for scitex.ui._backends module."""

import pytest

from scitex.ui._backends import (
    BACKENDS,
    AudioBackend,
    DesktopBackend,
    EmailBackend,
    NotifyLevel,
    NotifyResult,
    WebhookBackend,
    available_backends,
    get_backend,
)


class TestNotifyLevel:
    """Tests for NotifyLevel enum."""

    def test_level_values(self):
        assert NotifyLevel.INFO.value == "info"
        assert NotifyLevel.WARNING.value == "warning"
        assert NotifyLevel.ERROR.value == "error"
        assert NotifyLevel.CRITICAL.value == "critical"

    def test_level_from_string(self):
        assert NotifyLevel("info") == NotifyLevel.INFO
        assert NotifyLevel("critical") == NotifyLevel.CRITICAL


class TestNotifyResult:
    """Tests for NotifyResult dataclass."""

    def test_result_creation(self):
        result = NotifyResult(
            success=True,
            backend="audio",
            message="test",
            timestamp="2026-01-13T00:00:00",
        )
        assert result.success is True
        assert result.backend == "audio"
        assert result.message == "test"
        assert result.error is None

    def test_result_with_error(self):
        result = NotifyResult(
            success=False,
            backend="email",
            message="test",
            timestamp="",
            error="Connection failed",
        )
        assert result.success is False
        assert result.error == "Connection failed"


class TestBackendRegistry:
    """Tests for backend registry."""

    def test_backends_dict_exists(self):
        assert "audio" in BACKENDS
        assert "email" in BACKENDS
        assert "desktop" in BACKENDS
        assert "webhook" in BACKENDS

    def test_get_backend_audio(self):
        backend = get_backend("audio")
        assert isinstance(backend, AudioBackend)
        assert backend.name == "audio"

    def test_get_backend_email(self):
        backend = get_backend("email")
        assert isinstance(backend, EmailBackend)
        assert backend.name == "email"

    def test_get_backend_invalid(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("invalid_backend")

    def test_available_backends_returns_list(self):
        backends = available_backends()
        assert isinstance(backends, list)
        # At minimum, audio should be available
        assert "audio" in backends


class TestAudioBackend:
    """Tests for AudioBackend."""

    def test_init_defaults(self):
        backend = AudioBackend()
        assert backend.tts_backend == "gtts"
        assert backend.speed == 1.5
        assert backend.rate == 180

    def test_init_custom(self):
        backend = AudioBackend(backend="elevenlabs", speed=2.0, rate=200)
        assert backend.tts_backend == "elevenlabs"
        assert backend.speed == 2.0
        assert backend.rate == 200

    def test_is_available(self):
        backend = AudioBackend()
        # Should return True if scitex.audio is installed
        result = backend.is_available()
        assert isinstance(result, bool)


class TestEmailBackend:
    """Tests for EmailBackend."""

    def test_init_defaults(self):
        backend = EmailBackend()
        assert backend.name == "email"

    def test_is_available_without_env(self, monkeypatch):
        # Delete all email address env vars
        monkeypatch.delenv("SCITEX_SCHOLAR_EMAIL_NOREPLY", raising=False)
        monkeypatch.delenv("SCITEX_SCHOLAR_FROM_EMAIL_ADDRESS", raising=False)
        monkeypatch.delenv("SCITEX_EMAIL_NOREPLY", raising=False)
        monkeypatch.delenv("SCITEX_EMAIL_AGENT", raising=False)
        # Delete all password env vars
        monkeypatch.delenv("SCITEX_SCHOLAR_EMAIL_PASSWORD", raising=False)
        monkeypatch.delenv("SCITEX_SCHOLAR_FROM_EMAIL_PASSWORD", raising=False)
        monkeypatch.delenv("SCITEX_EMAIL_PASSWORD", raising=False)
        backend = EmailBackend()
        assert backend.is_available() is False


class TestDesktopBackend:
    """Tests for DesktopBackend."""

    def test_init(self):
        backend = DesktopBackend()
        assert backend.name == "desktop"

    def test_is_available(self):
        backend = DesktopBackend()
        result = backend.is_available()
        assert isinstance(result, bool)


class TestWebhookBackend:
    """Tests for WebhookBackend."""

    def test_init_no_url(self, monkeypatch):
        monkeypatch.delenv("SCITEX_NOTIFY_WEBHOOK_URL", raising=False)
        backend = WebhookBackend()
        assert backend.url is None

    def test_is_available_without_url(self, monkeypatch):
        monkeypatch.delenv("SCITEX_NOTIFY_WEBHOOK_URL", raising=False)
        backend = WebhookBackend()
        assert backend.is_available() is False

    def test_is_available_with_url(self):
        backend = WebhookBackend(url="https://hooks.example.com/test")
        assert backend.is_available() is True


@pytest.mark.asyncio
class TestAsyncSend:
    """Async tests for backend send methods."""

    async def test_audio_send_returns_result(self):
        backend = AudioBackend()
        if backend.is_available():
            result = await backend.send("Test message", level=NotifyLevel.INFO)
            assert isinstance(result, NotifyResult)
            assert result.backend == "audio"

    async def test_webhook_send_without_url_fails(self):
        backend = WebhookBackend(url=None)
        result = await backend.send("Test", level=NotifyLevel.INFO)
        assert result.success is False
        assert "No webhook URL" in result.error

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ui/_backends.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2026-01-13 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/ui/_backends.py
# 
# """Re-export from _backends package for backward compatibility."""
# 
# from ._backends import (
#     BACKENDS,
#     AudioBackend,
#     BaseNotifyBackend,
#     DesktopBackend,
#     EmailBackend,
#     MatplotlibBackend,
#     NotifyLevel,
#     NotifyResult,
#     PlaywrightBackend,
#     WebhookBackend,
#     available_backends,
#     get_backend,
# )
# 
# __all__ = [
#     "NotifyLevel",
#     "NotifyResult",
#     "BaseNotifyBackend",
#     "AudioBackend",
#     "EmailBackend",
#     "DesktopBackend",
#     "WebhookBackend",
#     "MatplotlibBackend",
#     "PlaywrightBackend",
#     "BACKENDS",
#     "get_backend",
#     "available_backends",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ui/_backends.py
# --------------------------------------------------------------------------------
