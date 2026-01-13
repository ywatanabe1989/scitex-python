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
# """Notification backend implementations.
#
# Supports: audio, email, desktop, webhook.
# SMS and call backends can be added via external services.
# """
#
# from __future__ import annotations
#
# import asyncio
# import os
# from abc import ABC, abstractmethod
# from dataclasses import dataclass
# from datetime import datetime
# from enum import Enum
# from typing import Optional
#
#
# class NotifyLevel(Enum):
#     """Notification urgency levels."""
#
#     INFO = "info"
#     WARNING = "warning"
#     ERROR = "error"
#     CRITICAL = "critical"
#
#
# @dataclass
# class NotifyResult:
#     """Result of a notification attempt."""
#
#     success: bool
#     backend: str
#     message: str
#     timestamp: str
#     error: Optional[str] = None
#     details: Optional[dict] = None
#
#
# class BaseNotifyBackend(ABC):
#     """Base class for notification backends."""
#
#     name: str = "base"
#
#     @abstractmethod
#     async def send(
#         self,
#         message: str,
#         title: Optional[str] = None,
#         level: NotifyLevel = NotifyLevel.INFO,
#         **kwargs,
#     ) -> NotifyResult:
#         """Send a notification."""
#         pass
#
#     @abstractmethod
#     def is_available(self) -> bool:
#         """Check if this backend is available."""
#         pass
#
#
# class AudioBackend(BaseNotifyBackend):
#     """Audio notification via scitex.audio TTS."""
#
#     name = "audio"
#
#     def __init__(
#         self,
#         backend: str = "gtts",
#         speed: float = 1.5,
#         rate: int = 180,
#     ):
#         self.tts_backend = backend
#         self.speed = speed
#         self.rate = rate
#
#     def is_available(self) -> bool:
#         try:
#             from scitex.audio import available_backends
#
#             return len(available_backends()) > 0
#         except ImportError:
#             return False
#
#     async def send(
#         self,
#         message: str,
#         title: Optional[str] = None,
#         level: NotifyLevel = NotifyLevel.INFO,
#         **kwargs,
#     ) -> NotifyResult:
#         try:
#             from scitex.audio import speak
#
#             # Prepend title if provided
#             full_message = f"{title}. {message}" if title else message
#
#             # Add urgency prefix for critical/error levels
#             if level == NotifyLevel.CRITICAL:
#                 full_message = f"Critical alert! {full_message}"
#             elif level == NotifyLevel.ERROR:
#                 full_message = f"Error. {full_message}"
#
#             # Run TTS in executor to not block
#             loop = asyncio.get_event_loop()
#             await loop.run_in_executor(
#                 None,
#                 lambda: speak(
#                     full_message,
#                     backend=kwargs.get("tts_backend", self.tts_backend),
#                     speed=kwargs.get("speed", self.speed),
#                     rate=kwargs.get("rate", self.rate),
#                 ),
#             )
#
#             return NotifyResult(
#                 success=True,
#                 backend=self.name,
#                 message=message,
#                 timestamp=datetime.now().isoformat(),
#             )
#         except Exception as e:
#             return NotifyResult(
#                 success=False,
#                 backend=self.name,
#                 message=message,
#                 timestamp=datetime.now().isoformat(),
#                 error=str(e),
#             )
#
#
# class EmailBackend(BaseNotifyBackend):
#     """Email notification via scitex.utils._email."""
#
#     name = "email"
#
#     def __init__(
#         self,
#         recipient: Optional[str] = None,
#         sender: Optional[str] = None,
#     ):
#         self.recipient = recipient or os.getenv("SCITEX_NOTIFY_EMAIL_TO")
#         self.sender = sender or os.getenv("SCITEX_NOTIFY_EMAIL_FROM")
#
#     def is_available(self) -> bool:
#         return bool(
#             os.getenv("SCITEX_SCHOLAR_FROM_EMAIL_ADDRESS")
#             or os.getenv("SCITEX_EMAIL_AGENT")
#         ) and bool(
#             os.getenv("SCITEX_SCHOLAR_FROM_EMAIL_PASSWORD")
#             or os.getenv("SCITEX_EMAIL_PASSWORD")
#         )
#
#     async def send(
#         self,
#         message: str,
#         title: Optional[str] = None,
#         level: NotifyLevel = NotifyLevel.INFO,
#         **kwargs,
#     ) -> NotifyResult:
#         try:
#             from scitex.utils._notify import notify as email_notify
#
#             loop = asyncio.get_event_loop()
#             await loop.run_in_executor(
#                 None,
#                 lambda: email_notify(
#                     subject=title or f"[SciTeX] {level.value.upper()}",
#                     message=message,
#                     recipient_email=kwargs.get("recipient", self.recipient),
#                 ),
#             )
#
#             return NotifyResult(
#                 success=True,
#                 backend=self.name,
#                 message=message,
#                 timestamp=datetime.now().isoformat(),
#             )
#         except Exception as e:
#             return NotifyResult(
#                 success=False,
#                 backend=self.name,
#                 message=message,
#                 timestamp=datetime.now().isoformat(),
#                 error=str(e),
#             )
#
#
# class DesktopBackend(BaseNotifyBackend):
#     """Desktop notification via native OS APIs."""
#
#     name = "desktop"
#
#     def is_available(self) -> bool:
#         # Check for notify-send (Linux) or other notification tools
#         import shutil
#
#         return shutil.which("notify-send") is not None
#
#     async def send(
#         self,
#         message: str,
#         title: Optional[str] = None,
#         level: NotifyLevel = NotifyLevel.INFO,
#         **kwargs,
#     ) -> NotifyResult:
#         import subprocess
#
#         try:
#             urgency_map = {
#                 NotifyLevel.INFO: "normal",
#                 NotifyLevel.WARNING: "normal",
#                 NotifyLevel.ERROR: "critical",
#                 NotifyLevel.CRITICAL: "critical",
#             }
#
#             cmd = [
#                 "notify-send",
#                 "-u",
#                 urgency_map.get(level, "normal"),
#                 title or "SciTeX",
#                 message,
#             ]
#
#             loop = asyncio.get_event_loop()
#             await loop.run_in_executor(
#                 None,
#                 lambda: subprocess.run(cmd, capture_output=True, timeout=5),
#             )
#
#             return NotifyResult(
#                 success=True,
#                 backend=self.name,
#                 message=message,
#                 timestamp=datetime.now().isoformat(),
#             )
#         except Exception as e:
#             return NotifyResult(
#                 success=False,
#                 backend=self.name,
#                 message=message,
#                 timestamp=datetime.now().isoformat(),
#                 error=str(e),
#             )
#
#
# class WebhookBackend(BaseNotifyBackend):
#     """Webhook notification for Slack, Discord, etc."""
#
#     name = "webhook"
#
#     def __init__(self, url: Optional[str] = None):
#         self.url = url or os.getenv("SCITEX_NOTIFY_WEBHOOK_URL")
#
#     def is_available(self) -> bool:
#         return bool(self.url)
#
#     async def send(
#         self,
#         message: str,
#         title: Optional[str] = None,
#         level: NotifyLevel = NotifyLevel.INFO,
#         **kwargs,
#     ) -> NotifyResult:
#         import json
#         import urllib.request
#
#         try:
#             url = kwargs.get("url", self.url)
#             if not url:
#                 raise ValueError("No webhook URL configured")
#
#             # Format for Slack/Discord compatibility
#             payload = {
#                 "text": f"*{title or 'SciTeX'}*\n{message}",
#                 "content": f"**{title or 'SciTeX'}**\n{message}",
#             }
#
#             data = json.dumps(payload).encode("utf-8")
#             req = urllib.request.Request(
#                 url,
#                 data=data,
#                 headers={"Content-Type": "application/json"},
#             )
#
#             loop = asyncio.get_event_loop()
#             await loop.run_in_executor(
#                 None,
#                 lambda: urllib.request.urlopen(req, timeout=10),
#             )
#
#             return NotifyResult(
#                 success=True,
#                 backend=self.name,
#                 message=message,
#                 timestamp=datetime.now().isoformat(),
#             )
#         except Exception as e:
#             return NotifyResult(
#                 success=False,
#                 backend=self.name,
#                 message=message,
#                 timestamp=datetime.now().isoformat(),
#                 error=str(e),
#             )
#
#
# # Registry of available backends
# BACKENDS: dict[str, type[BaseNotifyBackend]] = {
#     "audio": AudioBackend,
#     "email": EmailBackend,
#     "desktop": DesktopBackend,
#     "webhook": WebhookBackend,
# }
#
#
# def get_backend(name: str, **kwargs) -> BaseNotifyBackend:
#     """Get a notification backend by name."""
#     if name not in BACKENDS:
#         raise ValueError(f"Unknown backend: {name}. Available: {list(BACKENDS.keys())}")
#     return BACKENDS[name](**kwargs)
#
#
# def available_backends() -> list[str]:
#     """Return list of available notification backends."""
#     available = []
#     for name, cls in BACKENDS.items():
#         try:
#             backend = cls()
#             if backend.is_available():
#                 available.append(name)
#         except Exception:
#             pass
#     return available
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ui/_backends.py
# --------------------------------------------------------------------------------
