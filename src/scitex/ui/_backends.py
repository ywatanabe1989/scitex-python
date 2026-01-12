#!/usr/bin/env python3
# Timestamp: "2026-01-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/ui/_backends.py

"""Notification backend implementations.

Supports: audio, email, desktop, webhook.
SMS and call backends can be added via external services.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class NotifyLevel(Enum):
    """Notification urgency levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class NotifyResult:
    """Result of a notification attempt."""

    success: bool
    backend: str
    message: str
    timestamp: str
    error: Optional[str] = None
    details: Optional[dict] = None


class BaseNotifyBackend(ABC):
    """Base class for notification backends."""

    name: str = "base"

    @abstractmethod
    async def send(
        self,
        message: str,
        title: Optional[str] = None,
        level: NotifyLevel = NotifyLevel.INFO,
        **kwargs,
    ) -> NotifyResult:
        """Send a notification."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass


class AudioBackend(BaseNotifyBackend):
    """Audio notification via scitex.audio TTS."""

    name = "audio"

    def __init__(
        self,
        backend: str = "gtts",
        speed: float = 1.5,
        rate: int = 180,
    ):
        self.tts_backend = backend
        self.speed = speed
        self.rate = rate

    def is_available(self) -> bool:
        try:
            from scitex.audio import available_backends

            return len(available_backends()) > 0
        except ImportError:
            return False

    async def send(
        self,
        message: str,
        title: Optional[str] = None,
        level: NotifyLevel = NotifyLevel.INFO,
        **kwargs,
    ) -> NotifyResult:
        try:
            from scitex.audio import speak

            # Prepend title if provided
            full_message = f"{title}. {message}" if title else message

            # Add urgency prefix for critical/error levels
            if level == NotifyLevel.CRITICAL:
                full_message = f"Critical alert! {full_message}"
            elif level == NotifyLevel.ERROR:
                full_message = f"Error. {full_message}"

            # Run TTS in executor to not block
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: speak(
                    full_message,
                    backend=kwargs.get("tts_backend", self.tts_backend),
                    speed=kwargs.get("speed", self.speed),
                    rate=kwargs.get("rate", self.rate),
                ),
            )

            return NotifyResult(
                success=True,
                backend=self.name,
                message=message,
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            return NotifyResult(
                success=False,
                backend=self.name,
                message=message,
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )


class EmailBackend(BaseNotifyBackend):
    """Email notification via scitex.utils._email."""

    name = "email"

    def __init__(
        self,
        recipient: Optional[str] = None,
        sender: Optional[str] = None,
    ):
        self.recipient = recipient or os.getenv("SCITEX_NOTIFY_EMAIL_TO")
        self.sender = sender or os.getenv("SCITEX_NOTIFY_EMAIL_FROM")

    def is_available(self) -> bool:
        return bool(
            os.getenv("SCITEX_SCHOLAR_EMAIL_NOREPLY")  # New name
            or os.getenv("SCITEX_SCHOLAR_FROM_EMAIL_ADDRESS")  # Deprecated
            or os.getenv("SCITEX_EMAIL_AGENT")  # Fallback
        ) and bool(
            os.getenv("SCITEX_SCHOLAR_EMAIL_PASSWORD")  # New name
            or os.getenv("SCITEX_SCHOLAR_FROM_EMAIL_PASSWORD")  # Deprecated
            or os.getenv("SCITEX_EMAIL_PASSWORD")  # Fallback
        )

    async def send(
        self,
        message: str,
        title: Optional[str] = None,
        level: NotifyLevel = NotifyLevel.INFO,
        **kwargs,
    ) -> NotifyResult:
        try:
            from scitex.utils._notify import notify as email_notify

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: email_notify(
                    subject=title or f"[SciTeX] {level.value.upper()}",
                    message=message,
                    recipient_email=kwargs.get("recipient", self.recipient),
                ),
            )

            return NotifyResult(
                success=True,
                backend=self.name,
                message=message,
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            return NotifyResult(
                success=False,
                backend=self.name,
                message=message,
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )


class DesktopBackend(BaseNotifyBackend):
    """Desktop notification via native OS APIs.

    Supports:
    - Linux: notify-send
    - WSL/Windows: PowerShell toast notifications
    """

    name = "desktop"

    def _is_wsl(self) -> bool:
        """Check if running in WSL."""
        try:
            with open("/proc/version") as f:
                return "microsoft" in f.read().lower()
        except Exception:
            return False

    def _has_powershell(self) -> bool:
        """Check if PowerShell is available."""
        import shutil

        return shutil.which("powershell.exe") is not None

    def is_available(self) -> bool:
        import shutil

        # Check for notify-send (Linux)
        if shutil.which("notify-send") is not None:
            return True
        # Check for PowerShell (WSL/Windows)
        if self._is_wsl() and self._has_powershell():
            return True
        return False

    async def send(
        self,
        message: str,
        title: Optional[str] = None,
        level: NotifyLevel = NotifyLevel.INFO,
        **kwargs,
    ) -> NotifyResult:
        import shutil

        try:
            title = title or "SciTeX"

            # Use PowerShell for WSL
            if self._is_wsl() and self._has_powershell():
                return await self._send_windows_toast(message, title, level)

            # Use notify-send for Linux
            if shutil.which("notify-send"):
                return await self._send_notify_send(message, title, level)

            return NotifyResult(
                success=False,
                backend=self.name,
                message=message,
                timestamp=datetime.now().isoformat(),
                error="No desktop notification method available",
            )
        except Exception as e:
            return NotifyResult(
                success=False,
                backend=self.name,
                message=message,
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )

    async def _send_notify_send(
        self, message: str, title: str, level: NotifyLevel
    ) -> NotifyResult:
        """Send notification via notify-send (Linux)."""
        import subprocess

        urgency_map = {
            NotifyLevel.INFO: "normal",
            NotifyLevel.WARNING: "normal",
            NotifyLevel.ERROR: "critical",
            NotifyLevel.CRITICAL: "critical",
        }

        cmd = [
            "notify-send",
            "-u",
            urgency_map.get(level, "normal"),
            title,
            message,
        ]

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, timeout=5),
        )

        return NotifyResult(
            success=True,
            backend=self.name,
            message=message,
            timestamp=datetime.now().isoformat(),
        )

    async def _send_windows_toast(
        self, message: str, title: str, level: NotifyLevel
    ) -> NotifyResult:
        """Send Windows toast notification via PowerShell."""
        import subprocess
        import tempfile

        # Escape for XML
        title_escaped = (
            title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        message_escaped = (
            message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

        # Write PowerShell script to temp file to avoid escaping issues
        ps_script = f"""[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$template = '<toast><visual><binding template="ToastGeneric"><text>{title_escaped}</text><text>{message_escaped}</text></binding></visual></toast>'
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml($template)
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("SciTeX").Show($toast)
"""

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ps1", delete=False) as f:
            f.write(ps_script)
            ps_file = f.name

        try:
            # Use Popen to avoid blocking - PowerShell can hang on subprocess.run
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: subprocess.Popen(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        ps_file,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                ),
            )

            # Give PowerShell time to read the file before cleanup
            await asyncio.sleep(0.5)

            return NotifyResult(
                success=True,
                backend=self.name,
                message=message,
                timestamp=datetime.now().isoformat(),
                details={"method": "windows_toast"},
            )
        finally:
            # Clean up temp file after delay
            import os as _os

            await asyncio.sleep(1)
            try:
                _os.unlink(ps_file)
            except Exception:
                pass


class WebhookBackend(BaseNotifyBackend):
    """Webhook notification for Slack, Discord, etc."""

    name = "webhook"

    def __init__(self, url: Optional[str] = None):
        self.url = url or os.getenv("SCITEX_NOTIFY_WEBHOOK_URL")

    def is_available(self) -> bool:
        return bool(self.url)

    async def send(
        self,
        message: str,
        title: Optional[str] = None,
        level: NotifyLevel = NotifyLevel.INFO,
        **kwargs,
    ) -> NotifyResult:
        import json
        import urllib.request

        try:
            url = kwargs.get("url", self.url)
            if not url:
                raise ValueError("No webhook URL configured")

            # Format for Slack/Discord compatibility
            payload = {
                "text": f"*{title or 'SciTeX'}*\n{message}",
                "content": f"**{title or 'SciTeX'}**\n{message}",
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=10),
            )

            return NotifyResult(
                success=True,
                backend=self.name,
                message=message,
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            return NotifyResult(
                success=False,
                backend=self.name,
                message=message,
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )


# Registry of available backends
BACKENDS: dict[str, type[BaseNotifyBackend]] = {
    "audio": AudioBackend,
    "email": EmailBackend,
    "desktop": DesktopBackend,
    "webhook": WebhookBackend,
}


def get_backend(name: str, **kwargs) -> BaseNotifyBackend:
    """Get a notification backend by name."""
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(BACKENDS.keys())}")
    return BACKENDS[name](**kwargs)


def available_backends() -> list[str]:
    """Return list of available notification backends."""
    available = []
    for name, cls in BACKENDS.items():
        try:
            backend = cls()
            if backend.is_available():
                available.append(name)
        except Exception:
            pass
    return available


# EOF
