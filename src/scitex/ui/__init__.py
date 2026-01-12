#!/usr/bin/env python3
# Timestamp: "2026-01-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/ui/__init__.py

"""SciTeX UI Module - User alerts and feedback.

Usage:
    import scitex

    # Simple alert (default: audio)
    scitex.ui.alert("2FA required!")

    # Specify backend
    scitex.ui.alert("Error", backend="email")

    # Multiple backends
    scitex.ui.alert("Critical", backend=["audio", "email"])

Environment Variables:
    SCITEX_UI_DEFAULT_BACKEND: audio, email, desktop, webhook
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional, Union

from ._backends import NotifyLevel as _AlertLevel
from ._backends import available_backends as _available_backends
from ._backends import get_backend as _get_backend

__all__ = ["alert", "alert_async", "available_backends"]


def available_backends() -> list[str]:
    """Return list of available alert backends."""
    return _available_backends()


async def alert_async(
    message: str,
    title: Optional[str] = None,
    backend: Optional[Union[str, list[str]]] = None,
    level: str = "info",
    **kwargs,
) -> bool:
    """Send alert asynchronously. Returns True if any backend succeeded."""
    try:
        lvl = _AlertLevel(level.lower())
    except ValueError:
        lvl = _AlertLevel.INFO

    if backend is None:
        backend = os.getenv("SCITEX_UI_DEFAULT_BACKEND", "audio")

    backends = [backend] if isinstance(backend, str) else backend

    success = False
    for name in backends:
        try:
            b = _get_backend(name, **kwargs)
            result = await b.send(message, title=title, level=lvl, **kwargs)
            if result.success:
                success = True
        except Exception:
            pass
    return success


def alert(
    message: str,
    title: Optional[str] = None,
    backend: Optional[Union[str, list[str]]] = None,
    level: str = "info",
    **kwargs,
) -> bool:
    """Send alert synchronously. Returns True if any backend succeeded."""
    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                alert_async(message, title, backend, level, **kwargs),
            )
            return future.result(timeout=30)
    except RuntimeError:
        return asyncio.run(alert_async(message, title, backend, level, **kwargs))


# EOF
