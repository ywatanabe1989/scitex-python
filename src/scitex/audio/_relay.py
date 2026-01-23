#!/usr/bin/env python3
"""Audio relay client for remote TTS playback.

Forwards speak requests to a remote audio server via HTTP.
This allows agents running on remote machines (e.g., NAS) to play
audio through the user's local speakers.

Usage:
    from scitex.audio._relay import RelayClient

    # Connect to local relay server
    client = RelayClient("http://localhost:31293")

    # Speak through relay
    client.speak("Hello from remote!")

Environment Variables:
    SCITEX_AUDIO_RELAY_URL: Relay server URL
    SCITEX_AUDIO_RELAY_HOST: Relay server host (builds URL with port)
    SCITEX_AUDIO_RELAY_PORT: Relay server port (default: 31293)
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from ._branding import DEFAULT_PORT, get_relay_url


class RelayClient:
    """HTTP client for audio relay server.

    Forwards TTS requests to a remote server running the audio MCP
    in HTTP mode. Enables remote agents to play audio locally.

    Example:
        >>> client = RelayClient("http://localhost:31293")
        >>> client.speak("Hello!")
        {'success': True, 'text': 'Hello!', ...}
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize relay client.

        Args:
            base_url: Relay server URL. Auto-detects from env if None.
            timeout: Request timeout in seconds.
        """
        self.base_url = (
            base_url or get_relay_url() or f"http://localhost:{DEFAULT_PORT}"
        ).rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        method: str = "POST",
    ) -> Dict[str, Any]:
        """Make HTTP request to relay server."""
        url = f"{self.base_url}{endpoint}"

        try:
            if data is not None:
                req_data = json.dumps(data).encode("utf-8")
                req = urllib.request.Request(url, data=req_data, method=method)
                req.add_header("Content-Type", "application/json")
            else:
                req = urllib.request.Request(url, method=method)

            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            raise ConnectionError(f"Relay request failed: {e.code} {e.reason}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot connect to relay at {self.base_url}: {e.reason}"
            ) from e

    def health(self) -> Dict[str, Any]:
        """Check relay server health."""
        try:
            # Try MCP health endpoint
            return self._request("/health", method="GET")
        except Exception:
            # Relay may not have /health, try simple request
            return {"status": "unknown", "url": self.base_url}

    def is_available(self) -> bool:
        """Check if relay server is reachable."""
        try:
            self.health()
            return True
        except Exception:
            return False

    def speak(
        self,
        text: str,
        backend: Optional[str] = None,
        voice: Optional[str] = None,
        rate: int = 150,
        speed: float = 1.5,
        play: bool = True,
        save: bool = False,
        fallback: bool = True,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forward speak request to relay server.

        Args:
            text: Text to speak
            backend: TTS backend (auto if None)
            voice: Voice/language
            rate: Speech rate (pyttsx3)
            speed: Speed multiplier (gtts)
            play: Play audio
            save: Save audio file
            fallback: Try fallback backends
            agent_id: Agent identifier

        Returns
        -------
            Response dict with success status
        """
        data = {
            "text": text,
            "backend": backend,
            "voice": voice,
            "rate": rate,
            "speed": speed,
            "play": play,
            "save": save,
            "fallback": fallback,
            "agent_id": agent_id,
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        return self._request("/speak", data)

    def list_backends(self) -> Dict[str, Any]:
        """Get available backends from relay server."""
        return self._request("/list_backends", method="GET")


# Module-level client singleton
_client: Optional[RelayClient] = None


def get_relay_client(base_url: Optional[str] = None) -> RelayClient:
    """Get or create relay client singleton."""
    global _client
    if _client is None or (base_url and _client.base_url != base_url):
        _client = RelayClient(base_url)
    return _client


def reset_relay_client() -> None:
    """Reset relay client singleton."""
    global _client
    _client = None


def relay_speak(
    text: str,
    backend: Optional[str] = None,
    voice: Optional[str] = None,
    rate: int = 150,
    speed: float = 1.5,
    play: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function to speak via relay.

    Args:
        text: Text to speak
        backend: TTS backend
        voice: Voice/language
        rate: Speech rate
        speed: Speed multiplier
        play: Play audio
        **kwargs: Additional options

    Returns
    -------
        Response dict
    """
    client = get_relay_client()
    return client.speak(
        text=text,
        backend=backend,
        voice=voice,
        rate=rate,
        speed=speed,
        play=play,
        **kwargs,
    )


def is_relay_available() -> bool:
    """Check if relay server is available."""
    try:
        client = get_relay_client()
        return client.is_available()
    except Exception:
        return False


__all__ = [
    "RelayClient",
    "get_relay_client",
    "reset_relay_client",
    "relay_speak",
    "is_relay_available",
]
