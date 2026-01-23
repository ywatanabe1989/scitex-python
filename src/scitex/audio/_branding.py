#!/usr/bin/env python3
"""Configuration constants for scitex.audio.

Provides environment variable helpers and default values for the audio module.
"""

import os
from typing import Optional

# Fixed branding (scitex.audio is part of scitex, not rebranded)
BRAND_NAME = "scitex.audio"
BRAND_ALIAS = "audio"
ENV_PREFIX = "SCITEX_AUDIO"

# Default port: 31293 (SCITEX port scheme: 3129X where X=3 for audio)
DEFAULT_PORT = 31293
DEFAULT_HOST = "0.0.0.0"


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with SCITEX_AUDIO_ prefix.

    Args:
        key: Variable name without prefix (e.g., "PORT", "HOST", "MODE")
        default: Default value if not found

    Returns
    -------
        Environment variable value or default

    Examples
    --------
        get_env("PORT")  # Checks SCITEX_AUDIO_PORT
        get_env("MODE", "local")  # With default
    """
    return os.environ.get(f"SCITEX_AUDIO_{key}", default)


def get_port() -> int:
    """Get configured port number."""
    port_str = get_env("PORT", str(DEFAULT_PORT))
    return int(port_str)


def get_host() -> str:
    """Get configured host."""
    return get_env("HOST", DEFAULT_HOST)


def get_mode() -> str:
    """Get audio mode: 'local', 'remote', or 'auto'.

    - local: Always play locally (direct TTS)
    - remote: Always forward to relay server
    - auto: Try local first, fall back to remote
    """
    return get_env("MODE", "auto").lower()


def get_ssh_client_ip() -> Optional[str]:
    """Get IP address of SSH client if running in SSH session.

    Extracts client IP from SSH_CLIENT or SSH_CONNECTION env vars.
    Returns None if not in SSH session.
    """
    # SSH_CLIENT format: "client_ip client_port server_port"
    ssh_client = os.environ.get("SSH_CLIENT", "")
    if ssh_client:
        parts = ssh_client.split()
        if parts:
            return parts[0]

    # SSH_CONNECTION format: "client_ip client_port server_ip server_port"
    ssh_connection = os.environ.get("SSH_CONNECTION", "")
    if ssh_connection:
        parts = ssh_connection.split()
        if parts:
            return parts[0]

    return None


def _check_relay_reachable(url: str, timeout: float = 1.0) -> bool:
    """Quick check if relay URL is reachable."""
    import socket
    import urllib.parse

    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or DEFAULT_PORT
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def get_relay_url() -> Optional[str]:
    """Get relay server URL for remote mode.

    Priority:
    1. SCITEX_AUDIO_RELAY_URL env var
    2. SCITEX_AUDIO_RELAY_HOST env var + port
    3. localhost:31293 (SSH reverse tunnel - if reachable)
    4. Auto-detect from SSH_CLIENT (if in SSH session)

    Returns URL like 'http://192.168.1.100:31293' or None.
    """
    url = get_env("RELAY_URL")
    if url:
        return url.rstrip("/")

    # Build from host/port if relay host is set
    relay_host = get_env("RELAY_HOST")
    if relay_host:
        relay_port = get_env("RELAY_PORT", str(DEFAULT_PORT))
        return f"http://{relay_host}:{relay_port}"

    # Check localhost first (SSH reverse tunnel)
    localhost_url = f"http://localhost:{DEFAULT_PORT}"
    if _check_relay_reachable(localhost_url):
        return localhost_url

    # Auto-detect from SSH client IP
    ssh_client_ip = get_ssh_client_ip()
    if ssh_client_ip:
        return f"http://{ssh_client_ip}:{DEFAULT_PORT}"

    return None


def get_mcp_server_name() -> str:
    """Get the MCP server name."""
    return "scitex-audio"


def get_mcp_instructions() -> str:
    """Get MCP server instructions."""
    return """\
scitex.audio - Text-to-Speech with Multiple Backends

Backends (fallback order): elevenlabs -> gtts -> pyttsx3

## Quick Start
```python
from scitex.audio import speak
speak("Hello, world!")  # Auto-selects backend
speak("Fast", backend="gtts", speed=1.5)
```

## MCP Tools
- **speak**: Convert text to speech
- **list_backends**: Show available TTS backends
- **check_audio_status**: Check WSL audio connectivity
- **announce_context**: Announce current directory and git branch

## Remote Audio Relay
Run locally: `scitex audio serve -t http --port 31293`
Remote agents connect via HTTP to play audio on local speakers.

## Environment Variables
- SCITEX_AUDIO_MODE: 'local', 'remote', or 'auto' (default: auto)
- SCITEX_AUDIO_RELAY_URL: Remote relay server URL
- SCITEX_AUDIO_PORT: Server port (default: 31293)
"""


__all__ = [
    "BRAND_NAME",
    "BRAND_ALIAS",
    "ENV_PREFIX",
    "DEFAULT_PORT",
    "DEFAULT_HOST",
    "get_env",
    "get_port",
    "get_host",
    "get_mode",
    "get_relay_url",
    "get_ssh_client_ip",
    "get_mcp_server_name",
    "get_mcp_instructions",
]
