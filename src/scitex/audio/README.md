# SciTeX Audio

Text-to-Speech with automatic fallback: pyttsx3 -> gtts -> elevenlabs

## Usage

```python
import scitex

# Basic
scitex.audio.speak("Hello!")

# Faster speech (rate in words per minute)
scitex.audio.speak("Hello!", rate=200)

# Specific backend
scitex.audio.speak("Hello", backend="pyttsx3")

# Stop speech
scitex.audio.stop_speech()
```

## CLI

```bash
scitex audio speak "Hello world"
scitex audio speak "Bonjour" --backend gtts --voice fr
scitex audio backends       # List available backends
scitex audio check          # Check audio status (WSL)
```

## MCP Server

### Local (stdio - Claude Desktop)

Add to `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "scitex-audio": {
      "command": "python",
      "args": ["-m", "scitex.audio"]
    }
  }
}
```

### Remote Audio (SSH tunnel)

Enable remote agents to play audio on local speakers.

**Architecture:**
```
┌─────────────────────────┐              ┌─────────────────────────┐
│  Remote (e.g., NAS)     │              │  Local (WSL/Windows)    │
│                         │              │                         │
│  Claude Agent ──────────┼─ SSH ───────▶│  scitex audio serve     │
│  connects to            │ RemoteForward│  -t sse --port 8084     │
│  localhost:8084         │              │         │               │
│                         │              │         ▼               │
│                         │              │     🔊 Speakers         │
└─────────────────────────┘              └─────────────────────────┘
```

**Step 1: Local machine - Start audio server**
```bash
scitex audio serve -t sse --port 8084
```

**Step 2: SSH config - Add RemoteForward**

In `~/.ssh/config` (on local machine):
```
Host nas
  HostName 192.168.x.x
  User youruser
  RemoteForward 8084 127.0.0.1:8084  # Audio: remote -> local speakers
```

**Step 3: Remote machine - MCP config**

On remote machine, add to `~/.claude/mcp.json`:
```json
{
  "mcpServers": {
    "scitex-audio-remote": {
      "type": "sse",
      "url": "http://localhost:8084/sse"
    }
  }
}
```

**Step 4: Connect via SSH**
```bash
ssh nas  # RemoteForward creates tunnel automatically
```

Now Claude agents on the remote machine can use `mcp__scitex-audio-remote__speak` to play audio on your local speakers.

### Server Transports

| Transport | Command | Use Case |
|-----------|---------|----------|
| stdio | `scitex audio serve` | Claude Desktop (default) |
| sse | `scitex audio serve -t sse --port 8084` | Remote agents via SSH |
| http | `scitex audio serve -t http --port 8084` | HTTP clients |

### Tools

| Tool | Description |
|------|-------------|
| `speak` | Text to speech (supports `rate`, `speed` params) |
| `list_backends` | Show available backends |
| `check_audio_status` | Check WSL audio connectivity |
| `announce_context` | Announce current directory and git branch |

## Backends

| Backend | Cost | Internet | Install |
|---------|------|----------|---------|
| pyttsx3 | Free | No | `pip install pyttsx3` + `apt install espeak-ng` |
| gtts | Free | Yes | `pip install gTTS` |
| elevenlabs | Paid | Yes | `pip install elevenlabs` + API key |

## Cross-Process Locking

The MCP server uses FIFO locking to ensure only one audio plays at a time across all Claude Code sessions. This prevents audio overlap when multiple agents are running.
