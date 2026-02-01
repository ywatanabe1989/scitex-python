# SciTeX Audio

Text-to-Speech with automatic fallback and smart routing.

**Fallback order:** elevenlabs -> gtts -> pyttsx3

## Smart Routing

The `speak()` function automatically routes audio based on availability:

| Local Sink | Relay Available | Result (mode=auto) |
|------------|-----------------|-------------------|
| SUSPENDED | Yes | Uses relay |
| SUSPENDED | No | Returns error |
| RUNNING | Yes | Prefers relay |
| RUNNING | No | Uses local |

## Usage

```python
import scitex

# Basic - auto mode (smart routing)
result = scitex.audio.speak("Hello!")
if result["played"]:
    print(f"Played via {result['mode']}")

# Force local playback (fails if sink unavailable)
result = scitex.audio.speak("Hello!", mode="local")

# Force remote relay
result = scitex.audio.speak("Hello!", mode="remote")

# Faster speech (speed multiplier for gtts)
scitex.audio.speak("Hello!", speed=1.5)

# Specific backend
scitex.audio.speak("Hello", backend="gtts")

# Check local audio availability
status = scitex.audio.check_local_audio_available()
print(status)  # {'available': False, 'state': 'SUSPENDED', 'reason': '...'}

# Stop speech
scitex.audio.stop_speech()
```

## CLI

```bash
scitex audio speak "Hello world"
scitex audio speak "Bonjour" --backend gtts --voice fr
scitex audio backends       # List available backends
scitex audio check          # Check audio status (WSL)
scitex audio relay          # Start HTTP relay server (for remote audio)
scitex audio serve          # Start MCP server
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

### Remote Audio (HTTP Relay)

Enable remote agents to play audio on local speakers using a simple HTTP relay.

**Architecture:**
```
┌─────────────────────────┐              ┌─────────────────────────┐
│  Remote (e.g., NAS)     │              │  Local (WSL/Windows)    │
│                         │              │                         │
│  Claude Agent uses      │              │  scitex audio relay     │
│  audio_speak ───────────┼─ SSH ───────▶│  --port 31293           │
│  (auto-routes to relay) │ Reverse      │         │               │
│  localhost:31293        │ Tunnel       │         ▼               │
│                         │              │     🔊 Speakers         │
└─────────────────────────┘              └─────────────────────────┘
```

**Step 1: Local machine - Start relay server**
```bash
scitex audio relay --port 31293
```

**Step 2: SSH with reverse tunnel**
```bash
ssh -R 31293:localhost:31293 remote-server
```

Or add to `~/.ssh/config`:
```
Host nas
  HostName 192.168.x.x
  User youruser
  RemoteForward 31293 127.0.0.1:31293
```

**Step 3: Remote agent uses relay**

The unified `audio_speak` MCP tool auto-detects relay:
1. `SCITEX_AUDIO_RELAY_URL` env var
2. Localhost:31293 (SSH reverse tunnel)
3. SSH_CLIENT IP (auto-detected from SSH session)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCITEX_AUDIO_PORT` | 31293 | Server/relay port |
| `SCITEX_AUDIO_MODE` | auto | `local`, `remote`, or `auto` |
| `SCITEX_AUDIO_RELAY_URL` | (auto) | Full relay URL |
| `SCITEX_AUDIO_RELAY_HOST` | (none) | Relay host |
| `SCITEX_AUDIO_RELAY_PORT` | 31293 | Relay port |

### Server Transports

| Transport | Command | Use Case |
|-----------|---------|----------|
| stdio | `scitex audio serve` | Claude Desktop (default) |
| sse | `scitex audio serve -t sse --port 31293` | Remote MCP agents |
| http | `scitex audio serve -t http --port 31293` | HTTP MCP clients |
| relay | `scitex audio relay --port 31293` | Simple HTTP relay |

### MCP Tools

| Tool | Description |
|------|-------------|
| `audio_speak` | **Unified TTS with smart routing** (auto-selects local/relay) |
| `audio_list_backends` | Show available backends |
| `audio_check_audio_status` | Check WSL audio connectivity |
| `audio_announce_context` | Announce current directory and git branch |

The `audio_speak` tool automatically:
- Checks if local audio sink is available (not SUSPENDED)
- Uses relay server when local audio unavailable
- Returns clear error messages when neither is available

## Backends

| Backend | Cost | Internet | Install |
|---------|------|----------|---------|
| pyttsx3 | Free | No | `pip install pyttsx3` + `apt install espeak-ng` |
| gtts | Free | Yes | `pip install gTTS` |
| elevenlabs | Paid | Yes | `pip install elevenlabs` + API key |

## Cross-Process Locking

The relay server uses FIFO locking to ensure only one audio plays at a time across all Claude Code sessions. This prevents audio overlap when multiple agents are running.
