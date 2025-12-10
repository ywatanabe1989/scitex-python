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

## MCP Server

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "scitex-audio": {
      "command": "/path/to/python",
      "args": ["-m", "scitex.audio", "--mcp"]
    }
  }
}
```

### Tools

| Tool | Description |
|------|-------------|
| `speak` | Text to speech (supports `rate` param for speed) |
| `generate_audio` | Save audio to file |
| `list_backends` | Show available backends |

## Backends

| Backend | Cost | Internet | Install |
|---------|------|----------|---------|
| pyttsx3 | Free | No | `pip install pyttsx3` + `apt install espeak-ng` |
| gtts | Free | Yes | `pip install gTTS` |
| elevenlabs | Paid | Yes | `pip install elevenlabs` + API key |
