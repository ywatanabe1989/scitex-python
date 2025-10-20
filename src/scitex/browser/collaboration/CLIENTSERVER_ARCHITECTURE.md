# Client-Server Architecture for Browser Collaboration
**Date:** 2025-10-19
**Architecture:** Server (runs browser) ↔ Clients (send commands)

## Concept

```
┌─────────────────────────────────────────────────────┐
│         Browser Server (Long-Running)                │
│  • Keeps browser open                                │
│  • Exposes API (get_info, send_command)             │
│  • Watches for user actions                          │
│  • Executes commands                                 │
└──────────────────┬──────────────────────────────────┘
                   │ API (HTTP/WebSocket)
          ┌────────┴────────┬────────────┐
          ▼                 ▼            ▼
    ┌──────────┐      ┌──────────┐  ┌──────────┐
    │ Client 1 │      │ Client 2 │  │  Human   │
    │ (Claude) │      │  (GPT-4) │  │ (Panel)  │
    └──────────┘      └──────────┘  └──────────┘
```

## API Design

### Server Endpoints

```python
# GET /info
{
    "url": "http://127.0.0.1:8000/",
    "title": "SciTeX Cloud",
    "logged_in": true,
    "panel_data": {"email": "ywata1989@gmail.com"},
    "running": true,
}

# POST /command
{
    "action": "navigate",
    "params": {"url": "http://..."}
}

{
    "action": "click",
    "params": {"selector": "button"}
}

{
    "action": "type",
    "params": {"selector": "#input", "text": "hello"}
}

{
    "action": "screenshot",
    "params": {"message": "test"}
}
```

## Implementation

### Server (FastAPI)

```python
from fastapi import FastAPI
from scitex.browser.collaboration import SharedBrowserSession
import uvicorn

app = FastAPI()

# Global browser session
browser_session = None

@app.on_event("startup")
async def startup():
    global browser_session
    browser_session = SharedBrowserSession()
    await browser_session.start()

@app.get("/info")
async def get_info():
    """Get current browser state."""
    return browser_session.get_info()

@app.post("/command")
async def send_command(command: dict):
    """Execute command in browser."""
    action = command["action"]
    params = command.get("params", {})

    if action == "navigate":
        await browser_session.navigate(**params)
    elif action == "click":
        await browser_session.click(**params)
    elif action == "type":
        await browser_session.type(**params)
    elif action == "screenshot":
        path = await browser_session.screenshot(**params)
        return {"screenshot": path}

    return {"success": True}

@app.post("/shutdown")
async def shutdown():
    await browser_session.close()
    return {"success": True}

# Run server
# uvicorn server:app --port 8001
```

### Client (Simple HTTP)

```python
import requests

# Get info
info = requests.get("http://localhost:8001/info").json()
print(info)

# Send command
requests.post("http://localhost:8001/command", json={
    "action": "navigate",
    "params": {"url": "http://127.0.0.1:8000"}
})

# Screenshot
result = requests.post("http://localhost:8001/command", json={
    "action": "screenshot",
    "params": {"message": "test"}
})
```

## Simpler: Message Queue Pattern

For now, use **file-based messaging** (simpler than HTTP):

```python
# Server watches: ~/.scitex/browser/commands/
# Clients write: ~/.scitex/browser/commands/command_001.json

# Command file
{
    "id": "001",
    "action": "navigate",
    "params": {"url": "..."},
    "timestamp": 1234567890,
}

# Response file
{
    "id": "001",
    "success": true,
    "result": {...},
}
```

---

Should I implement:
**A) FastAPI server** (proper HTTP API)
**B) File-based queue** (simpler, no dependencies)
**C) WebSocket** (real-time bidirectional)

What fits better for your workflow?
