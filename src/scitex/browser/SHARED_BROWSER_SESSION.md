# Shared Browser Session for AI Agents and Humans
**Date:** 2025-10-19
**Goal:** Real-time shared browser experience between AI agents and human users

## Concept

Instead of starting/stopping browser for each task, keep a **persistent browser session** that:
1. **AI agents** can control programmatically
2. **Human users** can see and interact with visually
3. **Share state** in real-time (cookies, navigation, DOM changes)
4. **Coordinate actions** (agents can see what humans do, vice versa)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│          Shared Browser Session Manager                 │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │ AI Agent 1 │───▶│ Shared       │◀───│ Human User  │ │
│  └────────────┘    │ Browser      │    └─────────────┘ │
│                     │ Instance     │                     │
│  ┌────────────┐    │              │    ┌─────────────┐ │
│  │ AI Agent 2 │───▶│ (Playwright) │◀───│ Remote      │ │
│  └────────────┘    └──────────────┘    │ Debugger    │ │
│                                          └─────────────┘ │
│                                                           │
│  Features:                                                │
│  • Event streaming (page changes, clicks, etc.)          │
│  • Action queue (coordinated control)                     │
│  • State synchronization                                  │
│  • Screenshot streaming                                   │
└─────────────────────────────────────────────────────────┘
```

## Implementation

### 1. SharedBrowserSession Class

```python
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
import asyncio
from typing import Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum

class ActionType(Enum):
    """Types of browser actions."""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    SCREENSHOT = "screenshot"
    EXECUTE_JS = "execute_js"
    WAIT = "wait"

@dataclass
class BrowserAction:
    """Action to be performed in browser."""
    type: ActionType
    params: dict
    source: str  # "ai_agent", "human", "system"
    timestamp: float = field(default_factory=time.time)

@dataclass
class BrowserEvent:
    """Event that occurred in browser."""
    type: str  # "navigation", "click", "dom_change", etc.
    data: dict
    timestamp: float = field(default_factory=time.time)


class SharedBrowserSession:
    """
    Persistent browser session shared between AI agents and humans.

    Features:
    - Long-lived browser instance
    - Event streaming (what's happening)
    - Action queue (coordinated control)
    - Multi-agent coordination
    - Human observation/control

    Usage:
        # Start shared session
        session = SharedBrowserSession()
        await session.start(headless=False)  # Visible for humans

        # AI agent uses session
        await session.navigate("https://example.com")

        # Human can see and interact with same browser
        # Agent can see what human does

        # Keep running
        await session.wait_until_closed()
    """

    def __init__(
        self,
        session_id: str = "default",
        port: int = 9222,  # Chrome DevTools Protocol port
        enable_recording: bool = True,
        screenshot_interval: float = 1.0,  # Screenshots every 1 second
    ):
        self.session_id = session_id
        self.port = port
        self.enable_recording = enable_recording
        self.screenshot_interval = screenshot_interval

        # Browser state
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

        # Coordination
        self.action_queue: asyncio.Queue = asyncio.Queue()
        self.event_subscribers: List[Callable] = []
        self.running = False

        # Recording
        self.events: List[BrowserEvent] = []
        self.screenshots: List[str] = []

    async def start(
        self,
        browser_type: str = "chromium",
        headless: bool = False,  # False so humans can see
        viewport: dict = {"width": 1920, "height": 1080"},
    ):
        """
        Start shared browser session.

        Uses non-headless mode so humans can see and interact.
        """
        self.playwright = await async_playwright().start()

        # Launch browser with remote debugging
        browser_launcher = getattr(self.playwright, browser_type)
        self.browser = await browser_launcher.launch(
            headless=headless,
            args=[
                f"--remote-debugging-port={self.port}",
                "--disable-blink-features=AutomationControlled",
            ],
        )

        # Create persistent context
        self.context = await self.browser.new_context(
            viewport=viewport,
            user_data_dir=f"~/.scitex/browser/sessions/{self.session_id}",
        )

        # Create page
        self.page = await self.context.new_page()

        # Set up event listeners
        await self._setup_event_listeners()

        # Start action processor
        self.running = True
        asyncio.create_task(self._process_actions())

        # Start screenshot recorder
        if self.enable_recording:
            asyncio.create_task(self._record_screenshots())

        print(f"✅ Shared browser session started: {self.session_id}")
        print(f"   Remote debugging: http://localhost:{self.port}")
        print(f"   Humans can connect to: chrome://inspect")

    async def _setup_event_listeners(self):
        """Set up event listeners for the page."""

        # Navigation events
        self.page.on("load", lambda: self._emit_event("page_loaded", {"url": self.page.url}))
        self.page.on("framenavigated", lambda frame: self._emit_event("navigation", {"url": frame.url}))

        # Console messages
        self.page.on("console", lambda msg: self._emit_event("console", {"text": msg.text, "type": msg.type}))

        # Dialog events
        self.page.on("dialog", lambda dialog: self._emit_event("dialog", {"message": dialog.message, "type": dialog.type}))

        # Request/Response
        self.page.on("request", lambda req: self._emit_event("request", {"url": req.url, "method": req.method}))
        self.page.on("response", lambda res: self._emit_event("response", {"url": res.url, "status": res.status}))

    def _emit_event(self, event_type: str, data: dict):
        """Emit event to all subscribers."""
        event = BrowserEvent(type=event_type, data=data)
        self.events.append(event)

        # Notify subscribers
        for subscriber in self.event_subscribers:
            asyncio.create_task(subscriber(event))

    def subscribe(self, callback: Callable):
        """Subscribe to browser events."""
        self.event_subscribers.append(callback)

    async def navigate(self, url: str, wait_until: str = "networkidle"):
        """Navigate to URL (coordinated via action queue)."""
        await self.action_queue.put(BrowserAction(
            type=ActionType.NAVIGATE,
            params={"url": url, "wait_until": wait_until},
            source="ai_agent",
        ))

    async def click(self, selector: str):
        """Click element (coordinated via action queue)."""
        await self.action_queue.put(BrowserAction(
            type=ActionType.CLICK,
            params={"selector": selector},
            source="ai_agent",
        ))

    async def type_text(self, selector: str, text: str):
        """Type text (coordinated via action queue)."""
        await self.action_queue.put(BrowserAction(
            type=ActionType.TYPE,
            params={"selector": selector, "text": text},
            source="ai_agent",
        ))

    async def execute_javascript(self, script: str):
        """Execute JavaScript (coordinated via action queue)."""
        await self.action_queue.put(BrowserAction(
            type=ActionType.EXECUTE_JS,
            params={"script": script},
            source="ai_agent",
        ))

    async def take_screenshot(self) -> str:
        """Take screenshot immediately."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = f"$SCITEX_DIR/capture/{self.session_id}_{timestamp}.jpg"
        await self.page.screenshot(path=os.path.expanduser(path), quality=90)
        return path

    async def _process_actions(self):
        """Process action queue (coordination between agents and humans)."""
        while self.running:
            try:
                action = await asyncio.wait_for(
                    self.action_queue.get(),
                    timeout=0.1
                )

                # Execute action
                if action.type == ActionType.NAVIGATE:
                    await self.page.goto(**action.params)
                elif action.type == ActionType.CLICK:
                    await self.page.click(action.params["selector"])
                elif action.type == ActionType.TYPE:
                    await self.page.fill(
                        action.params["selector"],
                        action.params["text"]
                    )
                elif action.type == ActionType.EXECUTE_JS:
                    await self.page.evaluate(action.params["script"])
                elif action.type == ActionType.SCREENSHOT:
                    await self.take_screenshot()

                # Emit event
                self._emit_event("action_completed", {"action": action.type.value})

            except asyncio.TimeoutError:
                pass  # No actions in queue
            except Exception as e:
                self._emit_event("error", {"error": str(e)})

    async def _record_screenshots(self):
        """Continuously record screenshots for timeline."""
        while self.running:
            try:
                screenshot_path = await self.take_screenshot()
                self.screenshots.append(screenshot_path)
                await asyncio.sleep(self.screenshot_interval)
            except Exception as e:
                print(f"Screenshot error: {e}")

    async def wait_until_closed(self):
        """Wait until session is manually closed."""
        while self.running:
            await asyncio.sleep(1)

    async def close(self):
        """Close shared session."""
        self.running = False

        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

        print(f"✅ Shared browser session closed: {self.session_id}")

    def get_remote_debug_url(self) -> str:
        """Get URL for remote debugging (for humans to connect)."""
        return f"http://localhost:{self.port}"

    def get_timeline(self) -> List[dict]:
        """Get timeline of all events and screenshots."""
        return {
            'events': [
                {'time': e.timestamp, 'type': e.type, 'data': e.data}
                for e in self.events
            ],
            'screenshots': self.screenshots,
        }
```

### 2. Multi-Agent Coordination

```python
class AgentCoordinator:
    """
    Coordinate multiple AI agents using shared browser.

    Prevents conflicts, coordinates actions.
    """

    def __init__(self, session: SharedBrowserSession):
        self.session = session
        self.agents: Dict[str, "AIAgent"] = {}
        self.active_agent: Optional[str] = None
        self.lock = asyncio.Lock()

    def register_agent(self, agent_id: str, agent: "AIAgent"):
        """Register an AI agent."""
        self.agents[agent_id] = agent

    async def request_control(self, agent_id: str) -> bool:
        """Agent requests control of browser."""
        async with self.lock:
            if self.active_agent is None or self.active_agent == agent_id:
                self.active_agent = agent_id
                return True
            return False

    async def release_control(self, agent_id: str):
        """Agent releases control."""
        async with self.lock:
            if self.active_agent == agent_id:
                self.active_agent = None

    @asynccontextmanager
    async def agent_turn(self, agent_id: str):
        """Context manager for agent to take turn."""
        success = await self.request_control(agent_id)
        if not success:
            raise RuntimeError(f"Agent {agent_id} cannot get control")
        try:
            yield
        finally:
            await self.release_control(agent_id)


# Usage
async def main():
    session = SharedBrowserSession()
    await session.start(headless=False)

    coordinator = AgentCoordinator(session)

    # Agent 1: Login
    async with coordinator.agent_turn("agent1"):
        await session.navigate("http://127.0.0.1:8000/auth/login/")
        await session.type_text("#id_username", "user")
        await session.type_text("#id_password", "pass")
        await session.click("button[type='submit']")

    # Agent 2: Create project
    async with coordinator.agent_turn("agent2"):
        await session.navigate("http://127.0.0.1:8000/new/")
        await session.type_text("#id_name", "New Project")
        await session.click("button[type='submit']")

    # Human can see and interact at any time
    await session.wait_until_closed()
```

### 3. Human Observation Interface

```python
class HumanObserver:
    """
    Interface for humans to observe and optionally control browser.

    Provides:
    - Live event stream
    - Screenshot viewer
    - Manual control option
    """

    def __init__(self, session: SharedBrowserSession):
        self.session = session
        self.event_log: List[str] = []

        # Subscribe to events
        session.subscribe(self.on_event)

    async def on_event(self, event: BrowserEvent):
        """Handle browser event."""
        log_entry = f"[{event.type}] {event.data}"
        self.event_log.append(log_entry)
        print(log_entry)  # Or display in UI

    def get_live_view_url(self) -> str:
        """Get URL for human to view browser."""
        return self.session.get_remote_debug_url()

    async def take_manual_control(self):
        """Human takes manual control."""
        # Pause AI agents
        print("🧑 Human has taken control")
        print(f"   Connect to: {self.get_live_view_url()}")
        print("   Press Enter when done...")
        input()
        print("🤖 Returning control to AI agents")


# Usage
async def main():
    session = SharedBrowserSession()
    await session.start(headless=False)

    observer = HumanObserver(session)

    # AI agent works
    await session.navigate("http://127.0.0.1:8000")

    # Human can observe in real-time at:
    # chrome://inspect or http://localhost:9222
    print(f"👁️  Human can observe at: {observer.get_live_view_url()}")

    await session.wait_until_closed()
```

### 4. Integration with AuthenticatedBrowser

```python
class AuthenticatedBrowser:
    """Extended to support shared sessions."""

    def __init__(
        self,
        shared_session: Optional[SharedBrowserSession] = None,
        **kwargs
    ):
        self.shared_session = shared_session
        # ... existing code ...

    async def start(self):
        """Use shared session if available."""
        if self.shared_session:
            # Use existing session
            self.browser = self.shared_session.browser
            self.context = self.shared_session.context
            self.page = self.shared_session.page
        else:
            # Create new session (existing code)
            await super().start()


# Usage with shared session
session = SharedBrowserSession()
await session.start(headless=False)

# Multiple agents share same browser
async with AuthenticatedBrowser(
    shared_session=session,
    auth_strategy=DjangoAuthStrategy(...),
) as browser1:
    result1 = await browser1.navigate("http://127.0.0.1:8000/projects/")

async with AuthenticatedBrowser(
    shared_session=session,
) as browser2:
    result2 = await browser2.navigate("http://127.0.0.1:8000/new/")

# Same browser instance, shared cookies/state!
```

### 5. MCP Integration for Shared Sessions

```python
# In scitex/capture/mcp_server.py

class CaptureServer:
    def __init__(self):
        # ... existing code ...
        self.shared_sessions: Dict[str, SharedBrowserSession] = {}

    async def start_shared_session(
        self,
        session_id: str = "default",
        headless: bool = False,
    ):
        """Start a persistent shared browser session."""
        if session_id in self.shared_sessions:
            return {"error": "Session already exists"}

        session = SharedBrowserSession(session_id=session_id)
        await session.start(headless=headless)

        self.shared_sessions[session_id] = session

        return {
            "success": True,
            "session_id": session_id,
            "remote_debug_url": session.get_remote_debug_url(),
            "message": "Shared session started. Humans can connect to remote debug URL.",
        }

    async def shared_session_navigate(
        self,
        session_id: str,
        url: str,
    ):
        """Navigate in shared session."""
        session = self.shared_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        await session.navigate(url)
        screenshot = await session.take_screenshot()

        return {
            "success": True,
            "url": url,
            "screenshot": screenshot,
        }

    async def close_shared_session(self, session_id: str):
        """Close shared session."""
        session = self.shared_sessions.pop(session_id, None)
        if session:
            await session.close()
            return {"success": True}
        return {"error": "Session not found"}
```

## Usage Scenarios

### Scenario 1: AI Agent + Human Collaboration

```python
# Start shared session (visible)
session = SharedBrowserSession()
await session.start(headless=False)

# AI agent logs in
await session.navigate("http://127.0.0.1:8000/auth/login/")
await session.type_text("#id_username", "user")
await session.type_text("#id_password", "pass")
await session.click("button[type='submit']")

# AI agent starts creating project
await session.navigate("http://127.0.0.1:8000/new/")
await session.type_text("#id_name", "AI Generated Project")

# Human sees this happening in real-time
# Human can take over if AI gets stuck
# Or human can just observe and verify
```

### Scenario 2: Multiple AI Agents Coordinating

```python
# Shared session
session = SharedBrowserSession()
await session.start(headless=False)

coordinator = AgentCoordinator(session)

# Agent 1: Data collection
async def agent1_task():
    async with coordinator.agent_turn("collector"):
        await session.navigate("http://127.0.0.1:8000/projects/")
        # Extract project data
        data = await session.page.evaluate("/* extract JS */")
        return data

# Agent 2: Project creation
async def agent2_task():
    async with coordinator.agent_turn("creator"):
        await session.navigate("http://127.0.0.1:8000/new/")
        await session.type_text("#id_name", "New Project")
        await session.click("button[type='submit']")

# Run in sequence, sharing same authenticated session
data = await agent1_task()
await agent2_task()
```

### Scenario 3: Debugging with Human Oversight

```python
# AI agent runs with human watching
session = SharedBrowserSession(enable_recording=True)
await session.start(headless=False)

observer = HumanObserver(session)

# AI performs complex workflow
await session.navigate("http://127.0.0.1:8000")
# ... many actions ...

# If AI gets stuck, human can see exact state
if problem_detected:
    await observer.take_manual_control()
    # Human fixes issue manually
    # AI resumes

# Get timeline of what happened
timeline = session.get_timeline()
# Review screenshots and events
```

## Benefits

1. **Real-time Collaboration:** AI and humans work together
2. **Persistent State:** Single session, no re-authentication
3. **Visual Feedback:** Humans can see what AI is doing
4. **Multi-Agent:** Multiple AI agents coordinate
5. **Debugging:** Full timeline of events and screenshots
6. **Remote Access:** Chrome DevTools for remote viewing
7. **Efficiency:** No browser restart overhead

## Implementation Steps

1. ✅ Design shared session architecture
2. ⏳ Implement SharedBrowserSession
3. ⏳ Add AgentCoordinator
4. ⏳ Add HumanObserver interface
5. ⏳ Integrate with AuthenticatedBrowser
6. ⏳ Add MCP tools
7. ⏳ Create examples

---

**Status:** Architecture complete
**Priority:** Very High - enables real-time AI-human collaboration
**Unique Feature:** No other browser automation tool offers this!
