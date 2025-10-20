# Versatile Human-Like Interaction Strategy
**Date:** 2025-10-19
**Philosophy:** Agents should interact like humans, not follow framework-specific patterns

## Problem with Framework-Specific Approach

```python
# ❌ Too specific - only works for Django
DjangoAuthStrategy(login_url, username_field, password_field)

# ❌ What about OAuth? SAML? Custom auth?
# ❌ What about complex multi-step flows?
# ❌ What about CAPTCHAs?
```

## Better: Human-Like Interaction Primitives

```python
# ✅ Universal - works for ANY website
session.type("#username", "myuser")
session.click("button.submit")
session.wait_for(".dashboard")

# Works for:
# - Django, Flask, FastAPI, any framework
# - OAuth flows (click "Login with Google")
# - Complex multi-step forms
# - Any web application
```

## Core Concept: Record-Replay Pattern

### 1. Human Demonstrates (Record)
```python
# Human performs task manually
recorder = InteractionRecorder(session)
await recorder.start()

# Human:
# - Navigates to login page
# - Types username
# - Types password
# - Clicks submit
# - Waits for redirect

actions = await recorder.stop()
# Save actions to file
actions.save("login_workflow.json")
```

### 2. AI Replays (Automate)
```python
# AI replays recorded actions
replayer = InteractionReplayer(session)
await replayer.load("login_workflow.json")
await replayer.replay()

# Or with variations
await replayer.replay(variables={
    'username': 'different_user',
    'password': 'different_pass',
})
```

## Natural Interaction API

```python
class NaturalInteraction:
    """
    Human-like browser interaction.

    No framework assumptions - just natural actions.
    """

    def __init__(self, page: Page):
        self.page = page

    # Basic actions
    async def type(self, selector: str, text: str, delay: int = 50):
        """Type text like a human (with delay between keys)."""
        await self.page.type(selector, text, delay=delay)

    async def click(self, selector: str):
        """Click element."""
        await self.page.click(selector)

    async def hover(self, selector: str):
        """Hover over element."""
        await self.page.hover(selector)

    async def scroll_to(self, selector: str):
        """Scroll element into view."""
        await self.page.locator(selector).scroll_into_view_if_needed()

    async def press_key(self, key: str):
        """Press keyboard key (Enter, Tab, Escape, etc.)."""
        await self.page.keyboard.press(key)

    async def select_option(self, selector: str, value: str):
        """Select dropdown option."""
        await self.page.select_option(selector, value)

    async def upload_file(self, selector: str, file_path: str):
        """Upload file to input."""
        await self.page.set_input_files(selector, file_path)

    async def wait_for_element(self, selector: str, timeout: int = 5000):
        """Wait for element to appear."""
        await self.page.wait_for_selector(selector, timeout=timeout)

    async def wait_for_text(self, text: str, timeout: int = 5000):
        """Wait for text to appear."""
        await self.page.wait_for_selector(f"text={text}", timeout=timeout)

    async def wait_for_url(self, pattern: str, timeout: int = 5000):
        """Wait for URL to match pattern."""
        await self.page.wait_for_url(pattern, timeout=timeout)

    # Smart waiting
    async def wait_for_navigation(self):
        """Wait for navigation to complete."""
        await self.page.wait_for_load_state("load")

    async def wait_for_idle(self):
        """Wait until network is idle."""
        await self.page.wait_for_load_state("networkidle")

    # Extraction
    async def get_text(self, selector: str) -> str:
        """Get text content of element."""
        element = await self.page.query_selector(selector)
        return await element.text_content() if element else ""

    async def get_value(self, selector: str) -> str:
        """Get value of input element."""
        return await self.page.input_value(selector)

    async def get_attribute(self, selector: str, attr: str) -> str:
        """Get attribute value."""
        return await self.page.get_attribute(selector, attr)

    async def is_visible(self, selector: str) -> bool:
        """Check if element is visible."""
        return await self.page.is_visible(selector)

    async def is_enabled(self, selector: str) -> bool:
        """Check if element is enabled."""
        return await self.page.is_enabled(selector)
```

## Record-Replay Implementation

```python
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import json

@dataclass
class RecordedAction:
    """Single recorded action."""
    type: str  # type, click, scroll, wait, etc.
    target: Optional[str] = None  # Selector
    value: Optional[Any] = None  # Text, option, etc.
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class InteractionRecorder:
    """
    Record human interactions for later replay.

    Human demonstrates task → AI learns and repeats.
    """

    def __init__(self, session: SharedBrowserSession):
        self.session = session
        self.page = session.page
        self.recording = False
        self.actions: List[RecordedAction] = []
        self.start_time = 0.0

    async def start(self):
        """Start recording interactions."""
        self.recording = True
        self.start_time = time.time()
        self.actions = []

        # Inject recording script
        await self.page.evaluate("""
            () => {
                window.scitexRecording = {
                    actions: [],
                    startTime: Date.now(),

                    record: function(type, target, value, metadata = {}) {
                        this.actions.push({
                            type: type,
                            target: target,
                            value: value,
                            timestamp: Date.now() - this.startTime,
                            metadata: metadata,
                        });
                    }
                };

                // Record clicks
                document.addEventListener('click', (e) => {
                    const target = e.target;
                    const selector = getSelectorForElement(target);
                    window.scitexRecording.record('click', selector, null, {
                        tagName: target.tagName,
                        text: target.textContent?.substring(0, 50),
                    });
                });

                // Record typing
                document.addEventListener('input', (e) => {
                    const target = e.target;
                    const selector = getSelectorForElement(target);
                    window.scitexRecording.record('type', selector, target.value, {
                        inputType: target.type,
                    });
                });

                // Record scrolling
                let scrollTimeout;
                window.addEventListener('scroll', (e) => {
                    clearTimeout(scrollTimeout);
                    scrollTimeout = setTimeout(() => {
                        window.scitexRecording.record('scroll', null, {
                            x: window.scrollX,
                            y: window.scrollY,
                        });
                    }, 200);
                });

                // Helper to generate selector
                function getSelectorForElement(element) {
                    if (element.id) return `#${element.id}`;
                    if (element.name) return `[name="${element.name}"]`;

                    let path = [];
                    while (element && element.nodeType === Node.ELEMENT_NODE) {
                        let selector = element.nodeName.toLowerCase();
                        if (element.className) {
                            selector += '.' + element.className.split(' ').join('.');
                        }
                        path.unshift(selector);
                        element = element.parentNode;
                        if (path.length > 3) break;
                    }
                    return path.join(' > ');
                }

                console.log('🔴 Recording started...');
            }
        """)

        print("🔴 Recording started - perform your actions in the browser")

    async def stop(self) -> List[RecordedAction]:
        """Stop recording and return actions."""
        self.recording = False

        # Get recorded actions from page
        recorded_data = await self.page.evaluate("window.scitexRecording?.actions")

        if recorded_data:
            self.actions = [
                RecordedAction(**action) for action in recorded_data
            ]

        print(f"⏹️  Recording stopped - {len(self.actions)} actions recorded")

        return self.actions

    def save(self, filepath: str):
        """Save recording to file."""
        data = {
            'version': '1.0',
            'created_at': time.time(),
            'actions': [asdict(action) for action in self.actions],
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Saved recording: {filepath}")


class InteractionReplayer:
    """
    Replay recorded interactions.

    AI learns from human demonstration and repeats.
    """

    def __init__(self, session: SharedBrowserSession):
        self.session = session
        self.page = session.page
        self.actions: List[RecordedAction] = []

    def load(self, filepath: str):
        """Load recording from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.actions = [
            RecordedAction(**action) for action in data['actions']
        ]

        print(f"📂 Loaded recording: {filepath} ({len(self.actions)} actions)")

    async def replay(
        self,
        variables: Optional[Dict[str, str]] = None,
        speed: float = 1.0,  # 1.0 = real-time, 2.0 = 2x speed
        visual: Optional[VisualFeedback] = None,
    ):
        """
        Replay recorded actions.

        Args:
            variables: Replace recorded values (e.g., {'username': 'new_user'})
            speed: Playback speed multiplier
            visual: Visual feedback instance
        """
        variables = variables or {}

        print(f"▶️  Replaying {len(self.actions)} actions...")

        if visual:
            await visual.show_message("🔄 Replaying recorded actions...", "info")

        for i, action in enumerate(self.actions):
            # Show progress
            if visual:
                await visual.show_action(
                    "AI Agent",
                    f"{action.type} ({i+1}/{len(self.actions)})"
                )

            # Wait proportional to recorded timing
            if i > 0:
                delay = (action.timestamp - self.actions[i-1].timestamp) / 1000
                await asyncio.sleep(delay / speed)

            # Execute action
            await self._execute_action(action, variables)

        print(f"✅ Replay complete!")

        if visual:
            await visual.show_message("✅ Replay complete!", "success")

    async def _execute_action(
        self,
        action: RecordedAction,
        variables: Dict[str, str],
    ):
        """Execute a single recorded action."""
        try:
            if action.type == "click":
                await self.page.click(action.target)

            elif action.type == "type":
                # Apply variable substitution
                value = action.value
                for var_name, var_value in variables.items():
                    value = value.replace(f"{{{var_name}}}", var_value)

                await self.page.fill(action.target, value)

            elif action.type == "scroll":
                await self.page.evaluate(
                    f"window.scrollTo({action.value['x']}, {action.value['y']})"
                )

            elif action.type == "wait":
                await asyncio.sleep(action.value)

        except Exception as e:
            print(f"⚠️  Action failed: {action.type} - {e}")
```

## Usage Examples

### Example 1: Learn from Human

```python
# 1. Human demonstrates
session = SharedBrowserSession()
await session.start()

recorder = InteractionRecorder(session)
await recorder.start()

# Human performs login manually in browser:
# - Types username
# - Types password
# - Clicks submit

await asyncio.sleep(30)  # Give human time

actions = await recorder.stop()
recorder.save("workflows/login.json")

# 2. AI replays
replayer = InteractionReplayer(session)
replayer.load("workflows/login.json")

# Replay with different credentials
await replayer.replay(variables={
    'username': 'another_user',
    'password': 'another_pass',
})
```

### Example 2: Generic Interaction (No Recording Needed)

```python
# Just use natural primitives directly
from scitex.browser.collaboration import SharedBrowserSession

async with SharedBrowserSession() as session:
    await session.navigate("http://127.0.0.1:8000/auth/login/")

    # Type like a human
    await session.type("#id_username", "myuser")
    await session.type("#id_password", "mypass")

    # Click like a human
    await session.click("button[type='submit']")

    # Wait for result
    await session.wait_for_url("**/core/**")

    # Navigate to protected page
    await session.navigate("http://127.0.0.1:8000/new/")
```

### Example 3: Complex Multi-Step Flow

```python
# Works for ANY auth flow - OAuth, SAML, custom, etc.
session = SharedBrowserSession()
await session.start()

# OAuth example
await session.navigate("https://app.com/login")
await session.click("button:has-text('Login with Google')")
await session.wait_for_url("**/accounts.google.com/**")
await session.type("input[type='email']", "user@gmail.com")
await session.click("#identifierNext")
await session.wait_for("input[type='password']")
await session.type("input[type='password']", "password")
await session.click("#passwordNext")

# Works for ANYTHING!
```

## Proposed Architecture

```python
class SharedBrowserSession:
    """Enhanced with natural interaction primitives."""

    # Navigation
    async def navigate(self, url: str) -> str
    async def go_back(self)
    async def go_forward(self)
    async def reload(self)

    # Typing (human-like)
    async def type(self, selector: str, text: str, delay: int = 50)
    async def clear(self, selector: str)
    async def press(self, key: str)  # Enter, Tab, Escape, etc.

    # Clicking (human-like)
    async def click(self, selector: str, button: str = "left")
    async def double_click(self, selector: str)
    async def right_click(self, selector: str)
    async def hover(self, selector: str)

    # Scrolling (human-like)
    async def scroll_down(self, amount: int = 500)
    async def scroll_up(self, amount: int = 500)
    async def scroll_to_element(self, selector: str)
    async def scroll_to_top(self)
    async def scroll_to_bottom(self)

    # Waiting (flexible)
    async def wait_for(self, selector: str, timeout: int = 5000)
    async def wait_for_text(self, text: str, timeout: int = 5000)
    async def wait_for_url(self, pattern: str, timeout: int = 5000)
    async def wait(self, seconds: float)

    # Extraction (read data)
    async def get_text(self, selector: str) -> str
    async def get_value(self, selector: str) -> str
    async def get_attribute(self, selector: str, attr: str) -> str
    async def is_visible(self, selector: str) -> bool
    async def is_checked(self, selector: str) -> bool

    # Forms (human-like)
    async def fill_form(self, form_data: Dict[str, str])
    async def check_checkbox(self, selector: str)
    async def uncheck_checkbox(self, selector: str)
    async def select_dropdown(self, selector: str, value: str)

    # Advanced
    async def drag_and_drop(self, source: str, target: str)
    async def execute_js(self, script: str) -> Any
```

## Benefits

1. **Universal:** Works with ANY website/framework
2. **Intuitive:** Natural human-like actions
3. **Flexible:** Complex flows possible
4. **Learnable:** AI can learn from human demonstrations
5. **Reusable:** Record once, replay many times
6. **Adaptable:** Easy to modify for variations
7. **Framework-agnostic:** No assumptions about backend

## Comparison

### Old Approach (Framework-Specific)
```python
# ❌ Need different class for each framework
DjangoAuthStrategy(...)
OAuth2Strategy(...)
SAMLStrategy(...)
CustomStrategy(...)
```

### New Approach (Universal Primitives)
```python
# ✅ Same primitives work everywhere
await session.type(selector, text)
await session.click(selector)
await session.wait_for(selector)
```

---

**This is much better!** Should I implement this instead?
