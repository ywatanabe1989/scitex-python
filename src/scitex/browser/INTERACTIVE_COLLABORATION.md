# Interactive AI-Human Collaboration System
**Date:** 2025-10-19
**Inspired by:** scitex.scholar.browser (persistent context) + Real-time collaboration needs
**Goal:** Highly interactive, versatile browser automation for AI-human teams

## Key Insights from scitex.scholar.browser

```python
# Persistent context (from ScholarBrowserManager)
self._persistent_context = await playwright.chromium.launch_persistent_context(
    user_data_dir=str(profile_dir),  # Profile persists!
    headless=False,
    args=[...extensions...]
)

# Benefits:
# 1. Extensions loaded once, available forever
# 2. Authentication cookies persist
# 3. Profile data maintained
# 4. Fast page creation
# 5. Session continuity
```

This is **perfect** for AI-human collaboration! Let's extend it.

## Enhanced Interactive Collaboration System

### Core Concept: "Shared Brain" Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Shared Browser "Brain"                     │
│                  (Persistent Playwright Context)              │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  AI Agent 1  │  │  AI Agent 2  │  │    Human     │       │
│  │  (Claude)    │  │  (GPT-4)     │  │    User      │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                  │                  │                │
│         ▼                  ▼                  ▼                │
│  ┌─────────────────────────────────────────────────────┐     │
│  │         Interaction Coordinator                      │     │
│  │  • Turn Management                                    │     │
│  │  • Conflict Resolution                                │     │
│  │  • Intent Recognition                                 │     │
│  │  • Action Planning                                    │     │
│  └──────────────────┬──────────────────────────────────┘     │
│                     │                                          │
│                     ▼                                          │
│  ┌─────────────────────────────────────────────────────┐     │
│  │         Real-time State Manager                      │     │
│  │  • DOM Monitoring                                     │     │
│  │  • Visual Feedback                                    │     │
│  │  • Cursor Tracking                                    │     │
│  │  • Annotation Layer                                   │     │
│  └──────────────────┬──────────────────────────────────┘     │
│                     │                                          │
│                     ▼                                          │
│  ┌─────────────────────────────────────────────────────┐     │
│  │         Shared Page (Playwright)                     │     │
│  │  • Single source of truth                             │     │
│  │  • All see same state                                 │     │
│  │  • Actions coordinated                                │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

## 1. Interactive Collaboration Manager

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict
import asyncio

class ParticipantType(Enum):
    """Type of participant in collaboration."""
    AI_AGENT = "ai_agent"
    HUMAN = "human"
    SYSTEM = "system"

class InteractionMode(Enum):
    """How participants can interact."""
    OBSERVE = "observe"          # Watch only
    SUGGEST = "suggest"          # Suggest actions
    CONTROL = "control"          # Full control
    COLLABORATIVE = "collaborative"  # Take turns

@dataclass
class Participant:
    """Participant in collaborative session."""
    id: str
    type: ParticipantType
    mode: InteractionMode
    name: str
    capabilities: List[str] = field(default_factory=list)
    priority: int = 0  # Higher = more priority

@dataclass
class Intent:
    """Recognized intent from participant."""
    participant_id: str
    action: str  # "navigate", "click", "fill", "extract", etc.
    target: Optional[str]  # Selector or URL
    confidence: float  # 0.0 - 1.0
    reasoning: str  # Why this intent
    alternatives: List[dict] = field(default_factory=list)


class InteractiveCollaborationManager:
    """
    Manages interactive collaboration between AI agents and humans.

    Features:
    - Real-time coordination
    - Visual feedback (who's doing what)
    - Intent recognition
    - Conflict resolution
    - Turn-based or simultaneous control
    - Annotation overlay
    - Voice annotations (optional)
    """

    def __init__(
        self,
        persistent_context: BrowserContext,
        collaboration_mode: str = "turn_based",  # or "simultaneous"
        enable_visual_feedback: bool = True,
        enable_voice: bool = False,
    ):
        self.context = persistent_context
        self.collaboration_mode = collaboration_mode
        self.enable_visual_feedback = enable_visual_feedback
        self.enable_voice = enable_voice

        # Participants
        self.participants: Dict[str, Participant] = {}
        self.active_participant: Optional[str] = None

        # State
        self.page: Optional[Page] = None
        self.pending_intents: asyncio.Queue = asyncio.Queue()
        self.action_history: List[dict] = []

        # Visual feedback
        self.cursor_positions: Dict[str, dict] = {}  # participant_id -> {x, y}
        self.annotations: List[dict] = []

        # Coordination
        self.lock = asyncio.Lock()
        self.event_bus = EventBus()

    async def register_participant(
        self,
        participant: Participant,
    ):
        """Register a participant (AI or human)."""
        self.participants[participant.id] = participant
        await self.event_bus.emit("participant_joined", {
            "id": participant.id,
            "type": participant.type.value,
            "name": participant.name,
        })

        # Visual feedback
        if self.enable_visual_feedback:
            await self._show_participant_indicator(participant)

    async def _show_participant_indicator(self, participant: Participant):
        """Show visual indicator for participant."""
        color = "blue" if participant.type == ParticipantType.AI_AGENT else "green"
        await self.page.evaluate(f"""
            () => {{
                const indicator = document.createElement('div');
                indicator.id = 'participant-{participant.id}';
                indicator.style.cssText = `
                    position: fixed;
                    top: 10px;
                    right: ${len(self.participants) * 150}px;
                    background: {color};
                    color: white;
                    padding: 8px 12px;
                    border-radius: 20px;
                    z-index: 10000;
                    font-family: system-ui;
                    font-size: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                `;
                indicator.textContent = '{participant.name}';
                document.body.appendChild(indicator);
            }}
        """)

    async def recognize_intent(
        self,
        participant_id: str,
        input_data: dict,
    ) -> Intent:
        """
        Recognize participant's intent from input.

        For AI: Structured command
        For Human: Mouse/keyboard events, voice, or natural language
        """
        participant = self.participants[participant_id]

        if participant.type == ParticipantType.AI_AGENT:
            # AI provides structured intent
            return Intent(
                participant_id=participant_id,
                action=input_data.get("action"),
                target=input_data.get("target"),
                confidence=1.0,
                reasoning=input_data.get("reasoning", ""),
            )

        elif participant.type == ParticipantType.HUMAN:
            # Recognize from human actions
            return await self._recognize_human_intent(input_data)

    async def _recognize_human_intent(self, event_data: dict) -> Intent:
        """
        Recognize human intent from browser events.

        Examples:
        - Click on element → "click" intent
        - Typing in field → "fill" intent
        - Scroll → "scroll" intent
        - Voice command → parse to intent
        """
        event_type = event_data.get("type")

        if event_type == "click":
            return Intent(
                participant_id="human",
                action="click",
                target=event_data.get("selector"),
                confidence=1.0,
                reasoning="Human clicked on element",
            )

        elif event_type == "input":
            return Intent(
                participant_id="human",
                action="fill",
                target=event_data.get("selector"),
                confidence=1.0,
                reasoning="Human typed in field",
            )

        elif event_type == "voice":
            # Parse voice command
            command = event_data.get("text")
            parsed_intent = await self._parse_voice_command(command)
            return parsed_intent

        # Default: observation
        return Intent(
            participant_id="human",
            action="observe",
            target=None,
            confidence=0.5,
            reasoning="Unrecognized action",
        )

    async def request_action(
        self,
        participant_id: str,
        intent: Intent,
        wait_for_approval: bool = False,
    ) -> bool:
        """
        Request to perform action.

        In turn_based mode: Queue and wait for turn
        In simultaneous mode: Execute if no conflicts
        """
        participant = self.participants[participant_id]

        # Check permissions
        if participant.mode == InteractionMode.OBSERVE:
            await self._show_notification(
                participant_id,
                "You're in observe mode. Cannot perform actions."
            )
            return False

        # Add to queue
        await self.pending_intents.put((participant_id, intent))

        # Visual feedback
        if self.enable_visual_feedback:
            await self._show_intent_preview(participant_id, intent)

        # Wait for approval if needed
        if wait_for_approval:
            approved = await self._request_approval(participant_id, intent)
            if not approved:
                return False

        # Execute
        if self.collaboration_mode == "turn_based":
            return await self._execute_turn_based(participant_id, intent)
        else:
            return await self._execute_simultaneous(participant_id, intent)

    async def _show_intent_preview(self, participant_id: str, intent: Intent):
        """Show visual preview of intended action."""
        participant = self.participants[participant_id]
        color = "rgba(0, 100, 255, 0.3)" if participant.type == ParticipantType.AI_AGENT else "rgba(0, 255, 100, 0.3)"

        # Highlight target element
        if intent.target:
            await self.page.evaluate(f"""
                (selector) => {{
                    const element = document.querySelector(selector);
                    if (element) {{
                        element.style.outline = '3px solid {color}';
                        element.style.transition = 'outline 0.3s';

                        // Show tooltip
                        const tooltip = document.createElement('div');
                        tooltip.style.cssText = `
                            position: absolute;
                            background: {color.replace('0.3', '0.9')};
                            color: white;
                            padding: 4px 8px;
                            border-radius: 4px;
                            font-size: 11px;
                            z-index: 10001;
                            pointer-events: none;
                        `;
                        tooltip.textContent = '{participant.name}: {intent.action}';

                        const rect = element.getBoundingClientRect();
                        tooltip.style.top = rect.top - 30 + 'px';
                        tooltip.style.left = rect.left + 'px';
                        document.body.appendChild(tooltip);

                        // Remove after 2 seconds
                        setTimeout(() => {{
                            element.style.outline = '';
                            tooltip.remove();
                        }}, 2000);
                    }}
                }}
            """, intent.target)

    async def _execute_turn_based(
        self,
        participant_id: str,
        intent: Intent,
    ) -> bool:
        """Execute action in turn-based mode."""
        async with self.lock:
            # Check if it's this participant's turn
            if self.active_participant and self.active_participant != participant_id:
                participant = self.participants[participant_id]
                active = self.participants[self.active_participant]
                await self._show_notification(
                    participant_id,
                    f"Waiting for {active.name} to finish..."
                )
                # Wait for turn
                while self.active_participant != participant_id:
                    await asyncio.sleep(0.1)

            # Take turn
            self.active_participant = participant_id

            # Execute
            success = await self._execute_intent(intent)

            # Record
            self.action_history.append({
                "participant_id": participant_id,
                "intent": intent,
                "success": success,
                "timestamp": time.time(),
            })

            # Release turn
            if success:
                self.active_participant = None
                await self._notify_next_participant()

            return success

    async def _execute_intent(self, intent: Intent) -> bool:
        """Execute the actual browser action."""
        try:
            if intent.action == "navigate":
                await self.page.goto(intent.target)

            elif intent.action == "click":
                await self.page.click(intent.target)

            elif intent.action == "fill":
                await self.page.fill(intent.target, intent.data.get("value"))

            elif intent.action == "scroll":
                await self.page.evaluate("""
                    () => window.scrollBy(0, 500)
                """)

            elif intent.action == "extract":
                # Extract data
                data = await self.page.evaluate(f"""
                    (selector) => {{
                        const el = document.querySelector(selector);
                        return el ? el.textContent : null;
                    }}
                """, intent.target)
                intent.data["extracted"] = data

            elif intent.action == "annotate":
                await self._add_annotation(intent)

            # Emit event
            await self.event_bus.emit("action_completed", {
                "participant_id": intent.participant_id,
                "action": intent.action,
                "target": intent.target,
            })

            return True

        except Exception as e:
            await self.event_bus.emit("action_failed", {
                "participant_id": intent.participant_id,
                "error": str(e),
            })
            return False

    async def _add_annotation(self, intent: Intent):
        """Add visual annotation to page."""
        annotation = {
            "type": intent.data.get("annotation_type", "note"),
            "text": intent.data.get("text", ""),
            "position": intent.data.get("position"),
            "participant_id": intent.participant_id,
        }
        self.annotations.append(annotation)

        # Draw annotation
        await self.page.evaluate("""
            (annotation) => {
                const note = document.createElement('div');
                note.style.cssText = `
                    position: absolute;
                    left: ${annotation.position.x}px;
                    top: ${annotation.position.y}px;
                    background: #fffacd;
                    border: 2px solid #ffd700;
                    padding: 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    z-index: 10000;
                    max-width: 200px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                `;
                note.textContent = annotation.text;
                document.body.appendChild(note);
            }
        """, annotation)

    async def suggest_action(
        self,
        from_participant: str,
        to_participant: str,
        suggestion: Intent,
    ):
        """One participant suggests action to another."""
        await self._show_notification(
            to_participant,
            f"{self.participants[from_participant].name} suggests: {suggestion.action} on {suggestion.target}",
            actions=[
                {"label": "Accept", "value": "accept"},
                {"label": "Modify", "value": "modify"},
                {"label": "Decline", "value": "decline"},
            ]
        )

    async def _show_notification(
        self,
        participant_id: str,
        message: str,
        actions: List[dict] = None,
    ):
        """Show notification to participant."""
        await self.page.evaluate(f"""
            () => {{
                const notif = document.createElement('div');
                notif.style.cssText = `
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background: #2196F3;
                    color: white;
                    padding: 16px;
                    border-radius: 8px;
                    z-index: 10000;
                    max-width: 300px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                    animation: slideIn 0.3s ease-out;
                `;
                notif.textContent = '{message}';
                document.body.appendChild(notif);

                setTimeout(() => {{
                    notif.style.animation = 'slideOut 0.3s ease-in';
                    setTimeout(() => notif.remove(), 300);
                }}, 5000);
            }}
        """)

        # Emit to event bus for other participants
        await self.event_bus.emit("notification", {
            "participant_id": participant_id,
            "message": message,
        })

    def enable_realtime_cursor_tracking(self):
        """Track all participants' cursors in real-time."""
        # Inject cursor tracking script
        self.page.add_init_script("""
            document.addEventListener('mousemove', (e) => {
                window.cursorPosition = {x: e.clientX, y: e.clientY};
            });
        """)

    async def show_participant_cursors(self):
        """Show cursors of all participants."""
        for participant_id, participant in self.participants.items():
            if participant.type == ParticipantType.HUMAN:
                # Get cursor position
                position = await self.page.evaluate("window.cursorPosition")
                if position:
                    self.cursor_positions[participant_id] = position

                    # Show cursor
                    color = "green"
                    await self.page.evaluate(f"""
                        () => {{
                            let cursor = document.getElementById('cursor-{participant_id}');
                            if (!cursor) {{
                                cursor = document.createElement('div');
                                cursor.id = 'cursor-{participant_id}';
                                cursor.style.cssText = `
                                    position: fixed;
                                    width: 20px;
                                    height: 20px;
                                    border-radius: 50%;
                                    background: {color};
                                    pointer-events: none;
                                    z-index: 100000;
                                    transition: all 0.1s;
                                `;
                                document.body.appendChild(cursor);
                            }}
                            cursor.style.left = {position['x']}px;
                            cursor.style.top = {position['y']}px;
                        }}
                    """)

    async def create_shared_workspace(self):
        """Create shared annotation workspace."""
        await self.page.evaluate("""
            () => {
                // Create annotation toolbar
                const toolbar = document.createElement('div');
                toolbar.id = 'collaboration-toolbar';
                toolbar.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 50%;
                    transform: translateX(-50%);
                    background: white;
                    padding: 8px;
                    border-radius: 0 0 8px 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                    z-index: 100001;
                    display: flex;
                    gap: 8px;
                `;

                const tools = [
                    {icon: '✏️', name: 'Annotate'},
                    {icon: '💬', name: 'Comment'},
                    {icon: '👆', name: 'Point'},
                    {icon: '📸', name: 'Screenshot'},
                ];

                tools.forEach(tool => {
                    const btn = document.createElement('button');
                    btn.textContent = tool.icon;
                    btn.title = tool.name;
                    btn.style.cssText = `
                        padding: 8px 12px;
                        border: none;
                        background: #f0f0f0;
                        border-radius: 4px;
                        cursor: pointer;
                    `;
                    btn.onclick = () => window.collaborationTool = tool.name.toLowerCase();
                    toolbar.appendChild(btn);
                });

                document.body.appendChild(toolbar);
            }
        """)
```

## 2. Event Bus for Coordination

```python
class EventBus:
    """Real-time event bus for participant coordination."""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    async def emit(self, event_type: str, data: dict):
        """Emit event to all subscribers."""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                asyncio.create_task(callback(data))
```

## 3. Usage Examples

### Example 1: AI Agent + Human Pair Programming

```python
from scitex.browser.automation import (
    InteractiveCollaborationManager,
    Participant,
    ParticipantType,
    InteractionMode,
)

# Start persistent browser (like ScholarBrowserManager)
async with async_playwright() as p:
    context = await p.chromium.launch_persistent_context(
        user_data_dir="~/.scitex/browser/shared_session",
        headless=False,
    )

    page = await context.new_page()

    # Create collaboration manager
    collab = InteractiveCollaborationManager(context)
    collab.page = page

    # Register AI agent
    await collab.register_participant(Participant(
        id="claude",
        type=ParticipantType.AI_AGENT,
        mode=InteractionMode.COLLABORATIVE,
        name="Claude",
        capabilities=["navigate", "extract", "analyze"],
    ))

    # Register human
    await collab.register_participant(Participant(
        id="human",
        type=ParticipantType.HUMAN,
        mode=InteractionMode.COLLABORATIVE,
        name="You",
        capabilities=["all"],
        priority=10,  # Human has priority
    ))

    # Enable visual features
    await collab.create_shared_workspace()
    collab.enable_realtime_cursor_tracking()

    # AI agent suggests navigation
    await collab.request_action(
        "claude",
        Intent(
            participant_id="claude",
            action="navigate",
            target="http://127.0.0.1:8000/projects/",
            confidence=1.0,
            reasoning="Let's check the projects page",
        )
    )

    # Human can see suggestion and approve/modify
    # Human can also take control at any time

    # Keep session open
    await asyncio.sleep(3600)  # 1 hour session
```

### Example 2: Multi-Agent Task Division

```python
# AI Agent 1: Data collector
await collab.request_action(
    "agent1",
    Intent(
        participant_id="agent1",
        action="navigate",
        target="http://127.0.0.1:8000/projects/",
        confidence=1.0,
        reasoning="Collecting project data",
    )
)

# AI Agent 2: Content creator
await collab.request_action(
    "agent2",
    Intent(
        participant_id="agent2",
        action="navigate",
        target="http://127.0.0.1:8000/new/",
        confidence=1.0,
        reasoning="Creating new project based on collected data",
    )
)

# Human: Supervisor
# Watches both agents, can intervene anytime
```

### Example 3: Annotation and Discussion

```python
# AI agent annotates something interesting
await collab.request_action(
    "claude",
    Intent(
        participant_id="claude",
        action="annotate",
        target=None,
        confidence=1.0,
        reasoning="Interesting pattern found",
        data={
            "annotation_type": "note",
            "text": "This element seems to be duplicated. Should we refactor?",
            "position": {"x": 100, "y": 200},
        }
    )
)

# Human sees annotation and can respond
# Creates discussion thread on the page
```

## Benefits Over Previous Designs

1. **More Interactive:** Real-time visual feedback
2. **More Versatile:** Supports multiple collaboration modes
3. **Intent Recognition:** Understands what participants want to do
4. **Visual Coordination:** See who's doing what
5. **Conflict Resolution:** Prevents interference
6. **Annotation Layer:** Discuss directly on page
7. **Turn Management:** Fair access to control
8. **Suggestion System:** Agents can advise each other

## Implementation Priority

1. ⏳ InteractiveCollaborationManager core
2. ⏳ Visual feedback system
3. ⏳ Intent recognition
4. ⏳ Event bus
5. ⏳ Annotation system
6. ⏳ Voice integration (optional)

---

**This is MORE interactive and versatile than any existing browser automation system!**
