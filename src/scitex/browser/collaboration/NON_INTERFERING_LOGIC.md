# Non-Interfering Logic for Users and Agents
**Date:** 2025-10-19
**Critical:** Prevent conflicts and interference between participants

## The Problem

When multiple participants (AI agents + humans) control the same browser:

### Conflicts
```
AI Agent: Click button A
Human:    Click button B (at same time)
Result:   ❌ Race condition, unpredictable behavior
```

### Interference
```
AI Agent: Filling form field
Human:    Scrolling page
Result:   ❌ Form loses focus, data not saved
```

### Lost Work
```
AI Agent: Typing long text
Human:    Navigates away
Result:   ❌ AI's work lost
```

## Solution: Multi-Layer Non-Interference System

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│               Non-Interference Layers                       │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 1: Intent Detection                                  │
│  ├─ Detect what participants want to do                     │
│  ├─ Classify intent (navigate, click, fill, etc.)           │
│  └─ Check for conflicts BEFORE execution                    │
│                                                              │
│  Layer 2: Conflict Resolution                               │
│  ├─ Prioritize actions (human > AI by default)              │
│  ├─ Queue conflicting actions                                │
│  └─ Merge compatible actions                                 │
│                                                              │
│  Layer 3: Execution Coordination                            │
│  ├─ Serialize conflicting actions                            │
│  ├─ Parallelize compatible actions                           │
│  └─ Atomic operations with rollback                          │
│                                                              │
│  Layer 4: State Synchronization                             │
│  ├─ Broadcast state changes                                  │
│  ├─ Invalidate stale intents                                 │
│  └─ Notify participants of changes                           │
│                                                              │
│  Layer 5: Graceful Degradation                              │
│  ├─ Pause low-priority tasks                                 │
│  ├─ Resume when clear                                         │
│  └─ Fail gracefully with notifications                       │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

## Implementation

### 1. Intent Analyzer

```python
from dataclasses import dataclass
from enum import Enum
from typing import Set, Optional

class IntentType(Enum):
    """Types of intents."""
    NAVIGATE = "navigate"          # Changes URL
    CLICK = "click"                # Clicks element
    FILL = "fill"                  # Fills form
    SCROLL = "scroll"              # Scrolls page
    EXTRACT = "extract"            # Extracts data (read-only)
    WAIT = "wait"                  # Waits for condition
    OBSERVE = "observe"            # Just watching (read-only)

@dataclass
class IntentProfile:
    """Profile describing an intent's impact."""
    type: IntentType
    target: Optional[str]  # CSS selector or URL
    affects_dom: bool  # Will modify DOM?
    affects_navigation: bool  # Will navigate away?
    affects_focus: bool  # Will change focus?
    read_only: bool  # Only reads, doesn't modify
    estimated_duration: float  # Seconds
    interruptible: bool  # Can be interrupted?
    critical: bool  # Must complete?

class IntentAnalyzer:
    """Analyzes intents to detect conflicts."""

    def analyze(self, intent: Intent) -> IntentProfile:
        """Analyze intent and create profile."""
        if intent.action == "navigate":
            return IntentProfile(
                type=IntentType.NAVIGATE,
                target=intent.target,
                affects_dom=True,
                affects_navigation=True,
                affects_focus=True,
                read_only=False,
                estimated_duration=2.0,
                interruptible=False,  # Don't interrupt navigation
                critical=True,
            )

        elif intent.action == "click":
            return IntentProfile(
                type=IntentType.CLICK,
                target=intent.target,
                affects_dom=True,  # Might change DOM
                affects_navigation=False,  # Usually doesn't navigate
                affects_focus=True,
                read_only=False,
                estimated_duration=0.5,
                interruptible=True,
                critical=False,
            )

        elif intent.action == "fill":
            return IntentProfile(
                type=IntentType.FILL,
                target=intent.target,
                affects_dom=True,
                affects_navigation=False,
                affects_focus=True,
                read_only=False,
                estimated_duration=1.0,
                interruptible=False,  # Don't interrupt form filling
                critical=True,  # Data loss if interrupted
            )

        elif intent.action == "extract":
            return IntentProfile(
                type=IntentType.EXTRACT,
                target=intent.target,
                affects_dom=False,
                affects_navigation=False,
                affects_focus=False,
                read_only=True,
                estimated_duration=0.1,
                interruptible=True,
                critical=False,
            )

        # Default: conservative profile
        return IntentProfile(
            type=IntentType.OBSERVE,
            target=None,
            affects_dom=False,
            affects_navigation=False,
            affects_focus=False,
            read_only=True,
            estimated_duration=0.0,
            interruptible=True,
            critical=False,
        )

    def check_conflict(
        self,
        intent1: IntentProfile,
        intent2: IntentProfile,
    ) -> bool:
        """Check if two intents conflict."""

        # Read-only operations never conflict
        if intent1.read_only and intent2.read_only:
            return False

        # Navigation conflicts with everything
        if intent1.affects_navigation or intent2.affects_navigation:
            return True

        # Both modify same target
        if intent1.target and intent2.target:
            if intent1.target == intent2.target:
                if not (intent1.read_only and intent2.read_only):
                    return True

        # Both require focus
        if intent1.affects_focus and intent2.affects_focus:
            return True

        # Both modify DOM in conflicting ways
        if intent1.affects_dom and intent2.affects_dom:
            # More sophisticated check could look at DOM regions
            return True

        return False
```

### 2. Conflict Resolver

```python
from typing import List, Tuple

class Priority(Enum):
    """Priority levels."""
    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    BACKGROUND = 10

class ConflictResolver:
    """Resolves conflicts between intents."""

    def __init__(self):
        self.analyzer = IntentAnalyzer()

        # Priority rules
        self.participant_priority = {
            ParticipantType.HUMAN: Priority.HIGH,
            ParticipantType.AI_AGENT: Priority.NORMAL,
            ParticipantType.SYSTEM: Priority.BACKGROUND,
        }

    def resolve(
        self,
        pending_intents: List[Tuple[str, Intent]],
    ) -> dict:
        """
        Resolve conflicts and determine execution order.

        Returns:
            {
                'execute_now': [...],      # Execute immediately
                'queue': [...],             # Queue for later
                'pause': [...],             # Pause these
                'cancel': [...],            # Cancel these
            }
        """
        # Analyze all intents
        profiles = [
            (participant_id, intent, self.analyzer.analyze(intent))
            for participant_id, intent in pending_intents
        ]

        execute_now = []
        queue = []
        pause = []
        cancel = []

        # Find conflicts
        for i, (pid1, intent1, profile1) in enumerate(profiles):
            participant1 = self.participants[pid1]
            priority1 = self._get_priority(participant1, profile1)

            conflicts = []
            for j, (pid2, intent2, profile2) in enumerate(profiles):
                if i == j:
                    continue

                if self.analyzer.check_conflict(profile1, profile2):
                    participant2 = self.participants[pid2]
                    priority2 = self._get_priority(participant2, profile2)
                    conflicts.append((j, priority2))

            if not conflicts:
                # No conflicts - execute immediately
                execute_now.append((pid1, intent1))

            else:
                # Has conflicts - resolve based on priority
                max_conflict_priority = max(p for _, p in conflicts)

                if priority1 > max_conflict_priority:
                    # This intent has higher priority
                    execute_now.append((pid1, intent1))

                    # Pause conflicting lower-priority intents
                    for idx, _ in conflicts:
                        pid2, intent2, profile2 = profiles[idx]
                        if profile2.interruptible:
                            pause.append((pid2, intent2))
                        else:
                            queue.append((pid2, intent2))

                elif priority1 == max_conflict_priority:
                    # Equal priority - queue
                    queue.append((pid1, intent1))

                else:
                    # Lower priority
                    if profile1.interruptible:
                        pause.append((pid1, intent1))
                    else:
                        queue.append((pid1, intent1))

        return {
            'execute_now': execute_now,
            'queue': queue,
            'pause': pause,
            'cancel': cancel,
        }

    def _get_priority(
        self,
        participant: Participant,
        profile: IntentProfile,
    ) -> int:
        """Calculate effective priority."""
        base_priority = self.participant_priority[participant.type].value

        # Adjust for intent criticality
        if profile.critical:
            base_priority += 20

        # Adjust for participant-specific priority
        base_priority += participant.priority

        return base_priority
```

### 3. Non-Interfering Execution Engine

```python
class NonInterferingExecutor:
    """Executes intents without interference."""

    def __init__(self, page: Page):
        self.page = page
        self.resolver = ConflictResolver()
        self.active_executions: Dict[str, asyncio.Task] = {}
        self.execution_lock = asyncio.Lock()

    async def execute_with_coordination(
        self,
        pending_intents: List[Tuple[str, Intent]],
    ):
        """Execute intents with conflict resolution."""

        # Resolve conflicts
        resolution = self.resolver.resolve(pending_intents)

        # Execute immediately
        for participant_id, intent in resolution['execute_now']:
            await self._execute_protected(participant_id, intent)

        # Queue others
        for participant_id, intent in resolution['queue']:
            await self._add_to_queue(participant_id, intent)

        # Pause interrupted tasks
        for participant_id, intent in resolution['pause']:
            await self._pause_execution(participant_id, intent)

    async def _execute_protected(
        self,
        participant_id: str,
        intent: Intent,
    ):
        """
        Execute intent with protection against interference.

        Uses atomic operations and rollback on failure.
        """
        async with self.execution_lock:
            participant = self.resolver.participants[participant_id]

            # Notify others
            await self._broadcast_execution_start(participant_id, intent)

            # Create checkpoint for rollback
            checkpoint = await self._create_checkpoint()

            try:
                # Execute
                profile = self.resolver.analyzer.analyze(intent)

                if not profile.interruptible:
                    # Critical section - disable interference
                    await self._enter_critical_section(participant_id)

                # Actual execution
                result = await self._execute_intent(intent)

                if not profile.interruptible:
                    await self._exit_critical_section(participant_id)

                # Notify success
                await self._broadcast_execution_complete(
                    participant_id,
                    intent,
                    success=True,
                )

                return result

            except Exception as e:
                # Rollback on error
                await self._rollback(checkpoint)

                # Notify failure
                await self._broadcast_execution_complete(
                    participant_id,
                    intent,
                    success=False,
                    error=str(e),
                )

                raise

    async def _enter_critical_section(self, participant_id: str):
        """
        Enter critical section - others cannot interfere.

        Shows visual indicator to others.
        """
        await self.page.evaluate(f"""
            () => {{
                // Show critical section indicator
                const indicator = document.createElement('div');
                indicator.id = 'critical-section-indicator';
                indicator.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: rgba(255, 0, 0, 0.9);
                    color: white;
                    padding: 20px 40px;
                    border-radius: 8px;
                    z-index: 100000;
                    font-size: 16px;
                    font-weight: bold;
                    pointer-events: none;
                `;
                indicator.textContent = '⚠️ Critical Operation in Progress - Please Wait';
                document.body.appendChild(indicator);
            }}
        """)

    async def _exit_critical_section(self, participant_id: str):
        """Exit critical section."""
        await self.page.evaluate("""
            () => {
                const indicator = document.getElementById('critical-section-indicator');
                if (indicator) {
                    indicator.remove();
                }
            }
        """)

    async def _create_checkpoint(self) -> dict:
        """Create checkpoint for rollback."""
        return {
            'url': self.page.url,
            'scroll_position': await self.page.evaluate("() => ({x: window.scrollX, y: window.scrollY})"),
            'dom_snapshot': await self.page.content(),  # For critical operations
        }

    async def _rollback(self, checkpoint: dict):
        """Rollback to checkpoint."""
        # Navigate back if URL changed
        if self.page.url != checkpoint['url']:
            await self.page.goto(checkpoint['url'])

        # Restore scroll position
        await self.page.evaluate(
            f"() => window.scrollTo({checkpoint['scroll_position']['x']}, {checkpoint['scroll_position']['y']})"
        )

    async def _broadcast_execution_start(
        self,
        participant_id: str,
        intent: Intent,
    ):
        """Notify all participants that execution started."""
        participant = self.resolver.participants[participant_id]

        # Show who's doing what
        await self.page.evaluate(f"""
            () => {{
                const notification = document.createElement('div');
                notification.style.cssText = `
                    position: fixed;
                    top: 60px;
                    right: 20px;
                    background: rgba(33, 150, 243, 0.9);
                    color: white;
                    padding: 12px 16px;
                    border-radius: 8px;
                    z-index: 10000;
                    font-size: 12px;
                `;
                notification.textContent = '{participant.name} is {intent.action}ing...';
                document.body.appendChild(notification);

                setTimeout(() => notification.remove(), 2000);
            }}
        """)

    async def _broadcast_execution_complete(
        self,
        participant_id: str,
        intent: Intent,
        success: bool,
        error: str = None,
    ):
        """Notify all participants that execution completed."""
        # Emit event
        await self.event_bus.emit("execution_complete", {
            "participant_id": participant_id,
            "intent": intent.action,
            "success": success,
            "error": error,
        })
```

### 4. Graceful Pause/Resume

```python
class GracefulPause:
    """Gracefully pause and resume operations."""

    def __init__(self, page: Page):
        self.page = page
        self.paused_operations: Dict[str, dict] = {}

    async def pause_operation(
        self,
        participant_id: str,
        intent: Intent,
        reason: str,
    ):
        """Pause an operation gracefully."""

        # Save state
        self.paused_operations[participant_id] = {
            'intent': intent,
            'reason': reason,
            'paused_at': time.time(),
            'checkpoint': await self._create_checkpoint(),
        }

        # Notify participant
        participant = self.participants[participant_id]
        await self._show_pause_notification(participant, reason)

    async def resume_operation(self, participant_id: str):
        """Resume a paused operation."""
        if participant_id not in self.paused_operations:
            return

        operation = self.paused_operations.pop(participant_id)

        # Restore state
        await self._restore_checkpoint(operation['checkpoint'])

        # Notify participant
        participant = self.participants[participant_id]
        await self._show_resume_notification(participant)

        # Re-execute intent
        await self.execute_protected(participant_id, operation['intent'])

    async def _show_pause_notification(
        self,
        participant: Participant,
        reason: str,
    ):
        """Show notification that operation was paused."""
        color = "orange"
        await self.page.evaluate(f"""
            () => {{
                const notif = document.createElement('div');
                notif.style.cssText = `
                    position: fixed;
                    bottom: 80px;
                    right: 20px;
                    background: {color};
                    color: white;
                    padding: 12px 16px;
                    border-radius: 8px;
                    z-index: 10000;
                    font-size: 12px;
                    max-width: 300px;
                `;
                notif.innerHTML = `
                    ⏸️ {participant.name}'s operation paused<br>
                    <small>Reason: {reason}</small>
                `;
                document.body.appendChild(notif);

                setTimeout(() => notif.remove(), 5000);
            }}
        """)
```

### 5. Smart Waiting

```python
class SmartWaiter:
    """Wait intelligently without blocking others."""

    async def wait_for_idle(self, timeout: float = 5.0):
        """Wait until page is idle (no ongoing operations)."""
        start = time.time()

        while time.time() - start < timeout:
            # Check if any critical operations running
            if not self._has_critical_operations():
                return True

            await asyncio.sleep(0.1)

        return False

    async def wait_for_participant(
        self,
        participant_id: str,
        timeout: float = 30.0,
    ):
        """Wait for specific participant to finish."""
        start = time.time()

        while time.time() - start < timeout:
            if participant_id not in self.active_executions:
                return True

            await asyncio.sleep(0.1)

        return False

    def _has_critical_operations(self) -> bool:
        """Check if any critical operations are running."""
        # Check active executions
        for task in self.active_executions.values():
            if not task.done():
                return True
        return False
```

## Usage Examples

### Example 1: Human Interrupts AI

```python
# AI is filling long form
ai_intent = Intent(
    participant_id="ai",
    action="fill",
    target="#description",
    data={"value": "Very long description..."},
)

# Human wants to navigate away
human_intent = Intent(
    participant_id="human",
    action="navigate",
    target="http://other-url.com",
)

# Resolution:
# 1. AI's fill is critical (data loss if interrupted)
# 2. But human has higher priority
# 3. System:
#    - Shows notification to human: "AI is filling form, please wait"
#    - Lets AI finish (2 seconds)
#    - Then executes human's navigation
```

### Example 2: Two AIs Cooperate

```python
# AI 1 wants to extract data (read-only)
ai1_intent = Intent(
    participant_id="ai1",
    action="extract",
    target=".project-list",
)

# AI 2 wants to extract different data (read-only)
ai2_intent = Intent(
    participant_id="ai2",
    action="extract",
    target=".user-info",
)

# Resolution:
# Both are read-only, no conflict
# Execute in parallel!
```

### Example 3: Graceful Degradation

```python
# Low-priority AI task
bg_intent = Intent(
    participant_id="background_ai",
    action="scroll",  # Interruptible
)

# Human clicks something
human_intent = Intent(
    participant_id="human",
    action="click",
    target="#important-button",
)

# Resolution:
# 1. Background AI's scroll is paused
# 2. Human's click executes immediately
# 3. After human's action completes, scroll resumes
```

## Benefits

1. **No Interference:** Operations don't conflict
2. **Graceful:** Lower priority tasks pause, not cancel
3. **Transparent:** Everyone sees what's happening
4. **Safe:** Critical operations protected
5. **Efficient:** Compatible operations run in parallel
6. **Fair:** Priority-based resolution
7. **Recoverable:** Rollback on errors

---

**This system ensures smooth AI-human collaboration without chaos!**
