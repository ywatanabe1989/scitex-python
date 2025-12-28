# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/collaboration/shared_session.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-19 05:20:00 (ywatanabe)"
# # File: ./src/scitex/browser/collaboration/shared_session.py
# # ----------------------------------------
# """
# SharedBrowserSession - Persistent browser session for AI-human collaboration.
# 
# Inspired by scitex.scholar.browser.ScholarBrowserManager's persistent context pattern.
# 
# Key features:
# - Persistent browser that stays open
# - Session continuity (cookies, auth persist)
# - Multiple participants can use same browser
# - Screenshots integrated with scitex.capture
# """
# 
# import asyncio
# import os
# import time
# from datetime import datetime
# from pathlib import Path
# from typing import Optional, List, Callable
# from dataclasses import dataclass, field
# 
# from playwright.async_api import async_playwright, Browser, BrowserContext, Page
# 
# 
# @dataclass
# class SessionConfig:
#     """Configuration for shared browser session."""
# 
#     session_id: str = "default"
#     browser_type: str = "chromium"  # chromium, firefox, webkit
#     headless: bool = False  # False so humans can see
#     viewport: dict = field(default_factory=lambda: {"width": 1920, "height": 1080})
#     user_data_dir: Optional[str] = None  # For persistent profile
#     enable_screenshots: bool = True
#     screenshot_interval: Optional[float] = None  # Auto-screenshot every N seconds
# 
# 
# class SharedBrowserSession:
#     """
#     Persistent browser session shared between AI agents and humans.
# 
#     Simple, incremental start - just the basics.
# 
#     Example:
#         # Start session
#         session = SharedBrowserSession()
#         await session.start()
# 
#         # Navigate
#         await session.navigate("http://127.0.0.1:8000")
# 
#         # Take screenshot
#         screenshot = await session.screenshot()
# 
#         # Keep running
#         await session.wait()
# 
#         # Close when done
#         await session.close()
#     """
# 
#     def __init__(self, config: Optional[SessionConfig] = None):
#         """Initialize shared session."""
#         self.config = config or SessionConfig()
# 
#         # Browser state
#         self.playwright = None
#         self.browser: Optional[Browser] = None
#         self.context: Optional[BrowserContext] = None
#         self.page: Optional[Page] = None
# 
#         # Session state
#         self.running = False
#         self.start_time = None
# 
#         # Simple event log
#         self.events: List[dict] = []
#         self.screenshots: List[str] = []
# 
#         # Get screenshot directory from SCITEX_DIR
#         scitex_dir = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
#         self.screenshot_dir = scitex_dir / "capture"
#         self.screenshot_dir.mkdir(parents=True, exist_ok=True)
# 
#     async def start(self):
#         """
#         Start the persistent browser session.
# 
#         Uses persistent context pattern from ScholarBrowserManager.
#         """
#         if self.running:
#             raise RuntimeError("Session already running")
# 
#         # Get user data directory
#         if self.config.user_data_dir:
#             user_data_dir = self.config.user_data_dir
#         else:
#             scitex_dir = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
#             user_data_dir = str(
#                 scitex_dir / "browser" / "sessions" / self.config.session_id
#             )
# 
#         # Start playwright
#         self.playwright = await async_playwright().start()
# 
#         # Launch persistent context (like ScholarBrowserManager)
#         browser_type = getattr(self.playwright, self.config.browser_type)
# 
#         self.context = await browser_type.launch_persistent_context(
#             user_data_dir=user_data_dir,
#             headless=self.config.headless,
#             viewport=self.config.viewport,
#         )
# 
#         # Get or create page
#         if self.context.pages:
#             self.page = self.context.pages[0]
#         else:
#             self.page = await self.context.new_page()
# 
#         self.running = True
#         self.start_time = time.time()
# 
#         # Log event
#         self._log_event(
#             "session_started",
#             {
#                 "session_id": self.config.session_id,
#                 "user_data_dir": user_data_dir,
#             },
#         )
# 
#         print(f"✅ Shared browser session started: {self.config.session_id}")
#         print(f"   User data: {user_data_dir}")
#         print(f"   Screenshots: {self.screenshot_dir}")
# 
#         # Start auto-screenshot if configured
#         if self.config.screenshot_interval:
#             asyncio.create_task(self._auto_screenshot_loop())
# 
#     async def navigate(
#         self, url: str, wait_until: str = "load", timeout: int = 60000
#     ) -> str:
#         """
#         Navigate to URL.
# 
#         Args:
#             url: URL to navigate to
#             wait_until: "load", "domcontentloaded", or "networkidle" (default: "load")
#             timeout: Timeout in milliseconds (default: 60000)
# 
#         Returns:
#             Current URL after navigation
#         """
#         if not self.running:
#             raise RuntimeError("Session not started. Call start() first.")
# 
#         await self.page.goto(url, wait_until=wait_until, timeout=timeout)
# 
#         self._log_event("navigated", {"url": url})
# 
#         return self.page.url
# 
#     async def screenshot(self, message: str = "") -> str:
#         """
#         Take screenshot and save to scitex.capture directory.
# 
#         Returns:
#             Path to screenshot file
#         """
#         if not self.running:
#             raise RuntimeError("Session not started. Call start() first.")
# 
#         # Generate filename
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
# 
#         url_slug = self.page.url.replace("://", "_").replace("/", "_").replace("?", "_")
#         if len(url_slug) > 50:
#             url_slug = url_slug[:50]
# 
#         if message:
#             filename = f"{timestamp}-{self.config.session_id}-{message}.jpg"
#         else:
#             filename = f"{timestamp}-{self.config.session_id}-{url_slug}.jpg"
# 
#         screenshot_path = self.screenshot_dir / filename
# 
#         # Take screenshot (JPEG format supports quality)
#         await self.page.screenshot(
#             path=str(screenshot_path),
#             type="jpeg",
#             quality=90,
#         )
# 
#         self.screenshots.append(str(screenshot_path))
#         self._log_event("screenshot_taken", {"path": str(screenshot_path)})
# 
#         return str(screenshot_path)
# 
#     async def wait(self, duration: Optional[float] = None):
#         """
#         Wait for duration or until stopped.
# 
#         Args:
#             duration: Seconds to wait, or None to wait indefinitely
#         """
#         if duration:
#             await asyncio.sleep(duration)
#         else:
#             # Wait until closed
#             while self.running:
#                 await asyncio.sleep(1)
# 
#     async def close(self):
#         """Close the session."""
#         if not self.running:
#             return
# 
#         self.running = False
# 
#         # Close page and context
#         if self.page:
#             await self.page.close()
#         if self.context:
#             await self.context.close()
#         if self.playwright:
#             await self.playwright.stop()
# 
#         duration = time.time() - self.start_time if self.start_time else 0
# 
#         self._log_event(
#             "session_closed",
#             {
#                 "duration": duration,
#                 "screenshot_count": len(self.screenshots),
#             },
#         )
# 
#         print(f"✅ Shared browser session closed: {self.config.session_id}")
#         print(f"   Duration: {duration:.1f}s")
#         print(f"   Screenshots: {len(self.screenshots)}")
# 
#     async def __aenter__(self):
#         """Context manager entry."""
#         await self.start()
#         return self
# 
#     async def __aexit__(self, *args):
#         """Context manager exit."""
#         await self.close()
# 
#     def _log_event(self, event_type: str, data: dict):
#         """Simple event logging."""
#         event = {
#             "type": event_type,
#             "data": data,
#             "timestamp": time.time(),
#         }
#         self.events.append(event)
# 
#     async def _auto_screenshot_loop(self):
#         """Automatically take screenshots at intervals."""
#         while self.running:
#             try:
#                 await self.screenshot(message="auto")
#                 await asyncio.sleep(self.config.screenshot_interval)
#             except Exception as e:
#                 print(f"Auto-screenshot error: {e}")
# 
#     def get_info(self) -> dict:
#         """Get session information."""
#         return {
#             "session_id": self.config.session_id,
#             "running": self.running,
#             "current_url": self.page.url if self.page else None,
#             "uptime": time.time() - self.start_time if self.start_time else 0,
#             "screenshot_count": len(self.screenshots),
#             "event_count": len(self.events),
#         }
# 
#     # ========================================
#     # Natural Human-Like Interaction Primitives
#     # ========================================
# 
#     async def type(self, selector: str, text: str, delay: int = 50):
#         """Type text like a human (with delay between keys)."""
#         if not self.running:
#             raise RuntimeError("Session not started")
# 
#         await self.page.type(selector, text, delay=delay)
#         self._log_event("typed", {"selector": selector, "text": text})
# 
#     async def click(self, selector: str):
#         """Click element."""
#         if not self.running:
#             raise RuntimeError("Session not started")
# 
#         await self.page.click(selector)
#         self._log_event("clicked", {"selector": selector})
# 
#     async def hover(self, selector: str):
#         """Hover over element."""
#         await self.page.hover(selector)
# 
#     async def press(self, key: str):
#         """Press keyboard key (Enter, Tab, Escape, etc.)."""
#         await self.page.keyboard.press(key)
#         self._log_event("pressed_key", {"key": key})
# 
#     async def scroll_down(self, amount: int = 500):
#         """Scroll down."""
#         await self.page.evaluate(f"window.scrollBy(0, {amount})")
# 
#     async def scroll_to(self, selector: str):
#         """Scroll element into view."""
#         await self.page.locator(selector).scroll_into_view_if_needed()
# 
#     async def wait_for(self, selector: str, timeout: int = 5000):
#         """Wait for element to appear."""
#         await self.page.wait_for_selector(selector, timeout=timeout)
# 
#     async def wait_for_text(self, text: str, timeout: int = 5000):
#         """Wait for text to appear."""
#         await self.page.wait_for_selector(f"text={text}", timeout=timeout)
# 
#     async def wait_for_url(self, pattern: str, timeout: int = 5000):
#         """Wait for URL to match pattern."""
#         await self.page.wait_for_url(pattern, timeout=timeout)
# 
#     async def get_text(self, selector: str) -> str:
#         """Get text content of element."""
#         element = await self.page.query_selector(selector)
#         return await element.text_content() if element else ""
# 
#     async def get_value(self, selector: str) -> str:
#         """Get value of input element."""
#         return await self.page.input_value(selector)
# 
#     async def is_visible(self, selector: str) -> bool:
#         """Check if element is visible."""
#         return await self.page.is_visible(selector)
# 
#     # ========================================
#     # Interactive Prompts (Ask User in Browser)
#     # ========================================
# 
#     async def ask(self, question: str, options: Optional[List[str]] = None) -> str:
#         """
#         Ask user a question in the browser.
# 
#         Returns user's response.
# 
#         Example:
#             response = await session.ask("What username to use?")
#             response = await session.ask("Choose:", ["Option A", "Option B"])
#         """
#         if options:
#             # Multiple choice
#             result = await self.page.evaluate(f"""
#                 () => {{
#                     const choice = prompt('{question}\\n\\nOptions: {", ".join(options)}');
#                     return choice;
#                 }}
#             """)
#         else:
#             # Free text
#             result = await self.page.evaluate(f"""
#                 () => prompt('{question}')
#             """)
# 
#         self._log_event("user_input", {"question": question, "response": result})
#         return result
# 
#     async def confirm(self, message: str) -> bool:
#         """
#         Ask user for confirmation in browser.
# 
#         Returns True if user clicks OK, False if Cancel.
#         """
#         result = await self.page.evaluate(f"""
#             () => confirm('{message}')
#         """)
# 
#         self._log_event("user_confirmation", {"message": message, "confirmed": result})
#         return result
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/collaboration/shared_session.py
# --------------------------------------------------------------------------------
