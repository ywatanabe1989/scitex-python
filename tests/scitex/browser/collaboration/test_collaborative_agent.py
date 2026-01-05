# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/collaboration/collaborative_agent.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-19 06:10:00 (ywatanabe)"
# # File: ./src/scitex/browser/collaboration/collaborative_agent.py
# # ----------------------------------------
# """
# Collaborative Agent - Watches browser and responds to user actions.
# 
# Runs continuously, responds to:
# - User clicks panel buttons
# - User fills panel fields
# - User navigates
# - DOM changes
# 
# This is the "watching agent" pattern for true collaboration!
# """
# 
# import asyncio
# import signal
# from typing import Optional, Callable, Dict
# from playwright.async_api import Page
# 
# from scitex.browser.debugging import browser_logger
# from .interactive_panel import InteractivePanel
# 
# 
# class CollaborativeAgent:
#     """
#     Agent that watches browser and responds to user.
# 
#     Runs continuously, reacts to user input in real-time.
# 
#     Example:
#         agent = CollaborativeAgent(page)
# 
#         # Define what to do when user provides email
#         @agent.on_input("email")
#         async def handle_email(email):
#             print(f"Got email: {email}")
#             # Automatically navigate to login
# 
#         await agent.run()  # Runs forever, watching
#     """
# 
#     def __init__(
#         self,
#         page: Page,
#         panel: InteractivePanel,
#         check_interval: float = 0.5,
#     ):
#         self.page = page
#         self.panel = panel
#         self.check_interval = check_interval
# 
#         self.running = False
#         self.handlers: Dict[str, Callable] = {}
# 
#         # State
#         self.last_url = None
#         self.last_panel_data = {}
# 
#     def on_input(self, key: str):
#         """Decorator to register handler for input."""
# 
#         def decorator(func: Callable):
#             self.handlers[key] = func
#             return func
# 
#         return decorator
# 
#     def on_url_change(self, func: Callable):
#         """Decorator to register handler for URL changes."""
#         self.handlers["__url_change__"] = func
#         return func
# 
#     async def run(self):
#         """
#         Run agent - watch browser continuously.
# 
#         Responds to user actions automatically.
#         """
#         self.running = True
# 
#         print("🤖 Collaborative agent started")
#         print("   Watching browser for user actions...")
#         print("   Press Ctrl+C to stop")
# 
#         # Signal handling
#         loop = asyncio.get_event_loop()
# 
#         def stop_agent(sig, frame):
#             print("\n\n🛑 Stopping agent...")
#             self.running = False
# 
#         signal.signal(signal.SIGINT, stop_agent)
#         signal.signal(signal.SIGTERM, stop_agent)
# 
#         # Watch loop
#         try:
#             while self.running:
#                 await self._check_changes()
#                 await asyncio.sleep(self.check_interval)
# 
#         except Exception as e:
#             print(f"❌ Agent error: {e}")
# 
#         finally:
#             print("✅ Agent stopped")
# 
#     async def _check_changes(self):
#         """Check for changes and trigger handlers."""
# 
#         # Check URL change
#         current_url = self.page.url
#         if current_url != self.last_url:
#             self.last_url = current_url
# 
#             if "__url_change__" in self.handlers:
#                 await self.handlers["__url_change__"](current_url)
# 
#         # Check panel data changes
#         current_data = await self.page.evaluate("window.scitexPanel?.data || {}")
# 
#         for key, value in current_data.items():
#             # New or changed value
#             if key not in self.last_panel_data or self.last_panel_data[key] != value:
#                 self.last_panel_data[key] = value
# 
#                 # Trigger handler if registered
#                 if key in self.handlers:
#                     await self.handlers[key](value)
# 
#     async def stop(self):
#         """Stop the agent."""
#         self.running = False
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/collaboration/collaborative_agent.py
# --------------------------------------------------------------------------------
