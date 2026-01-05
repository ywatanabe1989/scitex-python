# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/collaboration/persistent_browser.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-19 05:55:00 (ywatanabe)"
# # File: ./src/scitex/browser/collaboration/persistent_browser.py
# # ----------------------------------------
# """
# Persistent Browser - Keep open forever, attach/detach anytime.
# 
# This is the key to true AI-human collaboration!
# 
# Usage:
#     # Terminal 1: Start browser server (once)
#     python3 -m scitex.browser.collaboration.server
# 
#     # Terminal 2+: AI agents attach
#     session = await SharedBrowserSession.attach()
#     await session.navigate("...")
#     # Script ends, browser stays open!
# 
#     # You (human): See everything in browser window, can interact!
# """
# 
# import asyncio
# import os
# import signal
# from pathlib import Path
# from typing import Optional
# 
# from playwright.async_api import async_playwright, Browser, BrowserContext, Page
# 
# 
# class PersistentBrowserServer:
#     """
#     Browser server that stays running.
# 
#     Keeps browser open, allows scripts to attach/detach.
#     """
# 
#     def __init__(
#         self,
#         port: int = 9222,
#         session_dir: Optional[str] = None,
#         browser_type: str = "chromium",
#         headless: bool = False,
#     ):
#         self.port = port
#         self.browser_type = browser_type
#         self.headless = headless
# 
#         # Get session directory from SCITEX_DIR
#         if session_dir:
#             self.session_dir = session_dir
#         else:
#             scitex_dir = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
#             self.session_dir = str(scitex_dir / "browser" / "persistent")
# 
#         self.playwright = None
#         self.context = None
#         self.running = False
# 
#     async def start(self):
#         """
#         Start persistent browser server.
# 
#         Browser stays open until explicitly stopped.
#         """
#         print("=" * 60)
#         print("🌐 Starting Persistent Browser Server")
#         print("=" * 60)
# 
#         self.playwright = await async_playwright().start()
# 
#         # Launch persistent context with remote debugging
#         browser_launcher = getattr(self.playwright, self.browser_type)
# 
#         self.context = await browser_launcher.launch_persistent_context(
#             user_data_dir=self.session_dir,
#             headless=self.headless,
#             args=[
#                 f"--remote-debugging-port={self.port}",
#                 "--disable-blink-features=AutomationControlled",
#             ],
#             viewport={"width": 1920, "height": 1080},
#         )
# 
#         # Create initial page
#         if not self.context.pages:
#             page = await self.context.new_page()
#             await page.goto("about:blank")
# 
#         self.running = True
# 
#         print(f"\n✅ Browser server started!")
#         print(f"   Port: {self.port}")
#         print(f"   CDP URL: http://localhost:{self.port}")
#         print(f"   Session dir: {self.session_dir}")
#         print(f"   Headless: {self.headless}")
#         print(f"\n💡 Scripts can now attach to this browser!")
#         print(f"   Chrome DevTools: chrome://inspect")
#         print(f"\n⏸️  Press Ctrl+C to stop the browser server")
#         print("=" * 60 + "\n")
# 
#         # Handle shutdown gracefully
#         loop = asyncio.get_event_loop()
# 
#         def handle_signal(sig, frame):
#             print("\n\n🛑 Shutting down browser server...")
#             self.running = False
# 
#         signal.signal(signal.SIGINT, handle_signal)
#         signal.signal(signal.SIGTERM, handle_signal)
# 
#         # Keep running
#         try:
#             while self.running:
#                 await asyncio.sleep(1)
#         finally:
#             await self.stop()
# 
#     async def stop(self):
#         """Stop browser server."""
#         if self.context:
#             await self.context.close()
#         if self.playwright:
#             await self.playwright.stop()
# 
#         print("\n✅ Browser server stopped")
# 
# 
# async def attach_to_browser(cdp_url: str = "http://localhost:9222") -> Page:
#     """
#     Attach to running persistent browser.
# 
#     Returns:
#         Page instance from the persistent browser
# 
#     Example:
#         page = await attach_to_browser()
#         await page.goto("https://example.com")
#         # Script ends, browser stays open!
#     """
#     playwright = await async_playwright().start()
# 
#     try:
#         # Connect via Chrome DevTools Protocol
#         browser = await playwright.chromium.connect_over_cdp(cdp_url)
# 
#         # Get context
#         contexts = browser.contexts
#         if not contexts:
#             raise RuntimeError("No browser context available")
# 
#         context = contexts[0]
# 
#         # Get or create page
#         pages = context.pages
#         if pages:
#             page = pages[0]
#         else:
#             page = await context.new_page()
# 
#         print(f"✅ Attached to browser at {cdp_url}")
#         return page
# 
#     except Exception as e:
#         await playwright.stop()
#         raise RuntimeError(f"Failed to attach to browser: {e}")
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/collaboration/persistent_browser.py
# --------------------------------------------------------------------------------
