# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/debugging/_sync_session.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: 2025-12-08
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/browser/debugging/_sync_session.py
# 
# """
# Sync browser session context manager for pytest-playwright E2E tests.
# 
# Ensures proper cleanup of browser processes to prevent zombies.
# 
# Usage in conftest.py:
#     from scitex.browser import SyncBrowserSession
# 
#     @pytest.fixture
#     def browser_session(page: Page):
#         with SyncBrowserSession(page) as session:
#             yield session
#         # Cleanup happens automatically even on exceptions
# 
# Or use the fixture factory:
#     from scitex.browser import create_browser_session_fixture
#     browser_session = create_browser_session_fixture()
# """
# 
# import atexit
# import os
# import signal
# import subprocess
# from contextlib import contextmanager
# from typing import TYPE_CHECKING, Callable, Optional
# 
# if TYPE_CHECKING:
#     from playwright.sync_api import Page
# 
# 
# class SyncBrowserSession:
#     """
#     Sync context manager for playwright browser sessions.
# 
#     Ensures zombie process cleanup on test failures, timeouts, or crashes.
#     Tracks browser PIDs and kills orphaned processes on exit.
#     """
# 
#     # Class-level tracking of active sessions for emergency cleanup
#     _active_sessions: list["SyncBrowserSession"] = []
#     _cleanup_registered = False
# 
#     def __init__(
#         self,
#         page: "Page",
#         timeout: int = 60,
#         on_enter: Optional[Callable[["Page"], None]] = None,
#         on_exit: Optional[Callable[["Page", bool], None]] = None,
#     ):
#         """
#         Initialize sync browser session.
# 
#         Args:
#             page: Playwright page instance from pytest-playwright
#             timeout: Default timeout for operations in seconds
#             on_enter: Callback when entering context
#             on_exit: Callback when exiting context (receives page and success flag)
#         """
#         self.page = page
#         self.timeout = timeout
#         self.on_enter = on_enter
#         self.on_exit = on_exit
#         self._browser_pid = None
#         self._context_pid = None
#         self._success = True
# 
#         # Register class-level emergency cleanup
#         if not SyncBrowserSession._cleanup_registered:
#             atexit.register(SyncBrowserSession._emergency_cleanup)
#             SyncBrowserSession._cleanup_registered = True
# 
#     def __enter__(self) -> "SyncBrowserSession":
#         """Enter context - track browser PIDs and run setup callback."""
#         # Track this session
#         SyncBrowserSession._active_sessions.append(self)
# 
#         # Try to get browser PID for tracking
#         try:
#             if self.page.context.browser:
#                 # Get the browser process
#                 browser = self.page.context.browser
#                 # Browser PID is available via internal _impl
#                 if hasattr(browser, "_impl"):
#                     impl = browser._impl
#                     if hasattr(impl, "_process"):
#                         self._browser_pid = impl._process.pid
#         except Exception:
#             pass  # PID tracking is best-effort
# 
#         # Run setup callback
#         if self.on_enter:
#             self.on_enter(self.page)
# 
#         return self
# 
#     def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
#         """Exit context - ensure cleanup happens."""
#         self._success = exc_type is None
# 
#         # Remove from active sessions
#         try:
#             SyncBrowserSession._active_sessions.remove(self)
#         except ValueError:
#             pass
# 
#         # Run exit callback
#         if self.on_exit:
#             try:
#                 self.on_exit(self.page, self._success)
#             except Exception:
#                 pass  # Don't fail on callback errors
# 
#         # If there was an exception, try to close gracefully
#         if exc_type is not None:
#             try:
#                 self.page.close()
#             except Exception:
#                 pass
# 
#             try:
#                 self.page.context.close()
#             except Exception:
#                 pass
# 
#         # Kill orphaned browser process if we have the PID
#         if self._browser_pid and not self._success:
#             self._kill_process_tree(self._browser_pid)
# 
#         # Don't suppress the exception
#         return False
# 
#     @staticmethod
#     def _kill_process_tree(pid: int):
#         """Kill a process and all its children (zombies)."""
#         try:
#             # Try SIGTERM first
#             os.kill(pid, signal.SIGTERM)
#         except ProcessLookupError:
#             return  # Already dead
#         except PermissionError:
#             return  # Can't kill
# 
#         # Give it a moment
#         import time
# 
#         time.sleep(0.5)
# 
#         # Force kill if still running
#         try:
#             os.kill(pid, signal.SIGKILL)
#         except (ProcessLookupError, PermissionError):
#             pass
# 
#     @classmethod
#     def _emergency_cleanup(cls):
#         """Emergency cleanup of all active sessions on process exit."""
#         for session in cls._active_sessions[:]:  # Copy list to avoid mutation
#             if session._browser_pid:
#                 cls._kill_process_tree(session._browser_pid)
#         cls._active_sessions.clear()
# 
#     @staticmethod
#     def kill_zombie_browsers():
#         """Kill all zombie chromium/chrome processes from failed tests.
# 
#         Call this at the start of test sessions to clean up from previous runs.
#         """
#         try:
#             # Find orphaned chromium processes
#             result = subprocess.run(
#                 ["pgrep", "-f", "chromium|chrome"],
#                 capture_output=True,
#                 text=True,
#             )
#             if result.returncode == 0:
#                 pids = result.stdout.strip().split("\n")
#                 for pid in pids:
#                     if pid:
#                         try:
#                             os.kill(int(pid), signal.SIGKILL)
#                         except (ProcessLookupError, PermissionError, ValueError):
#                             pass
#         except FileNotFoundError:
#             pass  # pgrep not available
# 
# 
# @contextmanager
# def sync_browser_session(
#     page: "Page",
#     timeout: int = 60,
#     on_enter: Optional[Callable[["Page"], None]] = None,
#     on_exit: Optional[Callable[["Page", bool], None]] = None,
# ):
#     """
#     Context manager for sync playwright sessions.
# 
#     Usage:
#         with sync_browser_session(page) as session:
#             session.page.goto(url)
#             # ... test code
#         # Cleanup happens automatically
#     """
#     session = SyncBrowserSession(page, timeout, on_enter, on_exit)
#     with session:
#         yield session
# 
# 
# def create_browser_session_fixture(
#     timeout: int = 60,
#     setup: Optional[Callable[["Page"], None]] = None,
#     teardown: Optional[Callable[["Page", bool], None]] = None,
#     kill_zombies_on_start: bool = True,
# ):
#     """
#     Create a pytest fixture for browser session with cleanup.
# 
#     Usage in conftest.py:
#         from scitex.browser import create_browser_session_fixture
# 
#         browser_session = create_browser_session_fixture(
#             timeout=60,
#             setup=lambda page: print(f"Starting test"),
#             teardown=lambda page, success: print(f"Test {'passed' if success else 'failed'}"),
#             kill_zombies_on_start=True,
#         )
# 
#     Args:
#         timeout: Default timeout for operations
#         setup: Callback when entering session
#         teardown: Callback when exiting (receives page and success flag)
#         kill_zombies_on_start: Kill orphaned browsers before first test
# 
#     Returns:
#         A pytest fixture function
#     """
#     import pytest
# 
#     _zombies_cleaned = False
# 
#     @pytest.fixture
#     def browser_session(page: "Page"):
#         nonlocal _zombies_cleaned
# 
#         # Clean up zombies from previous runs (once per session)
#         if kill_zombies_on_start and not _zombies_cleaned:
#             SyncBrowserSession.kill_zombie_browsers()
#             _zombies_cleaned = True
# 
#         with SyncBrowserSession(page, timeout, setup, teardown) as session:
#             yield session
# 
#     return browser_session
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/debugging/_sync_session.py
# --------------------------------------------------------------------------------
