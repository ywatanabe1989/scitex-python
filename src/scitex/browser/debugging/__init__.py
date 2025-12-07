#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SciTeX Browser Debugging Utilities
# ----------------------------------------

# from ._log_page import log_page_async, BrowserLogger
from ._browser_logger import browser_logger
from ._show_grid import show_grid_async
from ._highlight_element import highlight_element_async
from ._visual_cursor import (
    inject_visual_effects,
    inject_visual_effects_async,
    show_cursor_at,
    show_cursor_at_async,
    show_click_effect,
    show_click_effect_async,
    show_step,
    show_step_async,
    show_test_result,
    show_test_result_async,
)
from ._failure_capture import (
    setup_console_interceptor,
    collect_console_logs,
    collect_console_logs_detailed,
    format_logs_devtools_style,
    save_failure_artifacts,
    create_failure_capture_fixture,
)
from ._test_monitor import (
    TestMonitor,
    create_test_monitor_fixture,
    monitor_test,
)
from ._sync_session import (
    SyncBrowserSession,
    sync_browser_session,
    create_browser_session_fixture,
)

__all__ = [
    "log_page_async",
    "browser_logger",
    "show_grid_async",
    "highlight_element_async",
    # Visual cursor/feedback utilities
    "inject_visual_effects",
    "inject_visual_effects_async",
    "show_cursor_at",
    "show_cursor_at_async",
    "show_click_effect",
    "show_click_effect_async",
    "show_step",
    "show_step_async",
    "show_test_result",
    "show_test_result_async",
    # Failure capture utilities
    "setup_console_interceptor",
    "collect_console_logs",
    "collect_console_logs_detailed",
    "format_logs_devtools_style",
    "save_failure_artifacts",
    "create_failure_capture_fixture",
    # Test monitoring (periodic screenshots via scitex.capture)
    "TestMonitor",
    "create_test_monitor_fixture",
    "monitor_test",
    # Sync browser session for zombie prevention
    "SyncBrowserSession",
    "sync_browser_session",
    "create_browser_session_fixture",
]

# EOF
