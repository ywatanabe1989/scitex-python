#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SciTeX Browser Utilities - Universal Playwright helpers organized by category
# ----------------------------------------

# Debugging utilities
from .debugging import (
    browser_logger,
    show_grid_async,
    highlight_element_async,
    # Visual cursor/feedback utilities (sync and async)
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
    # Failure capture utilities (mirrors console-interceptor.ts)
    setup_console_interceptor,
    collect_console_logs,
    collect_console_logs_detailed,
    format_logs_devtools_style,
    save_failure_artifacts,
    create_failure_capture_fixture,
    # Test monitoring (periodic screenshots via scitex.capture)
    TestMonitor,
    create_test_monitor_fixture,
    monitor_test,
    # Sync browser session for zombie prevention
    SyncBrowserSession,
    sync_browser_session,
    create_browser_session_fixture,
)

# PDF utilities
from .pdf import (
    detect_chrome_pdf_viewer_async,
    click_download_for_chrome_pdf_viewer_async,
)

# Interaction utilities
from .interaction import (
    click_center_async,
    click_with_fallbacks_async,
    fill_with_fallbacks_async,
    PopupHandler,
    close_popups_async,
    ensure_no_popups_async,
)

__all__ = [
    # Debugging
    "browser_logger",
    "show_grid_async",
    "highlight_element_async",
    # Visual cursor/feedback (sync)
    "inject_visual_effects",
    "show_cursor_at",
    "show_click_effect",
    "show_step",
    "show_test_result",
    # Visual cursor/feedback (async)
    "inject_visual_effects_async",
    "show_cursor_at_async",
    "show_click_effect_async",
    "show_step_async",
    "show_test_result_async",
    # Failure capture utilities (mirrors console-interceptor.ts)
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
    # PDF
    "detect_chrome_pdf_viewer_async",
    "click_download_for_chrome_pdf_viewer_async",
    # Interaction
    "click_center_async",
    "click_with_fallbacks_async",
    "fill_with_fallbacks_async",
    "PopupHandler",
    "close_popups_async",
    "ensure_no_popups_async",
]

# EOF
