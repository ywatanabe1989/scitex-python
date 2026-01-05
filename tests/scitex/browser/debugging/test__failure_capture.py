# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/debugging/_failure_capture.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: 2025-12-08
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/browser/debugging/_failure_capture.py
# 
# """
# Automatic failure capture utilities for Playwright E2E tests.
# 
# Features:
# - Console log collection with source file/line tracking
# - Error interception (JS errors, unhandled promise rejections, resource failures)
# - Screenshot capture on test failure
# - Page HTML capture for debugging
# - DevTools-like formatted output
# - Pytest integration via fixtures
# 
# Based on scitex-cloud's console-interceptor.ts functionality.
# 
# Usage in conftest.py:
#     from scitex.browser.debugging import (
#         setup_console_interceptor,
#         collect_console_logs,
#         save_failure_artifacts,
#         create_failure_capture_fixture,
#     )
# """
# 
# from datetime import datetime
# from pathlib import Path
# from typing import TYPE_CHECKING
# 
# if TYPE_CHECKING:
#     from playwright.sync_api import Page
# 
# 
# # JavaScript code for advanced console interception
# # Mirrors functionality from scitex-cloud/static/shared/ts/utils/console-interceptor.ts
# CONSOLE_INTERCEPTOR_JS = """
# () => {
#     if (window._scitex_console_interceptor_setup) return;
# 
#     // Store for captured logs with full details
#     window._scitex_console_logs = [];
#     window._scitex_console_history = [];
#     const maxHistory = 2000;
# 
#     // Store original console methods
#     const originalConsole = {
#         log: console.log,
#         info: console.info,
#         warn: console.warn,
#         error: console.error,
#         debug: console.debug
#     };
# 
#     // Get source file and line number from stack trace
#     function getSource() {
#         try {
#             const stack = new Error().stack;
#             if (!stack) return '';
#             const lines = stack.split('\\n');
#             // Skip Error, getSource, capture, and intercepted console method
#             for (let i = 4; i < lines.length; i++) {
#                 const line = lines[i];
#                 const match = line.match(/(?:https?:\\/\\/[^\\/]+)?([^\\s]+):(\\d+):(\\d+)/);
#                 if (match) {
#                     const [, file, lineNum, col] = match;
#                     const cleanFile = file.split('/').slice(-2).join('/');
#                     return `${cleanFile}:${lineNum}:${col}`;
#                 }
#             }
#         } catch (e) {}
#         return '';
#     }
# 
#     // Format message from arguments
#     function formatMessage(args) {
#         return args.map(arg => {
#             if (typeof arg === 'object') {
#                 try { return JSON.stringify(arg, null, 2); }
#                 catch { return String(arg); }
#             }
#             return String(arg);
#         }).join(' ');
#     }
# 
#     // Capture log entry
#     function capture(level, args) {
#         const message = formatMessage(args);
#         const source = getSource();
#         const entry = {
#             level,
#             message,
#             source,
#             timestamp: Date.now(),
#             url: window.location.href
#         };
# 
#         window._scitex_console_history.push(entry);
#         if (window._scitex_console_history.length > maxHistory) {
#             window._scitex_console_history.shift();
#         }
# 
#         // Also store simple format for backwards compatibility
#         window._scitex_console_logs.push(`[${level.toUpperCase()}] ${source ? source + ' ' : ''}${message}`);
#         if (window._scitex_console_logs.length > 500) {
#             window._scitex_console_logs.shift();
#         }
#     }
# 
#     // Intercept console methods
#     ['log', 'info', 'warn', 'error', 'debug'].forEach(level => {
#         console[level] = function(...args) {
#             originalConsole[level].apply(console, args);
#             capture(level, args);
#         };
#     });
# 
#     // Capture unhandled JavaScript errors
#     window.addEventListener('error', (event) => {
#         let entry;
#         if (event.target && event.target.tagName) {
#             // Resource loading error
#             const target = event.target;
#             const src = target.src || target.href || '';
#             if (src) {
#                 entry = {
#                     level: 'error',
#                     message: `Failed to load resource: ${src}`,
#                     source: src.split('/').pop() || '',
#                     timestamp: Date.now(),
#                     url: window.location.href
#                 };
#             }
#         } else {
#             // JavaScript error
#             entry = {
#                 level: 'error',
#                 message: event.message,
#                 source: `${event.filename}:${event.lineno}:${event.colno}`,
#                 timestamp: Date.now(),
#                 url: window.location.href
#             };
#         }
#         if (entry) {
#             window._scitex_console_history.push(entry);
#             window._scitex_console_logs.push(`[ERROR] ${entry.source} ${entry.message}`);
#         }
#     }, true);
# 
#     // Capture unhandled promise rejections
#     window.addEventListener('unhandledrejection', (event) => {
#         const entry = {
#             level: 'error',
#             message: `Uncaught (in promise): ${event.reason}`,
#             source: '',
#             timestamp: Date.now(),
#             url: window.location.href
#         };
#         window._scitex_console_history.push(entry);
#         window._scitex_console_logs.push(`[ERROR] Uncaught (in promise): ${event.reason}`);
#     });
# 
#     window._scitex_console_interceptor_setup = true;
# }
# """
# 
# 
# def setup_console_interceptor(page: "Page") -> None:
#     """Set up console log interceptor with source tracking and error capture.
# 
#     Features (mirroring console-interceptor.ts):
#     - Intercepts console.log, info, warn, error, debug
#     - Captures source file and line number
#     - Captures unhandled JS errors
#     - Captures unhandled promise rejections
#     - Captures resource loading failures
# 
#     Call this at the start of each test to begin capturing logs.
#     """
#     try:
#         page.evaluate(CONSOLE_INTERCEPTOR_JS)
#     except Exception:
#         pass
# 
# 
# def collect_console_logs(page: "Page") -> list:
#     """Collect all captured console logs from the browser.
# 
#     Returns:
#         List of log strings in format "[LEVEL] source message"
#     """
#     try:
#         logs = page.evaluate("""
#         () => {
#             if (window._scitex_console_logs) {
#                 return window._scitex_console_logs;
#             }
#             return [];
#         }
#         """)
#         return logs or []
#     except Exception:
#         return []
# 
# 
# def collect_console_logs_detailed(page: "Page") -> list:
#     """Collect all captured console logs with full details.
# 
#     Returns:
#         List of dicts with keys: level, message, source, timestamp, url
#     """
#     try:
#         history = page.evaluate("""
#         () => {
#             if (window._scitex_console_history) {
#                 return window._scitex_console_history;
#             }
#             return [];
#         }
#         """)
#         return history or []
#     except Exception:
#         return []
# 
# 
# def format_logs_devtools_style(logs: list) -> str:
#     """Format logs in DevTools-like style.
# 
#     Args:
#         logs: List of detailed log entries from collect_console_logs_detailed()
# 
#     Returns:
#         Formatted string like browser DevTools output
#     """
#     if not logs:
#         return "No console logs captured."
# 
#     level_icons = {
#         "error": "[ERROR]",
#         "warn": "[WARN]",
#         "info": "[INFO]",
#         "debug": "[DEBUG]",
#         "log": "[LOG]",
#     }
# 
#     output = []
#     for entry in logs:
#         if isinstance(entry, dict):
#             level = entry.get("level", "log")
#             source = entry.get("source", "")
#             message = entry.get("message", "")
#             icon = level_icons.get(level, "[LOG]")
#             source_str = f" {source}" if source else ""
#             output.append(f"{icon}{source_str} {message}")
#         else:
#             output.append(str(entry))
# 
#     return "\n".join(output)
# 
# 
# def save_failure_artifacts(
#     page: "Page",
#     test_name: str,
#     artifacts_dir: Path | str,
#     console_logs: list | None = None,
# ) -> dict:
#     """Save screenshot, console logs, and page HTML on test failure.
# 
#     Args:
#         page: Playwright page object
#         test_name: Name of the failed test (e.g., request.node.nodeid)
#         artifacts_dir: Directory to save artifacts
#         console_logs: Pre-collected console logs (optional, will collect if None)
# 
#     Returns:
#         Dict with paths to saved artifacts
#     """
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     safe_test_name = (
#         test_name.replace("::", "_")
#         .replace("[", "_")
#         .replace("]", "")
#         .replace("/", "_")
#     )
# 
#     # Create artifacts directory with timestamp
#     artifacts_path = Path(artifacts_dir) / timestamp
#     artifacts_path.mkdir(parents=True, exist_ok=True)
# 
#     saved_files = {}
# 
#     # Collect console logs if not provided
#     if console_logs is None:
#         console_logs = collect_console_logs(page)
# 
#     # Save screenshot
#     try:
#         screenshot_path = artifacts_path / f"{safe_test_name}_screenshot.png"
#         page.screenshot(path=str(screenshot_path), full_page=True)
#         saved_files["screenshot"] = screenshot_path
#         print(f"\n[FAILURE] Screenshot saved: {screenshot_path}")
#     except Exception as e:
#         print(f"\n[FAILURE] Failed to save screenshot: {e}")
# 
#     # Save console logs
#     try:
#         logs_path = artifacts_path / f"{safe_test_name}_console.log"
#         with open(logs_path, "w") as f:
#             f.write(f"Test: {test_name}\n")
#             f.write(f"Timestamp: {timestamp}\n")
#             f.write(f"URL: {page.url}\n")
#             f.write("=" * 80 + "\n\n")
#             f.write("Console Logs:\n")
#             f.write("-" * 40 + "\n")
#             for log in console_logs:
#                 f.write(f"{log}\n")
#         saved_files["console_logs"] = logs_path
#         print(f"[FAILURE] Console logs saved: {logs_path}")
#     except Exception as e:
#         print(f"[FAILURE] Failed to save console logs: {e}")
# 
#     # Save page HTML
#     try:
#         html_path = artifacts_path / f"{safe_test_name}_page.html"
#         html_content = page.content()
#         with open(html_path, "w") as f:
#             f.write(html_content)
#         saved_files["page_html"] = html_path
#         print(f"[FAILURE] Page HTML saved: {html_path}")
#     except Exception as e:
#         print(f"[FAILURE] Failed to save page HTML: {e}")
# 
#     return saved_files
# 
# 
# def create_failure_capture_fixture(artifacts_dir: Path | str):
#     """Create a pytest fixture for automatic failure capture.
# 
#     Usage in conftest.py:
#         from scitex.browser.debugging import create_failure_capture_fixture
# 
#         capture_on_failure = create_failure_capture_fixture(
#             Path(__file__).parent / "artifacts"
#         )
# 
#     Args:
#         artifacts_dir: Directory to save failure artifacts
# 
#     Returns:
#         A pytest fixture function
#     """
#     import pytest
# 
#     @pytest.fixture(autouse=True)
#     def capture_on_failure(request, page):
#         """Automatically capture console logs and screenshot on test failure."""
#         setup_console_interceptor(page)
#         yield
#         if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
#             console_logs = collect_console_logs(page)
#             save_failure_artifacts(
#                 page, request.node.nodeid, artifacts_dir, console_logs
#             )
# 
#     return capture_on_failure
# 
# 
# # Pytest hook for capturing test results - add to conftest.py
# PYTEST_HOOK_CODE = '''
# @pytest.hookimpl(tryfirst=True, hookwrapper=True)
# def pytest_runtest_makereport(item, call):
#     """Hook to capture test outcome for use in fixture."""
#     outcome = yield
#     rep = outcome.get_result()
#     setattr(item, f"rep_{rep.when}", rep)
# '''
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/debugging/_failure_capture.py
# --------------------------------------------------------------------------------
