#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: /home/ywatanabe/proj/scitex-code/src/scitex/browser/debugging/_visual_cursor.py

"""
Visual cursor and click effects for E2E test feedback.

Provides visual feedback during browser automation:
- Visual cursor indicator that follows mouse movements
- Click ripple effects
- Drag state visualization
- Step progress messages

Works with both async and sync Playwright APIs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from playwright.async_api import Page as AsyncPage
    from playwright.sync_api import Page as SyncPage

# CSS styles for visual effects
VISUAL_EFFECTS_CSS = """
/* Visual cursor indicator */
#_scitex_cursor {
    position: fixed;
    width: 24px;
    height: 24px;
    border: 3px solid #FF4444;
    border-radius: 50%;
    pointer-events: none;
    z-index: 2147483647;
    transform: translate(-50%, -50%);
    transition: all 0.15s ease-out;
    box-shadow: 0 0 15px rgba(255, 68, 68, 0.6);
    display: none;
}
#_scitex_cursor.clicking {
    transform: translate(-50%, -50%) scale(0.6);
    background: rgba(255, 68, 68, 0.4);
    box-shadow: 0 0 25px rgba(255, 68, 68, 0.8);
}
#_scitex_cursor.dragging {
    border-color: #28A745;
    box-shadow: 0 0 15px rgba(40, 167, 69, 0.6);
    width: 28px;
    height: 28px;
}

/* Click ripple effect */
.scitex-click-ripple {
    position: fixed;
    border-radius: 50%;
    border: 3px solid #FF4444;
    pointer-events: none;
    z-index: 2147483646;
    animation: clickRipple 0.5s ease-out forwards;
}
@keyframes clickRipple {
    0% { width: 0; height: 0; opacity: 1; transform: translate(-50%, -50%); }
    100% { width: 80px; height: 80px; opacity: 0; transform: translate(-50%, -50%); }
}

/* Step message container */
#_scitex_step_messages {
    position: fixed;
    top: 10px;
    left: 10px;
    z-index: 2147483647;
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-width: 600px;
    pointer-events: none;
}
.scitex-step-msg {
    background: rgba(0, 0, 0, 0.9);
    color: white;
    padding: 14px 24px;
    border-radius: 8px;
    font-size: 16px;
    font-family: 'Courier New', monospace;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    word-wrap: break-word;
    animation: stepSlideIn 0.3s ease-out;
}
@keyframes stepSlideIn {
    0% { opacity: 0; transform: translateX(-20px); }
    100% { opacity: 1; transform: translateX(0); }
}

/* Test result banner */
#_scitex_result_banner {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    padding: 40px 80px;
    border-radius: 16px;
    font-size: 48px;
    font-weight: bold;
    font-family: 'Arial', sans-serif;
    z-index: 2147483647;
    pointer-events: none;
    animation: resultPulse 0.5s ease-out;
}
#_scitex_result_banner.success {
    background: rgba(40, 167, 69, 0.95);
    color: white;
    box-shadow: 0 0 50px rgba(40, 167, 69, 0.8);
}
#_scitex_result_banner.failure {
    background: rgba(220, 53, 69, 0.95);
    color: white;
    box-shadow: 0 0 50px rgba(220, 53, 69, 0.8);
}
@keyframes resultPulse {
    0% { transform: translate(-50%, -50%) scale(0.5); opacity: 0; }
    50% { transform: translate(-50%, -50%) scale(1.1); }
    100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
}
"""

# JavaScript to inject visual effects
INJECT_EFFECTS_JS = f"""
() => {{
    if (document.getElementById('_scitex_visual_effects')) return;

    const style = document.createElement('style');
    style.id = '_scitex_visual_effects';
    style.textContent = `{VISUAL_EFFECTS_CSS}`;
    document.head.appendChild(style);

    // Create cursor element
    const cursor = document.createElement('div');
    cursor.id = '_scitex_cursor';
    document.body.appendChild(cursor);

    // Create step messages container
    const msgContainer = document.createElement('div');
    msgContainer.id = '_scitex_step_messages';
    document.body.appendChild(msgContainer);
}}
"""


def inject_visual_effects(page: Union["AsyncPage", "SyncPage"]) -> None:
    """Inject CSS and elements for visual effects (sync version)."""
    page.evaluate(INJECT_EFFECTS_JS)


async def inject_visual_effects_async(page: "AsyncPage") -> None:
    """Inject CSS and elements for visual effects (async version)."""
    await page.evaluate(INJECT_EFFECTS_JS)


def show_cursor_at(
    page: Union["AsyncPage", "SyncPage"], x: float, y: float, state: str = "normal"
) -> None:
    """Move visual cursor to position (sync version).

    Args:
        page: Playwright page object
        x: X coordinate
        y: Y coordinate
        state: Cursor state - "normal", "clicking", or "dragging"
    """
    page.evaluate(
        """
    ([x, y, state]) => {
        let cursor = document.getElementById('_scitex_cursor');
        if (!cursor) {
            cursor = document.createElement('div');
            cursor.id = '_scitex_cursor';
            document.body.appendChild(cursor);
        }
        cursor.style.display = 'block';
        cursor.style.left = x + 'px';
        cursor.style.top = y + 'px';
        cursor.className = state === 'clicking' ? 'clicking' :
                          state === 'dragging' ? 'dragging' : '';
        if (state === 'dragging') {
            cursor.style.borderColor = '#28A745';
            cursor.style.boxShadow = '0 0 15px rgba(40, 167, 69, 0.6)';
        } else {
            cursor.style.borderColor = '#FF4444';
            cursor.style.boxShadow = '0 0 15px rgba(255, 68, 68, 0.6)';
        }
    }
    """,
        [x, y, state],
    )


async def show_cursor_at_async(
    page: "AsyncPage", x: float, y: float, state: str = "normal"
) -> None:
    """Move visual cursor to position (async version)."""
    await page.evaluate(
        """
    ([x, y, state]) => {
        let cursor = document.getElementById('_scitex_cursor');
        if (!cursor) {
            cursor = document.createElement('div');
            cursor.id = '_scitex_cursor';
            document.body.appendChild(cursor);
        }
        cursor.style.display = 'block';
        cursor.style.left = x + 'px';
        cursor.style.top = y + 'px';
        cursor.className = state === 'clicking' ? 'clicking' :
                          state === 'dragging' ? 'dragging' : '';
        if (state === 'dragging') {
            cursor.style.borderColor = '#28A745';
            cursor.style.boxShadow = '0 0 15px rgba(40, 167, 69, 0.6)';
        } else {
            cursor.style.borderColor = '#FF4444';
            cursor.style.boxShadow = '0 0 15px rgba(255, 68, 68, 0.6)';
        }
    }
    """,
        [x, y, state],
    )


def show_click_effect(page: Union["AsyncPage", "SyncPage"], x: float, y: float) -> None:
    """Show click ripple effect at position (sync version)."""
    page.evaluate(
        """
    ([x, y]) => {
        const ripple = document.createElement('div');
        ripple.className = 'scitex-click-ripple';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        document.body.appendChild(ripple);
        setTimeout(() => ripple.remove(), 600);

        const cursor = document.getElementById('_scitex_cursor');
        if (cursor) {
            cursor.classList.add('clicking');
            setTimeout(() => cursor.classList.remove('clicking'), 150);
        }
    }
    """,
        [x, y],
    )


async def show_click_effect_async(page: "AsyncPage", x: float, y: float) -> None:
    """Show click ripple effect at position (async version)."""
    await page.evaluate(
        """
    ([x, y]) => {
        const ripple = document.createElement('div');
        ripple.className = 'scitex-click-ripple';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        document.body.appendChild(ripple);
        setTimeout(() => ripple.remove(), 600);

        const cursor = document.getElementById('_scitex_cursor');
        if (cursor) {
            cursor.classList.add('clicking');
            setTimeout(() => cursor.classList.remove('clicking'), 150);
        }
    }
    """,
        [x, y],
    )


def show_step(
    page: Union["AsyncPage", "SyncPage"],
    step: int,
    total: int,
    message: str,
    level: str = "info",
) -> None:
    """Show numbered step message in browser (sync version).

    Args:
        page: Playwright page object
        step: Current step number
        total: Total number of steps
        message: Message to display
        level: Message level - "info", "success", "warning", or "error"
    """
    color_map = {
        "info": "#17A2B8",
        "success": "#28A745",
        "warning": "#FFC107",
        "error": "#DC3545",
    }
    color = color_map.get(level, color_map["info"])

    page.evaluate(
        """
    ([step, total, message, color]) => {
        let container = document.getElementById('_scitex_step_messages');
        if (!container) {
            container = document.createElement('div');
            container.id = '_scitex_step_messages';
            container.style.cssText = `
                position: fixed; top: 10px; left: 10px; z-index: 2147483647;
                display: flex; flex-direction: column; gap: 8px;
                max-width: 600px; pointer-events: none;
            `;
            document.body.appendChild(container);
        }
        const popup = document.createElement('div');
        popup.className = 'scitex-step-msg';
        popup.innerHTML = `<strong>[${step}/${total}] ${message}</strong>`;
        popup.style.cssText = `
            background: rgba(0, 0, 0, 0.9); color: white;
            padding: 14px 24px; border-radius: 8px; font-size: 16px;
            font-family: 'Courier New', monospace;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
            border-left: 6px solid ${color}; word-wrap: break-word;
        `;
        container.appendChild(popup);
        while (container.children.length > 5) container.removeChild(container.firstChild);
        setTimeout(() => { if (popup.parentNode) popup.parentNode.removeChild(popup); }, 8000);
    }
    """,
        [step, total, message, color],
    )
    page.wait_for_timeout(200)


async def show_step_async(
    page: "AsyncPage", step: int, total: int, message: str, level: str = "info"
) -> None:
    """Show numbered step message in browser (async version)."""
    color_map = {
        "info": "#17A2B8",
        "success": "#28A745",
        "warning": "#FFC107",
        "error": "#DC3545",
    }
    color = color_map.get(level, color_map["info"])

    await page.evaluate(
        """
    ([step, total, message, color]) => {
        let container = document.getElementById('_scitex_step_messages');
        if (!container) {
            container = document.createElement('div');
            container.id = '_scitex_step_messages';
            container.style.cssText = `
                position: fixed; top: 10px; left: 10px; z-index: 2147483647;
                display: flex; flex-direction: column; gap: 8px;
                max-width: 600px; pointer-events: none;
            `;
            document.body.appendChild(container);
        }
        const popup = document.createElement('div');
        popup.className = 'scitex-step-msg';
        popup.innerHTML = `<strong>[${step}/${total}] ${message}</strong>`;
        popup.style.cssText = `
            background: rgba(0, 0, 0, 0.9); color: white;
            padding: 14px 24px; border-radius: 8px; font-size: 16px;
            font-family: 'Courier New', monospace;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
            border-left: 6px solid ${color}; word-wrap: break-word;
        `;
        container.appendChild(popup);
        while (container.children.length > 5) container.removeChild(container.firstChild);
        setTimeout(() => { if (popup.parentNode) popup.parentNode.removeChild(popup); }, 8000);
    }
    """,
        [step, total, message, color],
    )
    await page.wait_for_timeout(200)


def show_test_result(
    page: Union["AsyncPage", "SyncPage"],
    success: bool,
    message: str = "",
    delay_ms: int = 3000,
) -> None:
    """Show test result banner (PASS/FAIL) and wait (sync version).

    Args:
        page: Playwright page object
        success: True for PASS, False for FAIL
        message: Optional message to display
        delay_ms: How long to display before continuing
    """
    status = "PASS" if success else "FAIL"
    css_class = "success" if success else "failure"
    display_text = f"{status}" + (f": {message}" if message else "")

    page.evaluate(
        """
    ([displayText, cssClass]) => {
        // Remove existing banner
        const existing = document.getElementById('_scitex_result_banner');
        if (existing) existing.remove();

        const banner = document.createElement('div');
        banner.id = '_scitex_result_banner';
        banner.className = cssClass;
        banner.textContent = displayText;
        document.body.appendChild(banner);
    }
    """,
        [display_text, css_class],
    )
    page.wait_for_timeout(delay_ms)


async def show_test_result_async(
    page: "AsyncPage", success: bool, message: str = "", delay_ms: int = 3000
) -> None:
    """Show test result banner (PASS/FAIL) and wait (async version)."""
    status = "PASS" if success else "FAIL"
    css_class = "success" if success else "failure"
    display_text = f"{status}" + (f": {message}" if message else "")

    await page.evaluate(
        """
    ([displayText, cssClass]) => {
        const existing = document.getElementById('_scitex_result_banner');
        if (existing) existing.remove();

        const banner = document.createElement('div');
        banner.id = '_scitex_result_banner';
        banner.className = cssClass;
        banner.textContent = displayText;
        document.body.appendChild(banner);
    }
    """,
        [display_text, css_class],
    )
    await page.wait_for_timeout(delay_ms)


# EOF
