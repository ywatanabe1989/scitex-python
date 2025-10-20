#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-19 05:42:00 (ywatanabe)"
# File: ./src/scitex/browser/collaboration/visual_feedback.py
# ----------------------------------------
"""
Visual feedback for collaborative browser sessions.

Shows who's controlling the browser and what they're doing.
"""

from playwright.async_api import Page
from typing import Optional


class VisualFeedback:
    """
    Simple visual feedback overlay on browser.

    Shows:
    - Who's in control
    - What action is being performed
    - Status messages
    """

    def __init__(self, page: Page):
        self.page = page
        self.initialized = False

    async def initialize(self):
        """Initialize visual feedback overlay."""
        if self.initialized:
            return

        await self.page.add_init_script("""
            () => {
                // Create feedback container
                const container = document.createElement('div');
                container.id = 'scitex-feedback-container';
                container.style.cssText = `
                    position: fixed;
                    top: 10px;
                    right: 10px;
                    z-index: 2147483647;
                    font-family: system-ui, -apple-system, sans-serif;
                    pointer-events: none;
                `;
                document.body?.appendChild(container) || window.addEventListener('load', () => {
                    document.body.appendChild(container);
                });

                // Make globally accessible
                window.scitexFeedback = {
                    show: function(message, type = 'info', duration = 3000) {
                        const colors = {
                            'info': '#2196F3',
                            'success': '#4CAF50',
                            'warning': '#FF9800',
                            'error': '#F44336',
                            'agent': '#9C27B0',
                            'human': '#00BCD4',
                        };

                        const badge = document.createElement('div');
                        badge.style.cssText = `
                            background: ${colors[type] || colors.info};
                            color: white;
                            padding: 10px 16px;
                            border-radius: 8px;
                            margin-bottom: 8px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                            animation: slideIn 0.3s ease-out;
                            font-size: 13px;
                            max-width: 300px;
                        `;
                        badge.textContent = message;

                        const container = document.getElementById('scitex-feedback-container');
                        if (container) {
                            container.appendChild(badge);

                            if (duration > 0) {
                                setTimeout(() => {
                                    badge.style.animation = 'slideOut 0.3s ease-in';
                                    setTimeout(() => badge.remove(), 300);
                                }, duration);
                            }
                        }
                    },

                    showParticipant: function(name, type = 'agent') {
                        const indicator = document.createElement('div');
                        indicator.id = `scitex-participant-${name}`;
                        indicator.style.cssText = `
                            background: ${type === 'agent' ? '#9C27B0' : '#00BCD4'};
                            color: white;
                            padding: 6px 12px;
                            border-radius: 16px;
                            margin-bottom: 6px;
                            font-size: 12px;
                            font-weight: 500;
                        `;
                        indicator.textContent = `${type === 'agent' ? '🤖' : '👤'} ${name}`;

                        const container = document.getElementById('scitex-feedback-container');
                        if (container) {
                            container.appendChild(indicator);
                        }
                    },

                    clear: function() {
                        const container = document.getElementById('scitex-feedback-container');
                        if (container) {
                            container.innerHTML = '';
                        }
                    }
                };

                // Add animations
                const style = document.createElement('style');
                style.textContent = `
                    @keyframes slideIn {
                        from {
                            transform: translateX(400px);
                            opacity: 0;
                        }
                        to {
                            transform: translateX(0);
                            opacity: 1;
                        }
                    }
                    @keyframes slideOut {
                        from {
                            transform: translateX(0);
                            opacity: 1;
                        }
                        to {
                            transform: translateX(400px);
                            opacity: 0;
                        }
                    }
                `;
                document.head?.appendChild(style) || window.addEventListener('load', () => {
                    document.head.appendChild(style);
                });
            }
        """)

        self.initialized = True

    async def show_message(
        self,
        message: str,
        type: str = "info",  # info, success, warning, error, agent, human
        duration: int = 3000,
    ):
        """Show feedback message."""
        await self.page.evaluate(
            f"window.scitexFeedback?.show('{message}', '{type}', {duration})"
        )

    async def show_participant(self, name: str, type: str = "agent"):
        """Show participant indicator."""
        await self.page.evaluate(
            f"window.scitexFeedback?.showParticipant('{name}', '{type}')"
        )

    async def show_action(self, participant_name: str, action: str):
        """Show what participant is doing."""
        await self.show_message(
            f"{participant_name} is {action}...",
            type="agent" if "agent" in participant_name.lower() else "human",
            duration=2000,
        )

    async def clear(self):
        """Clear all feedback."""
        await self.page.evaluate("window.scitexFeedback?.clear()")


# EOF
