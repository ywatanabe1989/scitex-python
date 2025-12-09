#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-19 06:18:00 (ywatanabe)"
# File: ./src/scitex/browser/collaboration/standard_interactions.py
# ----------------------------------------
"""
Standardized user interaction patterns for smooth AI-human collaboration.

Provides consistent, reusable interaction patterns.
"""

from enum import Enum
from typing import Optional, Any
from playwright.async_api import Page


class UserResponse(Enum):
    """Standardized user responses."""

    YES = "yes"
    NO = "no"
    ACCEPT = "accept"
    REJECT = "reject"
    SKIP = "skip"
    RETRY = "retry"
    CANCEL = "cancel"
    MEMORIZE_SESSION = "memorize_session"  # Remember for this session
    MEMORIZE_PERSISTENT = "memorize_persistent"  # Remember forever
    HINT = "hint"  # Show me a hint
    HELP = "help"  # Show help
    CUSTOM = "custom"  # User provides custom input


class StandardInteractions:
    """
    Standardized interaction patterns.

    Makes AI-human communication smooth and predictable.
    """

    def __init__(self, page: Page):
        self.page = page
        self.initialized = False

    async def initialize(self):
        """Initialize standard interactions UI."""
        if self.initialized:
            return

        await self.page.add_init_script("""
            () => {
                window.scitexStandard = {
                    // Show choice dialog with standard options
                    askChoice: function(question, options) {
                        return new Promise((resolve) => {
                            const overlay = document.createElement('div');
                            overlay.id = 'scitex-choice-overlay';
                            overlay.style.cssText = `
                                position: fixed;
                                top: 0;
                                left: 0;
                                right: 0;
                                bottom: 0;
                                background: rgba(0, 0, 0, 0.5);
                                z-index: 2147483647;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                            `;

                            const dialog = document.createElement('div');
                            dialog.style.cssText = `
                                background: white;
                                padding: 24px;
                                border-radius: 12px;
                                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                                max-width: 500px;
                                width: 90%;
                            `;

                            const questionEl = document.createElement('div');
                            questionEl.style.cssText = `
                                font-size: 16px;
                                font-weight: 500;
                                margin-bottom: 20px;
                                color: #333;
                            `;
                            questionEl.textContent = question;

                            const buttonsContainer = document.createElement('div');
                            buttonsContainer.style.cssText = `
                                display: grid;
                                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                                gap: 8px;
                            `;

                            const buttonStyles = {
                                yes: { bg: '#4CAF50', text: '✓ Yes' },
                                no: { bg: '#F44336', text: '✗ No' },
                                accept: { bg: '#4CAF50', text: '✓ Accept' },
                                reject: { bg: '#F44336', text: '✗ Reject' },
                                skip: { bg: '#9E9E9E', text: '→ Skip' },
                                retry: { bg: '#FF9800', text: '↻ Retry' },
                                cancel: { bg: '#757575', text: '✗ Cancel' },
                                memorize_session: { bg: '#2196F3', text: '💾 This Session' },
                                memorize_persistent: { bg: '#673AB7', text: '💾 Forever' },
                                hint: { bg: '#00BCD4', text: '💡 Hint' },
                                help: { bg: '#607D8B', text: '❓ Help' },
                                custom: { bg: '#795548', text: '✏️ Custom' },
                            };

                            options.forEach(option => {
                                const btn = document.createElement('button');
                                const style = buttonStyles[option] || { bg: '#9E9E9E', text: option };

                                btn.style.cssText = `
                                    padding: 10px 16px;
                                    background: ${style.bg};
                                    color: white;
                                    border: none;
                                    border-radius: 6px;
                                    cursor: pointer;
                                    font-size: 13px;
                                    font-weight: 500;
                                    transition: transform 0.1s, box-shadow 0.1s;
                                `;

                                btn.textContent = style.text;
                                btn.value = option;

                                btn.onmouseover = () => {
                                    btn.style.transform = 'translateY(-2px)';
                                    btn.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
                                };

                                btn.onmouseout = () => {
                                    btn.style.transform = 'translateY(0)';
                                    btn.style.boxShadow = 'none';
                                };

                                btn.onclick = () => {
                                    overlay.remove();
                                    resolve(option);
                                };

                                buttonsContainer.appendChild(btn);
                            });

                            dialog.appendChild(questionEl);
                            dialog.appendChild(buttonsContainer);
                            overlay.appendChild(dialog);
                            document.body.appendChild(overlay);
                        });
                    },

                    // Quick yes/no
                    askYesNo: function(question) {
                        return this.askChoice(question, ['yes', 'no']);
                    },

                    // Accept/Reject with memory options
                    askWithMemory: function(question) {
                        return this.askChoice(question, [
                            'accept',
                            'reject',
                            'memorize_session',
                            'memorize_persistent'
                        ]);
                    },

                    // Action approval
                    askApproval: function(action) {
                        return this.askChoice(`Approve: ${action}?`, [
                            'accept',
                            'reject',
                            'hint',
                            'custom'
                        ]);
                    },
                };

                console.log('✅ Standard interactions initialized');
            }
        """)

        self.initialized = True
        print("✅ Standard interactions initialized")

    async def ask_yes_no(self, question: str) -> bool:
        """
        Ask yes/no question.

        Returns:
            True for yes, False for no
        """
        result = await self.page.evaluate(
            f"window.scitexStandard?.askYesNo('{self._escape_js(question)}')"
        )
        return result == "yes"

    async def ask_choice(self, question: str, options: list) -> str:
        """
        Ask user to choose from options.

        Args:
            question: Question to ask
            options: List of option keys (yes, no, accept, etc.)

        Returns:
            Selected option key
        """
        options_json = str(options).replace("'", '"')
        result = await self.page.evaluate(
            f"window.scitexStandard?.askChoice('{self._escape_js(question)}', {options_json})"
        )
        return result

    async def ask_with_memory(self, question: str) -> str:
        """
        Ask with memory options.

        Returns:
            'accept', 'reject', 'memorize_session', or 'memorize_persistent'
        """
        result = await self.page.evaluate(
            f"window.scitexStandard?.askWithMemory('{self._escape_js(question)}')"
        )
        return result

    async def ask_approval(self, action: str) -> str:
        """
        Ask approval for action.

        Returns:
            'accept', 'reject', 'hint', or 'custom'
        """
        result = await self.page.evaluate(
            f"window.scitexStandard?.askApproval('{self._escape_js(action)}')"
        )
        return result

    def _escape_js(self, text: str) -> str:
        """Escape text for JavaScript."""
        return text.replace("'", "\\'").replace("\n", "\\n")


# EOF
