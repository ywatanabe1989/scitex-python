# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/collaboration/interactive_panel.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-19 06:05:00 (ywatanabe)"
# # File: ./src/scitex/browser/collaboration/interactive_panel.py
# # ----------------------------------------
# """
# Interactive Control Panel - UI in browser for AI-human interaction.
# 
# User can:
# - Input credentials
# - Choose options
# - Give commands
# - See AI's intentions
# - Approve/reject actions
# 
# AI can:
# - Ask for input
# - Show intentions
# - Request approval
# - Remember user preferences
# """
# 
# import json
# import os
# from pathlib import Path
# from typing import Optional, Dict, Any
# from playwright.async_api import Page
# 
# 
# class InteractivePanel:
#     """
#     Interactive control panel injected into browser.
# 
#     Features:
#     - Input fields for credentials
#     - Action approval buttons
#     - Session memory (remembers during session)
#     - Persistent memory (remembers between sessions)
#     - Two-way communication (AI ↔ Human)
#     """
# 
#     def __init__(
#         self,
#         page: Page,
#         session_id: str = "default",
#         enable_persistence: bool = True,
#     ):
#         self.page = page
#         self.session_id = session_id
#         self.enable_persistence = enable_persistence
# 
#         # Memory
#         self.session_memory: Dict[str, Any] = {}  # Lost when session ends
#         self.persistent_memory_file = self._get_memory_file()
# 
#         # Load persistent memory
#         if self.enable_persistence:
#             self._load_persistent_memory()
# 
#     def _get_memory_file(self) -> Path:
#         """Get path to persistent memory file."""
#         scitex_dir = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
#         memory_dir = scitex_dir / "browser" / "memory"
#         memory_dir.mkdir(parents=True, exist_ok=True)
#         return memory_dir / f"{self.session_id}_memory.json"
# 
#     def _load_persistent_memory(self):
#         """Load persistent memory from disk."""
#         if self.persistent_memory_file.exists():
#             with open(self.persistent_memory_file, "r") as f:
#                 self.session_memory = json.load(f)
# 
#     def _save_persistent_memory(self):
#         """Save persistent memory to disk."""
#         if self.enable_persistence:
#             with open(self.persistent_memory_file, "w") as f:
#                 json.dump(self.session_memory, f, indent=2)
# 
#     async def initialize(self):
#         """Inject interactive panel into browser."""
#         await self.page.evaluate("""
#             () => {
#                 // Create panel container
#                 const panel = document.createElement('div');
#                 panel.id = 'scitex-interactive-panel';
#                 panel.style.cssText = `
#                     position: fixed;
#                     top: 50px;
#                     right: 20px;
#                     width: 320px;
#                     background: white;
#                     border: 2px solid #2196F3;
#                     border-radius: 12px;
#                     box-shadow: 0 4px 20px rgba(0,0,0,0.3);
#                     z-index: 2147483646;
#                     font-family: system-ui, -apple-system, sans-serif;
#                     max-height: 80vh;
#                     overflow-y: auto;
#                     cursor: move;
#                 `;
# 
#                 // Make draggable
#                 let isDragging = false;
#                 let currentX, currentY, initialX, initialY;
# 
#                 panel.addEventListener('mousedown', (e) => {
#                     if (e.target.closest('#panel-content')) return; // Don't drag when clicking content
#                     isDragging = true;
#                     initialX = e.clientX - panel.offsetLeft;
#                     initialY = e.clientY - panel.offsetTop;
#                 });
# 
#                 document.addEventListener('mousemove', (e) => {
#                     if (!isDragging) return;
#                     e.preventDefault();
#                     currentX = e.clientX - initialX;
#                     currentY = e.clientY - initialY;
#                     panel.style.left = currentX + 'px';
#                     panel.style.top = currentY + 'px';
#                     panel.style.right = 'auto'; // Remove right positioning
#                 });
# 
#                 document.addEventListener('mouseup', () => {
#                     isDragging = false;
#                 });
# 
#                 panel.innerHTML = `
#                     <div style="background: #2196F3; color: white; padding: 12px; border-radius: 10px 10px 0 0; font-weight: 600; cursor: move;">
#                         🤖 Claude Control Panel <span style="float: right; opacity: 0.7;">⋮⋮</span>
#                     </div>
#                     <div id="panel-content" style="padding: 16px;">
#                         <div style="color: #666; font-size: 12px; margin-bottom: 12px;">
#                             AI-Human Collaboration Interface
#                         </div>
#                         <div id="panel-messages"></div>
#                         <div id="panel-inputs"></div>
#                         <div id="panel-actions"></div>
#                     </div>
#                 `;
# 
#                 document.body.appendChild(panel);
# 
#                 // Global API
#                 window.scitexPanel = {
#                     data: {},
# 
#                     showMessage: function(message, type = 'info') {
#                         const colors = {
#                             info: '#2196F3',
#                             success: '#4CAF50',
#                             warning: '#FF9800',
#                             error: '#F44336',
#                         };
# 
#                         const msgDiv = document.createElement('div');
#                         msgDiv.style.cssText = `
#                             background: ${colors[type] || colors.info};
#                             color: white;
#                             padding: 8px 12px;
#                             border-radius: 6px;
#                             margin-bottom: 8px;
#                             font-size: 13px;
#                         `;
#                         msgDiv.textContent = message;
# 
#                         const container = document.getElementById('panel-messages');
#                         container.appendChild(msgDiv);
#                         container.scrollTop = container.scrollHeight;
# 
#                         // Keep last 5 messages
#                         while (container.children.length > 5) {
#                             container.removeChild(container.firstChild);
#                         }
#                     },
# 
#                     askInput: function(key, label, type = 'text', remember = true) {
#                         return new Promise((resolve) => {
#                             const container = document.getElementById('panel-inputs');
#                             container.innerHTML = '';
# 
#                             const inputGroup = document.createElement('div');
#                             inputGroup.style.marginBottom = '12px';
# 
#                             const labelEl = document.createElement('label');
#                             labelEl.textContent = label;
#                             labelEl.style.cssText = 'display: block; margin-bottom: 4px; font-size: 13px; font-weight: 500;';
# 
#                             const input = document.createElement('input');
#                             input.type = type;
#                             input.id = `input-${key}`;
#                             input.style.cssText = `
#                                 width: 100%;
#                                 padding: 8px;
#                                 border: 1px solid #ddd;
#                                 border-radius: 4px;
#                                 font-size: 13px;
#                                 box-sizing: border-box;
#                             `;
# 
#                             // Load from memory if exists
#                             if (window.scitexPanel.data[key]) {
#                                 input.value = window.scitexPanel.data[key];
#                             }
# 
#                             const btnContainer = document.createElement('div');
#                             btnContainer.style.cssText = 'display: flex; gap: 8px; margin-top: 8px;';
# 
#                             const submitBtn = document.createElement('button');
#                             submitBtn.textContent = 'Submit';
#                             submitBtn.style.cssText = `
#                                 flex: 1;
#                                 padding: 8px;
#                                 background: #4CAF50;
#                                 color: white;
#                                 border: none;
#                                 border-radius: 4px;
#                                 cursor: pointer;
#                                 font-size: 13px;
#                             `;
# 
#                             submitBtn.onclick = () => {
#                                 const value = input.value;
#                                 if (remember) {
#                                     window.scitexPanel.data[key] = value;
#                                 }
#                                 container.innerHTML = '';
#                                 resolve(value);
#                             };
# 
#                             const cancelBtn = document.createElement('button');
#                             cancelBtn.textContent = 'Cancel';
#                             cancelBtn.style.cssText = `
#                                 padding: 8px 16px;
#                                 background: #f0f0f0;
#                                 color: #666;
#                                 border: none;
#                                 border-radius: 4px;
#                                 cursor: pointer;
#                                 font-size: 13px;
#                             `;
# 
#                             cancelBtn.onclick = () => {
#                                 container.innerHTML = '';
#                                 resolve(null);
#                             };
# 
#                             btnContainer.appendChild(submitBtn);
#                             btnContainer.appendChild(cancelBtn);
# 
#                             inputGroup.appendChild(labelEl);
#                             inputGroup.appendChild(input);
#                             inputGroup.appendChild(btnContainer);
#                             container.appendChild(inputGroup);
# 
#                             input.focus();
# 
#                             // Submit on Enter
#                             input.addEventListener('keypress', (e) => {
#                                 if (e.key === 'Enter') {
#                                     submitBtn.click();
#                                 }
#                             });
#                         });
#                     },
# 
#                     askConfirm: function(message) {
#                         return new Promise((resolve) => {
#                             const container = document.getElementById('panel-actions');
#                             container.innerHTML = '';
# 
#                             const msg = document.createElement('div');
#                             msg.textContent = message;
#                             msg.style.cssText = 'margin-bottom: 8px; font-size: 13px;';
# 
#                             const btnContainer = document.createElement('div');
#                             btnContainer.style.cssText = 'display: flex; gap: 8px;';
# 
#                             const yesBtn = document.createElement('button');
#                             yesBtn.textContent = '✓ Yes';
#                             yesBtn.style.cssText = `
#                                 flex: 1;
#                                 padding: 8px;
#                                 background: #4CAF50;
#                                 color: white;
#                                 border: none;
#                                 border-radius: 4px;
#                                 cursor: pointer;
#                             `;
#                             yesBtn.onclick = () => {
#                                 container.innerHTML = '';
#                                 resolve(true);
#                             };
# 
#                             const noBtn = document.createElement('button');
#                             noBtn.textContent = '✗ No';
#                             noBtn.style.cssText = `
#                                 flex: 1;
#                                 padding: 8px;
#                                 background: #F44336;
#                                 color: white;
#                                 border: none;
#                                 border-radius: 4px;
#                                 cursor: pointer;
#                             `;
#                             noBtn.onclick = () => {
#                                 container.innerHTML = '';
#                                 resolve(false);
#                             };
# 
#                             btnContainer.appendChild(yesBtn);
#                             btnContainer.appendChild(noBtn);
#                             container.appendChild(msg);
#                             container.appendChild(btnContainer);
#                         });
#                     },
# 
#                     clear: function() {
#                         document.getElementById('panel-messages').innerHTML = '';
#                         document.getElementById('panel-inputs').innerHTML = '';
#                         document.getElementById('panel-actions').innerHTML = '';
#                     }
#                 };
# 
#                 console.log('✅ Interactive panel initialized');
#             }
#         """)
# 
#         print("✅ Interactive panel initialized in browser")
# 
#     async def show_message(self, message: str, type: str = "info"):
#         """Show message in panel."""
#         await self.page.evaluate(
#             f"window.scitexPanel?.showMessage('{message}', '{type}')"
#         )
# 
#     async def ask_input(
#         self,
#         key: str,
#         label: str,
#         input_type: str = "text",
#         remember: bool = True,
#     ) -> Optional[str]:
#         """
#         Ask user for input via panel.
# 
#         Args:
#             key: Key to store value (for memory)
#             label: Label to show user
#             input_type: "text", "password", "email", etc.
#             remember: Remember value for future use
# 
#         Returns:
#             User's input or None if cancelled
#         """
#         # Check if already in memory
#         if key in self.session_memory:
#             print(f"🔑 Using remembered {key}")
#             return self.session_memory[key]
# 
#         # Ask user
#         value = await self.page.evaluate(f"""
#             window.scitexPanel?.askInput('{key}', '{label}', '{input_type}', {str(remember).lower()})
#         """)
# 
#         # Store in memory
#         if value and remember:
#             self.session_memory[key] = value
#             self._save_persistent_memory()
# 
#         return value
# 
#     async def ask_confirm(self, message: str) -> bool:
#         """Ask user for confirmation via panel."""
#         result = await self.page.evaluate(f"""
#             window.scitexPanel?.askConfirm('{message}')
#         """)
#         return result if result is not None else False
# 
#     async def clear(self):
#         """Clear panel."""
#         await self.page.evaluate("window.scitexPanel?.clear()")
# 
#     def get_memory(self, key: str, default: Any = None) -> Any:
#         """Get value from memory."""
#         return self.session_memory.get(key, default)
# 
#     def set_memory(self, key: str, value: Any, persistent: bool = True):
#         """Set value in memory."""
#         self.session_memory[key] = value
#         if persistent:
#             self._save_persistent_memory()
# 
#     def clear_memory(self, persistent: bool = False):
#         """Clear memory."""
#         self.session_memory = {}
#         if persistent and self.persistent_memory_file.exists():
#             self.persistent_memory_file.unlink()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/collaboration/interactive_panel.py
# --------------------------------------------------------------------------------
