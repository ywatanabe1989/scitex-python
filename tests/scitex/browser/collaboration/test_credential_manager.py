# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/collaboration/credential_manager.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-19 05:50:00 (ywatanabe)"
# # File: ./src/scitex/browser/collaboration/credential_manager.py
# # ----------------------------------------
# """
# Flexible credential management for browser automation.
# 
# Safe, clear communication with user.
# Multiple input methods: env vars, terminal, browser.
# """
# 
# import os
# import getpass
# from typing import Optional, Dict
# from playwright.async_api import Page
# 
# 
# class CredentialManager:
#     """
#     Flexible credential retrieval.
# 
#     Tries multiple sources in order:
#     1. Explicitly provided credentials
#     2. Environment variables
#     3. Terminal prompt (if terminal available)
#     4. Browser prompt (if browser available)
# 
#     Always clearly communicates what it's doing!
#     """
# 
#     def __init__(self, verbose: bool = True):
#         self.verbose = verbose
#         self.cache: Dict[str, str] = {}  # Session cache (not persistent)
# 
#     async def get_credential(
#         self,
#         name: str,
#         env_var: Optional[str] = None,
#         prompt_text: Optional[str] = None,
#         page: Optional[Page] = None,
#         mask: bool = False,  # For passwords
#     ) -> str:
#         """
#         Get credential from best available source.
# 
#         Args:
#             name: Credential name (for caching)
#             env_var: Environment variable to check
#             prompt_text: Text to show in prompt
#             page: Playwright page (for browser prompts)
#             mask: Whether to mask input (for passwords)
# 
#         Returns:
#             Credential value
# 
#         Example:
#             username = await creds.get_credential(
#                 name="username",
#                 env_var="SCITEX_CLOUD_USERNAME",
#                 prompt_text="Django username",
#                 page=page,
#             )
#         """
#         # Check cache first
#         if name in self.cache:
#             if self.verbose:
#                 print(f"🔑 Using cached {name}")
#             return self.cache[name]
# 
#         # Try environment variable
#         if env_var:
#             value = os.getenv(env_var)
#             if value:
#                 if self.verbose:
#                     display_value = "***" if mask else value
#                     print(f"🔑 Using {name} from ${env_var}: {display_value}")
#                 self.cache[name] = value
#                 return value
# 
#         # Try terminal prompt
#         if self._is_terminal_available():
#             value = await self._prompt_terminal(name, prompt_text, mask)
#             if value:
#                 self.cache[name] = value
#                 return value
# 
#         # Try browser prompt
#         if page:
#             value = await self._prompt_browser(page, name, prompt_text)
#             if value:
#                 self.cache[name] = value
#                 return value
# 
#         raise ValueError(f"Could not get credential: {name}")
# 
#     def _is_terminal_available(self) -> bool:
#         """Check if we can prompt in terminal."""
#         try:
#             return os.isatty(0)  # stdin is a terminal
#         except:
#             return False
# 
#     async def _prompt_terminal(
#         self,
#         name: str,
#         prompt_text: Optional[str],
#         mask: bool,
#     ) -> Optional[str]:
#         """Prompt user in terminal."""
#         prompt_text = prompt_text or name
# 
#         print(f"\n🔑 Credential needed: {name}")
#         print(f"   (No environment variable found)")
# 
#         if mask:
#             value = getpass.getpass(f"   Enter {prompt_text}: ")
#         else:
#             value = input(f"   Enter {prompt_text}: ")
# 
#         return value if value else None
# 
#     async def _prompt_browser(
#         self,
#         page: Page,
#         name: str,
#         prompt_text: Optional[str],
#     ) -> Optional[str]:
#         """Prompt user in browser window."""
#         prompt_text = prompt_text or name
# 
#         print(f"\n🔑 Asking for {name} in browser...")
# 
#         # Wait for page to be ready
#         try:
#             await page.wait_for_load_state("domcontentloaded", timeout=2000)
#         except:
#             pass  # Continue anyway
# 
#         value = await page.evaluate(f"""
#             () => {{
#                 const response = prompt('🔑 Credential needed: {prompt_text}\\n\\n(You can also set ${name.upper()} environment variable)');
#                 return response;
#             }}
#         """)
# 
#         return value if value else None
# 
#     async def get_login_credentials(
#         self,
#         page: Optional[Page] = None,
#         username_env: str = "SCITEX_CLOUD_USERNAME",
#         password_env: str = "SCITEX_CLOUD_PASSWORD",
#     ) -> Dict[str, str]:
#         """
#         Get both username and password.
# 
#         Convenient helper for login flows.
# 
#         Returns:
#             {'username': '...', 'password': '...'}
#         """
#         username = await self.get_credential(
#             name="username",
#             env_var=username_env,
#             prompt_text="Username",
#             page=page,
#             mask=False,
#         )
# 
#         password = await self.get_credential(
#             name="password",
#             env_var=password_env,
#             prompt_text="Password",
#             page=page,
#             mask=True,
#         )
# 
#         return {"username": username, "password": password}
# 
#     def clear_cache(self):
#         """Clear credential cache."""
#         self.cache = {}
#         if self.verbose:
#             print("🔑 Credential cache cleared")
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/browser/collaboration/credential_manager.py
# --------------------------------------------------------------------------------
