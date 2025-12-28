# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/browser/utils/close_unwanted_pages.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-13 11:22:44 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/scholar/browser/utils/close_unwanted_pages.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/scholar/browser/utils/close_unwanted_pages.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import asyncio
# 
# __FILE__ = __file__
# 
# from playwright.async_api import BrowserContext, Page
# 
# from scitex.browser.debugging import browser_logger
# 
# 
# async def close_unwanted_pages(
#     context: BrowserContext,
#     delay_sec=1,
#     max_attempts: int = 20,
#     func_name="close_unwanted_pages",
#     continuous: bool = False,
# ):
#     """Close unwanted extension and blank pages while keeping at least one page open.
# 
#     Args:
#         context: Browser context to clean up
#         delay_sec: Initial delay before starting cleanup
#         max_attempts: Maximum cleanup attempts (for single-run mode)
#         func_name: Name for logging purposes
#         continuous: If True, run continuously monitoring for new unwanted pages
#     """
# 
#     # URL patterns to identify unwanted pages
#     UNWANTED_PAGE_EXPRESSIONS = [
#         "chrome-extension://",
#         "pbapi.xyz",
#         "options.html",
#         # "newtab",
#         # "about:blank",
#         # "chrome://newtab",
#     ]
# 
#     await asyncio.sleep(delay_sec)
# 
#     attempts = 0
#     while True:
#         if not continuous and attempts >= max_attempts:
#             break
# 
#         attempts += 1
# 
#         try:
#             # Get current pages first to avoid stale references
#             current_pages = list(context.pages)
# 
#             # Get a valid page for browser_logger (prefer non-extension pages)
#             valid_page = None
#             for page in current_pages:
#                 if "chrome-extension://" not in page.url and not page.is_closed():
#                     valid_page = page
#                     break
# 
#             # If no valid page, use the first page
#             if not valid_page and current_pages:
#                 valid_page = current_pages[0]
# 
#             # Find unwanted pages by checking if any expression matches
#             unwanted_pages = [
#                 page
#                 for page in current_pages
#                 if any(
#                     unwanted_exp in page.url
#                     for unwanted_exp in UNWANTED_PAGE_EXPRESSIONS
#                 )
#             ]
# 
#             if not unwanted_pages:
#                 if valid_page and not continuous:
#                     await browser_logger.debug(
#                         valid_page, f"{func_name}: Extension cleanup completed"
#                     )
#                 if not continuous:
#                     break
#                 # In continuous mode, wait before checking again
#                 await asyncio.sleep(2)
#                 continue
# 
#             # Log what we're about to close
#             if valid_page and unwanted_pages:
#                 await browser_logger.debug(
#                     valid_page,
#                     f"{func_name}: Closing {len(unwanted_pages)} unwanted page(s)",
#                 )
# 
#             # Ensure context stays alive - create new page BEFORE closing if needed
#             if len(current_pages) == len(unwanted_pages):
#                 new_page = await context.new_page()
#                 if valid_page:
#                     await browser_logger.debug(
#                         valid_page,
#                         f"{func_name}: Created new page to keep context alive",
#                     )
#                 valid_page = new_page  # Use new page for logging
# 
#             # Close unwanted pages
#             closed_count = 0
#             for page in unwanted_pages:
#                 try:
#                     await page.close()
#                     closed_count += 1
#                 except Exception as e:
#                     if valid_page:
#                         await browser_logger.warning(
#                             valid_page, f"{func_name}: Failed to close page"
#                         )
# 
#             if closed_count > 0 and valid_page:
#                 await browser_logger.debug(
#                     valid_page,
#                     f"{func_name}: Closed {closed_count} page(s) (attempt {attempts + 1})",
#                 )
# 
#         except Exception as e:
#             if valid_page:
#                 await browser_logger.warning(
#                     valid_page,
#                     f"{func_name}: Cleanup attempt {attempt + 1} failed",
#                 )
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/browser/utils/close_unwanted_pages.py
# --------------------------------------------------------------------------------
