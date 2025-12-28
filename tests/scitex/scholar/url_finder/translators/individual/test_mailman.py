# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/mailman.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """Mailman translator."""
# 
# import re
# from typing import List
# from playwright.async_api import Page
# from ..core.base import BaseTranslator
# 
# 
# class MailmanTranslator(BaseTranslator):
#     """Mailman."""
# 
#     LABEL = "Mailman"
#     URL_TARGET_PATTERN = r"/(pipermail|archives)/[A-Za-z0-9_-]*/[0-9]{4}-(January|February|March|April|May|June|July|August|September|October|November|December)/[0-9]{6}\.html"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         return []

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/mailman.py
# --------------------------------------------------------------------------------
