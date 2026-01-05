# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/ubiquity_journals.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """Ubiquity Journals translator."""
# 
# import re
# from typing import List
# from playwright.async_api import Page
# from ..core.base import BaseTranslator
# 
# 
# class UbiquityJournalsTranslator(BaseTranslator):
#     """Ubiquity Journals."""
# 
#     LABEL = "Ubiquity Journals"
#     URL_TARGET_PATTERN = r"^https?://[^/]+(/articles/10\.\d{4,9}/[-._;()/:a-z0-9A-Z]+|/articles/?$|/\d+/volume/\d+/issue/\d+)"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         return []

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/ubiquity_journals.py
# --------------------------------------------------------------------------------
