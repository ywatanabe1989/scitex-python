# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/thesesfr.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """ThesesFR translator."""
# 
# import re
# from typing import List
# from playwright.async_api import Page
# from ..core.base import BaseTranslator
# 
# 
# class ThesesfrTranslator(BaseTranslator):
#     """ThesesFR."""
# 
#     LABEL = "ThesesFR"
#     URL_TARGET_PATTERN = r"^https?://(www\.)?theses\.fr/([a-z]{2}/)?((s\d+|\d{4}.{8}|\d{8}X|\d{9})(?!\.(rdf|xml)$)|(sujets/\?q=|\?q=))(?!.*&format=(json|xml))"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         return []

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/thesesfr.py
# --------------------------------------------------------------------------------
