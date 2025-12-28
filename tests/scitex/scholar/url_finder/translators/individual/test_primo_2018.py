# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/primo_2018.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """Primo 2018 translator."""
# 
# import re
# from typing import List
# from playwright.async_api import Page
# from ..core.base import BaseTranslator
# 
# 
# class Primo2018Translator(BaseTranslator):
#     """Primo 2018."""
# 
#     LABEL = "Primo 2018"
#     URL_TARGET_PATTERN = r"(/primo-explore/|/(discovery|nde)/(search|fulldisplay|jsearch|dbsearch|npsearch|openurl|jfulldisplay|dbfulldisplay|npfulldisplay|collectionDiscovery)\?)"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         return []

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/primo_2018.py
# --------------------------------------------------------------------------------
