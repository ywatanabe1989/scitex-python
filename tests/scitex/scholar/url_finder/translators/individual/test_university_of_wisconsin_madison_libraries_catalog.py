# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/university_of_wisconsin_madison_libraries_catalog.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """University of Wisconsin-Madison Libraries Catalog translator."""
# 
# import re
# from typing import List
# from playwright.async_api import Page
# from ..core.base import BaseTranslator
# 
# 
# class UniversityOfWisconsinMadisonLibrariesCatalogTranslator(BaseTranslator):
#     """University of Wisconsin-Madison Libraries Catalog."""
# 
#     LABEL = "University of Wisconsin-Madison Libraries Catalog"
#     URL_TARGET_PATTERN = r"^https://search\.library\.wisc\.edu/(catalog|search)/"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         return []

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/university_of_wisconsin_madison_libraries_catalog.py
# --------------------------------------------------------------------------------
