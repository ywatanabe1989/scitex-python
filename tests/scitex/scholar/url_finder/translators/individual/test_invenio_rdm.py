# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/invenio_rdm.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """InvenioRDM translator."""
# 
# import re
# from typing import List
# from playwright.async_api import Page
# from ..core.base import BaseTranslator
# 
# 
# class InvenioRDMTranslator(BaseTranslator):
#     """InvenioRDM."""
# 
#     LABEL = "InvenioRDM"
#     URL_TARGET_PATTERN = r"^https?://(zenodo\.org|sandbox\.zenodo\.org|data\.caltech\.edu|repository\.tugraz\.at|researchdata\.tuwien\.at|ultraviolet\.library\.nyu\.edu|adc\.ei-basel\.hasdai\.org|fdat\.uni-tuebingen\.de|www\.fdr\.uni-hamburg\.de|rodare\.hzdr\.de|aperta\.ulakbim\.gov\.tr|www\.openaccessrepository\.it|eic-zenodo\.sdcc\.bnl\.gov)"
# 
#     @classmethod
#     def matches_url(cls, url: str) -> bool:
#         return bool(re.match(cls.URL_TARGET_PATTERN, url))
# 
#     @classmethod
#     async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
#         return []

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/url_finder/translators/individual/invenio_rdm.py
# --------------------------------------------------------------------------------
