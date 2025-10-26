#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NewsnetTamedia translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class NewsnettamediaTranslator(BaseTranslator):
    """NewsnetTamedia."""

    LABEL = "NewsnetTamedia"
    URL_TARGET_PATTERN = r"^https?://((www\.)?(tagesanzeiger|(bo\.)?bernerzeitung|bazonline|derbund|lematin|24heures|landbote|zuonline|zsz|tdg|letemps)\.ch/.)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
