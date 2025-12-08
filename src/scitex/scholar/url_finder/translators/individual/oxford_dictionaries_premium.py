#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Oxford Dictionaries Premium translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class OxfordDictionariesPremiumTranslator(BaseTranslator):
    """Oxford Dictionaries Premium."""

    LABEL = "Oxford Dictionaries Premium"
    URL_TARGET_PATTERN = (
        r"^https?://premium\.oxforddictionaries\.com/(translate|definition)/"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
