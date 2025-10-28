#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMS MathSciNet translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class AMSMathSciNetTranslator(BaseTranslator):
    """AMS MathSciNet."""

    LABEL = "AMS MathSciNet"
    URL_TARGET_PATTERN = r"^https?://mathscinet\.ams\.[^/]*/mathscinet/(article\?|publications-search\?|author\?)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
