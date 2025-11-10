#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMS MathSciNet (Legacy) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class AMSMathSciNetLegacyTranslator(BaseTranslator):
    """AMS MathSciNet (Legacy)."""

    LABEL = "AMS MathSciNet (Legacy)"
    URL_TARGET_PATTERN = r"^https?://(mathscinet\.)?ams\.[^/]*/(mathscinet/2006/)?mathscinet(\-getitem\?|/search/(publications\.html|publdoc\.html))"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
