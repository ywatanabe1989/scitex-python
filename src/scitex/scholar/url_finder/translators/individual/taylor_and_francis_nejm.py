#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Taylor and Francis+NEJM translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class TaylorAndFrancisNejmTranslator(BaseTranslator):
    """Taylor and Francis+NEJM."""

    LABEL = "Taylor and Francis+NEJM"
    URL_TARGET_PATTERN = r"^https?://(www\.)?(tandfonline\.com|nejm\.org)/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
