#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TV by the Numbers translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class TvByTheNumbersTranslator(BaseTranslator):
    """TV by the Numbers."""

    LABEL = "TV by the Numbers"
    URL_TARGET_PATTERN = r"^https?://tvbythenumbers\.zap2it\.com/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
