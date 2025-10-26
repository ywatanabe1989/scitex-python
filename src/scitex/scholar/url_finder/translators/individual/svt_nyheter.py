#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVT Nyheter translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class SvtNyheterTranslator(BaseTranslator):
    """SVT Nyheter."""

    LABEL = "SVT Nyheter"
    URL_TARGET_PATTERN = r"^https?://www\.svt\.se/nyheter/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
