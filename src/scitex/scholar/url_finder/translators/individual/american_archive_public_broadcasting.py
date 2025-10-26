#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""American Archive of Public Broadcasting translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class AmericanArchivePublicBroadcastingTranslator(BaseTranslator):
    """American Archive of Public Broadcasting."""

    LABEL = "American Archive of Public Broadcasting"
    URL_TARGET_PATTERN = r"^https?://(www\.)?americanarchive\.org/catalog"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
