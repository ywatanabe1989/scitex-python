#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AGRIS translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class AGRISTranslator(BaseTranslator):
    """AGRIS."""

    LABEL = "AGRIS"
    URL_TARGET_PATTERN = r"^https?://agris\.fao\.org/agris-search/search"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
