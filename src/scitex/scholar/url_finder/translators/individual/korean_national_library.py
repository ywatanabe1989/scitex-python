#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Korean National Library translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class KoreanNationalLibraryTranslator(BaseTranslator):
    """Korean National Library."""

    LABEL = "Korean National Library"
    URL_TARGET_PATTERN = (
        r"^https?://www\.nl\.go\.kr/(EN|NL)/contents/(eng)?[sS]earch\.do"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
