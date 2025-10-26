#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ProQuest Ebook Central translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class ProquestEbookCentralTranslator(BaseTranslator):
    """ProQuest Ebook Central."""

    LABEL = "ProQuest Ebook Central"
    URL_TARGET_PATTERN = r"^https?://ebookcentral\.proquest\.com/(ebc/)?lib/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
