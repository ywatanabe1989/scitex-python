#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Douban translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class DoubanTranslator(BaseTranslator):
    """Douban."""

    LABEL = "Douban"
    URL_TARGET_PATTERN = r"^https?://(www|book)\.douban\.com/(subject|doulist|people/[a-zA-Z0-9._]*/(do|wish|collect)|.*?status=(do|wish|collect)|group/[0-9]*?/collection|tag)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
