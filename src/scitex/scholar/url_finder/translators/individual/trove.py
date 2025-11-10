#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trove translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class TroveTranslator(BaseTranslator):
    """Trove."""

    LABEL = "Trove"
    URL_TARGET_PATTERN = r"^https?://trove\.nla\.gov\.au/(?:newspaper|gazette|work|book|article|picture|music|map|collection|search)/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
