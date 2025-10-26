#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aluka translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class AlukaTranslator(BaseTranslator):
    """Aluka."""

    LABEL = "Aluka"
    URL_TARGET_PATTERN = r"^https?://(www\.)?aluka\.org/(stable/|struggles/search\?|struggles/collection/)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
