#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web of Science Tagged translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class WebOfScienceTaggedTranslator(BaseTranslator):
    """Web of Science Tagged."""

    LABEL = "Web of Science Tagged"
    URL_TARGET_PATTERN = r"txt"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
