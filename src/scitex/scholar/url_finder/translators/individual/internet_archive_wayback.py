#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Internet Archive Wayback Machine translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class InternetArchiveWaybackTranslator(BaseTranslator):
    """Internet Archive Wayback Machine."""

    LABEL = "Internet Archive Wayback Machine"
    URL_TARGET_PATTERN = r"^https?://web\.archive\.org/web/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
