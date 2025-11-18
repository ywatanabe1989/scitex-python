#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NYTimes.com translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class NytimesComTranslator(BaseTranslator):
    """NYTimes.com."""

    LABEL = "NYTimes.com"
    URL_TARGET_PATTERN = r"^https?://(query\.nytimes\.com/(search|gst)/|(select\.|www\.|mobile\.|[^\/.]*\.blogs\.)?nytimes\.com/)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
