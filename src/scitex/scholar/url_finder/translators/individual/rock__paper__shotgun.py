#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rock, Paper, Shotgun translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class RockPaperShotgunTranslator(BaseTranslator):
    """Rock, Paper, Shotgun."""

    LABEL = "Rock, Paper, Shotgun"
    URL_TARGET_PATTERN = r"^https?://(www\.)?rockpapershotgun\.(com|de)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
