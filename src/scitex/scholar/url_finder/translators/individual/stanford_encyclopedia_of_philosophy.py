#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stanford Encyclopedia of Philosophy translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class StanfordEncyclopediaOfPhilosophyTranslator(BaseTranslator):
    """Stanford Encyclopedia of Philosophy."""

    LABEL = "Stanford Encyclopedia of Philosophy"
    URL_TARGET_PATTERN = (
        r"^https?://plato\.stanford\.edu/(archives/[a-z]{3}\d{4}/)?(entries|search)"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
