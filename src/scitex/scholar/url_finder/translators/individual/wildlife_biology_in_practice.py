#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wildlife Biology in Practice translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class WildlifeBiologyInPracticeTranslator(BaseTranslator):
    """Wildlife Biology in Practice."""

    LABEL = "Wildlife Biology in Practice"
    URL_TARGET_PATTERN = r"^https?://[^/]*socpvs\.org/journals/index\.php/wbp"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
