#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SORA translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class SoraTranslator(BaseTranslator):
    """SORA."""

    LABEL = "SORA"
    URL_TARGET_PATTERN = (
        r"^https?://sora\.unm\.edu/(node/|search/node/|advancedsearch\?)"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
