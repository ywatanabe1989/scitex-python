#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gasyrlar Awazy translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class GasyrlarAwazyTranslator(BaseTranslator):
    """Gasyrlar Awazy."""

    LABEL = "Gasyrlar Awazy"
    URL_TARGET_PATTERN = r"^https?://www\.archive\.gov\.tatarstan\.ru/magazine/go/anonymous/main/\?path=mg:/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
