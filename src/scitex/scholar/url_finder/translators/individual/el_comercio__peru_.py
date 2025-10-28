#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""El Comercio (Peru) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class ElComercioPeruTranslator(BaseTranslator):
    """El Comercio (Peru)."""

    LABEL = "El Comercio (Peru)"
    URL_TARGET_PATTERN = r"^https?://elcomercio\.pe"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
