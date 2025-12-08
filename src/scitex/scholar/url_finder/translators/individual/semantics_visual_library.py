#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""semantics Visual Library translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class SemanticsVisualLibraryTranslator(BaseTranslator):
    """semantics Visual Library."""

    LABEL = "semantics Visual Library"
    URL_TARGET_PATTERN = (
        r"^https?://www\.(blldb-online\.de/blldb|bdsl-online\.de/BDSL-DB)/suche/"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
