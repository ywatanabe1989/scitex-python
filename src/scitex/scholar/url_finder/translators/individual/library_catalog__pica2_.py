#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library Catalog (PICA2) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class LibraryCatalogPica2Translator(BaseTranslator):
    """Library Catalog (PICA2)."""

    LABEL = "Library Catalog (PICA2)"
    URL_TARGET_PATTERN = r"^https?://[^/]+/DB=[\dA-Z]"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
