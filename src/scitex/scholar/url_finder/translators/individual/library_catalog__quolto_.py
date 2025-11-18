#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library Catalog (Quolto) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class LibraryCatalogQuoltoTranslator(BaseTranslator):
    """Library Catalog (Quolto)."""

    LABEL = "Library Catalog (Quolto)"
    URL_TARGET_PATTERN = r"/record/-/record/|results/-/results|^https?://(www\.)?(mokka\.hu/|odrportal\.hu/).+results"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
