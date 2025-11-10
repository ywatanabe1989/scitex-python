#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library Catalog (Voyager 7) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class LibraryCatalogVoyager7Translator(BaseTranslator):
    """Library Catalog (Voyager 7)."""

    LABEL = "Library Catalog (Voyager 7)"
    URL_TARGET_PATTERN = r"/vwebv/(holdingsInfo|search)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
