#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library Catalog (TIND ILS) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class LibraryCatalogTindIlsTranslator(BaseTranslator):
    """Library Catalog (TIND ILS)."""

    LABEL = "Library Catalog (TIND ILS)"
    URL_TARGET_PATTERN = r"/search.+p=|record/[0-9]+"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
