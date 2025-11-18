#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library Catalog (RERO ILS) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class LibraryCatalogReroIlsTranslator(BaseTranslator):
    """Library Catalog (RERO ILS)."""

    LABEL = "Library Catalog (RERO ILS)"
    URL_TARGET_PATTERN = r"^https://((ils\.test|bib)\.rero\.ch|ils\.bib\.uclouvain\.be)/[^/]+/(documents/\d+|search/documents\?)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
