#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library Catalog (SIRSI eLibrary) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class LibraryCatalogSirsiElibraryTranslator(BaseTranslator):
    """Library Catalog (SIRSI eLibrary)."""

    LABEL = "Library Catalog (SIRSI eLibrary)"
    URL_TARGET_PATTERN = r"/uhtbin/(cgisirsi|quick_keyword)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
