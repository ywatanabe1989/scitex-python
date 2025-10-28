#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library Catalog (Dynix) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class LibraryCatalogDynixTranslator(BaseTranslator):
    """Library Catalog (Dynix)."""

    LABEL = "Library Catalog (Dynix)"
    URL_TARGET_PATTERN = r"ipac\.jsp\?.*(uri=(link|full)=[0-9]|menu=search|term=)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
