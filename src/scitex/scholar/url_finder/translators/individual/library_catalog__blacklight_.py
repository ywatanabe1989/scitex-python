#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library Catalog (Blacklight) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class LibraryCatalogBlacklightTranslator(BaseTranslator):
    """Library Catalog (Blacklight)."""

    LABEL = "Library Catalog (Blacklight)"
    URL_TARGET_PATTERN = r"^https?://(catalog\.libraries\.psu|clio\.columbia|searchworks\.stanford|search\.library\.brown)\.edu/(view|catalog|\?search)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
