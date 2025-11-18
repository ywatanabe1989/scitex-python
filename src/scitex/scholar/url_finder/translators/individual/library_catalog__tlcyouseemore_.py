#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library Catalog (TLCYouSeeMore) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class LibraryCatalogTlcyouseemoreTranslator(BaseTranslator):
    """Library Catalog (TLCYouSeeMore)."""

    LABEL = "Library Catalog (TLCYouSeeMore)"
    URL_TARGET_PATTERN = r"TLCScripts/interpac\.dll\?(.*LabelDisplay.*RecordNumber=[0-9]|Search|ItemTitles)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
