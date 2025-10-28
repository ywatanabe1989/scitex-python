#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library Catalog (Aleph) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class LibraryCatalogAlephTranslator(BaseTranslator):
    """Library Catalog (Aleph)."""

    LABEL = "Library Catalog (Aleph)"
    URL_TARGET_PATTERN = r"^https?://[^/]+/F(/?[A-Z0-9\-]*(\?.*)?$|\?func=find|\?func=scan|\?func=short|\?local_base=)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
