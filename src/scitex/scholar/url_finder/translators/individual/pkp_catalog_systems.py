#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PKP Catalog Systems translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class PKPCatalogSystemsTranslator(BaseTranslator):
    """PKP Catalog Systems."""

    LABEL = "PKP Catalog Systems"
    URL_TARGET_PATTERN = r"/(article|preprint|issue)/view/|/catalog/book/|/search/search|/index\.php/default"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
