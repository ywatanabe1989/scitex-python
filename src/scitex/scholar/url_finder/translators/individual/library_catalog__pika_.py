#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library Catalog (Pika) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class LibraryCatalogPikaTranslator(BaseTranslator):
    """Library Catalog (Pika)."""

    LABEL = "Library Catalog (Pika)"
    URL_TARGET_PATTERN = r"/Record/\.[a-z]|/GroupedWork/[a-z0-9-]+|/Union/Search\?"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
