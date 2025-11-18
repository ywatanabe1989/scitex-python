#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library Catalog (Visual Library 2021) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class LibraryCatalogVisualLibrary2021Translator(BaseTranslator):
    """Library Catalog (Visual Library 2021)."""

    LABEL = "Library Catalog (Visual Library 2021)"
    URL_TARGET_PATTERN = r"/search(/quick)?\?|/nav/index/|/(content|periodical)/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
