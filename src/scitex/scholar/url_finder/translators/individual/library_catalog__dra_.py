#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library Catalog (DRA) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class LibraryCatalogDraTranslator(BaseTranslator):
    """Library Catalog (DRA)."""

    LABEL = "Library Catalog (DRA)"
    URL_TARGET_PATTERN = r"/web2/tramp2\.exe/(see\_record/|authority\_hits/|do_keyword_search|form/|goto/.*\?.*screen=(MARC)?Record\.html)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
