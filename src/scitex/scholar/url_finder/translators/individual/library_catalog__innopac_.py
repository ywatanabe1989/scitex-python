#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Library Catalog (InnoPAC) translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class LibraryCatalogInnopacTranslator(BaseTranslator):
    """Library Catalog (InnoPAC)."""

    LABEL = "Library Catalog (InnoPAC)"
    URL_TARGET_PATTERN = r"(search~|/search\?|(a|X|t|Y|w)\?|\?(searchtype|searchscope)|frameset&FF|record=[bi]?[0-9]+(~S[0-9])?|/search/q\?)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
