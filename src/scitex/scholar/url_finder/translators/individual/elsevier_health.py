#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Elsevier Health Journals translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class ElsevierHealthTranslator(BaseTranslator):
    """Elsevier Health Journals."""

    LABEL = "Elsevier Health Journals"
    URL_TARGET_PATTERN = r"^https?://[^/]+/(?:action/doSearch\?|(?:journals/[^/]+/)?article/[^/]+/(abstract|fulltext|references|images))"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
