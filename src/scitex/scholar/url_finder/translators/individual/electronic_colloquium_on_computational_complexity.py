#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Electronic Colloquium on Computational Complexity translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class ElectronicColloquiumOnComputationalComplexityTranslator(BaseTranslator):
    """Electronic Colloquium on Computational Complexity."""

    LABEL = "Electronic Colloquium on Computational Complexity"
    URL_TARGET_PATTERN = (
        r"^https?://eccc\.weizmann\.ac\.il/(title|year|keyword|report|search)"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
