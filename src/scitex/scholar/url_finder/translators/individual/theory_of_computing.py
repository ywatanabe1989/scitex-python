#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Theory of Computing translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class TheoryOfComputingTranslator(BaseTranslator):
    """Theory of Computing."""

    LABEL = "Theory of Computing"
    URL_TARGET_PATTERN = r"^https?://(theoryofcomputing\.org|toc\.cse\.iitk\.ac\.in|www\.cims\.nyu\.edu/~regev/toc|toc\.ilab\.sztaki\.hu|toc\.nada\.kth\.se|tocmirror\.cs\.tau\.ac\.il)/articles/[vg].*(/|html?)$"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
