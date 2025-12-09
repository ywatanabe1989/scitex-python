#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""University of Chicago Press Books translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class UniversityOfChicagoPressBooksTranslator(BaseTranslator):
    """University of Chicago Press Books."""

    LABEL = "University of Chicago Press Books"
    URL_TARGET_PATTERN = (
        r"^https?://(www\.)?press\.uchicago\.edu/(ucp/books/|press/search.html)"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
