#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pleade translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class PleadeTranslator(BaseTranslator):
    """Pleade."""

    LABEL = "Pleade"
    URL_TARGET_PATTERN = r"(list-results|results|ead)\.html\?.*(base=ead|mode=|id=)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
