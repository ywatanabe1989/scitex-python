#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPO Access e-CFR translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class GpoAccessECfrTranslator(BaseTranslator):
    """GPO Access e-CFR."""

    LABEL = "GPO Access e-CFR"
    URL_TARGET_PATTERN = r"^https?://(www\.)?ecfr\.gov/cgi-bin/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
