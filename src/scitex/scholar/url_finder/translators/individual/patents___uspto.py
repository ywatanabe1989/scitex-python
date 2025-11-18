#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patents - USPTO translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class PatentsUsptoTranslator(BaseTranslator):
    """Patents - USPTO."""

    LABEL = "Patents - USPTO"
    URL_TARGET_PATTERN = r"^https?://(patft|appft1)\.uspto\.gov/netacgi/nph-Parser.+"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
