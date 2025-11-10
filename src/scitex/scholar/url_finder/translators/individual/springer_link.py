#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Springer Link translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class SpringerLinkTranslator(BaseTranslator):
    """Springer Link."""

    LABEL = "Springer Link"
    URL_TARGET_PATTERN = r"^https?://link\.springer\.com/(search(/page/\d+)?\?|(article|chapter|book|referenceworkentry|protocol|journal|referencework)/.+)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
