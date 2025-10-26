#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Talis Aspire translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class TalisAspireTranslator(BaseTranslator):
    """Talis Aspire."""

    LABEL = "Talis Aspire"
    URL_TARGET_PATTERN = r"^https?://([^/]+\.)?(((my)?reading|resource|lib|cyprus|)lists|aspire\.surrey|rl\.talis)\..+/(lists|items)/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
