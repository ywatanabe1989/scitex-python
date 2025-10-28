#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SIRS Knowledge Source translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class SirsKnowledgeSourceTranslator(BaseTranslator):
    """SIRS Knowledge Source."""

    LABEL = "SIRS Knowledge Source"
    URL_TARGET_PATTERN = r"^https?://([^/]+\.)?sks\.sirs\.com/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
