#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Microbiology Society Journals translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class MicrobiologySocietyJournalsTranslator(BaseTranslator):
    """Microbiology Society Journals."""

    LABEL = "Microbiology Society Journals"
    URL_TARGET_PATTERN = (
        r"^https?://(www\.)?(\w)+\.microbiologyresearch\.org/(content/journal/|search?)"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
