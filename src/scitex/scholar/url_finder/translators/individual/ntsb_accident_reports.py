#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NTSB Accident Reports translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class NtsbAccidentReportsTranslator(BaseTranslator):
    """NTSB Accident Reports."""

    LABEL = "NTSB Accident Reports"
    URL_TARGET_PATTERN = (
        r"^https?://(www\.)?ntsb\.gov/investigations/AccidentReports/Pages/"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
