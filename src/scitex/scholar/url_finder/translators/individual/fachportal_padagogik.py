#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fachportal Pädagogik translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class FachportalPadagogikTranslator(BaseTranslator):
    """Fachportal Pädagogik."""

    LABEL = "Fachportal Pädagogik"
    URL_TARGET_PATTERN = (
        r"^https?://(www\.fachportal-paedagogik\.de/literatur/|www\.pedocs\.de/)"
    )

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
