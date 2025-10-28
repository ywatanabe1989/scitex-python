#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Google Books translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class GoogleBooksTranslator(BaseTranslator):
    """Google Books."""

    LABEL = "Google Books"
    URL_TARGET_PATTERN = r"^https?://(books|www)\.google\.[a-z]+(\.[a-z]+)?/(books(/.*)?\?(.*id=.*|.*q=.*)|search\?.*?(btnG=Search\+Books|tbm=bks)|books/edition/)|^https?://play\.google\.[a-z]+(\.[a-z]+)?/(store/)?(books|search\?.*c=books)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
