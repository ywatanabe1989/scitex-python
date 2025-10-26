#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACLWeb translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class ACLWebTranslator(BaseTranslator):
    """ACLWeb."""

    LABEL = "ACLWeb"
    URL_TARGET_PATTERN = r"^https?://(www\.)?(aclanthology\.org|aclweb\.org/anthology)/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
