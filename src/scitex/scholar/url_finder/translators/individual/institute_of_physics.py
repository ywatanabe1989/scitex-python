#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Institute of Physics translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class InstituteOfPhysicsTranslator(BaseTranslator):
    """Institute of Physics."""

    LABEL = "Institute of Physics"
    URL_TARGET_PATTERN = r"^https?://iopscience\.iop\.org/((article/10\.[^/]+/)?[0-9-X]+/.+|n?search\?.+)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
