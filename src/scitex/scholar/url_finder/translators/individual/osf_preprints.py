#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OSF Preprints translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class OSFPreprintsTranslator(BaseTranslator):
    """OSF Preprints."""

    LABEL = "OSF Preprints"
    URL_TARGET_PATTERN = r"^https?://(osf\.io|psyarxiv\.com|arabixiv\.org|biohackrxiv\.org|eartharxiv\.org|ecoevorxiv\.org|ecsarxiv\.org|edarxiv\.org|engrxiv\.org|frenxiv\.org|indiarxiv\.org|mediarxiv\.org|paleorxiv\.org)"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
