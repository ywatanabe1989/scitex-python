#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""News Corp Australia translator."""

import re
from typing import List
from playwright.async_api import Page
from ..core.base import BaseTranslator


class NewsCorpAustraliaTranslator(BaseTranslator):
    """News Corp Australia."""

    LABEL = "News Corp Australia"
    URL_TARGET_PATTERN = r"^https?://(www\.)?(news|theaustralian|couriermail|adelaidenow|heraldsun|dailytelegraph|goldcoastbulletin|themercury|dailymercury|ntnews|northshoretimes|geelongadvertiser|townsvillebulletin|cairnspost|themorningbulletin|gladstoneobserver|sunshinecoastdaily|qt|thechronicle)\.com\.au/"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return bool(re.match(cls.URL_TARGET_PATTERN, url))

    @classmethod
    async def extract_pdf_urls_async(cls, page: Page) -> List[str]:
        return []
