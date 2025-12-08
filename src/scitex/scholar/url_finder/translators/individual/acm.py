#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACM Digital Library translator.

This is a redirect to the main acm_digital_library module.
Kept for backward compatibility with existing registry imports.
"""

from .acm_digital_library import ACMDigitalLibraryTranslator as ACMTranslator

__all__ = ["ACMTranslator"]
