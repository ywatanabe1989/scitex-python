"""Python implementations of Zotero translators.

For translators that are problematic in JavaScript (hanging, scoping issues),
we provide simplified Python implementations that extract PDF URLs directly.

Usage:
    from scitex.scholar.url_finder.translators import TranslatorRegistry

    # Find translator for URL
    translator = TranslatorRegistry.get_translator_for_url(url)
    if translator:
        pdf_urls = await translator.extract_pdf_urls_async(page)

Public API:
    - TranslatorRegistry: Main entry point for finding and using translators
    - BaseTranslator: Base class for creating custom translators

Internal modules (not part of public API):
    - .core: Core implementation details
    - .individual: Individual translator implementations
"""

# Public API - only expose what users need
from .core.registry import TranslatorRegistry
from .core.base import BaseTranslator

# Everything else is internal implementation
__all__ = [
    "TranslatorRegistry",
    "BaseTranslator",
]
