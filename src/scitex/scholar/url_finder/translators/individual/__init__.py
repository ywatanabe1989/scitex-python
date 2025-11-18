"""Individual translator implementations.

Each translator handles a specific publisher or platform.

Internal module - not part of public API
-----------------------------------------
Users should NOT import individual translators directly.
All translators are accessed via TranslatorRegistry.

Good:
    from scitex.scholar.url_finder.translators import TranslatorRegistry
    translator = TranslatorRegistry.get_translator_for_url(url)

Bad:
    from scitex.scholar.url_finder.translators.individual.arxiv import ArXivTranslator

Why?
- Implementation details may change
- Translators are registered automatically
- Registry handles URL matching and prioritization
- Importing individual translators bypasses the registry system
"""

# Nothing exported - use TranslatorRegistry instead
__all__ = []
