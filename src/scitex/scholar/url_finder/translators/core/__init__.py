"""Core components for Zotero translators."""

from .base import BaseTranslator
from .registry import TranslatorRegistry

__all__ = ["BaseTranslator", "TranslatorRegistry"]
