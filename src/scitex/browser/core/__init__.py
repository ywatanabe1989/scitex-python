"""Core browser management components."""

# Optional: BrowserMixin requires playwright
try:
    from .BrowserMixin import BrowserMixin
except ImportError:
    BrowserMixin = None

# Optional: ChromeProfileManager may have dependencies
try:
    from .ChromeProfileManager import ChromeProfileManager
except ImportError:
    ChromeProfileManager = None

__all__ = [
    "BrowserMixin",
    "ChromeProfileManager",
]
