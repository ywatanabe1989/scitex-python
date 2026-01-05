"""Browser automation utilities."""

# Optional: CookieAutoAcceptor requires playwright
try:
    from .CookieHandler import CookieAutoAcceptor
except ImportError:
    CookieAutoAcceptor = None

__all__ = [
    "CookieAutoAcceptor",
]
