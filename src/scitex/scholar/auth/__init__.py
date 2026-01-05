"""Authentication module for Scholar."""

# Optional: requires browser dependencies (playwright, aiohttp)
try:
    from .ScholarAuthManager import ScholarAuthManager
except ImportError:
    ScholarAuthManager = None

try:
    from .core.AuthenticationGateway import AuthenticationGateway, URLContext
except ImportError:
    AuthenticationGateway = None
    URLContext = None

__all__ = [
    "ScholarAuthManager",
    "AuthenticationGateway",
    "URLContext",
]

# EOF
