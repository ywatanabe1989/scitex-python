"""Authentication provider implementations."""

from .BaseAuthenticator import BaseAuthenticator
from .EZProxyAuthenticator import EZProxyAuthenticator, EZProxyError
from .OpenAthensAuthenticator import OpenAthensAuthenticator, OpenAthensError
from .ShibbolethAuthenticator import ShibbolethAuthenticator, ShibbolethError

__all__ = [
    "BaseAuthenticator",
    "OpenAthensAuthenticator",
    "OpenAthensError",
    "EZProxyAuthenticator",
    "EZProxyError",
    "ShibbolethAuthenticator",
    "ShibbolethError",
]

# EOF
