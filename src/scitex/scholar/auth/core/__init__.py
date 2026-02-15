"""Core authentication modules."""

from .AuthenticationGateway import AuthenticationGateway, URLContext
from .BrowserAuthenticator import BrowserAuthenticator
from .StrategyResolver import (
    AuthenticationMethod,
    AuthenticationStrategy,
    AuthenticationStrategyResolver,
)

__all__ = [
    "AuthenticationGateway",
    "URLContext",
    "BrowserAuthenticator",
    "AuthenticationStrategyResolver",
    "AuthenticationStrategy",
    "AuthenticationMethod",
]

# EOF
