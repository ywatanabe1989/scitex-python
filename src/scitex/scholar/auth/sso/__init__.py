"""SSO automation modules."""

from .BaseSSOAutomator import BaseSSOAutomator
from .OpenAthensSSOAutomator import OpenAthensSSOAutomator
from .SSOAutomator import SSOAutomator
from .UniversityOfMelbourneSSOAutomator import UniversityOfMelbourneSSOAutomator

__all__ = [
    "BaseSSOAutomator",
    "SSOAutomator",
    "OpenAthensSSOAutomator",
    "UniversityOfMelbourneSSOAutomator",
]

# EOF
