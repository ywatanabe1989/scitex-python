"""Remote browser components (ZenRows, CAPTCHA handling)."""

from .CaptchaHandler import CaptchaHandler
from .ZenRowsAPIClient import ZenRowsAPIBrowser
from .ZenRowsBrowserManager import ZenRowsRemoteScholarBrowserManager

__all__ = [
    "ZenRowsAPIBrowser",
    "ZenRowsRemoteScholarBrowserManager",
    "CaptchaHandler",
]
