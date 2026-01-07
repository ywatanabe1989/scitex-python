#!/usr/bin/env python3
# Timestamp: "2025-07-14 15:28:49 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/SciTeX-Code/src/scitex/__init__.py
# ----------------------------------------
import os

__FILE__ = "./src/scitex/__init__.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Minimal scitex initialization.
Modules are imported on-demand to avoid circular dependencies.
"""

import warnings

# Show deprecation warnings from scitex modules (educational for migration)
warnings.filterwarnings("default", category=DeprecationWarning, module="scitex.*")

# Version
from .__version__ import __version__

# Installation guide - show users what modules are available
from ._install_guide import show_install_guide


# Sentinel object for decorator-injected parameters
class _InjectedSentinel:
    """Sentinel value indicating a parameter will be injected by a decorator"""

    def __repr__(self):
        return "<INJECTED>"


INJECTED = _InjectedSentinel()


# Lazy loading for all modules
class _LazyModule:
    def __init__(self, name):
        self._name = name
        self._module = None

    def __getattr__(self, attr):
        if self._module is None:
            import importlib

            self._module = importlib.import_module(f".{self._name}", package="scitex")
        return getattr(self._module, attr)


class _CallableModuleWrapper:
    """Callable module wrapper that acts as both a decorator and a module.

    This allows:
    - @scitex.session (new clean API)
    - @scitex.session.session (old API for backwards compatibility)
    - scitex.session.start() and other module functions

    Example:
        import scitex

        @scitex.session  # Clean! Calls __call__()
        def main(): pass

        @scitex.session.session  # Backwards compatible
        def main(): pass

        scitex.session.start(...)  # Access other functions
    """

    def __init__(self, module_name, main_decorator_name="session"):
        self._module_name = module_name
        self._main_decorator_name = main_decorator_name
        self._module = None
        self._parent_name = None
        self._attr_name = None

    def _setup_persistence(self, parent_name, attr_name):
        """Set up persistence information to prevent replacement."""
        self._parent_name = parent_name
        self._attr_name = attr_name

    def _load_module(self):
        """Lazy load the actual module."""
        if self._module is None:
            import importlib
            import sys

            # Import the module
            self._module = importlib.import_module(
                f".{self._module_name}", package="scitex"
            )

            # Restore ourselves in the parent module's __dict__ to prevent replacement
            if self._parent_name and self._attr_name:
                parent_module = sys.modules.get(self._parent_name)
                if parent_module is not None:
                    setattr(parent_module, self._attr_name, self)

        return self._module

    def __call__(self, *args, **kwargs):
        """When used as @scitex.session"""
        module = self._load_module()
        main_decorator = getattr(module, self._main_decorator_name)
        return main_decorator(*args, **kwargs)

    def __getattr__(self, name):
        """When accessed as scitex.session.session or scitex.session.start"""
        if name == self._main_decorator_name:
            # Return self so @scitex.session.session works
            return self

        # Otherwise, delegate to the actual module
        module = self._load_module()
        return getattr(module, name)


# Create lazy modules
io = _LazyModule("io")
gen = _LazyModule("gen")
plt = _LazyModule("plt")
ai = _LazyModule("ai")
ml = _LazyModule("ai")  # Alias for machine learning - same as ai
pd = _LazyModule("pd")
str = _LazyModule("str")
stats = _LazyModule("stats")
path = _LazyModule("path")
dict = _LazyModule("dict")
decorators = _LazyModule("decorators")
dsp = _LazyModule("dsp")
nn = _LazyModule("nn")
torch = _LazyModule("torch")
web = _LazyModule("web")
db = _LazyModule("db")
repro = _LazyModule("repro")
reproduce = _LazyModule("reproduce")
rng = _LazyModule("rng")
scholar = _LazyModule("scholar")
writer = _LazyModule("writer")
fig = _LazyModule("fig")
resource = _LazyModule("resource")
tex = _LazyModule("tex")
linalg = _LazyModule("linalg")
parallel = _LazyModule("parallel")
datetime = _LazyModule("datetime")
dt = _LazyModule("dt")  # Alias for datetime (shorter name)
types = _LazyModule("types")
utils = _LazyModule("utils")
etc = _LazyModule("etc")
context = _LazyModule("context")
dev = _LazyModule("dev")
gists = _LazyModule("gists")
errors = _LazyModule("errors")
units = _LazyModule("units")
logging = _LazyModule("logging")
session = _CallableModuleWrapper("session", main_decorator_name="session")
session._setup_persistence("scitex", "session")
capture = _LazyModule("capture")
template = _LazyModule("template")
cloud = _LazyModule("cloud")
config = _LazyModule("config")
audio = _LazyModule("audio")
msword = _LazyModule("msword")
fts = _LazyModule("fts")  # Bundle schemas module

# Centralized path configuration - eager loaded for convenience
# Usage: scitex.PATHS.logs, scitex.PATHS.cache, etc.
from .config import ScitexPaths as _ScitexPaths

PATHS = _ScitexPaths()

# Auto-load cloud hooks if in cloud environment
import os as _os

if _os.environ.get("SCITEX_CLOUD_CODE_WORKSPACE") == "true":
    try:
        from .cloud import _matplotlib_hook
    except Exception:
        pass  # Silently fail if matplotlib not available

__all__ = [
    "io",
    "gen",
    "plt",
    "ai",
    "ml",
    "pd",
    "str",
    "stats",
    "path",
    "dict",
    "decorators",
    "__version__",
    "sh",
    "errors",
    "units",
    "logging",
    "session",
    "rng",
    "capture",
    "template",
    "torch",
    "dsp",
    "nn",
    "web",
    "db",
    "repro",
    "reproduce",
    "scholar",
    "writer",
    "fig",
    "resource",
    "tex",
    "linalg",
    "parallel",
    "datetime",
    "dt",  # Alias for datetime (shorter name)
    "types",
    "utils",
    "etc",
    "context",
    "dev",
    "gists",
    "cloud",
    "config",
    "audio",
    "msword",
    "fts",
    "fsb",  # Legacy alias
    "PATHS",
    "INJECTED",
]

# EOF
