# External Package Branding Guide

When scitex wraps external packages (like `figrecipe` for `scitex.plt` or `crossref-local` for `scitex.scholar`), those packages should support configurable branding so documentation and error messages show the scitex namespace.

## When to Use Branding

- **Use branding**: External packages that scitex wraps (figrecipe, crossref-local, etc.)
- **Don't use branding**: Internal scitex modules (scitex.audio, scitex.stats, etc.) - just hardcode `SCITEX_*` prefix

## Pattern Overview

The external package provides a `_branding.py` module that:
1. Reads brand name from environment variable
2. Derives environment variable prefix from brand name
3. Provides helper functions for rebranding text/docstrings

## Implementation Template

```python
# external_package/_branding.py

import os
import re
from typing import Optional

# Environment variables for branding
# Parent package sets these before importing
BRAND_NAME = os.environ.get("{PACKAGE}_BRAND", "{package}")
BRAND_ALIAS = os.environ.get("{PACKAGE}_ALIAS", "{alias}")

# Original values for replacement
_ORIGINAL_NAME = "{package}"
_ORIGINAL_ALIAS = "{alias}"


def _brand_to_env_prefix(brand: str) -> str:
    """Convert brand name to environment variable prefix.

    Examples:
        "figrecipe" -> "FIGRECIPE"
        "scitex.plt" -> "SCITEX_PLT"
        "crossref-local" -> "CROSSREF_LOCAL"
    """
    return brand.upper().replace(".", "_").replace("-", "_")


# Environment variable prefix based on brand
ENV_PREFIX = _brand_to_env_prefix(BRAND_NAME)


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with brand-aware prefix.

    Checks {ENV_PREFIX}_{key} first, then falls back to original prefix.
    """
    value = os.environ.get(f"{ENV_PREFIX}_{key}")
    if value is not None:
        return value

    # Fall back to original prefix if different
    original_prefix = _brand_to_env_prefix(_ORIGINAL_NAME)
    if ENV_PREFIX != original_prefix:
        value = os.environ.get(f"{original_prefix}_{key}")
        if value is not None:
            return value

    return default


def rebrand_text(text: Optional[str]) -> Optional[str]:
    """Apply branding to a text string (docstrings, error messages)."""
    if text is None:
        return None

    if BRAND_NAME == _ORIGINAL_NAME and BRAND_ALIAS == _ORIGINAL_ALIAS:
        return text

    result = text

    # Replace import statements
    result = re.sub(
        rf"import\s+{_ORIGINAL_NAME}\s+as\s+{_ORIGINAL_ALIAS}",
        f"import {BRAND_NAME} as {BRAND_ALIAS}",
        result,
    )

    # Replace "from package" statements
    result = re.sub(
        rf"from\s+{_ORIGINAL_NAME}(\s+import|\s*\.)",
        lambda m: f"from {BRAND_NAME}{m.group(1)}",
        result,
    )

    return result


def get_mcp_server_name() -> str:
    """Get the MCP server name based on branding."""
    return BRAND_NAME.replace(".", "-")
```

## Usage in Parent Package (scitex)

```python
# scitex/plt/__init__.py
import os

# Set branding BEFORE importing the external package
os.environ["FIGRECIPE_BRAND"] = "scitex.plt"
os.environ["FIGRECIPE_ALIAS"] = "plt"

# Now import - docstrings will show scitex.plt instead of figrecipe
from figrecipe import *
```

## Port Scheme

SciTeX uses port scheme 3129X (TEX → te-ku-su → 2-9-3 in Japanese):

| Port  | Service          |
|-------|------------------|
| 31290 | scitex-cloud     |
| 31291 | crossref-local   |
| 31292 | openalex         |
| 31293 | scitex-audio     |

## Environment Variable Pattern

External packages should use `{ENV_PREFIX}_{SETTING}`:

```
SCITEX_PLT_MODE=local
CROSSREF_LOCAL_API_URL=http://localhost:8333
SCITEX_AUDIO_RELAY_URL=http://localhost:31293
```

## Example: crossref-local

See GitHub Issue: https://github.com/ywatanabe1989/crossref-local/issues/11

The crossref-local package should implement:
- `CROSSREF_LOCAL_BRAND` / `CROSSREF_LOCAL_ALIAS` env vars
- Dynamic `ENV_PREFIX` derived from brand name
- When used via scitex.scholar, shows `scitex.scholar` in docs

## References

- figrecipe/_branding.py - Reference implementation
- scitex/audio/_branding.py - Simple internal module (no rebranding needed)
