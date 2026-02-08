#!/usr/bin/env python3
"""Minimal session script template."""

TEMPLATE = {
    "name": "Minimal Session Script",
    "description": "Minimal @stx.session script with essential injections only",
    "filename": "script.py",
    "usage": """
Usage:
  python script.py
  python script.py --help
""",
    "content": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "{timestamp}"
# File: {filepath}

"""
{docstring}
"""

import scitex as stx


@stx.session
def main(
    CONFIG=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Script description for --help."""
    logger.info(f"Session: {{CONFIG.ID}}")
    logger.info(f"Output: {{CONFIG.SDIR_OUT}}")
    return 0


if __name__ == "__main__":
    main()

# EOF
''',
}

# EOF
