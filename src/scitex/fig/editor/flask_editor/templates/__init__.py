#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/templates/__init__.py
"""Template components for Flask editor."""

from ._styles import CSS_STYLES
from ._scripts import JS_SCRIPTS
from ._html import HTML_BODY


def build_html_template() -> str:
    """Build the complete HTML template from components."""
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SciTeX Figure Editor - {{{{ filename }}}}</title>
    <style>
{CSS_STYLES}
    </style>
</head>
<body>
{HTML_BODY}
    <script>
{JS_SCRIPTS}
    </script>
</body>
</html>
"""


# EOF
