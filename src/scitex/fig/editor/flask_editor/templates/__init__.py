#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/templates/__init__.py
"""Template components for Flask editor.

Supports two modes:
- Static files mode (default): Uses external CSS/JS files from static/
- Inline mode (fallback): Embeds CSS/JS directly in HTML
"""

from ._html import HTML_BODY

# Configuration flag - set to False to use inline mode for debugging
USE_STATIC_FILES = True


def build_html_template(use_static: bool = None) -> str:
    """Build the complete HTML template from components.

    Args:
        use_static: Override static file usage. If None, uses USE_STATIC_FILES.

    Returns:
        Complete HTML template string.
    """
    if use_static is None:
        use_static = USE_STATIC_FILES

    if use_static:
        return _build_static_template()
    else:
        return _build_inline_template()


def _build_static_template() -> str:
    """Build template using external static CSS/JS files."""
    # Get list of JS files in correct load order
    js_files = [
        # Dev tools (load first to capture console logs)
        'js/dev/element-inspector.js',
        # Core modules first (dependencies)
        'js/core/state.js',
        'js/core/utils.js',
        'js/core/api.js',
        # Editor modules
        'js/editor/bbox.js',
        'js/editor/overlay.js',
        'js/editor/preview.js',
        'js/editor/element-drag.js',
        # Canvas modules
        'js/canvas/selection.js',
        'js/canvas/resize.js',
        'js/canvas/dragging.js',
        'js/canvas/canvas.js',
        # Alignment modules
        'js/alignment/basic.js',
        'js/alignment/axis.js',
        'js/alignment/distribute.js',
        # UI modules
        'js/ui/theme.js',
        'js/ui/help.js',
        'js/ui/download.js',
        'js/ui/controls.js',
        # Shortcuts
        'js/shortcuts/context-menu.js',
        'js/shortcuts/keyboard.js',
        # Main entry (last)
        'js/main.js',
    ]

    # Generate script tags
    script_tags = '\n    '.join([
        f'<script src="{{{{ url_for(\'static\', filename=\'{f}\') }}}}"></script>'
        for f in js_files
    ])

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SciTeX Figure Editor - {{{{ filename }}}}</title>
    <link rel="stylesheet" href="{{{{ url_for('static', filename='css/index.css') }}}}">
</head>
<body>
{HTML_BODY}
    <!-- Initial data injection (must be before module scripts) -->
    <script>
        // Flask injects overrides data here
        var overrides = {{{{ overrides|safe }}}};
    </script>
    {script_tags}
</body>
</html>
"""


def _build_inline_template() -> str:
    """Build template with inline CSS/JS (fallback mode)."""
    from ._styles import CSS_STYLES
    from ._scripts import JS_SCRIPTS

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
