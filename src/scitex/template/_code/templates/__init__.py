#!/usr/bin/env python3
"""Code templates registry - loads all template definitions."""

# Core templates (priority 1-3)
from .audio import TEMPLATE as AUDIO
from .canvas import TEMPLATE as CANVAS
from .capture import TEMPLATE as CAPTURE
from .config import TEMPLATE as CONFIG
from .diagram import TEMPLATE as DIAGRAM
from .io import TEMPLATE as IO

# Module template
from .module import TEMPLATE as MODULE

# Module usage templates (priority 4+)
from .plt import TEMPLATE as PLT
from .scholar import TEMPLATE as SCHOLAR
from .session import TEMPLATE as SESSION

# Session variants
from .session_minimal import TEMPLATE as SESSION_MINIMAL
from .session_plot import TEMPLATE as SESSION_PLOT
from .session_stats import TEMPLATE as SESSION_STATS
from .stats import TEMPLATE as STATS
from .writer import TEMPLATE as WRITER

CODE_TEMPLATES = {
    # Priority 1-3: Core templates
    "session": SESSION,
    "io": IO,
    "config": CONFIG,
    # Session variants
    "session-minimal": SESSION_MINIMAL,
    "session-plot": SESSION_PLOT,
    "session-stats": SESSION_STATS,
    # Module template
    "module": MODULE,
    # Module usage templates
    "plt": PLT,
    "stats": STATS,
    "scholar": SCHOLAR,
    "audio": AUDIO,
    "capture": CAPTURE,
    "diagram": DIAGRAM,
    "canvas": CANVAS,
    "writer": WRITER,
}

__all__ = ["CODE_TEMPLATES"]

# EOF
