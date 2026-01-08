# Alignment API Proposal

## Overview

Add alignment and positioning capabilities to scitex at the **backend/API level**, with MCP tools as thin wrappers.

**Reference Implementation**: `figrecipe._composition._alignment`

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  MCP Tools (thin wrappers)                          │
│  mcp__scitex__canvas_align_entities()               │
│  mcp__scitex__plt_reposition_legend()               │
└─────────────────────┬───────────────────────────────┘
                      │ calls
┌─────────────────────▼───────────────────────────────┐
│  Python API (user-facing)                           │
│  scitex.canvas.align()                              │
│  scitex.canvas.distribute()                         │
│  scitex.plt.reposition_legend()                     │
└─────────────────────┬───────────────────────────────┘
                      │ uses
┌─────────────────────▼───────────────────────────────┐
│  Backend Core (internal)                            │
│  scitex.canvas.alignment._calculate_aligned_bbox()  │
│  scitex.canvas.alignment._distribute_entities()     │
│  scitex.plt.legend._find_optimal_position()         │
└─────────────────────────────────────────────────────┘
```

## Scope

### scitex.canvas (Multi-panel composition)
- Panel/annotation alignment
- Entity distribution
- Grid snapping
- Panel label positioning

### scitex.plt (Individual plot elements)
- Legend repositioning
- Significance bracket styling

## Backend API Design

### Canvas Module

```python
# src/scitex/canvas/alignment.py

from enum import Enum
from typing import List, Optional, Tuple, Union

class AlignmentMode(Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    CENTER_H = "center_h"
    CENTER_V = "center_v"

def align(
    parent_dir: str,
    canvas_name: str,
    entity_names: List[str],
    mode: Union[str, AlignmentMode],
    reference: str = "first",
) -> dict:
    """Align canvas entities to a reference.

    Parameters
    ----------
    parent_dir : str
        Canvas parent directory
    canvas_name : str
        Canvas name
    entity_names : list of str
        Names of entities to align (panels, annotations)
    mode : str or AlignmentMode
        Alignment mode
    reference : str
        Reference entity: "first", "last", or entity name

    Returns
    -------
    dict
        Updated entity positions
    """

def distribute(
    parent_dir: str,
    canvas_name: str,
    entity_names: List[str],
    direction: str = "horizontal",
    spacing_mm: Optional[float] = None,
) -> dict:
    """Distribute entities evenly.

    Parameters
    ----------
    direction : str
        "horizontal" or "vertical"
    spacing_mm : float, optional
        Fixed spacing. Auto-calculate if None.
    """

def snap_to_grid(
    parent_dir: str,
    canvas_name: str,
    entity_name: str,
    grid_mm: float = 5.0,
) -> dict:
    """Snap entity to nearest grid point."""

def set_panel_label(
    parent_dir: str,
    canvas_name: str,
    panel_name: str,
    label: str,
    position: str = "top-left",
    offset_mm: Tuple[float, float] = (-2, 2),
) -> dict:
    """Set panel label with precise positioning."""
```

### Plt Module

```python
# src/scitex/plt/legend.py

def reposition_legend(
    ax,
    position: str = "best",
    offset_mm: Tuple[float, float] = (0, 0),
    avoid_data: bool = True,
) -> None:
    """Reposition legend with optional collision avoidance."""

# src/scitex/plt/stats_annotations.py

def set_bracket_style(
    h_offset_mm: float = 1.0,
    v_offset_mm: float = 2.0,
    bar_height_mm: float = 1.5,
    line_width_mm: float = 0.2,
    text_offset_mm: float = 0.5,
) -> dict:
    """Configure significance bracket appearance."""
```

## MCP Handlers (Thin Wrappers)

```python
# src/scitex/canvas/_mcp_handlers.py

async def align_entities_handler(
    parent_dir: str,
    canvas_name: str,
    entity_names: List[str],
    alignment: str,
    reference: str = "first",
) -> dict:
    """MCP wrapper for canvas.align()"""
    from scitex.canvas import alignment

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: alignment.align(
            parent_dir, canvas_name, entity_names, alignment, reference
        )
    )
    return {"success": True, **result}
```

## File Structure

```
src/scitex/canvas/
├── alignment/
│   ├── __init__.py           # Public API exports
│   ├── _modes.py             # AlignmentMode enum
│   ├── _align.py             # align() implementation
│   ├── _distribute.py        # distribute() implementation
│   └── _snap.py              # snap_to_grid() implementation
├── _mcp_handlers.py          # Add alignment handlers
└── _mcp_tool_schemas.py      # Add tool schemas

src/scitex/plt/
├── legend/
│   ├── __init__.py
│   └── _reposition.py        # reposition_legend()
├── stats_annotations/
│   ├── __init__.py
│   └── _bracket_style.py     # set_bracket_style()
└── _mcp_handlers.py          # Add handlers
```

## Implementation Plan

### Phase 1: Canvas Alignment Backend
1. Create `scitex.canvas.alignment` module
2. Port logic from `figrecipe._composition._alignment`
3. Adapt for canvas JSON schema (mm coordinates)
4. Add unit tests

### Phase 2: Canvas MCP Wrappers
1. Add async handlers in `_mcp_handlers.py`
2. Add tool schemas
3. Register in mcp_server.py

### Phase 3: Plt Positioning
1. Create `scitex.plt.legend` module
2. Create `scitex.plt.stats_annotations` module
3. Add MCP wrappers

## Usage Examples

### Python API
```python
import scitex.canvas as canvas

# Align panels
canvas.align("/tmp", "my_figure", ["panel_a", "panel_b"], mode="left")

# Distribute evenly with 10mm spacing
canvas.distribute("/tmp", "my_figure", ["panel_a", "panel_b", "panel_c"],
                  direction="horizontal", spacing_mm=10)
```

### MCP Tools
```
mcp__scitex__canvas_align_entities(
    parent_dir="/tmp",
    canvas_name="my_figure",
    entity_names=["panel_a", "panel_b"],
    alignment="left"
)
```

## Notes

- Backend handles all coordinate calculations
- MCP tools are pure async wrappers
- Coordinate system: mm externally, figure fraction internally
- Canvas JSON schema stores positions in mm
