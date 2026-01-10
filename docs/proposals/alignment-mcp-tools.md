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
# src/scitex/canvas/_mcp.handlers.py

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
├── _mcp.handlers.py          # Add alignment handlers
└── _mcp.tool_schemas.py      # Add tool schemas

src/scitex/plt/
├── legend/
│   ├── __init__.py
│   └── _reposition.py        # reposition_legend()
├── stats_annotations/
│   ├── __init__.py
│   └── _bracket_style.py     # set_bracket_style()
└── _mcp.handlers.py          # Add handlers
```

## Implementation Plan

### Phase 1: Canvas Alignment Backend
1. Create `scitex.canvas.alignment` module
2. Port logic from `figrecipe._composition._alignment`
3. Adapt for canvas JSON schema (mm coordinates)
4. Add unit tests

### Phase 2: Canvas MCP Wrappers
1. Add async handlers in `_mcp.handlers.py`
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

## Overlap Detection & Collision Avoidance

### Purpose

Enable agents and users to programmatically detect overlapping elements (legend, traces, annotations, brackets) and auto-adjust positions for better visual clarity.

### Backend API Design

```python
# src/scitex/plt/overlap/

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class BBox:
    """Bounding box in axes coordinates (0-1) or pixel coordinates."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)


def bboxes_overlap(
    bbox1: BBox,
    bbox2: BBox,
    margin: float = 0.0,
) -> bool:
    """Check if two bounding boxes overlap.

    Parameters
    ----------
    bbox1 : BBox
        First bounding box
    bbox2 : BBox
        Second bounding box
    margin : float
        Minimum margin between boxes (same units as bbox)

    Returns
    -------
    bool
        True if boxes overlap (including margin)
    """
    return not (
        bbox1.x1 + margin < bbox2.x0 or
        bbox2.x1 + margin < bbox1.x0 or
        bbox1.y1 + margin < bbox2.y0 or
        bbox2.y1 + margin < bbox1.y0
    )


def extract_element_bbox(
    ax,
    element_type: str,
    element_id: Optional[str] = None,
) -> Optional[BBox]:
    """Extract bounding box for a plot element.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing the element
    element_type : str
        Type: "legend", "title", "xlabel", "ylabel", "annotation", "line", "scatter"
    element_id : str, optional
        Specific element identifier (for annotations/lines)

    Returns
    -------
    BBox or None
        Bounding box in axes coordinates, or None if not found
    """
    pass  # Implementation uses get_window_extent() / get_tightbbox()


def detect_overlaps(
    ax,
    element_types: Optional[List[str]] = None,
    margin_mm: float = 1.0,
) -> List[Tuple[str, str, float]]:
    """Detect all overlapping elements in axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to analyze
    element_types : list of str, optional
        Types to check. Default: ["legend", "annotation", "title"]
    margin_mm : float
        Minimum margin in mm

    Returns
    -------
    list of tuple
        Each tuple: (element1_id, element2_id, overlap_area)
    """
    pass


def find_non_overlapping_position(
    ax,
    bbox_size: Tuple[float, float],
    preferred_positions: List[str] = None,
    avoid_elements: List[str] = None,
    margin_mm: float = 2.0,
) -> Tuple[float, float]:
    """Find optimal position that avoids overlaps.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes
    bbox_size : tuple
        (width, height) in axes fraction
    preferred_positions : list of str
        Preference order: ["upper right", "upper left", "lower right", ...]
    avoid_elements : list of str
        Element types to avoid: ["legend", "data", "annotation"]
    margin_mm : float
        Minimum margin from other elements

    Returns
    -------
    tuple
        (x, y) position in axes fraction
    """
    pass


def auto_adjust_legend(
    ax,
    avoid_data: bool = True,
    avoid_annotations: bool = True,
    margin_mm: float = 2.0,
) -> Dict[str, Any]:
    """Automatically reposition legend to avoid overlaps.

    Returns
    -------
    dict
        {"moved": bool, "old_position": ..., "new_position": ..., "avoided": [...]}
    """
    pass


def auto_stack_annotations(
    ax,
    annotations: List[dict],
    direction: str = "vertical",
    spacing_mm: float = 1.5,
) -> List[dict]:
    """Stack overlapping annotations vertically or horizontally.

    Used for significance brackets that would otherwise overlap.

    Returns
    -------
    list of dict
        Updated annotation positions
    """
    pass
```

### Canvas Overlap Detection

```python
# src/scitex/canvas/overlap/

def detect_panel_overlaps(
    parent_dir: str,
    canvas_name: str,
    margin_mm: float = 0.0,
) -> List[Tuple[str, str, float]]:
    """Detect overlapping panels in canvas.

    Returns
    -------
    list of tuple
        Each tuple: (panel1_name, panel2_name, overlap_area_mm2)
    """
    pass


def auto_distribute_overlapping(
    parent_dir: str,
    canvas_name: str,
    direction: str = "horizontal",
    margin_mm: float = 5.0,
) -> dict:
    """Automatically distribute panels that overlap.

    Returns
    -------
    dict
        {"adjusted": [...], "new_positions": {...}}
    """
    pass
```

### MCP Tools

```python
# Thin wrappers for overlap detection

mcp__scitex__plt_detect_overlaps(
    axes_id: str,
    element_types: List[str] = ["legend", "annotation"],
    margin_mm: float = 1.0,
) -> dict

mcp__scitex__plt_auto_adjust_legend(
    axes_id: str,
    avoid_data: bool = True,
    margin_mm: float = 2.0,
) -> dict

mcp__scitex__canvas_detect_overlaps(
    parent_dir: str,
    canvas_name: str,
    margin_mm: float = 0.0,
) -> dict
```

### Reference Implementation

Existing overlap detection in `figrecipe._wrappers._stat_annotation`:

```python
# Check for overlaps with existing annotations
for ann in existing_annotations:
    ann_x1, ann_x2 = ann.get("x1", 0), ann.get("x2", 0)
    ann_y = ann.get("y", 0)
    # Check if x ranges overlap
    if not (x2 < ann_x1 or x1 > ann_x2):
        # Overlapping x range, need to stack
        y = max(y, ann_y + pad * 2)
```

### Implementation Plan Update

**Phase 4: Overlap Detection & Auto-Adjust**
1. Create `scitex.plt.overlap` module with bbox extraction
2. Port overlap detection from figrecipe stat_annotation
3. Implement `detect_overlaps()` for all element types
4. Add `auto_adjust_legend()` with collision avoidance
5. Add canvas-level overlap detection
6. Add MCP wrappers
7. Unit tests with visual validation

## Notes

- Backend handles all coordinate calculations
- MCP tools are pure async wrappers
- Coordinate system: mm externally, figure fraction internally
- Canvas JSON schema stores positions in mm
- Overlap detection uses matplotlib's `get_window_extent()` and `get_tightbbox()`
- Auto-adjust prioritizes least visual disruption
