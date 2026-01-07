#!/usr/bin/env python3
# File: ./src/scitex/vis/model/figure.py
"""Figure JSON model for scitex.canvas."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FigureModel:
    """
    Top-level figure model representing a complete publication figure.

    This model captures all information needed to recreate a figure:
    - Physical dimensions (mm units)
    - Subplot layout
    - Individual axes configurations
    - Global figure properties

    All measurements use mm units for publication compatibility.
    """

    # Physical dimensions (mm)
    width_mm: float
    height_mm: float

    # Subplot layout
    nrows: int = 1
    ncols: int = 1

    # Axes configurations (list of AxesModel)
    axes: List[Dict[str, Any]] = field(default_factory=list)

    # Figure-level properties
    dpi: int = 300
    facecolor: str = "white"
    edgecolor: str = "none"

    # Spacing (mm)
    left_mm: Optional[float] = None
    right_mm: Optional[float] = None
    top_mm: Optional[float] = None
    bottom_mm: Optional[float] = None
    wspace_mm: Optional[float] = None
    hspace_mm: Optional[float] = None

    # Suptitle
    suptitle: Optional[str] = None
    suptitle_fontsize: Optional[float] = None
    suptitle_fontweight: Optional[str] = None
    suptitle_y: Optional[float] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Figure ID for tracking
    figure_id: Optional[str] = None

    # Version for schema evolution
    schema_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FigureModel":
        """Create FigureModel from dictionary."""
        return cls(**data)

    def validate(self) -> bool:
        """
        Validate the figure model.

        Returns
        -------
        bool
            True if valid, raises ValueError otherwise
        """
        if self.width_mm <= 0:
            raise ValueError(f"width_mm must be positive, got {self.width_mm}")

        if self.height_mm <= 0:
            raise ValueError(f"height_mm must be positive, got {self.height_mm}")

        if self.nrows <= 0:
            raise ValueError(f"nrows must be positive, got {self.nrows}")

        if self.ncols <= 0:
            raise ValueError(f"ncols must be positive, got {self.ncols}")

        if self.dpi <= 0:
            raise ValueError(f"dpi must be positive, got {self.dpi}")

        # Validate that axes count matches layout
        expected_axes = self.nrows * self.ncols
        if len(self.axes) > expected_axes:
            raise ValueError(
                f"Too many axes: expected {expected_axes} for {self.nrows}x{self.ncols} layout, "
                f"got {len(self.axes)}"
            )

        return True

    def get_axes_by_position(self, row: int, col: int) -> Optional[Dict[str, Any]]:
        """
        Get axes configuration by subplot position.

        Parameters
        ----------
        row : int
            Row index (0-based)
        col : int
            Column index (0-based)

        Returns
        -------
        Optional[Dict[str, Any]]
            Axes configuration or None if not found
        """
        idx = row * self.ncols + col
        if idx < len(self.axes):
            return self.axes[idx]
        return None

    def add_axes(self, axes_config: Dict[str, Any]) -> None:
        """
        Add an axes configuration to the figure.

        Parameters
        ----------
        axes_config : Dict[str, Any]
            Axes configuration dictionary
        """
        self.axes.append(axes_config)


# EOF
