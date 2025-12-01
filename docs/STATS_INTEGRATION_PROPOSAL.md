# SciTeX Stats Integration Proposal

## Executive Summary

Standardize statistics integration across `scitex.plt`, `scitex.vis`, and `scitex.stats` with unified metadata schema, automatic tracking, JSON serialization, and GUI-ready positioning.

## Current State

### Existing Infrastructure ✅
- **TrackingMixin**: Captures all plot operations in `_ax_history`
- **Metadata System**: `collect_figure_metadata()` with `_scitex_metadata`
- **JSON Export**: vis module with declarative FigureModel/AxesModel
- **CSV Export**: `export_as_csv()` with 50+ formatters
- **Annotation System**: AnnotationModel for text/arrows

### Gaps ❌
- No automatic stats calculation on plot
- No stats field in vis models
- Manual p-value annotation positioning
- Separate metadata formats (plt vs vis)
- No GUI positioning hints

## Proposed Architecture

### 1. Unified Stats Metadata Schema

```python
# Location: src/scitex/stats/_schema.py

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import json

@dataclass
class StatResult:
    """Standardized statistical test result."""

    # Core test information
    test_type: str  # "t-test", "pearson", "anova", etc.
    test_category: str  # "parametric", "non-parametric", "correlation"

    # Primary results
    statistic: Dict[str, float]  # {"name": "t", "value": 3.45}
    p_value: float
    stars: str  # "***", "**", "*", "ns"

    # Effect size
    effect_size: Optional[Dict[str, Any]] = None  # {"name": "cohens_d", "value": 0.85, "interpretation": "large", "ci_95": [0.42, 1.28]}

    # Corrections
    correction: Optional[Dict[str, Any]] = None  # {"method": "bonferroni", "n_comparisons": 10, "corrected_p": 0.010}

    # Sample information
    samples: Dict[str, Any] = None  # {"group1": {"n": 30, "mean": 5.2}, "group2": {"n": 32, "mean": 6.8}}

    # Assumptions
    assumptions: Optional[Dict[str, Dict]] = None  # {"normality": {"test": "shapiro", "passed": True, "p": 0.23}}

    # Confidence intervals
    ci_95: Optional[List[float]] = None  # For correlation, regression, etc.

    # Visualization hints (GUI-ready)
    annotation: Optional[Dict[str, Any]] = None  # {"position": "auto", "style": {"fontsize": 6}}

    # Metadata
    created_at: Optional[str] = None
    software_version: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> 'StatResult':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'StatResult':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def format_text(self, style: str = "compact") -> str:
        """Format for display on plot."""
        if style == "compact":
            return f"{self.statistic['name']} = {self.statistic['value']:.3f}{self.stars}"
        elif style == "detailed":
            es = f", d = {self.effect_size['value']:.2f}" if self.effect_size else ""
            return f"{self.statistic['name']} = {self.statistic['value']:.3f}, p = {self.p_value:.3e}{es}"
        elif style == "publication":
            return f"({self.statistic['name']} = {self.statistic['value']:.2f}, p {self._format_p()})"

    def _format_p(self) -> str:
        """Format p-value for publication."""
        if self.p_value < 0.001:
            return "< 0.001"
        elif self.p_value < 0.01:
            return f"< 0.01"
        elif self.p_value < 0.05:
            return f"< 0.05"
        else:
            return f"= {self.p_value:.3f}"
```

### 2. Extension to TrackingMixin

```python
# Location: src/scitex/plt/_subplots/_AxisWrapperMixins/_TrackingMixin.py

class TrackingMixin:
    """Existing tracking with stats support."""

    def __init__(self):
        self._ax_history = OrderedDict()
        self._stats_registry = {}  # plot_id -> StatResult

    def add_stats(
        self,
        stat_result: 'StatResult',
        plot_id: Optional[str] = None,
        annotate: bool = True,
        annotation_kwargs: Optional[Dict] = None
    ):
        """
        Add statistical test result to plot.

        Parameters
        ----------
        stat_result : StatResult
            Statistical test result object
        plot_id : str, optional
            ID of plot to associate with (default: last plot)
        annotate : bool
            Whether to add visual annotation to plot
        annotation_kwargs : dict
            Override annotation positioning/styling
        """
        # Get target plot
        if plot_id is None:
            if not self._ax_history:
                raise ValueError("No plots to add stats to")
            plot_id = list(self._ax_history.keys())[-1]

        if plot_id not in self._ax_history:
            raise ValueError(f"Plot ID '{plot_id}' not found in history")

        # Store in registry
        self._stats_registry[plot_id] = stat_result

        # Add to history record
        id_, method_name, tracked_dict, kwargs = self._ax_history[plot_id]
        tracked_dict['stats'] = stat_result.to_dict()
        self._ax_history[plot_id] = (id_, method_name, tracked_dict, kwargs)

        # Add to metadata
        if not hasattr(self, '_scitex_metadata'):
            self._scitex_metadata = {}
        if 'stats' not in self._scitex_metadata:
            self._scitex_metadata['stats'] = []
        self._scitex_metadata['stats'].append({
            'plot_id': plot_id,
            **stat_result.to_dict()
        })

        # Auto-annotate if requested
        if annotate:
            self._auto_annotate_stats(stat_result, plot_id, annotation_kwargs)

        return self

    def _auto_annotate_stats(
        self,
        stat_result: 'StatResult',
        plot_id: str,
        kwargs: Optional[Dict] = None
    ):
        """Automatically add stats annotation to plot."""
        kwargs = kwargs or {}

        # Determine position based on plot type
        position = self._get_smart_position(plot_id)

        # Format text
        text = stat_result.format_text(style=kwargs.get('style', 'compact'))

        # Add annotation
        self._axis_mpl.text(
            position['x'],
            position['y'],
            text,
            transform=self._axis_mpl.transAxes,
            fontsize=kwargs.get('fontsize', 6),
            verticalalignment=position.get('va', 'top'),
            horizontalalignment=position.get('ha', 'right')
        )

    def _get_smart_position(self, plot_id: str) -> Dict[str, float]:
        """Calculate smart annotation position based on plot type and data."""
        # Get plot type
        id_, method_name, tracked_dict, kwargs = self._ax_history[plot_id]

        # Default positions by plot type
        positions = {
            'scatter': {'x': 0.95, 'y': 0.05, 'ha': 'right', 'va': 'bottom'},
            'boxplot': {'x': 0.95, 'y': 0.95, 'ha': 'right', 'va': 'top'},
            'bar': {'x': 0.95, 'y': 0.95, 'ha': 'right', 'va': 'top'},
            'plot': {'x': 0.05, 'y': 0.95, 'ha': 'left', 'va': 'top'},
        }

        return positions.get(method_name, {'x': 0.95, 'y': 0.95, 'ha': 'right', 'va': 'top'})

    def get_stats(self, plot_id: Optional[str] = None) -> 'StatResult':
        """Retrieve stats for a plot."""
        if plot_id is None:
            plot_id = list(self._ax_history.keys())[-1]
        return self._stats_registry.get(plot_id)

    def export_stats_json(self) -> str:
        """Export all stats as JSON."""
        stats_list = [s.to_dict() for s in self._stats_registry.values()]
        return json.dumps(stats_list, indent=2)
```

### 3. Extension to scitex.vis Models

```python
# Location: src/scitex/vis/model/stats.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class StatsModel:
    """Statistical test specification for vis."""

    # Test configuration
    test_type: str  # "t-test", "pearson", etc.
    variables: List[str]  # Variable names/plot IDs to test

    # Results (filled after rendering)
    result: Optional[Dict[str, Any]] = None

    # Display options
    show_annotation: bool = True
    annotation_position: str = "auto"  # "auto", "top-right", "bottom-left", etc.
    annotation_style: str = "compact"  # "compact", "detailed", "publication"

    # Correction
    correction: Optional[str] = None  # "bonferroni", "fdr", etc.

    # Assumptions to check
    check_assumptions: bool = True


@dataclass
class AxesModel:
    """Extended with stats support."""

    row: int = 0
    col: int = 0
    plots: List[Dict] = field(default_factory=list)

    # Labels and limits (existing)
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    title: Optional[str] = None
    xlim: Optional[List[float]] = None
    ylim: Optional[List[float]] = None

    # NEW: Stats field
    stats: List[StatsModel] = field(default_factory=list)

    # Existing fields
    annotations: List[Dict] = field(default_factory=list)
    guides: List[Dict] = field(default_factory=list)
    style: Optional[Dict] = None
```

### 4. High-Level API Functions

```python
# Location: src/scitex/plt/ax/_plot/_plot_with_stats.py

def plot_with_comparison(
    ax,
    groups: List[np.ndarray],
    labels: Optional[List[str]] = None,
    test: str = "auto",
    correction: Optional[str] = None,
    plot_type: str = "boxplot",
    show_stats: bool = True,
    **kwargs
) -> 'StatResult':
    """
    Create plot with automatic statistical comparison.

    Parameters
    ----------
    ax : AxisWrapper
        Axes to plot on
    groups : list of arrays
        Data for each group
    labels : list of str
        Group labels
    test : str
        Statistical test ("auto", "t-test", "anova", "kruskal", etc.)
    correction : str
        Multiple comparison correction
    plot_type : str
        "boxplot", "violin", "bar", etc.
    show_stats : bool
        Whether to annotate significance

    Returns
    -------
    StatResult
        Statistical test result

    Examples
    --------
    >>> fig, ax = stx.plt.subplots(**stx.plt.presets.SCITEX_STYLE)
    >>> result = stx.plt.ax.plot_with_comparison(
    ...     ax,
    ...     groups=[control, treatment],
    ...     labels=["Control", "Treatment"],
    ...     test="t-test"
    ... )
    >>> # Stats automatically annotated and embedded in metadata
    """
    from scitex.stats.tests import auto_test

    # Create plot
    plot_funcs = {
        'boxplot': ax.boxplot,
        'violin': ax.violinplot,
        'bar': ax.bar,
    }

    plot_id = plot_funcs[plot_type](groups, labels=labels, **kwargs)

    # Run statistical test
    if test == "auto":
        test = _select_test(groups)

    result = auto_test(groups, test_type=test, correction=correction)

    # Add to axis
    ax.add_stats(result, plot_id=plot_id, annotate=show_stats)

    return result


def add_pairwise_comparisons(
    ax,
    groups: List[np.ndarray],
    positions: List[float],
    test: str = "auto",
    correction: str = "bonferroni",
    **kwargs
) -> List['StatResult']:
    """
    Add pairwise comparison brackets to plot.

    Examples
    --------
    >>> fig, ax = stx.plt.subplots(**stx.plt.presets.SCITEX_STYLE)
    >>> ax.boxplot([g1, g2, g3], positions=[1, 2, 3])
    >>> results = stx.plt.ax.add_pairwise_comparisons(
    ...     ax,
    ...     groups=[g1, g2, g3],
    ...     positions=[1, 2, 3]
    ... )
    >>> # Adds brackets with *, **, *** above boxes
    """
    # Implementation...
```

### 5. JSON Schema Update

```json
{
  "figure": {
    "width_mm": 40,
    "height_mm": 28,
    "axes": [
      {
        "row": 0,
        "col": 0,
        "plots": [
          {
            "type": "boxplot",
            "data": "[[...], [...]]",
            "labels": ["Control", "Treatment"]
          }
        ],
        "stats": [
          {
            "test_type": "t-test",
            "variables": ["plot_0_group0", "plot_0_group1"],
            "result": {
              "statistic": {"name": "t", "value": 3.45},
              "p_value": 0.001,
              "stars": "***",
              "effect_size": {
                "name": "cohens_d",
                "value": 0.85,
                "interpretation": "large"
              }
            },
            "show_annotation": true,
            "annotation_position": "top-right",
            "annotation_style": "compact"
          }
        ]
      }
    ]
  }
}
```

### 6. GUI-Ready Positioning

```python
# Location: src/scitex/plt/utils/_auto_position.py

class AnnotationPositioner:
    """Smart positioning for stats annotations."""

    def __init__(self, ax):
        self.ax = ax
        self.occupied_regions = []

    def find_empty_region(
        self,
        preferred: str = "top-right",
        min_distance: float = 0.05
    ) -> Dict[str, float]:
        """
        Find empty region for annotation.

        Returns position dict with GUI-friendly metadata:
        {
            'x': 0.95,
            'y': 0.95,
            'ha': 'right',
            'va': 'top',
            'region': 'top-right',
            'priority': 1,  # For GUI reordering
            'collision': False
        }
        """
        # Check data density
        # Avoid overlapping with existing annotations
        # Return position + metadata

    def suggest_positions(self, n: int) -> List[Dict]:
        """Suggest N non-overlapping positions for multiple comparisons."""
        pass
```

## Implementation Phases

### Phase 1: Core Schema (Week 1)
- [ ] Create `StatResult` dataclass
- [ ] Add `to_dict()`, `from_dict()`, `to_json()` methods
- [ ] Write comprehensive tests

### Phase 2: TrackingMixin Extension (Week 2)
- [ ] Add `_stats_registry` to TrackingMixin
- [ ] Implement `add_stats()` method
- [ ] Implement `_auto_annotate_stats()`
- [ ] Add `export_stats_json()`

### Phase 3: Vis Integration (Week 3)
- [ ] Create `StatsModel` dataclass
- [ ] Extend `AxesModel` with stats field
- [ ] Update JSON parser to handle stats
- [ ] Update renderer to run tests on load

### Phase 4: High-Level API (Week 4)
- [ ] Implement `plot_with_comparison()`
- [ ] Implement `add_pairwise_comparisons()`
- [ ] Create example gallery
- [ ] Write documentation

### Phase 5: GUI Positioning (Week 5-6)
- [ ] Implement `AnnotationPositioner`
- [ ] Add collision detection
- [ ] Smart placement algorithm
- [ ] Export position metadata for GUI

## Benefits

1. **Reproducibility**: Full stats embedded in figure metadata
2. **Automation**: One-line statistical plots
3. **Consistency**: Same schema across plt, vis, stats
4. **GUI-Ready**: Position metadata for future tools
5. **Publication-Quality**: Automatic APA/Nature formatting
6. **JSON-First**: Full declarative workflow support

## Migration Path

All changes are backward compatible:
- Existing `ax.plot()` continues to work
- New `ax.add_stats()` is optional
- Stats only added to metadata if explicitly called
- vis JSON schema extended, not replaced

## Example Workflow

```python
# Traditional workflow (still works)
fig, ax = stx.plt.subplots(**stx.plt.presets.SCITEX_STYLE)
ax.boxplot([control, treatment], labels=["Control", "Treatment"])

# New integrated workflow
fig, ax = stx.plt.subplots(**stx.plt.presets.SCITEX_STYLE)
result = stx.plt.ax.plot_with_comparison(
    ax,
    groups=[control, treatment],
    labels=["Control", "Treatment"],
    test="t-test"
)
# Automatic: plot created, stats computed, annotated, metadata embedded

# Export everything
fig.savefig("figure.png")  # Stats embedded in PNG metadata
fig.export_as_csv("data.csv")  # Includes stats in CSV
fig.export_metadata_yaml("metadata.yaml")  # Full stats in YAML
ax.export_stats_json()  # Stats-only JSON

# Later: Reload from JSON
fig2 = stx.vis.load("figure.json")  # Includes stats, reruns tests
```

## Open Questions

1. Should stats be run on `savefig()` or only on explicit `add_stats()`?
2. How to handle stats updates when data changes?
3. Should we support interactive stats parameter tuning in future GUI?
4. How to version stats schema for backward compatibility?

## References

- scitex.plt TrackingMixin: `src/scitex/plt/_subplots/_AxisWrapperMixins/_TrackingMixin.py`
- scitex.vis AxesModel: `src/scitex/vis/model/axes.py`
- scitex.stats tests: `src/scitex/stats/tests/`
