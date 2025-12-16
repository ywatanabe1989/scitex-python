#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: ./src/scitex/vis/editor/_qt_editor.py
"""Qt-based figure editor with rich desktop UI."""

from pathlib import Path
from typing import Dict, Any, Optional
import copy


def _get_qt():
    """Get Qt bindings (PyQt6, PyQt5, PySide6, or PySide2)."""
    try:
        from PyQt6 import QtWidgets, QtCore, QtGui

        return QtWidgets, QtCore, QtGui, "PyQt6"
    except ImportError:
        pass

    try:
        from PyQt5 import QtWidgets, QtCore, QtGui

        return QtWidgets, QtCore, QtGui, "PyQt5"
    except ImportError:
        pass

    try:
        from PySide6 import QtWidgets, QtCore, QtGui

        return QtWidgets, QtCore, QtGui, "PySide6"
    except ImportError:
        pass

    try:
        from PySide2 import QtWidgets, QtCore, QtGui

        return QtWidgets, QtCore, QtGui, "PySide2"
    except ImportError:
        pass

    raise ImportError(
        "Qt backend requires PyQt6, PyQt5, PySide6, or PySide2. "
        "Install with: pip install PyQt6"
    )


class QtEditor:
    """
    Rich desktop figure editor using Qt (PyQt6/5 or PySide6/2).

    Features:
    - Native desktop UI with dockable panels
    - Embedded matplotlib canvas with navigation
    - Property editors with spinboxes, sliders, color pickers
    - Real-time preview updates
    - Save to .manual.json
    - SciTeX style defaults pre-filled
    - Hitmap-based element selection (when available)
    """

    def __init__(
        self,
        json_path: Path,
        metadata: Dict[str, Any],
        csv_data: Optional[Any] = None,
        png_path: Optional[Path] = None,
        manual_overrides: Optional[Dict[str, Any]] = None,
        hitmap_path: Optional[Path] = None,
        bundle_spec: Optional[Dict[str, Any]] = None,
    ):
        self.json_path = Path(json_path)
        self.metadata = metadata
        self.csv_data = csv_data
        self.png_path = Path(png_path) if png_path else None
        self.manual_overrides = manual_overrides or {}
        self.hitmap_path = Path(hitmap_path) if hitmap_path else None
        self.bundle_spec = bundle_spec or {}

        # Load hitmap image if available
        self.hitmap_array = None
        if self.hitmap_path and self.hitmap_path.exists():
            try:
                from PIL import Image
                import numpy as np
                hitmap_img = Image.open(self.hitmap_path).convert('RGB')
                self.hitmap_array = np.array(hitmap_img)
            except Exception:
                pass

        # Extract hit_regions and selectable_regions from spec
        self.hit_regions = self.bundle_spec.get('hit_regions', {})
        self.selectable_regions = self.bundle_spec.get('selectable_regions', {})
        self.color_map = self.hit_regions.get('color_map', {})

        # Get SciTeX defaults and merge with metadata
        from ._defaults import get_scitex_defaults, extract_defaults_from_metadata

        self.scitex_defaults = get_scitex_defaults()
        self.metadata_defaults = extract_defaults_from_metadata(metadata)

        # Start with defaults, then overlay manual overrides
        self.current_overrides = copy.deepcopy(self.scitex_defaults)
        self.current_overrides.update(self.metadata_defaults)
        self.current_overrides.update(self.manual_overrides)

        # Track modifications
        self._initial_overrides = copy.deepcopy(self.current_overrides)
        self._user_modified = False

        # Qt components (initialized in run())
        self.app = None
        self.main_window = None
        self.canvas = None
        self.fig = None
        self.ax = None

    def run(self):
        """Launch the Qt editor."""
        QtWidgets, QtCore, QtGui, qt_version = _get_qt()

        # Create application
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        # Create main window
        self.main_window = QtEditorWindow(self, QtWidgets, QtCore, QtGui, qt_version)
        self.main_window.show()

        # Start event loop
        self.app.exec() if hasattr(self.app, "exec") else self.app.exec_()


class QtEditorWindow:
    """Qt main window for the editor."""

    def __init__(self, editor, QtWidgets, QtCore, QtGui, qt_version):
        self.editor = editor
        self.QtWidgets = QtWidgets
        self.QtCore = QtCore
        self.QtGui = QtGui
        self.qt_version = qt_version

        # Create main window
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle(f"SciTeX Editor - {editor.json_path.name}")
        self.window.resize(1400, 900)

        # Central widget with splitter
        central_widget = QtWidgets.QWidget()
        self.window.setCentralWidget(central_widget)

        layout = QtWidgets.QHBoxLayout(central_widget)

        # Splitter for resizable panels
        splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Horizontal
            if hasattr(QtCore.Qt, "Orientation")
            else QtCore.Qt.Horizontal
        )
        layout.addWidget(splitter)

        # Left panel: Canvas
        self._create_canvas_panel(splitter)

        # Right panel: Controls
        self._create_control_panel(splitter)

        # Set splitter sizes
        splitter.setSizes([900, 400])

        # Create toolbar
        self._create_toolbar()

        # Create status bar
        self.status_bar = self.window.statusBar()
        self.status_bar.showMessage("Ready")

        # Initial render
        self._render_figure()

    def show(self):
        """Show the window."""
        self.window.show()

    def _create_canvas_panel(self, parent):
        """Create the matplotlib canvas panel with hitmap-based selection."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        from matplotlib.figure import Figure

        canvas_widget = self.QtWidgets.QWidget()
        canvas_layout = self.QtWidgets.QVBoxLayout(canvas_widget)

        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Create canvas
        self.canvas = FigureCanvasQTAgg(self.fig)
        canvas_layout.addWidget(self.canvas)

        # Navigation toolbar
        toolbar = NavigationToolbar2QT(self.canvas, canvas_widget)
        canvas_layout.addWidget(toolbar)

        # Connect click event for hitmap-based selection
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

        # Selection info label
        self.selection_label = self.QtWidgets.QLabel("Click on figure to select element")
        self.selection_label.setStyleSheet("padding: 5px; background: #f0f0f0; border-radius: 3px;")
        canvas_layout.addWidget(self.selection_label)

        parent.addWidget(canvas_widget)

    def _on_canvas_click(self, event):
        """Handle canvas click - use hitmap to identify clicked element."""
        if event.inaxes is None:
            return

        # Get pixel coordinates
        x_px = int(event.x)
        y_px = int(self.canvas.get_width_height()[1] - event.y)  # Flip Y

        # First check selectable_regions (bounding boxes for axis elements)
        selection_info = self._check_selectable_regions(x_px, y_px)

        # If no selectable_region hit, try hitmap for data elements
        if selection_info is None and self.editor.hitmap_array is not None:
            selection_info = self._check_hitmap(x_px, y_px)

        # Update selection display
        if selection_info:
            self._show_selection(selection_info)
        else:
            self.selection_label.setText(f"No element at ({x_px}, {y_px})")
            self.status_bar.showMessage(f"Click at ({x_px}, {y_px}) - no element found")

    def _check_selectable_regions(self, x_px: int, y_px: int) -> Optional[Dict[str, Any]]:
        """Check if click is within any selectable_region bounding box."""
        regions = self.editor.selectable_regions
        if not regions or 'axes' not in regions:
            return None

        for ax_info in regions['axes']:
            ax_idx = ax_info.get('index', 0)

            # Check title
            if 'title' in ax_info:
                bbox = ax_info['title'].get('bbox_px', [])
                if len(bbox) == 4 and self._point_in_bbox(x_px, y_px, bbox):
                    return {
                        'type': 'title',
                        'axes_index': ax_idx,
                        'text': ax_info['title'].get('text', ''),
                        'element': 'title',
                    }

            # Check xlabel
            if 'xlabel' in ax_info:
                bbox = ax_info['xlabel'].get('bbox_px', [])
                if len(bbox) == 4 and self._point_in_bbox(x_px, y_px, bbox):
                    return {
                        'type': 'xlabel',
                        'axes_index': ax_idx,
                        'text': ax_info['xlabel'].get('text', ''),
                        'element': 'xlabel',
                    }

            # Check ylabel
            if 'ylabel' in ax_info:
                bbox = ax_info['ylabel'].get('bbox_px', [])
                if len(bbox) == 4 and self._point_in_bbox(x_px, y_px, bbox):
                    return {
                        'type': 'ylabel',
                        'axes_index': ax_idx,
                        'text': ax_info['ylabel'].get('text', ''),
                        'element': 'ylabel',
                    }

            # Check legend
            if 'legend' in ax_info:
                bbox = ax_info['legend'].get('bbox_px', [])
                if len(bbox) == 4 and self._point_in_bbox(x_px, y_px, bbox):
                    return {
                        'type': 'legend',
                        'axes_index': ax_idx,
                        'element': 'legend',
                    }

            # Check xaxis elements
            if 'xaxis' in ax_info:
                xaxis = ax_info['xaxis']
                if 'spine' in xaxis and xaxis['spine']:
                    bbox = xaxis['spine'].get('bbox_px', [])
                    if len(bbox) == 4 and self._point_in_bbox(x_px, y_px, bbox):
                        return {'type': 'xaxis_spine', 'axes_index': ax_idx, 'element': 'xaxis'}

            # Check yaxis elements
            if 'yaxis' in ax_info:
                yaxis = ax_info['yaxis']
                if 'spine' in yaxis and yaxis['spine']:
                    bbox = yaxis['spine'].get('bbox_px', [])
                    if len(bbox) == 4 and self._point_in_bbox(x_px, y_px, bbox):
                        return {'type': 'yaxis_spine', 'axes_index': ax_idx, 'element': 'yaxis'}

        return None

    def _check_hitmap(self, x_px: int, y_px: int) -> Optional[Dict[str, Any]]:
        """Check hitmap for data element at pixel position."""
        hitmap = self.editor.hitmap_array
        color_map = self.editor.color_map

        if hitmap is None or not color_map:
            return None

        # Bounds check
        h, w = hitmap.shape[:2]
        if x_px < 0 or x_px >= w or y_px < 0 or y_px >= h:
            return None

        # Get RGB at position
        r, g, b = hitmap[y_px, x_px]

        # Skip background (black) and axes elements (dark gray)
        if (r, g, b) == (0, 0, 0):  # Background
            return None
        if (r, g, b) == (64, 64, 64):  # Axes color (#404040)
            return None

        # Convert RGB to ID
        element_id = r * 65536 + g * 256 + b

        # Look up in color_map
        if str(element_id) in color_map:
            info = color_map[str(element_id)]
            return {
                'type': 'data_element',
                'element_id': element_id,
                'element_type': info.get('type', 'unknown'),
                'label': info.get('label', ''),
                'axes_index': info.get('axes_index', 0),
                'rgb': [r, g, b],
            }

        return None

    def _point_in_bbox(self, x: int, y: int, bbox: list) -> bool:
        """Check if point (x, y) is within bounding box [x0, y0, x1, y1]."""
        x0, y0, x1, y1 = bbox
        return x0 <= x <= x1 and y0 <= y <= y1

    def _show_selection(self, info: Dict[str, Any]):
        """Display selection info in UI."""
        if info['type'] == 'data_element':
            text = f"Selected: {info['element_type']} - {info.get('label', 'unnamed')}"
            detail = f"ID: {info['element_id']}, Axes: {info['axes_index']}"
        else:
            text = f"Selected: {info['type']}"
            if 'text' in info:
                text += f" - \"{info['text']}\""
            detail = f"Axes: {info.get('axes_index', 0)}"

        self.selection_label.setText(f"{text}\n{detail}")
        self.status_bar.showMessage(text)

    def _create_control_panel(self, parent):
        """Create the control panel with property editors."""
        scroll_area = self.QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(350)

        control_widget = self.QtWidgets.QWidget()
        control_layout = self.QtWidgets.QVBoxLayout(control_widget)
        control_layout.setSpacing(10)

        # Labels Section
        self._create_labels_section(control_layout)

        # Axis Limits Section
        self._create_limits_section(control_layout)

        # Line Style Section
        self._create_line_style_section(control_layout)

        # Font Settings Section
        self._create_font_section(control_layout)

        # Tick Settings Section
        self._create_tick_section(control_layout)

        # Style Section
        self._create_style_section(control_layout)

        # Legend Section
        self._create_legend_section(control_layout)

        # Dimensions Section
        self._create_dimensions_section(control_layout)

        # Annotations Section
        self._create_annotations_section(control_layout)

        # Add stretch at bottom
        control_layout.addStretch()

        scroll_area.setWidget(control_widget)
        parent.addWidget(scroll_area)

    def _create_group_box(self, title: str) -> "QtWidgets.QGroupBox":
        """Create a collapsible group box."""
        group = self.QtWidgets.QGroupBox(title)
        group.setCheckable(True)
        group.setChecked(True)
        return group

    def _create_labels_section(self, parent_layout):
        """Create labels section."""
        group = self._create_group_box("Labels")
        layout = self.QtWidgets.QFormLayout(group)

        self.title_edit = self.QtWidgets.QLineEdit(
            self.editor.current_overrides.get("title", "")
        )
        self.title_edit.editingFinished.connect(self._on_value_change)
        layout.addRow("Title:", self.title_edit)

        self.xlabel_edit = self.QtWidgets.QLineEdit(
            self.editor.current_overrides.get("xlabel", "")
        )
        self.xlabel_edit.editingFinished.connect(self._on_value_change)
        layout.addRow("X Label:", self.xlabel_edit)

        self.ylabel_edit = self.QtWidgets.QLineEdit(
            self.editor.current_overrides.get("ylabel", "")
        )
        self.ylabel_edit.editingFinished.connect(self._on_value_change)
        layout.addRow("Y Label:", self.ylabel_edit)

        parent_layout.addWidget(group)

    def _create_limits_section(self, parent_layout):
        """Create axis limits section."""
        group = self._create_group_box("Axis Limits")
        layout = self.QtWidgets.QGridLayout(group)

        xlim = self.editor.current_overrides.get("xlim", [0, 1])
        ylim = self.editor.current_overrides.get("ylim", [0, 1])

        layout.addWidget(self.QtWidgets.QLabel("X Min:"), 0, 0)
        self.xmin_spin = self.QtWidgets.QDoubleSpinBox()
        self.xmin_spin.setRange(-1e9, 1e9)
        self.xmin_spin.setDecimals(4)
        self.xmin_spin.setValue(xlim[0] if xlim else 0)
        layout.addWidget(self.xmin_spin, 0, 1)

        layout.addWidget(self.QtWidgets.QLabel("X Max:"), 0, 2)
        self.xmax_spin = self.QtWidgets.QDoubleSpinBox()
        self.xmax_spin.setRange(-1e9, 1e9)
        self.xmax_spin.setDecimals(4)
        self.xmax_spin.setValue(xlim[1] if xlim else 1)
        layout.addWidget(self.xmax_spin, 0, 3)

        layout.addWidget(self.QtWidgets.QLabel("Y Min:"), 1, 0)
        self.ymin_spin = self.QtWidgets.QDoubleSpinBox()
        self.ymin_spin.setRange(-1e9, 1e9)
        self.ymin_spin.setDecimals(4)
        self.ymin_spin.setValue(ylim[0] if ylim else 0)
        layout.addWidget(self.ymin_spin, 1, 1)

        layout.addWidget(self.QtWidgets.QLabel("Y Max:"), 1, 2)
        self.ymax_spin = self.QtWidgets.QDoubleSpinBox()
        self.ymax_spin.setRange(-1e9, 1e9)
        self.ymax_spin.setDecimals(4)
        self.ymax_spin.setValue(ylim[1] if ylim else 1)
        layout.addWidget(self.ymax_spin, 1, 3)

        apply_btn = self.QtWidgets.QPushButton("Apply Limits")
        apply_btn.clicked.connect(self._apply_limits)
        layout.addWidget(apply_btn, 2, 0, 1, 4)

        parent_layout.addWidget(group)

    def _create_line_style_section(self, parent_layout):
        """Create line style section."""
        group = self._create_group_box("Line Style")
        layout = self.QtWidgets.QFormLayout(group)

        self.linewidth_spin = self.QtWidgets.QDoubleSpinBox()
        self.linewidth_spin.setRange(0.1, 10.0)
        self.linewidth_spin.setSingleStep(0.1)
        self.linewidth_spin.setValue(
            self.editor.current_overrides.get("linewidth", 1.0)
        )
        self.linewidth_spin.valueChanged.connect(self._on_value_change)
        layout.addRow("Line Width (pt):", self.linewidth_spin)

        parent_layout.addWidget(group)

    def _create_font_section(self, parent_layout):
        """Create font settings section."""
        group = self._create_group_box("Font Settings")
        layout = self.QtWidgets.QFormLayout(group)

        self.title_fontsize_spin = self.QtWidgets.QSpinBox()
        self.title_fontsize_spin.setRange(6, 24)
        self.title_fontsize_spin.setValue(
            self.editor.current_overrides.get("title_fontsize", 8)
        )
        self.title_fontsize_spin.valueChanged.connect(self._on_value_change)
        layout.addRow("Title Font Size:", self.title_fontsize_spin)

        self.axis_fontsize_spin = self.QtWidgets.QSpinBox()
        self.axis_fontsize_spin.setRange(6, 24)
        self.axis_fontsize_spin.setValue(
            self.editor.current_overrides.get("axis_fontsize", 7)
        )
        self.axis_fontsize_spin.valueChanged.connect(self._on_value_change)
        layout.addRow("Axis Font Size:", self.axis_fontsize_spin)

        self.tick_fontsize_spin = self.QtWidgets.QSpinBox()
        self.tick_fontsize_spin.setRange(6, 24)
        self.tick_fontsize_spin.setValue(
            self.editor.current_overrides.get("tick_fontsize", 7)
        )
        self.tick_fontsize_spin.valueChanged.connect(self._on_value_change)
        layout.addRow("Tick Font Size:", self.tick_fontsize_spin)

        self.legend_fontsize_spin = self.QtWidgets.QSpinBox()
        self.legend_fontsize_spin.setRange(4, 20)
        self.legend_fontsize_spin.setValue(
            self.editor.current_overrides.get("legend_fontsize", 6)
        )
        self.legend_fontsize_spin.valueChanged.connect(self._on_value_change)
        layout.addRow("Legend Font Size:", self.legend_fontsize_spin)

        parent_layout.addWidget(group)

    def _create_tick_section(self, parent_layout):
        """Create tick settings section."""
        group = self._create_group_box("Tick Settings")
        layout = self.QtWidgets.QFormLayout(group)

        self.n_ticks_spin = self.QtWidgets.QSpinBox()
        self.n_ticks_spin.setRange(2, 15)
        self.n_ticks_spin.setValue(self.editor.current_overrides.get("n_ticks", 4))
        self.n_ticks_spin.valueChanged.connect(self._on_value_change)
        layout.addRow("N Ticks:", self.n_ticks_spin)

        self.tick_length_spin = self.QtWidgets.QDoubleSpinBox()
        self.tick_length_spin.setRange(0.1, 5.0)
        self.tick_length_spin.setSingleStep(0.1)
        self.tick_length_spin.setValue(
            self.editor.current_overrides.get("tick_length", 0.8)
        )
        self.tick_length_spin.valueChanged.connect(self._on_value_change)
        layout.addRow("Tick Length (mm):", self.tick_length_spin)

        self.tick_width_spin = self.QtWidgets.QDoubleSpinBox()
        self.tick_width_spin.setRange(0.05, 2.0)
        self.tick_width_spin.setSingleStep(0.05)
        self.tick_width_spin.setValue(
            self.editor.current_overrides.get("tick_width", 0.2)
        )
        self.tick_width_spin.valueChanged.connect(self._on_value_change)
        layout.addRow("Tick Width (mm):", self.tick_width_spin)

        self.tick_direction_combo = self.QtWidgets.QComboBox()
        self.tick_direction_combo.addItems(["out", "in", "inout"])
        self.tick_direction_combo.setCurrentText(
            self.editor.current_overrides.get("tick_direction", "out")
        )
        self.tick_direction_combo.currentTextChanged.connect(self._on_value_change)
        layout.addRow("Tick Direction:", self.tick_direction_combo)

        parent_layout.addWidget(group)

    def _create_style_section(self, parent_layout):
        """Create style section."""
        group = self._create_group_box("Style")
        layout = self.QtWidgets.QFormLayout(group)

        self.grid_check = self.QtWidgets.QCheckBox()
        self.grid_check.setChecked(self.editor.current_overrides.get("grid", False))
        self.grid_check.stateChanged.connect(self._on_value_change)
        layout.addRow("Show Grid:", self.grid_check)

        self.hide_top_spine_check = self.QtWidgets.QCheckBox()
        self.hide_top_spine_check.setChecked(
            self.editor.current_overrides.get("hide_top_spine", True)
        )
        self.hide_top_spine_check.stateChanged.connect(self._on_value_change)
        layout.addRow("Hide Top Spine:", self.hide_top_spine_check)

        self.hide_right_spine_check = self.QtWidgets.QCheckBox()
        self.hide_right_spine_check.setChecked(
            self.editor.current_overrides.get("hide_right_spine", True)
        )
        self.hide_right_spine_check.stateChanged.connect(self._on_value_change)
        layout.addRow("Hide Right Spine:", self.hide_right_spine_check)

        self.transparent_check = self.QtWidgets.QCheckBox()
        self.transparent_check.setChecked(
            self.editor.current_overrides.get("transparent", True)
        )
        self.transparent_check.stateChanged.connect(self._on_value_change)
        layout.addRow("Transparent BG:", self.transparent_check)

        self.axis_width_spin = self.QtWidgets.QDoubleSpinBox()
        self.axis_width_spin.setRange(0.05, 2.0)
        self.axis_width_spin.setSingleStep(0.05)
        self.axis_width_spin.setValue(
            self.editor.current_overrides.get("axis_width", 0.2)
        )
        self.axis_width_spin.valueChanged.connect(self._on_value_change)
        layout.addRow("Axis Width (mm):", self.axis_width_spin)

        # Background color button
        self.bg_color = self.editor.current_overrides.get("facecolor", "#ffffff")
        self.bg_color_btn = self.QtWidgets.QPushButton("Choose...")
        self.bg_color_btn.clicked.connect(self._choose_bg_color)
        layout.addRow("Background Color:", self.bg_color_btn)

        parent_layout.addWidget(group)

    def _create_legend_section(self, parent_layout):
        """Create legend section."""
        group = self._create_group_box("Legend")
        layout = self.QtWidgets.QFormLayout(group)

        self.legend_visible_check = self.QtWidgets.QCheckBox()
        self.legend_visible_check.setChecked(
            self.editor.current_overrides.get("legend_visible", True)
        )
        self.legend_visible_check.stateChanged.connect(self._on_value_change)
        layout.addRow("Show Legend:", self.legend_visible_check)

        self.legend_frameon_check = self.QtWidgets.QCheckBox()
        self.legend_frameon_check.setChecked(
            self.editor.current_overrides.get("legend_frameon", False)
        )
        self.legend_frameon_check.stateChanged.connect(self._on_value_change)
        layout.addRow("Show Frame:", self.legend_frameon_check)

        self.legend_loc_combo = self.QtWidgets.QComboBox()
        self.legend_loc_combo.addItems(
            [
                "best",
                "upper right",
                "upper left",
                "lower right",
                "lower left",
                "center right",
                "center left",
                "upper center",
                "lower center",
                "center",
            ]
        )
        self.legend_loc_combo.setCurrentText(
            self.editor.current_overrides.get("legend_loc", "best")
        )
        self.legend_loc_combo.currentTextChanged.connect(self._on_value_change)
        layout.addRow("Position:", self.legend_loc_combo)

        parent_layout.addWidget(group)

    def _create_dimensions_section(self, parent_layout):
        """Create dimensions section."""
        group = self._create_group_box("Dimensions")
        layout = self.QtWidgets.QFormLayout(group)

        fig_size = self.editor.current_overrides.get("fig_size", [3.15, 2.68])

        self.fig_width_spin = self.QtWidgets.QDoubleSpinBox()
        self.fig_width_spin.setRange(1.0, 20.0)
        self.fig_width_spin.setSingleStep(0.1)
        self.fig_width_spin.setValue(fig_size[0])
        layout.addRow("Width (in):", self.fig_width_spin)

        self.fig_height_spin = self.QtWidgets.QDoubleSpinBox()
        self.fig_height_spin.setRange(1.0, 20.0)
        self.fig_height_spin.setSingleStep(0.1)
        self.fig_height_spin.setValue(fig_size[1])
        layout.addRow("Height (in):", self.fig_height_spin)

        self.dpi_spin = self.QtWidgets.QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(self.editor.current_overrides.get("dpi", 300))
        self.dpi_spin.valueChanged.connect(self._on_value_change)
        layout.addRow("DPI:", self.dpi_spin)

        parent_layout.addWidget(group)

    def _create_annotations_section(self, parent_layout):
        """Create annotations section."""
        group = self._create_group_box("Annotations")
        layout = self.QtWidgets.QVBoxLayout(group)

        # Input fields
        form_layout = self.QtWidgets.QFormLayout()

        self.annot_text_edit = self.QtWidgets.QLineEdit()
        form_layout.addRow("Text:", self.annot_text_edit)

        pos_layout = self.QtWidgets.QHBoxLayout()
        self.annot_x_spin = self.QtWidgets.QDoubleSpinBox()
        self.annot_x_spin.setRange(0, 1)
        self.annot_x_spin.setSingleStep(0.05)
        self.annot_x_spin.setValue(0.5)
        pos_layout.addWidget(self.QtWidgets.QLabel("X:"))
        pos_layout.addWidget(self.annot_x_spin)

        self.annot_y_spin = self.QtWidgets.QDoubleSpinBox()
        self.annot_y_spin.setRange(0, 1)
        self.annot_y_spin.setSingleStep(0.05)
        self.annot_y_spin.setValue(0.5)
        pos_layout.addWidget(self.QtWidgets.QLabel("Y:"))
        pos_layout.addWidget(self.annot_y_spin)

        form_layout.addRow("Position:", pos_layout)
        layout.addLayout(form_layout)

        add_btn = self.QtWidgets.QPushButton("Add Annotation")
        add_btn.clicked.connect(self._add_annotation)
        layout.addWidget(add_btn)

        # Annotation list
        self.annot_list = self.QtWidgets.QListWidget()
        self.annot_list.setMaximumHeight(100)
        layout.addWidget(self.annot_list)

        remove_btn = self.QtWidgets.QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_annotation)
        layout.addWidget(remove_btn)

        self._update_annotations_list()

        parent_layout.addWidget(group)

    def _create_toolbar(self):
        """Create the main toolbar."""
        toolbar = self.window.addToolBar("Main")

        # Update action
        update_action = toolbar.addAction("Update Preview")
        update_action.triggered.connect(self._render_figure)

        # Save action
        save_action = toolbar.addAction("Save")
        save_action.triggered.connect(self._save_manual)

        # Reset action
        reset_action = toolbar.addAction("Reset")
        reset_action.triggered.connect(self._reset_overrides)

        toolbar.addSeparator()

        # Export action
        export_action = toolbar.addAction("Export PNG")
        export_action.triggered.connect(self._export_png)

    def _on_value_change(self, value=None):
        """Handle value changes from widgets."""
        self.editor._user_modified = True
        self._collect_overrides()
        self._render_figure()

    def _collect_overrides(self):
        """Collect current values from all widgets."""
        o = self.editor.current_overrides

        # Labels
        o["title"] = self.title_edit.text()
        o["xlabel"] = self.xlabel_edit.text()
        o["ylabel"] = self.ylabel_edit.text()

        # Line style
        o["linewidth"] = self.linewidth_spin.value()

        # Font settings
        o["title_fontsize"] = self.title_fontsize_spin.value()
        o["axis_fontsize"] = self.axis_fontsize_spin.value()
        o["tick_fontsize"] = self.tick_fontsize_spin.value()
        o["legend_fontsize"] = self.legend_fontsize_spin.value()

        # Tick settings
        o["n_ticks"] = self.n_ticks_spin.value()
        o["tick_length"] = self.tick_length_spin.value()
        o["tick_width"] = self.tick_width_spin.value()
        o["tick_direction"] = self.tick_direction_combo.currentText()

        # Style
        o["grid"] = self.grid_check.isChecked()
        o["hide_top_spine"] = self.hide_top_spine_check.isChecked()
        o["hide_right_spine"] = self.hide_right_spine_check.isChecked()
        o["transparent"] = self.transparent_check.isChecked()
        o["axis_width"] = self.axis_width_spin.value()
        o["facecolor"] = self.bg_color

        # Legend
        o["legend_visible"] = self.legend_visible_check.isChecked()
        o["legend_frameon"] = self.legend_frameon_check.isChecked()
        o["legend_loc"] = self.legend_loc_combo.currentText()

        # Dimensions
        o["fig_size"] = [self.fig_width_spin.value(), self.fig_height_spin.value()]
        o["dpi"] = self.dpi_spin.value()

    def _apply_limits(self):
        """Apply axis limits."""
        xmin = self.xmin_spin.value()
        xmax = self.xmax_spin.value()
        ymin = self.ymin_spin.value()
        ymax = self.ymax_spin.value()

        if xmin < xmax:
            self.editor.current_overrides["xlim"] = [xmin, xmax]
        if ymin < ymax:
            self.editor.current_overrides["ylim"] = [ymin, ymax]

        self.editor._user_modified = True
        self._render_figure()

    def _choose_bg_color(self):
        """Open color dialog for background."""
        color = self.QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.bg_color = color.name()
            self.editor.current_overrides["facecolor"] = self.bg_color
            self._render_figure()

    def _add_annotation(self):
        """Add text annotation."""
        text = self.annot_text_edit.text()
        if not text:
            return

        x = self.annot_x_spin.value()
        y = self.annot_y_spin.value()

        if "annotations" not in self.editor.current_overrides:
            self.editor.current_overrides["annotations"] = []

        self.editor.current_overrides["annotations"].append(
            {
                "type": "text",
                "text": text,
                "x": x,
                "y": y,
                "fontsize": self.editor.current_overrides.get("axis_fontsize", 7),
            }
        )

        self.annot_text_edit.clear()
        self._update_annotations_list()
        self.editor._user_modified = True
        self._render_figure()

    def _remove_annotation(self):
        """Remove selected annotation."""
        row = self.annot_list.currentRow()
        annotations = self.editor.current_overrides.get("annotations", [])

        if row >= 0 and row < len(annotations):
            del annotations[row]
            self._update_annotations_list()
            self.editor._user_modified = True
            self._render_figure()

    def _update_annotations_list(self):
        """Update the annotations list widget."""
        self.annot_list.clear()
        for ann in self.editor.current_overrides.get("annotations", []):
            if ann.get("type") == "text":
                text = ann.get("text", "")[:20]
                x = ann.get("x", 0)
                y = ann.get("y", 0)
                self.annot_list.addItem(f"{text} ({x:.2f}, {y:.2f})")

    def _render_figure(self):
        """Render the figure with current overrides."""
        from matplotlib.ticker import MaxNLocator

        self.ax.clear()
        o = self.editor.current_overrides
        mm_to_pt = 2.83465

        # Background
        if o.get("transparent", True):
            self.fig.patch.set_facecolor("none")
            self.ax.patch.set_facecolor("none")
        else:
            self.fig.patch.set_facecolor(o.get("facecolor", "#ffffff"))
            self.ax.patch.set_facecolor(o.get("facecolor", "#ffffff"))

        # Plot from CSV
        if self.editor.csv_data is not None:
            self._plot_from_csv(o)
        else:
            self.ax.text(
                0.5,
                0.5,
                "No plot data available\n(CSV not found)",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )

        # Labels
        if o.get("title"):
            self.ax.set_title(o["title"], fontsize=o.get("title_fontsize", 8))
        if o.get("xlabel"):
            self.ax.set_xlabel(o["xlabel"], fontsize=o.get("axis_fontsize", 7))
        if o.get("ylabel"):
            self.ax.set_ylabel(o["ylabel"], fontsize=o.get("axis_fontsize", 7))

        # Ticks
        self.ax.tick_params(
            axis="both",
            labelsize=o.get("tick_fontsize", 7),
            length=o.get("tick_length", 0.8) * mm_to_pt,
            width=o.get("tick_width", 0.2) * mm_to_pt,
            direction=o.get("tick_direction", "out"),
        )

        self.ax.xaxis.set_major_locator(MaxNLocator(nbins=o.get("n_ticks", 4)))
        self.ax.yaxis.set_major_locator(MaxNLocator(nbins=o.get("n_ticks", 4)))

        # Grid
        if o.get("grid"):
            self.ax.grid(True, linewidth=o.get("axis_width", 0.2) * mm_to_pt, alpha=0.3)

        # Limits
        if o.get("xlim"):
            self.ax.set_xlim(o["xlim"])
        if o.get("ylim"):
            self.ax.set_ylim(o["ylim"])

        # Spines
        if o.get("hide_top_spine", True):
            self.ax.spines["top"].set_visible(False)
        if o.get("hide_right_spine", True):
            self.ax.spines["right"].set_visible(False)

        for spine in self.ax.spines.values():
            spine.set_linewidth(o.get("axis_width", 0.2) * mm_to_pt)

        # Annotations
        for ann in o.get("annotations", []):
            if ann.get("type") == "text":
                self.ax.text(
                    ann.get("x", 0.5),
                    ann.get("y", 0.5),
                    ann.get("text", ""),
                    transform=self.ax.transAxes,
                    fontsize=ann.get("fontsize", o.get("axis_fontsize", 7)),
                )

        self.fig.tight_layout()
        self.canvas.draw()
        self.status_bar.showMessage("Preview updated")

    def _plot_from_csv(self, o):
        """Reconstruct plot from CSV data."""
        import pandas as pd

        if not isinstance(self.editor.csv_data, pd.DataFrame):
            return

        df = self.editor.csv_data
        linewidth = o.get("linewidth", 1.0)
        legend_visible = o.get("legend_visible", True)
        legend_fontsize = o.get("legend_fontsize", 6)
        legend_frameon = o.get("legend_frameon", False)
        legend_loc = o.get("legend_loc", "best")

        traces = o.get("traces", [])

        if traces:
            for trace in traces:
                csv_cols = trace.get("csv_columns", {})
                x_col = csv_cols.get("x")
                y_col = csv_cols.get("y")

                if x_col in df.columns and y_col in df.columns:
                    self.ax.plot(
                        df[x_col],
                        df[y_col],
                        label=trace.get("label", trace.get("id", "")),
                        color=trace.get("color"),
                        linestyle=trace.get("linestyle", "-"),
                        linewidth=trace.get("linewidth", linewidth),
                    )

            if legend_visible and any(t.get("label") for t in traces):
                self.ax.legend(
                    fontsize=legend_fontsize, frameon=legend_frameon, loc=legend_loc
                )
        else:
            cols = df.columns.tolist()
            if len(cols) >= 2:
                x_col = cols[0]
                for y_col in cols[1:]:
                    try:
                        self.ax.plot(
                            df[x_col], df[y_col], label=str(y_col), linewidth=linewidth
                        )
                    except Exception:
                        pass
                if len(cols) > 2 and legend_visible:
                    self.ax.legend(
                        fontsize=legend_fontsize, frameon=legend_frameon, loc=legend_loc
                    )

    def _save_manual(self):
        """Save to .manual.json."""
        from .edit import save_manual_overrides

        try:
            self._collect_overrides()
            manual_path = save_manual_overrides(
                self.editor.json_path, self.editor.current_overrides
            )
            self.status_bar.showMessage(f"Saved: {manual_path.name}")
            self.QtWidgets.QMessageBox.information(
                self.window, "Saved", f"Manual overrides saved to:\n{manual_path}"
            )
        except Exception as e:
            self.QtWidgets.QMessageBox.critical(
                self.window, "Error", f"Failed to save: {e}"
            )

    def _reset_overrides(self):
        """Reset to initial overrides."""
        reply = self.QtWidgets.QMessageBox.question(
            self.window,
            "Reset",
            "Reset all changes to original values?",
            self.QtWidgets.QMessageBox.StandardButton.Yes
            | self.QtWidgets.QMessageBox.StandardButton.No
            if hasattr(self.QtWidgets.QMessageBox, "StandardButton")
            else self.QtWidgets.QMessageBox.Yes | self.QtWidgets.QMessageBox.No,
        )

        yes_val = (
            self.QtWidgets.QMessageBox.StandardButton.Yes
            if hasattr(self.QtWidgets.QMessageBox, "StandardButton")
            else self.QtWidgets.QMessageBox.Yes
        )

        if reply == yes_val:
            self.editor.current_overrides = copy.deepcopy(
                self.editor._initial_overrides
            )
            self.editor._user_modified = False

            # Update UI
            self.title_edit.setText(self.editor.current_overrides.get("title", ""))
            self.xlabel_edit.setText(self.editor.current_overrides.get("xlabel", ""))
            self.ylabel_edit.setText(self.editor.current_overrides.get("ylabel", ""))
            self.linewidth_spin.setValue(
                self.editor.current_overrides.get("linewidth", 1.0)
            )
            self.grid_check.setChecked(self.editor.current_overrides.get("grid", False))

            self._update_annotations_list()
            self._render_figure()
            self.status_bar.showMessage("Reset to original")

    def _export_png(self):
        """Export current view to PNG."""
        filepath, _ = self.QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Export PNG",
            str(self.editor.json_path.with_suffix(".edited.png")),
            "PNG files (*.png);;All files (*)",
        )

        if filepath:
            self._collect_overrides()
            o = self.editor.current_overrides
            dpi = o.get("dpi", 300)

            self.fig.savefig(
                filepath,
                dpi=dpi,
                bbox_inches="tight",
                transparent=o.get("transparent", True),
            )
            self.status_bar.showMessage(f"Exported: {Path(filepath).name}")


# EOF
