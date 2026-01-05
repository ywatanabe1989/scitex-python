#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/_tkinter_editor.py
"""Tkinter-based figure editor with matplotlib canvas."""

import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
from pathlib import Path
from typing import Dict, Any, Optional
import copy


class TkinterEditor:
    """
    Interactive figure editor using Tkinter GUI.

    Features:
    - Figure preview with embedded matplotlib canvas
    - Property editors for colors, line widths, fonts, labels
    - Real-time preview updates
    - Save to .manual.json
    - SciTeX style defaults pre-filled
    """

    def __init__(
        self,
        json_path: Path,
        metadata: Dict[str, Any],
        csv_data: Optional[Any] = None,
        manual_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.json_path = Path(json_path)
        self.metadata = metadata
        self.csv_data = csv_data
        self.manual_overrides = manual_overrides or {}

        # Get SciTeX defaults and merge with metadata
        from ._defaults import get_scitex_defaults, extract_defaults_from_metadata

        self.scitex_defaults = get_scitex_defaults()
        self.metadata_defaults = extract_defaults_from_metadata(metadata)

        # Track current overrides (modifications during session)
        # Start with defaults, then overlay manual overrides
        self.current_overrides = copy.deepcopy(self.scitex_defaults)
        self.current_overrides.update(self.metadata_defaults)
        self.current_overrides.update(self.manual_overrides)

        # UI state
        self.root = None
        self.canvas = None
        self.fig = None
        self.ax = None

    def run(self):
        """Launch the editor GUI."""
        self.root = tk.Tk()
        self.root.title(f"SciTeX Editor - {self.json_path.name}")
        self.root.geometry("1200x800")

        # Configure grid
        self.root.columnconfigure(0, weight=3)  # Canvas area
        self.root.columnconfigure(1, weight=1)  # Control panel
        self.root.rowconfigure(0, weight=1)

        # Create main frames
        self._create_canvas_frame()
        self._create_control_panel()
        self._create_toolbar()

        # Initial render
        self._render_figure()

        # Start main loop
        self.root.mainloop()

    def _create_canvas_frame(self):
        """Create the matplotlib canvas frame."""
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
        from matplotlib.figure import Figure

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Navigation toolbar
        toolbar_frame = ttk.Frame(canvas_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

    def _create_control_panel(self):
        """Create the property control panel."""
        panel = ttk.Frame(self.root, padding=10)
        panel.grid(row=0, column=1, sticky="nsew")

        # Notebook for tabbed controls
        notebook = ttk.Notebook(panel)
        notebook.pack(fill="both", expand=True)

        # Tab 1: Figure settings
        fig_tab = ttk.Frame(notebook, padding=10)
        notebook.add(fig_tab, text="Figure")
        self._create_figure_controls(fig_tab)

        # Tab 2: Axes settings
        axes_tab = ttk.Frame(notebook, padding=10)
        notebook.add(axes_tab, text="Axes")
        self._create_axes_controls(axes_tab)

        # Tab 3: Style settings
        style_tab = ttk.Frame(notebook, padding=10)
        notebook.add(style_tab, text="Style")
        self._create_style_controls(style_tab)

        # Tab 4: Annotations
        annot_tab = ttk.Frame(notebook, padding=10)
        notebook.add(annot_tab, text="Annotations")
        self._create_annotation_controls(annot_tab)

    def _create_figure_controls(self, parent):
        """Create figure-level controls."""
        # Title
        ttk.Label(parent, text="Title:").grid(row=0, column=0, sticky="w", pady=2)
        self.title_var = tk.StringVar(value=self._get_override("title", ""))
        title_entry = ttk.Entry(parent, textvariable=self.title_var, width=30)
        title_entry.grid(row=0, column=1, sticky="ew", pady=2)
        title_entry.bind("<Return>", lambda e: self._update_and_render())

        # X Label
        ttk.Label(parent, text="X Label:").grid(row=1, column=0, sticky="w", pady=2)
        self.xlabel_var = tk.StringVar(value=self._get_override("xlabel", ""))
        xlabel_entry = ttk.Entry(parent, textvariable=self.xlabel_var, width=30)
        xlabel_entry.grid(row=1, column=1, sticky="ew", pady=2)
        xlabel_entry.bind("<Return>", lambda e: self._update_and_render())

        # Y Label
        ttk.Label(parent, text="Y Label:").grid(row=2, column=0, sticky="w", pady=2)
        self.ylabel_var = tk.StringVar(value=self._get_override("ylabel", ""))
        ylabel_entry = ttk.Entry(parent, textvariable=self.ylabel_var, width=30)
        ylabel_entry.grid(row=2, column=1, sticky="ew", pady=2)
        ylabel_entry.bind("<Return>", lambda e: self._update_and_render())

        # Apply button
        ttk.Button(parent, text="Apply", command=self._update_and_render).grid(
            row=3, column=0, columnspan=2, pady=10
        )

        parent.columnconfigure(1, weight=1)

    def _create_axes_controls(self, parent):
        """Create axes-level controls."""
        # X limits
        ttk.Label(parent, text="X Min:").grid(row=0, column=0, sticky="w", pady=2)
        self.xmin_var = tk.StringVar(value="")
        ttk.Entry(parent, textvariable=self.xmin_var, width=10).grid(
            row=0, column=1, pady=2
        )

        ttk.Label(parent, text="X Max:").grid(
            row=0, column=2, sticky="w", pady=2, padx=(10, 0)
        )
        self.xmax_var = tk.StringVar(value="")
        ttk.Entry(parent, textvariable=self.xmax_var, width=10).grid(
            row=0, column=3, pady=2
        )

        # Y limits
        ttk.Label(parent, text="Y Min:").grid(row=1, column=0, sticky="w", pady=2)
        self.ymin_var = tk.StringVar(value="")
        ttk.Entry(parent, textvariable=self.ymin_var, width=10).grid(
            row=1, column=1, pady=2
        )

        ttk.Label(parent, text="Y Max:").grid(
            row=1, column=2, sticky="w", pady=2, padx=(10, 0)
        )
        self.ymax_var = tk.StringVar(value="")
        ttk.Entry(parent, textvariable=self.ymax_var, width=10).grid(
            row=1, column=3, pady=2
        )

        # Grid toggle
        self.grid_var = tk.BooleanVar(value=self._get_override("grid", False))
        ttk.Checkbutton(
            parent,
            text="Show Grid",
            variable=self.grid_var,
            command=self._update_and_render,
        ).grid(row=2, column=0, columnspan=2, pady=5)

        # Apply button
        ttk.Button(parent, text="Apply Limits", command=self._apply_limits).grid(
            row=3, column=0, columnspan=4, pady=10
        )

    def _create_style_controls(self, parent):
        """Create style controls."""
        # Line width
        ttk.Label(parent, text="Line Width:").grid(row=0, column=0, sticky="w", pady=2)
        self.linewidth_var = tk.DoubleVar(value=self._get_override("linewidth", 1.5))
        lw_spin = ttk.Spinbox(
            parent,
            from_=0.1,
            to=10,
            increment=0.1,
            textvariable=self.linewidth_var,
            width=8,
        )
        lw_spin.grid(row=0, column=1, sticky="w", pady=2)

        # Font size
        ttk.Label(parent, text="Font Size:").grid(row=1, column=0, sticky="w", pady=2)
        self.fontsize_var = tk.IntVar(value=self._get_override("fontsize", 10))
        fs_spin = ttk.Spinbox(
            parent, from_=6, to=24, increment=1, textvariable=self.fontsize_var, width=8
        )
        fs_spin.grid(row=1, column=1, sticky="w", pady=2)

        # Background color
        ttk.Label(parent, text="Background:").grid(row=2, column=0, sticky="w", pady=2)
        self.bg_color = self._get_override("facecolor", "white")
        self.bg_btn = ttk.Button(
            parent, text="Choose...", command=self._choose_bg_color
        )
        self.bg_btn.grid(row=2, column=1, sticky="w", pady=2)

        # Color frame to show current color
        self.bg_preview = tk.Frame(parent, width=20, height=20, bg=self.bg_color)
        self.bg_preview.grid(row=2, column=2, padx=5)

        # Apply button
        ttk.Button(parent, text="Apply Style", command=self._update_and_render).grid(
            row=5, column=0, columnspan=3, pady=10
        )

    def _create_annotation_controls(self, parent):
        """Create annotation controls."""
        ttk.Label(parent, text="Add annotations:").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )

        # Text annotation
        ttk.Label(parent, text="Text:").grid(row=1, column=0, sticky="w", pady=2)
        self.annot_text_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.annot_text_var, width=25).grid(
            row=1, column=1, pady=2
        )

        ttk.Label(parent, text="X:").grid(row=2, column=0, sticky="w", pady=2)
        self.annot_x_var = tk.StringVar(value="0.5")
        ttk.Entry(parent, textvariable=self.annot_x_var, width=10).grid(
            row=2, column=1, sticky="w", pady=2
        )

        ttk.Label(parent, text="Y:").grid(row=3, column=0, sticky="w", pady=2)
        self.annot_y_var = tk.StringVar(value="0.5")
        ttk.Entry(parent, textvariable=self.annot_y_var, width=10).grid(
            row=3, column=1, sticky="w", pady=2
        )

        ttk.Button(parent, text="Add Text", command=self._add_text_annotation).grid(
            row=4, column=0, columnspan=2, pady=5
        )

        # Annotation list
        ttk.Label(parent, text="Current annotations:").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(10, 2)
        )
        self.annot_listbox = tk.Listbox(parent, height=5, width=30)
        self.annot_listbox.grid(row=6, column=0, columnspan=2, sticky="ew")

        ttk.Button(
            parent, text="Remove Selected", command=self._remove_annotation
        ).grid(row=7, column=0, columnspan=2, pady=5)

        self._update_annotation_list()

    def _create_toolbar(self):
        """Create the main toolbar."""
        toolbar = ttk.Frame(self.root)
        toolbar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        ttk.Button(toolbar, text="Save", command=self._save_manual).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="Reset", command=self._reset_overrides).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="Export PNG", command=self._export_png).pack(
            side="left", padx=2
        )

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side="right", padx=10)

    def _get_override(self, key, default=None):
        """Get value from current overrides or default."""
        return self.current_overrides.get(key, default)

    def _render_figure(self):
        """Render the figure with current data and overrides."""
        import scitex as stx

        self.ax.clear()

        # Try to reconstruct from CSV data
        if self.csv_data is not None:
            self._plot_from_csv()
        else:
            # Show placeholder
            self.ax.text(
                0.5,
                0.5,
                "No plot data available\n(CSV not found)",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )

        # Apply overrides
        if self.current_overrides.get("title"):
            self.ax.set_title(self.current_overrides["title"])
        if self.current_overrides.get("xlabel"):
            self.ax.set_xlabel(self.current_overrides["xlabel"])
        if self.current_overrides.get("ylabel"):
            self.ax.set_ylabel(self.current_overrides["ylabel"])
        if self.current_overrides.get("grid"):
            self.ax.grid(True)
        if self.current_overrides.get("xlim"):
            self.ax.set_xlim(self.current_overrides["xlim"])
        if self.current_overrides.get("ylim"):
            self.ax.set_ylim(self.current_overrides["ylim"])
        if self.current_overrides.get("facecolor"):
            self.ax.set_facecolor(self.current_overrides["facecolor"])

        # Apply annotations
        for annot in self.current_overrides.get("annotations", []):
            if annot.get("type") == "text":
                self.ax.text(
                    annot.get("x", 0.5),
                    annot.get("y", 0.5),
                    annot.get("text", ""),
                    transform=self.ax.transAxes,
                    fontsize=annot.get("fontsize", 10),
                )

        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_from_csv(self):
        """Reconstruct plot from CSV data."""
        import pandas as pd

        if isinstance(self.csv_data, pd.DataFrame):
            df = self.csv_data
        else:
            return

        # Try to identify x and y columns
        cols = df.columns.tolist()

        # Simple heuristic: first column is x, rest are y series
        if len(cols) >= 2:
            x_col = cols[0]
            for y_col in cols[1:]:
                try:
                    self.ax.plot(df[x_col], df[y_col], label=str(y_col))
                except Exception:
                    pass
            if len(cols) > 2:
                self.ax.legend()
        elif len(cols) == 1:
            self.ax.plot(df[cols[0]])

    def _update_and_render(self):
        """Update overrides from UI and re-render."""
        # Collect values from UI
        if hasattr(self, "title_var") and self.title_var.get():
            self.current_overrides["title"] = self.title_var.get()
        if hasattr(self, "xlabel_var") and self.xlabel_var.get():
            self.current_overrides["xlabel"] = self.xlabel_var.get()
        if hasattr(self, "ylabel_var") and self.ylabel_var.get():
            self.current_overrides["ylabel"] = self.ylabel_var.get()
        if hasattr(self, "grid_var"):
            self.current_overrides["grid"] = self.grid_var.get()
        if hasattr(self, "linewidth_var"):
            self.current_overrides["linewidth"] = self.linewidth_var.get()
        if hasattr(self, "fontsize_var"):
            self.current_overrides["fontsize"] = self.fontsize_var.get()

        self._render_figure()
        self.status_var.set("Preview updated")

    def _apply_limits(self):
        """Apply axis limits."""
        try:
            if self.xmin_var.get() and self.xmax_var.get():
                self.current_overrides["xlim"] = [
                    float(self.xmin_var.get()),
                    float(self.xmax_var.get()),
                ]
            if self.ymin_var.get() and self.ymax_var.get():
                self.current_overrides["ylim"] = [
                    float(self.ymin_var.get()),
                    float(self.ymax_var.get()),
                ]
            self._render_figure()
            self.status_var.set("Limits applied")
        except ValueError:
            messagebox.showerror("Error", "Invalid limit values")

    def _choose_bg_color(self):
        """Open color chooser for background."""
        color = colorchooser.askcolor(
            title="Choose Background Color", color=self.bg_color
        )
        if color[1]:
            self.bg_color = color[1]
            self.bg_preview.configure(bg=self.bg_color)
            self.current_overrides["facecolor"] = self.bg_color
            self._render_figure()

    def _add_text_annotation(self):
        """Add a text annotation."""
        text = self.annot_text_var.get()
        if not text:
            return

        try:
            x = float(self.annot_x_var.get())
            y = float(self.annot_y_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid X or Y position")
            return

        if "annotations" not in self.current_overrides:
            self.current_overrides["annotations"] = []

        self.current_overrides["annotations"].append(
            {
                "type": "text",
                "text": text,
                "x": x,
                "y": y,
                "fontsize": self.fontsize_var.get()
                if hasattr(self, "fontsize_var")
                else 10,
            }
        )

        self.annot_text_var.set("")
        self._update_annotation_list()
        self._render_figure()
        self.status_var.set("Annotation added")

    def _remove_annotation(self):
        """Remove selected annotation."""
        selection = self.annot_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        annotations = self.current_overrides.get("annotations", [])
        if idx < len(annotations):
            del annotations[idx]
            self._update_annotation_list()
            self._render_figure()
            self.status_var.set("Annotation removed")

    def _update_annotation_list(self):
        """Update the annotation listbox."""
        self.annot_listbox.delete(0, tk.END)
        for annot in self.current_overrides.get("annotations", []):
            if annot.get("type") == "text":
                self.annot_listbox.insert(tk.END, f"Text: {annot.get('text', '')[:20]}")

    def _save_manual(self):
        """Save current overrides to .manual.json."""
        from .edit import save_manual_overrides

        try:
            manual_path = save_manual_overrides(self.json_path, self.current_overrides)
            self.status_var.set(f"Saved: {manual_path.name}")
            messagebox.showinfo("Saved", f"Manual overrides saved to:\n{manual_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def _reset_overrides(self):
        """Reset to original overrides."""
        if messagebox.askyesno("Reset", "Reset all changes?"):
            self.current_overrides = copy.deepcopy(self.manual_overrides)
            self._render_figure()
            self.status_var.set("Reset to original")

    def _export_png(self):
        """Export current view to PNG."""
        from tkinter import filedialog

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=f"{self.json_path.stem}_edited.png",
        )
        if filepath:
            self.fig.savefig(filepath, dpi=300, bbox_inches="tight")
            self.status_var.set(f"Exported: {Path(filepath).name}")


# EOF
