#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/_mpl_editor.py
"""Minimal matplotlib-based figure editor."""

from pathlib import Path
from typing import Dict, Any, Optional
import copy


class MplEditor:
    """
    Minimal interactive figure editor using matplotlib's built-in interactivity.

    Features:
    - Basic figure display with navigation toolbar
    - Text-based property editing via console
    - Save to .manual.json
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
        self.current_overrides = copy.deepcopy(self.manual_overrides)

    def run(self):
        """Launch the matplotlib editor."""
        import matplotlib

        matplotlib.use("TkAgg")  # Use interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox

        # Create figure with extra space for controls
        self.fig = plt.figure(figsize=(12, 8))

        # Main axes for plot
        self.ax = self.fig.add_axes([0.1, 0.25, 0.85, 0.65])

        # Initial render
        self._render()

        # Add control buttons
        self._add_controls()

        # Show instructions
        print("\n" + "=" * 50)
        print("SciTeX Matplotlib Editor")
        print("=" * 50)
        print(f"Editing: {self.json_path.name}")
        print("\nControls:")
        print("  - Use navigation toolbar for zoom/pan")
        print("  - Click buttons below figure for actions")
        print("  - Close window when done")
        print("=" * 50 + "\n")

        plt.show()

    def _render(self):
        """Render the figure."""
        self.ax.clear()

        # Plot from CSV data
        if self.csv_data is not None:
            self._plot_from_csv()
        else:
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

        self.fig.canvas.draw()

    def _plot_from_csv(self):
        """Reconstruct plot from CSV data."""
        import pandas as pd

        if isinstance(self.csv_data, pd.DataFrame):
            df = self.csv_data
        else:
            return

        cols = df.columns.tolist()
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

    def _add_controls(self):
        """Add control buttons."""
        from matplotlib.widgets import Button, TextBox

        # Title text box
        ax_title = self.fig.add_axes([0.15, 0.12, 0.3, 0.04])
        self.title_box = TextBox(
            ax_title, "Title:", initial=self.current_overrides.get("title", "")
        )
        self.title_box.on_submit(self._on_title_change)

        # Grid toggle button
        ax_grid = self.fig.add_axes([0.55, 0.12, 0.1, 0.04])
        self.grid_btn = Button(ax_grid, "Toggle Grid")
        self.grid_btn.on_clicked(self._toggle_grid)

        # Save button
        ax_save = self.fig.add_axes([0.7, 0.12, 0.1, 0.04])
        self.save_btn = Button(ax_save, "Save")
        self.save_btn.on_clicked(self._save)

        # Edit labels button
        ax_labels = self.fig.add_axes([0.15, 0.05, 0.15, 0.04])
        self.labels_btn = Button(ax_labels, "Edit Labels")
        self.labels_btn.on_clicked(self._edit_labels)

        # Add annotation button
        ax_annot = self.fig.add_axes([0.35, 0.05, 0.15, 0.04])
        self.annot_btn = Button(ax_annot, "Add Text")
        self.annot_btn.on_clicked(self._add_annotation)

        # Export PNG button
        ax_export = self.fig.add_axes([0.55, 0.05, 0.12, 0.04])
        self.export_btn = Button(ax_export, "Export PNG")
        self.export_btn.on_clicked(self._export_png)

    def _on_title_change(self, text):
        """Handle title change."""
        self.current_overrides["title"] = text
        self._render()

    def _toggle_grid(self, event):
        """Toggle grid visibility."""
        self.current_overrides["grid"] = not self.current_overrides.get("grid", False)
        self._render()

    def _edit_labels(self, event):
        """Edit axis labels via console."""
        print("\n--- Edit Labels ---")
        xlabel = input(
            f"X Label [{self.current_overrides.get('xlabel', '')}]: "
        ).strip()
        if xlabel:
            self.current_overrides["xlabel"] = xlabel

        ylabel = input(
            f"Y Label [{self.current_overrides.get('ylabel', '')}]: "
        ).strip()
        if ylabel:
            self.current_overrides["ylabel"] = ylabel

        self._render()
        print("Labels updated!")

    def _add_annotation(self, event):
        """Add text annotation via console."""
        print("\n--- Add Text Annotation ---")
        text = input("Text: ").strip()
        if not text:
            return

        try:
            x = float(input("X position (0-1) [0.5]: ").strip() or "0.5")
            y = float(input("Y position (0-1) [0.5]: ").strip() or "0.5")
        except ValueError:
            print("Invalid position, using defaults")
            x, y = 0.5, 0.5

        if "annotations" not in self.current_overrides:
            self.current_overrides["annotations"] = []

        self.current_overrides["annotations"].append(
            {
                "type": "text",
                "text": text,
                "x": x,
                "y": y,
                "fontsize": 10,
            }
        )

        self._render()
        print("Annotation added!")

    def _save(self, event):
        """Save to .manual.json."""
        from .edit import save_manual_overrides

        try:
            manual_path = save_manual_overrides(self.json_path, self.current_overrides)
            print(f"\nSaved: {manual_path}")
        except Exception as e:
            print(f"\nError saving: {e}")

    def _export_png(self, event):
        """Export current view to PNG."""
        output_path = self.json_path.with_suffix(".edited.png")
        self.fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nExported: {output_path}")


# EOF
