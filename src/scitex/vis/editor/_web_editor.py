#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/_web_editor.py
"""Web-based figure editor using Flask."""

from pathlib import Path
from typing import Dict, Any, Optional
import copy
import json
import io
import base64
import webbrowser
import threading


def _find_available_port(start_port: int = 5050, max_attempts: int = 10) -> int:
    """Find an available port, starting from start_port."""
    import socket

    for offset in range(max_attempts):
        port = start_port + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue

    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")


def _kill_process_on_port(port: int) -> bool:
    """Try to kill process using the specified port. Returns True if successful."""
    import subprocess
    import sys

    try:
        if sys.platform == 'win32':
            # Windows: netstat + taskkill
            result = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True, capture_output=True, text=True
            )
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
                return True
        else:
            # Linux/Mac: fuser or lsof
            result = subprocess.run(
                ['fuser', '-k', f'{port}/tcp'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return True

            # Fallback to lsof
            result = subprocess.run(
                ['lsof', '-t', f'-i:{port}'],
                capture_output=True, text=True
            )
            if result.stdout:
                for pid in result.stdout.strip().split('\n'):
                    if pid:
                        subprocess.run(['kill', '-9', pid], capture_output=True)
                return True
    except Exception:
        pass

    return False


class WebEditor:
    """
    Browser-based figure editor using Flask.

    Features:
    - Modern responsive UI
    - Real-time preview via WebSocket or polling
    - Property editors with sliders and color pickers
    - Save to .manual.json
    - SciTeX style defaults pre-filled
    - Auto-finds available port if default is in use
    """

    def __init__(
        self,
        json_path: Path,
        metadata: Dict[str, Any],
        csv_data: Optional[Any] = None,
        png_path: Optional[Path] = None,
        manual_overrides: Optional[Dict[str, Any]] = None,
        port: int = 5050,
    ):
        self.json_path = Path(json_path)
        self.metadata = metadata
        self.csv_data = csv_data
        self.png_path = Path(png_path) if png_path else None
        self.manual_overrides = manual_overrides or {}
        self._requested_port = port
        self.port = port  # Will be updated in run() if needed

        # Get SciTeX defaults and merge with metadata
        from ._defaults import get_scitex_defaults, extract_defaults_from_metadata
        self.scitex_defaults = get_scitex_defaults()
        self.metadata_defaults = extract_defaults_from_metadata(metadata)

        # Start with defaults, then overlay manual overrides
        self.current_overrides = copy.deepcopy(self.scitex_defaults)
        self.current_overrides.update(self.metadata_defaults)
        self.current_overrides.update(self.manual_overrides)

        # Track initial state to detect modifications
        self._initial_overrides = copy.deepcopy(self.current_overrides)
        self._user_modified = False  # Set to True when user makes changes

    def run(self):
        """Launch the web editor."""
        try:
            from flask import Flask, render_template_string, request, jsonify
        except ImportError:
            raise ImportError("Flask is required for web editor. Install: pip install flask")

        # Handle port conflicts: try to find available port or kill existing process
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', self._requested_port))
                self.port = self._requested_port
        except OSError:
            # Port in use - try to kill existing process first
            print(f"Port {self._requested_port} is in use. Attempting to free it...")
            if _kill_process_on_port(self._requested_port):
                import time
                time.sleep(0.5)  # Give process time to release port
                self.port = self._requested_port
                print(f"Successfully freed port {self.port}")
            else:
                # Find an alternative port
                self.port = _find_available_port(self._requested_port + 1)
                print(f"Using alternative port: {self.port}")

        app = Flask(__name__)

        # Store reference to self for routes
        editor = self

        @app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE,
                                         filename=editor.json_path.name,
                                         overrides=json.dumps(editor.current_overrides))

        @app.route('/preview')
        def preview():
            """Generate figure preview as base64 PNG."""
            img_data = editor._render_preview()
            return jsonify({'image': img_data})

        @app.route('/update', methods=['POST'])
        def update():
            """Update overrides and return new preview."""
            data = request.json
            editor.current_overrides.update(data.get('overrides', {}))
            editor._user_modified = True  # Mark as modified to regenerate from CSV
            img_data = editor._render_preview()
            return jsonify({'image': img_data, 'status': 'updated'})

        @app.route('/save', methods=['POST'])
        def save():
            """Save to .manual.json."""
            from ._edit import save_manual_overrides
            try:
                manual_path = save_manual_overrides(editor.json_path, editor.current_overrides)
                return jsonify({'status': 'saved', 'path': str(manual_path)})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @app.route('/shutdown', methods=['POST'])
        def shutdown():
            """Shutdown the server."""
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                raise RuntimeError('Not running with Werkzeug Server')
            func()
            return jsonify({'status': 'shutdown'})

        # Open browser after short delay
        def open_browser():
            import time
            time.sleep(0.5)
            webbrowser.open(f'http://127.0.0.1:{self.port}')

        threading.Thread(target=open_browser, daemon=True).start()

        print(f"Starting SciTeX Editor at http://127.0.0.1:{self.port}")
        print("Press Ctrl+C to stop")

        app.run(host='127.0.0.1', port=self.port, debug=False, use_reloader=False)

    def _has_modifications(self) -> bool:
        """Check if user has made any modifications to the figure."""
        return self._user_modified

    def _render_preview(self) -> str:
        """Render figure and return as base64 PNG.

        If original PNG exists and no overrides have been applied, return the original.
        Otherwise, regenerate from CSV with applied overrides using exact metadata.
        """
        # If PNG exists and this is initial load (no modifications), show original
        if self.png_path and self.png_path.exists() and not self._has_modifications():
            with open(self.png_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        # mm to pt conversion
        mm_to_pt = 2.83465

        # Get values from overrides (which includes metadata defaults)
        o = self.current_overrides

        # Dimensions
        dpi = o.get('dpi', 300)
        fig_size = o.get('fig_size', [3.15, 2.68])

        # Font sizes
        axis_fontsize = o.get('axis_fontsize', 7)
        tick_fontsize = o.get('tick_fontsize', 7)
        title_fontsize = o.get('title_fontsize', 8)
        legend_fontsize = o.get('legend_fontsize', 6)

        # Line/axis thickness (convert mm to pt)
        linewidth_pt = o.get('linewidth', 0.57)  # Already in pt or convert
        axis_width_pt = o.get('axis_width', 0.2) * mm_to_pt
        tick_length_pt = o.get('tick_length', 0.8) * mm_to_pt
        tick_width_pt = o.get('tick_width', 0.2) * mm_to_pt
        tick_direction = o.get('tick_direction', 'out')
        n_ticks = o.get('n_ticks', 4)

        # Transparent background
        transparent = o.get('transparent', True)

        # Create figure with dimensions from overrides
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        if transparent:
            fig.patch.set_facecolor('none')
            ax.patch.set_facecolor('none')
        elif o.get('facecolor'):
            fig.patch.set_facecolor(o['facecolor'])
            ax.patch.set_facecolor(o['facecolor'])

        # Plot from CSV data
        if self.csv_data is not None:
            self._plot_from_csv(ax, linewidth=linewidth_pt)
        else:
            ax.text(0.5, 0.5, "No plot data available\n(CSV not found)",
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=axis_fontsize)

        # Apply labels
        if o.get('title'):
            ax.set_title(o['title'], fontsize=title_fontsize)
        if o.get('xlabel'):
            ax.set_xlabel(o['xlabel'], fontsize=axis_fontsize)
        if o.get('ylabel'):
            ax.set_ylabel(o['ylabel'], fontsize=axis_fontsize)

        # Tick styling
        ax.tick_params(
            axis='both',
            labelsize=tick_fontsize,
            length=tick_length_pt,
            width=tick_width_pt,
            direction=tick_direction,
        )

        # Number of ticks
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

        # Grid
        if o.get('grid'):
            ax.grid(True, linewidth=axis_width_pt, alpha=0.3)

        # Axis limits
        if o.get('xlim'):
            ax.set_xlim(o['xlim'])
        if o.get('ylim'):
            ax.set_ylim(o['ylim'])

        # Spines visibility
        if o.get('hide_top_spine', True):
            ax.spines['top'].set_visible(False)
        if o.get('hide_right_spine', True):
            ax.spines['right'].set_visible(False)

        # Spine line width
        for spine in ax.spines.values():
            spine.set_linewidth(axis_width_pt)

        # Apply annotations
        for annot in o.get('annotations', []):
            if annot.get('type') == 'text':
                ax.text(
                    annot.get('x', 0.5),
                    annot.get('y', 0.5),
                    annot.get('text', ''),
                    transform=ax.transAxes,
                    fontsize=annot.get('fontsize', axis_fontsize),
                )

        fig.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', transparent=transparent)
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return img_data

    def _plot_from_csv(self, ax, linewidth=1.0):
        """Reconstruct plot from CSV data using trace info from overrides."""
        import pandas as pd

        if not isinstance(self.csv_data, pd.DataFrame):
            return

        df = self.csv_data
        o = self.current_overrides

        # Get legend settings from overrides
        legend_fontsize = o.get('legend_fontsize', 6)
        legend_visible = o.get('legend_visible', True)
        legend_frameon = o.get('legend_frameon', False)
        legend_loc = o.get('legend_loc', 'best')

        # Get traces from overrides (which may have been edited by user)
        traces = o.get('traces', [])

        if traces:
            # Use trace information to reconstruct plot correctly
            for trace in traces:
                csv_cols = trace.get('csv_columns', {})
                x_col = csv_cols.get('x')
                y_col = csv_cols.get('y')

                if x_col in df.columns and y_col in df.columns:
                    ax.plot(
                        df[x_col],
                        df[y_col],
                        label=trace.get('label', trace.get('id', '')),
                        color=trace.get('color'),
                        linestyle=trace.get('linestyle', '-'),
                        linewidth=trace.get('linewidth', linewidth),
                        marker=trace.get('marker', None),
                        markersize=trace.get('markersize', 6),
                    )

            # Add legend if there are labeled traces
            if legend_visible and any(t.get('label') for t in traces):
                ax.legend(
                    fontsize=legend_fontsize,
                    frameon=legend_frameon,
                    loc=legend_loc,
                )
        else:
            # Fallback: smart parsing of CSV column names
            # Format: ax_00_{id}_plot_x, ax_00_{id}_plot_y
            cols = df.columns.tolist()

            # Group columns by trace ID
            trace_groups = {}
            for col in cols:
                if col.endswith('_x'):
                    trace_id = col[:-2]  # Remove '_x'
                    y_col = trace_id + '_y'
                    if y_col in cols:
                        # Extract label from column name (e.g., ax_00_sine_plot -> sine)
                        parts = trace_id.split('_')
                        label = parts[2] if len(parts) > 2 else trace_id
                        trace_groups[trace_id] = {
                            'x_col': col,
                            'y_col': y_col,
                            'label': label,
                        }

            if trace_groups:
                for trace_id, info in trace_groups.items():
                    ax.plot(
                        df[info['x_col']],
                        df[info['y_col']],
                        label=info['label'],
                        linewidth=linewidth,
                    )
                if legend_visible:
                    ax.legend(fontsize=legend_fontsize, frameon=legend_frameon, loc=legend_loc)
            elif len(cols) >= 2:
                # Last resort: assume first column is x, rest are y
                x_col = cols[0]
                for y_col in cols[1:]:
                    try:
                        ax.plot(df[x_col], df[y_col], label=str(y_col), linewidth=linewidth)
                    except Exception:
                        pass
                if len(cols) > 2 and legend_visible:
                    ax.legend(fontsize=legend_fontsize, frameon=legend_frameon, loc=legend_loc)


# HTML template for web editor with light/dark mode based on scitex-cloud
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SciTeX Editor - {{ filename }}</title>
    <style>
        /* =============================================================================
         * SciTeX Color System - Based on scitex-cloud/static/shared/css
         * ============================================================================= */
        :root, [data-theme="light"] {
            /* Brand colors (light mode) */
            --scitex-01: #1a2a40;
            --scitex-02: #34495e;
            --scitex-03: #506b7a;
            --scitex-04: #6c8ba0;
            --scitex-05: #8fa4b0;
            --scitex-06: #b5c7d1;
            --scitex-07: #d4e1e8;
            --white: #fafbfc;
            --gray-subtle: #f6f8fa;

            /* Semantic tokens */
            --text-primary: var(--scitex-01);
            --text-secondary: var(--scitex-02);
            --text-muted: var(--scitex-04);
            --text-inverse: var(--white);

            --bg-page: #fefefe;
            --bg-surface: var(--white);
            --bg-muted: var(--gray-subtle);

            --border-default: var(--scitex-05);
            --border-muted: var(--scitex-06);

            /* Workspace colors */
            --workspace-bg-primary: #f8f9fa;
            --workspace-bg-secondary: #f3f4f6;
            --workspace-bg-tertiary: #ebedef;
            --workspace-bg-elevated: #ffffff;
            --workspace-border-subtle: #e0e4e8;
            --workspace-border-default: #b5c7d1;

            /* Status */
            --status-success: #4a9b7e;
            --status-warning: #b8956a;
            --status-error: #a67373;

            /* CTA */
            --color-cta: #3b82f6;
            --color-cta-hover: #2563eb;

            /* Preview background (checkered for transparency) */
            --preview-bg: linear-gradient(45deg, #e0e0e0 25%, transparent 25%),
                          linear-gradient(-45deg, #e0e0e0 25%, transparent 25%),
                          linear-gradient(45deg, transparent 75%, #e0e0e0 75%),
                          linear-gradient(-45deg, transparent 75%, #e0e0e0 75%);
        }

        [data-theme="dark"] {
            /* Semantic tokens (dark mode) */
            --text-primary: var(--scitex-07);
            --text-secondary: var(--scitex-05);
            --text-muted: var(--scitex-04);
            --text-inverse: var(--scitex-01);

            --bg-page: #0f1419;
            --bg-surface: var(--scitex-01);
            --bg-muted: var(--scitex-02);

            --border-default: var(--scitex-03);
            --border-muted: var(--scitex-02);

            /* Workspace colors */
            --workspace-bg-primary: #0d0d0d;
            --workspace-bg-secondary: #151515;
            --workspace-bg-tertiary: #1a1a1a;
            --workspace-bg-elevated: #1f1f1f;
            --workspace-border-subtle: #1a1a1a;
            --workspace-border-default: #3a3a3a;

            /* Status */
            --status-success: #6ba89a;
            --status-warning: #d4a87a;
            --status-error: #c08888;

            /* Preview background (darker checkered) */
            --preview-bg: linear-gradient(45deg, #2a2a2a 25%, transparent 25%),
                          linear-gradient(-45deg, #2a2a2a 25%, transparent 25%),
                          linear-gradient(45deg, transparent 75%, #2a2a2a 75%),
                          linear-gradient(-45deg, transparent 75%, #2a2a2a 75%);
        }

        /* =============================================================================
         * Base Styles
         * ============================================================================= */
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--workspace-bg-primary);
            color: var(--text-primary);
            transition: background 0.3s, color 0.3s;
        }

        .container { display: flex; height: 100vh; }

        /* =============================================================================
         * Preview Panel
         * ============================================================================= */
        .preview {
            flex: 2;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--workspace-bg-secondary);
        }

        .preview-wrapper {
            background: var(--preview-bg);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }

        .preview img {
            max-width: 100%;
            max-height: calc(100vh - 80px);
            display: block;
        }

        /* =============================================================================
         * Controls Panel
         * ============================================================================= */
        .controls {
            flex: 1;
            min-width: 320px;
            max-width: 420px;
            background: var(--workspace-bg-elevated);
            border-left: 1px solid var(--workspace-border-default);
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }

        .controls-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--workspace-border-subtle);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--bg-surface);
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .controls-header h1 {
            font-size: 1.1em;
            font-weight: 600;
            color: var(--status-success);
        }

        .controls-body {
            padding: 0 20px 20px;
            flex: 1;
        }

        .filename {
            font-size: 0.8em;
            color: var(--text-muted);
            margin-top: 4px;
            word-break: break-all;
        }

        /* Theme toggle */
        .theme-toggle {
            background: transparent;
            border: 1px solid var(--border-muted);
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 16px;
            padding: 6px 10px;
            border-radius: 6px;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .theme-toggle:hover {
            background: var(--bg-muted);
            border-color: var(--border-default);
        }

        /* =============================================================================
         * Section Headers
         * ============================================================================= */
        .section {
            margin-top: 16px;
        }

        .section-header {
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-inverse);
            background: var(--status-success);
            padding: 8px 12px;
            border-radius: 4px;
            margin-bottom: 12px;
        }

        /* =============================================================================
         * Form Fields
         * ============================================================================= */
        .field { margin-bottom: 12px; }

        .field label {
            display: block;
            font-size: 0.8em;
            font-weight: 500;
            margin-bottom: 4px;
            color: var(--text-secondary);
        }

        .field input[type="text"],
        .field input[type="number"],
        .field select {
            width: 100%;
            padding: 8px 10px;
            border: 1px solid var(--border-muted);
            border-radius: 4px;
            background: var(--bg-surface);
            color: var(--text-primary);
            font-size: 0.85em;
            transition: border-color 0.2s;
        }

        .field input:focus,
        .field select:focus {
            outline: none;
            border-color: var(--status-success);
        }

        .field input[type="color"] {
            width: 40px;
            height: 32px;
            padding: 2px;
            border: 1px solid var(--border-muted);
            border-radius: 4px;
            cursor: pointer;
            background: var(--bg-surface);
        }

        .field-row {
            display: flex;
            gap: 10px;
        }

        .field-row .field { flex: 1; }

        /* Checkbox styling */
        .checkbox-field {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            padding: 6px 0;
        }

        .checkbox-field input[type="checkbox"] {
            width: 16px;
            height: 16px;
            accent-color: var(--status-success);
        }

        .checkbox-field span {
            font-size: 0.85em;
            color: var(--text-primary);
        }

        /* Color field with input */
        .color-field {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .color-field input[type="text"] {
            flex: 1;
        }

        /* =============================================================================
         * Traces Section
         * ============================================================================= */
        .traces-list {
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid var(--border-muted);
            border-radius: 4px;
            background: var(--bg-muted);
        }

        .trace-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 10px;
            border-bottom: 1px solid var(--border-muted);
            font-size: 0.85em;
        }

        .trace-item:last-child { border-bottom: none; }

        .trace-color {
            width: 24px;
            height: 24px;
            border-radius: 4px;
            border: 1px solid var(--border-default);
            cursor: pointer;
        }

        .trace-label {
            flex: 1;
            color: var(--text-primary);
        }

        .trace-style select {
            padding: 4px 6px;
            font-size: 0.8em;
            border: 1px solid var(--border-muted);
            border-radius: 3px;
            background: var(--bg-surface);
            color: var(--text-primary);
        }

        /* =============================================================================
         * Annotations
         * ============================================================================= */
        .annotations-list {
            margin-top: 10px;
            max-height: 120px;
            overflow-y: auto;
        }

        .annotation-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 10px;
            background: var(--bg-muted);
            border-radius: 4px;
            margin-bottom: 5px;
            font-size: 0.85em;
        }

        .annotation-item span { color: var(--text-primary); }

        .annotation-item button {
            padding: 3px 8px;
            font-size: 0.75em;
            background: var(--status-error);
            border: none;
            border-radius: 3px;
            color: white;
            cursor: pointer;
        }

        /* =============================================================================
         * Buttons
         * ============================================================================= */
        .btn {
            width: 100%;
            padding: 10px 16px;
            margin-top: 8px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 500;
            transition: all 0.2s;
        }

        .btn-primary {
            background: var(--status-success);
            color: white;
        }

        .btn-primary:hover {
            filter: brightness(1.1);
        }

        .btn-secondary {
            background: var(--bg-muted);
            color: var(--text-primary);
            border: 1px solid var(--border-muted);
        }

        .btn-secondary:hover {
            background: var(--workspace-bg-tertiary);
        }

        .btn-cta {
            background: var(--color-cta);
            color: white;
        }

        .btn-cta:hover {
            background: var(--color-cta-hover);
        }

        /* =============================================================================
         * Status Bar
         * ============================================================================= */
        .status-bar {
            margin-top: 16px;
            padding: 10px 12px;
            border-radius: 4px;
            background: var(--bg-muted);
            font-size: 0.8em;
            color: var(--text-secondary);
            border-left: 3px solid var(--status-success);
        }

        .status-bar.error {
            border-left-color: var(--status-error);
        }

        /* =============================================================================
         * Collapsible Sections
         * ============================================================================= */
        .section-toggle {
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .section-toggle::before {
            content: "\\25BC";
            font-size: 0.7em;
            transition: transform 0.2s;
        }

        .section-toggle.collapsed::before {
            transform: rotate(-90deg);
        }

        .section-content {
            overflow: hidden;
            transition: max-height 0.3s ease;
        }

        .section-content.collapsed {
            max-height: 0 !important;
        }

        /* =============================================================================
         * Scrollbar Styling
         * ============================================================================= */
        .controls::-webkit-scrollbar,
        .traces-list::-webkit-scrollbar,
        .annotations-list::-webkit-scrollbar {
            width: 6px;
        }

        .controls::-webkit-scrollbar-track,
        .traces-list::-webkit-scrollbar-track,
        .annotations-list::-webkit-scrollbar-track {
            background: var(--bg-muted);
        }

        .controls::-webkit-scrollbar-thumb,
        .traces-list::-webkit-scrollbar-thumb,
        .annotations-list::-webkit-scrollbar-thumb {
            background: var(--border-default);
            border-radius: 3px;
        }

        .controls::-webkit-scrollbar-thumb:hover,
        .traces-list::-webkit-scrollbar-thumb:hover,
        .annotations-list::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="preview">
            <div class="preview-wrapper">
                <img id="preview-img" src="" alt="Figure Preview">
            </div>
        </div>
        <div class="controls">
            <div class="controls-header">
                <div>
                    <h1>SciTeX Editor</h1>
                    <div class="filename">{{ filename }}</div>
                </div>
                <button class="theme-toggle" onclick="toggleTheme()" title="Toggle light/dark mode">
                    <span id="theme-icon">&#9790;</span>
                </button>
            </div>

            <div class="controls-body">
                <!-- Labels Section -->
                <div class="section">
                    <div class="section-header section-toggle" onclick="toggleSection(this)">Labels</div>
                    <div class="section-content">
                        <div class="field">
                            <label>Title</label>
                            <input type="text" id="title" placeholder="Figure title">
                        </div>
                        <div class="field">
                            <label>X Label</label>
                            <input type="text" id="xlabel" placeholder="X axis label">
                        </div>
                        <div class="field">
                            <label>Y Label</label>
                            <input type="text" id="ylabel" placeholder="Y axis label">
                        </div>
                    </div>
                </div>

                <!-- Axis Limits Section -->
                <div class="section">
                    <div class="section-header section-toggle" onclick="toggleSection(this)">Axis Limits</div>
                    <div class="section-content">
                        <div class="field-row">
                            <div class="field">
                                <label>X Min</label>
                                <input type="number" id="xmin" step="any">
                            </div>
                            <div class="field">
                                <label>X Max</label>
                                <input type="number" id="xmax" step="any">
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>Y Min</label>
                                <input type="number" id="ymin" step="any">
                            </div>
                            <div class="field">
                                <label>Y Max</label>
                                <input type="number" id="ymax" step="any">
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Traces Section -->
                <div class="section">
                    <div class="section-header section-toggle" onclick="toggleSection(this)">Traces</div>
                    <div class="section-content">
                        <div class="traces-list" id="traces-list">
                            <!-- Dynamically populated -->
                        </div>
                        <div class="field" style="margin-top: 10px;">
                            <label>Default Line Width (pt)</label>
                            <input type="number" id="linewidth" value="1.0" min="0.1" max="5" step="0.1">
                        </div>
                    </div>
                </div>

                <!-- Legend Section -->
                <div class="section">
                    <div class="section-header section-toggle" onclick="toggleSection(this)">Legend</div>
                    <div class="section-content">
                        <label class="checkbox-field">
                            <input type="checkbox" id="legend_visible" checked>
                            <span>Show Legend</span>
                        </label>
                        <div class="field">
                            <label>Position</label>
                            <select id="legend_loc">
                                <option value="best">Best</option>
                                <option value="upper right">Upper Right</option>
                                <option value="upper left">Upper Left</option>
                                <option value="lower right">Lower Right</option>
                                <option value="lower left">Lower Left</option>
                                <option value="center right">Center Right</option>
                                <option value="center left">Center Left</option>
                                <option value="upper center">Upper Center</option>
                                <option value="lower center">Lower Center</option>
                                <option value="center">Center</option>
                            </select>
                        </div>
                        <label class="checkbox-field">
                            <input type="checkbox" id="legend_frameon">
                            <span>Show Frame</span>
                        </label>
                        <div class="field">
                            <label>Font Size (pt)</label>
                            <input type="number" id="legend_fontsize" value="6" min="4" max="16" step="1">
                        </div>
                    </div>
                </div>

                <!-- Ticks Section -->
                <div class="section">
                    <div class="section-header section-toggle" onclick="toggleSection(this)">Ticks</div>
                    <div class="section-content">
                        <div class="field-row">
                            <div class="field">
                                <label>N Ticks</label>
                                <input type="number" id="n_ticks" value="4" min="2" max="10" step="1">
                            </div>
                            <div class="field">
                                <label>Font Size (pt)</label>
                                <input type="number" id="tick_fontsize" value="7" min="4" max="16" step="1">
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>Tick Length (mm)</label>
                                <input type="number" id="tick_length" value="0.8" min="0.1" max="3" step="0.1">
                            </div>
                            <div class="field">
                                <label>Tick Width (mm)</label>
                                <input type="number" id="tick_width" value="0.2" min="0.05" max="1" step="0.05">
                            </div>
                        </div>
                        <div class="field">
                            <label>Direction</label>
                            <select id="tick_direction">
                                <option value="out">Out</option>
                                <option value="in">In</option>
                                <option value="inout">Both</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- Style Section -->
                <div class="section">
                    <div class="section-header section-toggle" onclick="toggleSection(this)">Style</div>
                    <div class="section-content">
                        <label class="checkbox-field">
                            <input type="checkbox" id="grid">
                            <span>Show Grid</span>
                        </label>
                        <label class="checkbox-field">
                            <input type="checkbox" id="hide_top_spine" checked>
                            <span>Hide Top Spine</span>
                        </label>
                        <label class="checkbox-field">
                            <input type="checkbox" id="hide_right_spine" checked>
                            <span>Hide Right Spine</span>
                        </label>
                        <div class="field-row">
                            <div class="field">
                                <label>Axis Width (mm)</label>
                                <input type="number" id="axis_width" value="0.2" min="0.05" max="1" step="0.05">
                            </div>
                            <div class="field">
                                <label>Label Size (pt)</label>
                                <input type="number" id="axis_fontsize" value="7" min="4" max="16" step="1">
                            </div>
                        </div>
                        <div class="field">
                            <label>Background Color</label>
                            <div class="color-field">
                                <input type="color" id="facecolor" value="#ffffff">
                                <input type="text" id="facecolor_text" value="#ffffff" placeholder="#ffffff">
                            </div>
                        </div>
                        <label class="checkbox-field">
                            <input type="checkbox" id="transparent" checked>
                            <span>Transparent Background</span>
                        </label>
                    </div>
                </div>

                <!-- Dimensions Section -->
                <div class="section">
                    <div class="section-header section-toggle" onclick="toggleSection(this)">Dimensions</div>
                    <div class="section-content">
                        <div class="field-row">
                            <div class="field">
                                <label>Width (inch)</label>
                                <input type="number" id="fig_width" value="3.15" min="1" max="12" step="0.1">
                            </div>
                            <div class="field">
                                <label>Height (inch)</label>
                                <input type="number" id="fig_height" value="2.68" min="1" max="12" step="0.1">
                            </div>
                        </div>
                        <div class="field">
                            <label>DPI</label>
                            <input type="number" id="dpi" value="300" min="72" max="600" step="1">
                        </div>
                    </div>
                </div>

                <!-- Annotations Section -->
                <div class="section">
                    <div class="section-header section-toggle" onclick="toggleSection(this)">Annotations</div>
                    <div class="section-content">
                        <div class="field">
                            <label>Text</label>
                            <input type="text" id="annot-text" placeholder="Annotation text">
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>X (0-1)</label>
                                <input type="number" id="annot-x" value="0.5" min="0" max="1" step="0.05">
                            </div>
                            <div class="field">
                                <label>Y (0-1)</label>
                                <input type="number" id="annot-y" value="0.5" min="0" max="1" step="0.05">
                            </div>
                            <div class="field">
                                <label>Size</label>
                                <input type="number" id="annot-size" value="8" min="4" max="24" step="1">
                            </div>
                        </div>
                        <button class="btn btn-secondary" onclick="addAnnotation()">Add Annotation</button>
                        <div class="annotations-list" id="annotations-list"></div>
                    </div>
                </div>

                <!-- Actions Section -->
                <div class="section">
                    <div class="section-header">Actions</div>
                    <button class="btn btn-cta" onclick="updatePreview()">Update Preview</button>
                    <button class="btn btn-primary" onclick="saveManual()">Save to .manual.json</button>
                    <button class="btn btn-secondary" onclick="resetOverrides()">Reset to Original</button>
                </div>

                <div class="status-bar" id="status">Ready</div>
            </div>
        </div>
    </div>

    <script>
        let overrides = {{ overrides|safe }};
        let traces = overrides.traces || [];

        // Theme management
        function toggleTheme() {
            const html = document.documentElement;
            const current = html.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', next);
            document.getElementById('theme-icon').innerHTML = next === 'dark' ? '&#9790;' : '&#9788;';
            localStorage.setItem('scitex-editor-theme', next);
        }

        // Load saved theme
        const savedTheme = localStorage.getItem('scitex-editor-theme');
        if (savedTheme) {
            document.documentElement.setAttribute('data-theme', savedTheme);
            document.getElementById('theme-icon').innerHTML = savedTheme === 'dark' ? '&#9790;' : '&#9788;';
        }

        // Collapsible sections
        function toggleSection(header) {
            header.classList.toggle('collapsed');
            const content = header.nextElementSibling;
            content.classList.toggle('collapsed');
        }

        // Initialize fields
        document.addEventListener('DOMContentLoaded', () => {
            // Labels
            if (overrides.title) document.getElementById('title').value = overrides.title;
            if (overrides.xlabel) document.getElementById('xlabel').value = overrides.xlabel;
            if (overrides.ylabel) document.getElementById('ylabel').value = overrides.ylabel;

            // Axis limits
            if (overrides.xlim) {
                document.getElementById('xmin').value = overrides.xlim[0];
                document.getElementById('xmax').value = overrides.xlim[1];
            }
            if (overrides.ylim) {
                document.getElementById('ymin').value = overrides.ylim[0];
                document.getElementById('ymax').value = overrides.ylim[1];
            }

            // Traces
            document.getElementById('linewidth').value = overrides.linewidth || 1.0;
            updateTracesList();

            // Legend
            document.getElementById('legend_visible').checked = overrides.legend_visible !== false;
            document.getElementById('legend_loc').value = overrides.legend_loc || 'best';
            document.getElementById('legend_frameon').checked = overrides.legend_frameon || false;
            document.getElementById('legend_fontsize').value = overrides.legend_fontsize || 6;

            // Ticks
            document.getElementById('n_ticks').value = overrides.n_ticks || 4;
            document.getElementById('tick_fontsize').value = overrides.tick_fontsize || 7;
            document.getElementById('tick_length').value = overrides.tick_length || 0.8;
            document.getElementById('tick_width').value = overrides.tick_width || 0.2;
            document.getElementById('tick_direction').value = overrides.tick_direction || 'out';

            // Style
            document.getElementById('grid').checked = overrides.grid || false;
            document.getElementById('hide_top_spine').checked = overrides.hide_top_spine !== false;
            document.getElementById('hide_right_spine').checked = overrides.hide_right_spine !== false;
            document.getElementById('axis_width').value = overrides.axis_width || 0.2;
            document.getElementById('axis_fontsize').value = overrides.axis_fontsize || 7;
            document.getElementById('facecolor').value = overrides.facecolor || '#ffffff';
            document.getElementById('facecolor_text').value = overrides.facecolor || '#ffffff';
            document.getElementById('transparent').checked = overrides.transparent !== false;

            // Dimensions
            if (overrides.fig_size) {
                document.getElementById('fig_width').value = overrides.fig_size[0];
                document.getElementById('fig_height').value = overrides.fig_size[1];
            }
            document.getElementById('dpi').value = overrides.dpi || 300;

            // Sync color inputs
            document.getElementById('facecolor').addEventListener('input', (e) => {
                document.getElementById('facecolor_text').value = e.target.value;
            });
            document.getElementById('facecolor_text').addEventListener('change', (e) => {
                document.getElementById('facecolor').value = e.target.value;
            });

            updateAnnotationsList();
            updatePreview();
        });

        // Traces list management
        function updateTracesList() {
            const list = document.getElementById('traces-list');
            if (!traces || traces.length === 0) {
                list.innerHTML = '<div style="padding: 10px; color: var(--text-muted); font-size: 0.85em;">No traces found in metadata</div>';
                return;
            }

            list.innerHTML = traces.map((t, i) => `
                <div class="trace-item">
                    <input type="color" class="trace-color" value="${t.color || '#1f77b4'}"
                           onchange="updateTraceColor(${i}, this.value)">
                    <span class="trace-label">${t.label || t.id || 'Trace ' + (i+1)}</span>
                    <div class="trace-style">
                        <select onchange="updateTraceStyle(${i}, this.value)">
                            <option value="-" ${t.linestyle === '-' ? 'selected' : ''}>Solid</option>
                            <option value="--" ${t.linestyle === '--' ? 'selected' : ''}>Dashed</option>
                            <option value=":" ${t.linestyle === ':' ? 'selected' : ''}>Dotted</option>
                            <option value="-." ${t.linestyle === '-.' ? 'selected' : ''}>Dash-dot</option>
                        </select>
                    </div>
                </div>
            `).join('');
        }

        function updateTraceColor(idx, color) {
            if (traces[idx]) {
                traces[idx].color = color;
            }
        }

        function updateTraceStyle(idx, style) {
            if (traces[idx]) {
                traces[idx].linestyle = style;
            }
        }

        function collectOverrides() {
            const o = {};

            // Labels
            const title = document.getElementById('title').value;
            const xlabel = document.getElementById('xlabel').value;
            const ylabel = document.getElementById('ylabel').value;
            if (title) o.title = title;
            if (xlabel) o.xlabel = xlabel;
            if (ylabel) o.ylabel = ylabel;

            // Axis limits
            const xmin = document.getElementById('xmin').value;
            const xmax = document.getElementById('xmax').value;
            if (xmin !== '' && xmax !== '') o.xlim = [parseFloat(xmin), parseFloat(xmax)];

            const ymin = document.getElementById('ymin').value;
            const ymax = document.getElementById('ymax').value;
            if (ymin !== '' && ymax !== '') o.ylim = [parseFloat(ymin), parseFloat(ymax)];

            // Traces
            o.linewidth = parseFloat(document.getElementById('linewidth').value) || 1.0;
            o.traces = traces;

            // Legend
            o.legend_visible = document.getElementById('legend_visible').checked;
            o.legend_loc = document.getElementById('legend_loc').value;
            o.legend_frameon = document.getElementById('legend_frameon').checked;
            o.legend_fontsize = parseInt(document.getElementById('legend_fontsize').value) || 6;

            // Ticks
            o.n_ticks = parseInt(document.getElementById('n_ticks').value) || 4;
            o.tick_fontsize = parseInt(document.getElementById('tick_fontsize').value) || 7;
            o.tick_length = parseFloat(document.getElementById('tick_length').value) || 0.8;
            o.tick_width = parseFloat(document.getElementById('tick_width').value) || 0.2;
            o.tick_direction = document.getElementById('tick_direction').value;

            // Style
            o.grid = document.getElementById('grid').checked;
            o.hide_top_spine = document.getElementById('hide_top_spine').checked;
            o.hide_right_spine = document.getElementById('hide_right_spine').checked;
            o.axis_width = parseFloat(document.getElementById('axis_width').value) || 0.2;
            o.axis_fontsize = parseInt(document.getElementById('axis_fontsize').value) || 7;
            o.facecolor = document.getElementById('facecolor').value;
            o.transparent = document.getElementById('transparent').checked;

            // Dimensions
            o.fig_size = [
                parseFloat(document.getElementById('fig_width').value) || 3.15,
                parseFloat(document.getElementById('fig_height').value) || 2.68
            ];
            o.dpi = parseInt(document.getElementById('dpi').value) || 300;

            // Annotations
            o.annotations = overrides.annotations || [];

            return o;
        }

        async function updatePreview() {
            setStatus('Updating...', false);
            overrides = collectOverrides();
            try {
                const resp = await fetch('/update', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({overrides})
                });
                const data = await resp.json();
                document.getElementById('preview-img').src = 'data:image/png;base64,' + data.image;
                setStatus('Preview updated', false);
            } catch (e) {
                setStatus('Error: ' + e.message, true);
            }
        }

        async function saveManual() {
            setStatus('Saving...', false);
            try {
                const resp = await fetch('/save', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                const data = await resp.json();
                if (data.status === 'saved') {
                    setStatus('Saved: ' + data.path.split('/').pop(), false);
                } else {
                    setStatus('Error: ' + data.message, true);
                }
            } catch (e) {
                setStatus('Error: ' + e.message, true);
            }
        }

        function resetOverrides() {
            if (confirm('Reset all changes to original values?')) {
                location.reload();
            }
        }

        function addAnnotation() {
            const text = document.getElementById('annot-text').value;
            if (!text) return;
            const x = parseFloat(document.getElementById('annot-x').value) || 0.5;
            const y = parseFloat(document.getElementById('annot-y').value) || 0.5;
            const size = parseInt(document.getElementById('annot-size').value) || 8;
            if (!overrides.annotations) overrides.annotations = [];
            overrides.annotations.push({type: 'text', text, x, y, fontsize: size});
            document.getElementById('annot-text').value = '';
            updateAnnotationsList();
            updatePreview();
        }

        function removeAnnotation(idx) {
            overrides.annotations.splice(idx, 1);
            updateAnnotationsList();
            updatePreview();
        }

        function updateAnnotationsList() {
            const list = document.getElementById('annotations-list');
            const annotations = overrides.annotations || [];
            if (annotations.length === 0) {
                list.innerHTML = '';
                return;
            }
            list.innerHTML = annotations.map((a, i) =>
                `<div class="annotation-item">
                    <span>${a.text.substring(0, 25)}${a.text.length > 25 ? '...' : ''} (${a.x.toFixed(2)}, ${a.y.toFixed(2)})</span>
                    <button onclick="removeAnnotation(${i})">Remove</button>
                </div>`
            ).join('');
        }

        function setStatus(msg, isError = false) {
            const el = document.getElementById('status');
            el.textContent = msg;
            el.classList.toggle('error', isError);
        }

        // Auto-update on Enter key in input fields
        document.querySelectorAll('input[type="text"], input[type="number"]').forEach(el => {
            el.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') updatePreview();
            });
        });
    </script>
</body>
</html>
'''

# EOF
