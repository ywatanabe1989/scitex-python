#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/templates/html.py
"""HTML structure for the Flask editor UI."""

HTML_BODY = """
<div class="container">
    <div class="preview">
        <!-- Panel Grid View (for multi-panel figz bundles) -->
        <div class="panel-grid-section" id="panel-grid-section" style="display: none;">
            <div class="panel-grid-header">
                <h3>All Panels</h3>
                <div class="canvas-controls">
                    <button class="btn btn-secondary btn-sm" id="panel-debug-btn" onclick="togglePanelDebugMode()">Show Hit Regions</button>
                </div>
            </div>
            <div class="panel-canvas" id="panel-canvas">
                <!-- Panels will be rendered here -->
            </div>
        </div>

        <div class="preview-wrapper">
            <div class="preview-header" id="preview-header" style="display: none;">
                <span id="current-panel-name">Panel A</span>
                <div class="panel-nav">
                    <button class="btn btn-sm" onclick="prevPanel()" id="prev-panel-btn">&laquo; Prev</button>
                    <span id="panel-indicator">1 / 6</span>
                    <button class="btn btn-sm" onclick="nextPanel()" id="next-panel-btn">Next &raquo;</button>
                    <button class="btn btn-secondary btn-sm" onclick="togglePanelGrid()" id="show-grid-btn">Show All</button>
                </div>
            </div>
            <div class="preview-container" id="preview-container">
                <img id="preview-img" src="" alt="Figure Preview">
                <svg id="hover-overlay" class="hover-overlay"></svg>
                <div id="loading-overlay" class="loading-overlay" style="display: none;">
                    <div class="spinner"></div>
                </div>
            </div>
            <button id="debug-toggle-btn" class="debug-toggle" onclick="toggleDebugMode()">Show Hit Areas</button>
        </div>
    </div>
    <div class="controls">
        <div class="controls-header">
            <div>
                <h1>SciTeX Figure Editor</h1>
                <div class="filename">{{ filename }}</div>
                {% if panel_path %}<div class="panel-path" id="panel-path-display">Panel: {{ panel_path }}</div>{% endif %}
            </div>
            <button class="theme-toggle" onclick="toggleTheme()" title="Toggle light/dark mode">
                <span id="theme-icon">&#9790;</span>
            </button>
        </div>

        <div class="controls-body">
            <!-- Selected Element Section (Dynamic) -->
            <div class="section" id="section-selected" style="display: none;">
                <div class="section-header section-toggle" onclick="toggleSection(this)">
                    <span id="selected-element-title">Selected: None</span>
                </div>
                <div class="section-content">
                    <div id="selected-element-info" class="selected-info">
                        <div class="element-type-badge" id="element-type-badge"></div>
                        <div class="element-axis-info" id="element-axis-info"></div>
                    </div>

                    <!-- Trace/Line Properties -->
                    <div id="selected-trace-props" class="element-props" style="display: none;">
                        <div class="field">
                            <label>Label</label>
                            <input type="text" id="sel-trace-label" placeholder="Trace label">
                        </div>
                        <div class="field">
                            <label>Color</label>
                            <div class="color-field">
                                <input type="color" id="sel-trace-color" value="#1f77b4">
                                <input type="text" id="sel-trace-color-text" value="#1f77b4">
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>Line Width (pt)</label>
                                <input type="number" id="sel-trace-linewidth" value="1.0" min="0.1" max="5" step="0.1">
                            </div>
                            <div class="field">
                                <label>Line Style</label>
                                <select id="sel-trace-linestyle">
                                    <option value="-">Solid</option>
                                    <option value="--">Dashed</option>
                                    <option value="-.">Dash-dot</option>
                                    <option value=":">Dotted</option>
                                </select>
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>Marker</label>
                                <select id="sel-trace-marker">
                                    <option value="">None</option>
                                    <option value="o">Circle</option>
                                    <option value="s">Square</option>
                                    <option value="^">Triangle</option>
                                    <option value="D">Diamond</option>
                                    <option value="x">X</option>
                                    <option value="+">Plus</option>
                                </select>
                            </div>
                            <div class="field">
                                <label>Marker Size</label>
                                <input type="number" id="sel-trace-markersize" value="4" min="1" max="20" step="0.5">
                            </div>
                        </div>
                        <div class="field">
                            <label>Alpha (0-1)</label>
                            <input type="range" id="sel-trace-alpha" min="0" max="1" step="0.1" value="1">
                        </div>
                    </div>

                    <!-- Scatter Properties -->
                    <div id="selected-scatter-props" class="element-props" style="display: none;">
                        <div class="field">
                            <label>Color</label>
                            <div class="color-field">
                                <input type="color" id="sel-scatter-color" value="#1f77b4">
                                <input type="text" id="sel-scatter-color-text" value="#1f77b4">
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>Marker Size</label>
                                <input type="number" id="sel-scatter-size" value="20" min="1" max="200" step="1">
                            </div>
                            <div class="field">
                                <label>Marker</label>
                                <select id="sel-scatter-marker">
                                    <option value="o">Circle</option>
                                    <option value="s">Square</option>
                                    <option value="^">Triangle</option>
                                    <option value="D">Diamond</option>
                                    <option value="x">X</option>
                                </select>
                            </div>
                        </div>
                        <div class="field">
                            <label>Alpha (0-1)</label>
                            <input type="range" id="sel-scatter-alpha" min="0" max="1" step="0.1" value="0.7">
                        </div>
                        <div class="field">
                            <label>Edge Color</label>
                            <div class="color-field">
                                <input type="color" id="sel-scatter-edgecolor" value="#000000">
                                <input type="text" id="sel-scatter-edgecolor-text" value="#000000">
                            </div>
                        </div>
                    </div>

                    <!-- Fill Properties -->
                    <div id="selected-fill-props" class="element-props" style="display: none;">
                        <div class="field">
                            <label>Fill Color</label>
                            <div class="color-field">
                                <input type="color" id="sel-fill-color" value="#1f77b4">
                                <input type="text" id="sel-fill-color-text" value="#1f77b4">
                            </div>
                        </div>
                        <div class="field">
                            <label>Alpha (0-1)</label>
                            <input type="range" id="sel-fill-alpha" min="0" max="1" step="0.05" value="0.3">
                        </div>
                    </div>

                    <!-- Bar Properties -->
                    <div id="selected-bar-props" class="element-props" style="display: none;">
                        <div class="field">
                            <label>Face Color</label>
                            <div class="color-field">
                                <input type="color" id="sel-bar-facecolor" value="#1f77b4">
                                <input type="text" id="sel-bar-facecolor-text" value="#1f77b4">
                            </div>
                        </div>
                        <div class="field">
                            <label>Edge Color</label>
                            <div class="color-field">
                                <input type="color" id="sel-bar-edgecolor" value="#000000">
                                <input type="text" id="sel-bar-edgecolor-text" value="#000000">
                            </div>
                        </div>
                        <div class="field">
                            <label>Alpha (0-1)</label>
                            <input type="range" id="sel-bar-alpha" min="0" max="1" step="0.1" value="1">
                        </div>
                    </div>

                    <!-- Label/Text Properties -->
                    <div id="selected-label-props" class="element-props" style="display: none;">
                        <div class="field">
                            <label>Text</label>
                            <input type="text" id="sel-label-text" placeholder="Label text">
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>Font Size (pt)</label>
                                <input type="number" id="sel-label-fontsize" value="7" min="4" max="24" step="1">
                            </div>
                            <div class="field">
                                <label>Font Weight</label>
                                <select id="sel-label-fontweight">
                                    <option value="normal">Normal</option>
                                    <option value="bold">Bold</option>
                                </select>
                            </div>
                        </div>
                        <div class="field">
                            <label>Color</label>
                            <div class="color-field">
                                <input type="color" id="sel-label-color" value="#000000">
                                <input type="text" id="sel-label-color-text" value="#000000">
                            </div>
                        </div>
                    </div>

                    <!-- Panel Properties -->
                    <div id="selected-panel-props" class="element-props" style="display: none;">
                        <div class="field">
                            <label>Panel Title</label>
                            <input type="text" id="sel-panel-title" placeholder="Panel title">
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>X Label</label>
                                <input type="text" id="sel-panel-xlabel" placeholder="X axis">
                            </div>
                            <div class="field">
                                <label>Y Label</label>
                                <input type="text" id="sel-panel-ylabel" placeholder="Y axis">
                            </div>
                        </div>
                        <div class="field">
                            <label>Background Color</label>
                            <div class="color-field">
                                <input type="color" id="sel-panel-facecolor" value="#ffffff">
                                <input type="text" id="sel-panel-facecolor-text" value="#ffffff">
                            </div>
                        </div>
                        <label class="checkbox-field">
                            <input type="checkbox" id="sel-panel-transparent" checked>
                            <span>Transparent</span>
                        </label>
                        <label class="checkbox-field">
                            <input type="checkbox" id="sel-panel-grid">
                            <span>Show Grid</span>
                        </label>
                    </div>

                    <!-- X-Axis Properties -->
                    <div id="selected-xaxis-props" class="element-props" style="display: none;">
                        <div class="field-row">
                            <div class="field">
                                <label>Tick Font Size (pt)</label>
                                <input type="number" id="sel-xaxis-fontsize" value="7" min="4" max="16" step="1">
                            </div>
                            <div class="field">
                                <label>Label Font Size (pt)</label>
                                <input type="number" id="sel-xaxis-label-fontsize" value="7" min="4" max="16" step="1">
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>Tick Direction</label>
                                <select id="sel-xaxis-direction">
                                    <option value="out">Out</option>
                                    <option value="in">In</option>
                                    <option value="inout">Both</option>
                                </select>
                            </div>
                            <div class="field">
                                <label>N Ticks</label>
                                <input type="number" id="sel-xaxis-nticks" value="5" min="2" max="15" step="1">
                            </div>
                        </div>
                        <label class="checkbox-field">
                            <input type="checkbox" id="sel-xaxis-hide-ticks">
                            <span>Hide Ticks</span>
                        </label>
                        <label class="checkbox-field">
                            <input type="checkbox" id="sel-xaxis-hide-label">
                            <span>Hide Label</span>
                        </label>
                        <label class="checkbox-field">
                            <input type="checkbox" id="sel-xaxis-hide-spine">
                            <span>Hide Spine</span>
                        </label>
                    </div>

                    <!-- Y-Axis Properties -->
                    <div id="selected-yaxis-props" class="element-props" style="display: none;">
                        <div class="field-row">
                            <div class="field">
                                <label>Tick Font Size (pt)</label>
                                <input type="number" id="sel-yaxis-fontsize" value="7" min="4" max="16" step="1">
                            </div>
                            <div class="field">
                                <label>Label Font Size (pt)</label>
                                <input type="number" id="sel-yaxis-label-fontsize" value="7" min="4" max="16" step="1">
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>Tick Direction</label>
                                <select id="sel-yaxis-direction">
                                    <option value="out">Out</option>
                                    <option value="in">In</option>
                                    <option value="inout">Both</option>
                                </select>
                            </div>
                            <div class="field">
                                <label>N Ticks</label>
                                <input type="number" id="sel-yaxis-nticks" value="5" min="2" max="15" step="1">
                            </div>
                        </div>
                        <label class="checkbox-field">
                            <input type="checkbox" id="sel-yaxis-hide-ticks">
                            <span>Hide Ticks</span>
                        </label>
                        <label class="checkbox-field">
                            <input type="checkbox" id="sel-yaxis-hide-label">
                            <span>Hide Label</span>
                        </label>
                        <label class="checkbox-field">
                            <input type="checkbox" id="sel-yaxis-hide-spine">
                            <span>Hide Spine</span>
                        </label>
                    </div>

                    <!-- Statistics (for data elements) -->
                    <div id="selected-stats" class="element-stats" style="display: none;">
                        <div class="stats-header">Statistics</div>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <span class="stat-label">N points</span>
                                <span class="stat-value" id="stat-n">-</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Mean</span>
                                <span class="stat-value" id="stat-mean">-</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Std</span>
                                <span class="stat-value" id="stat-std">-</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Min</span>
                                <span class="stat-value" id="stat-min">-</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Max</span>
                                <span class="stat-value" id="stat-max">-</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Range</span>
                                <span class="stat-value" id="stat-range">-</span>
                            </div>
                        </div>
                    </div>

                    <button class="btn btn-primary" onclick="applySelectedElementChanges()" style="margin-top: 10px;">Apply Changes</button>
                </div>
            </div>

            <!-- Dimensions Section (General - moved higher) -->
            <div class="section" id="section-dimensions">
                <div class="section-header section-toggle collapsed" onclick="toggleSection(this)">Dimensions</div>
                <div class="section-content collapsed">
                    <div class="field" style="margin-bottom: 8px;">
                        <label>Unit</label>
                        <div class="unit-toggle">
                            <button class="unit-btn active" id="unit-mm" onclick="setDimensionUnit('mm')">mm</button>
                            <button class="unit-btn" id="unit-inch" onclick="setDimensionUnit('inch')">inch</button>
                        </div>
                    </div>
                    <div class="field-row">
                        <div class="field">
                            <label id="fig_width_label">Width (mm)</label>
                            <input type="number" id="fig_width" value="80" min="10" max="300" step="1">
                        </div>
                        <div class="field">
                            <label id="fig_height_label">Height (mm)</label>
                            <input type="number" id="fig_height" value="68" min="10" max="300" step="1">
                        </div>
                    </div>
                    <div class="field">
                        <label>DPI</label>
                        <input type="number" id="dpi" value="300" min="72" max="600" step="1">
                    </div>
                </div>
            </div>

            <!-- Style Section (General - moved higher) -->
            <div class="section" id="section-style">
                <div class="section-header section-toggle collapsed" onclick="toggleSection(this)">Style</div>
                <div class="section-content collapsed">
                    <label class="checkbox-field">
                        <input type="checkbox" id="grid">
                        <span>Show Grid</span>
                    </label>
                    <div class="field">
                        <label>Label Size (pt)</label>
                        <input type="number" id="axis_fontsize" value="7" min="4" max="16" step="1">
                    </div>
                    <div class="field">
                        <label>Background</label>
                        <div class="bg-toggle">
                            <button class="bg-btn" id="bg-white" onclick="setBackgroundType('white')" title="White background">
                                <span class="bg-preview white"></span>
                                <span>White</span>
                            </button>
                            <button class="bg-btn active" id="bg-transparent" onclick="setBackgroundType('transparent')" title="Transparent background">
                                <span class="bg-preview transparent"></span>
                                <span>Transparent</span>
                            </button>
                            <button class="bg-btn" id="bg-black" onclick="setBackgroundType('black')" title="Black background">
                                <span class="bg-preview black"></span>
                                <span>Black</span>
                            </button>
                        </div>
                    </div>
                    <input type="hidden" id="facecolor" value="#ffffff">
                    <input type="hidden" id="transparent" value="true">
                </div>
            </div>

            <!-- Title, Labels & Caption Section -->
            <div class="section" id="section-labels">
                <div class="section-header section-toggle collapsed" onclick="toggleSection(this)">Title, Labels & Caption</div>
                <div class="section-content collapsed">
                    <p class="section-hint">For multi-panel figures, select a panel to edit its labels.</p>

                    <!-- Title row in table style -->
                    <table class="props-table" style="width: 100%; border-collapse: collapse; margin-bottom: 8px;">
                        <tr>
                            <td style="width: 70px; padding: 4px 0;"><label>Title</label></td>
                            <td style="padding: 4px 0;">
                                <input type="text" id="title" placeholder="Figure title" style="width: 100%;">
                            </td>
                            <td style="width: 50px; padding: 4px 0; text-align: right;">
                                <input type="number" id="title_fontsize" value="8" min="4" max="24" step="1" style="width: 45px;" title="Font size (pt)">
                            </td>
                            <td style="width: 24px; padding: 4px 0;">
                                <label class="checkbox-field" style="margin: 0;" title="Show title">
                                    <input type="checkbox" id="show_title" checked>
                                </label>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 4px 0;"><label>X Label</label></td>
                            <td style="padding: 4px 0;">
                                <input type="text" id="xlabel" placeholder="X axis label" style="width: 100%;">
                            </td>
                            <td colspan="2"></td>
                        </tr>
                        <tr>
                            <td style="padding: 4px 0;"><label>Y Label</label></td>
                            <td style="padding: 4px 0;">
                                <input type="text" id="ylabel" placeholder="Y axis label" style="width: 100%;">
                            </td>
                            <td colspan="2"></td>
                        </tr>
                        <tr>
                            <td style="padding: 4px 0; vertical-align: top;"><label>Caption</label></td>
                            <td style="padding: 4px 0;">
                                <textarea id="caption" rows="2" placeholder="Figure caption..." style="width: 100%; padding: 6px; border: 1px solid var(--border-muted); border-radius: 4px; background: var(--bg-surface); color: var(--text-primary); font-size: 0.85em; resize: vertical;"></textarea>
                            </td>
                            <td style="padding: 4px 0; text-align: right; vertical-align: top;">
                                <input type="number" id="caption_fontsize" value="7" min="4" max="16" step="1" style="width: 45px;" title="Font size (pt)">
                            </td>
                            <td style="padding: 4px 0; vertical-align: top;">
                                <label class="checkbox-field" style="margin: 0;" title="Show caption">
                                    <input type="checkbox" id="show_caption">
                                </label>
                            </td>
                        </tr>
                    </table>
                </div>
            </div>

            <!-- Axis & Ticks Section (merged with Axis Limits) -->
            <div class="section" id="section-ticks">
                <div class="section-header section-toggle collapsed" onclick="toggleSection(this)">Axis & Ticks</div>
                <div class="section-content collapsed">
                    <!-- Axis Limits (merged from separate section) -->
                    <div class="subsection-header">Limits</div>
                    <div class="field-row">
                        <div class="field">
                            <label>X Range</label>
                            <div class="field-row" style="gap: 4px; margin-top: 4px;">
                                <input type="number" id="xmin" step="any" placeholder="Min">
                                <input type="number" id="xmax" step="any" placeholder="Max">
                            </div>
                        </div>
                        <div class="field">
                            <label>Y Range</label>
                            <div class="field-row" style="gap: 4px; margin-top: 4px;">
                                <input type="number" id="ymin" step="any" placeholder="Min">
                                <input type="number" id="ymax" step="any" placeholder="Max">
                            </div>
                        </div>
                    </div>

                    <!-- Axis Tabs -->
                    <div class="subsection-header" style="margin-top: 12px;">Tick Settings</div>
                    <div class="axis-tabs">
                        <button class="axis-tab active" onclick="switchAxisTab('x')" id="axis-tab-x">X</button>
                        <button class="axis-tab" onclick="switchAxisTab('y')" id="axis-tab-y">Y</button>
                        <button class="axis-tab" onclick="switchAxisTab('z')" id="axis-tab-z">Z</button>
                    </div>

                    <!-- X Axis Panel -->
                    <div class="axis-panel" id="axis-panel-x">
                        <div class="subsection-header">Bottom (Primary)</div>
                        <div class="field-row">
                            <div class="field">
                                <label class="checkbox-field">
                                    <input type="checkbox" id="hide_x_ticks">
                                    <span>Hide</span>
                                </label>
                            </div>
                            <div class="field">
                                <label>N Ticks</label>
                                <input type="number" id="x_n_ticks" value="4" min="2" max="10" step="1">
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>Font Size (pt)</label>
                                <input type="number" id="x_tick_fontsize" value="7" min="4" max="16" step="1">
                            </div>
                            <div class="field">
                                <label>Direction</label>
                                <select id="x_tick_direction">
                                    <option value="out">Out</option>
                                    <option value="in">In</option>
                                    <option value="inout">Both</option>
                                </select>
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>Length (mm)</label>
                                <input type="number" id="x_tick_length" value="0.8" min="0.1" max="3" step="0.1">
                            </div>
                            <div class="field">
                                <label>Width (mm)</label>
                                <input type="number" id="x_tick_width" value="0.2" min="0.05" max="1" step="0.05">
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label class="checkbox-field">
                                    <input type="checkbox" id="hide_bottom_spine" unchecked>
                                    <span>Hide Spine</span>
                                </label>
                            </div>
                        </div>

                        <div class="subsection-header" style="margin-top: 12px;">Top (Secondary)</div>
                        <div class="field-row">
                            <div class="field">
                                <label class="checkbox-field">
                                    <input type="checkbox" id="show_x_top" unchecked>
                                    <span>Show</span>
                                </label>
                            </div>
                            <div class="field">
                                <label class="checkbox-field">
                                    <input type="checkbox" id="x_top_mirror">
                                    <span>Mirror</span>
                                </label>
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label class="checkbox-field">
                                    <input type="checkbox" id="hide_top_spine" checked>
                                    <span>Hide Spine</span>
                                </label>
                            </div>
                        </div>
                    </div>

                    <!-- Y Axis Panel -->
                    <div class="axis-panel" id="axis-panel-y" style="display: none;">
                        <div class="subsection-header">Left (Primary)</div>
                        <div class="field-row">
                            <div class="field">
                                <label class="checkbox-field">
                                    <input type="checkbox" id="hide_y_ticks">
                                    <span>Hide</span>
                                </label>
                            </div>
                            <div class="field">
                                <label>N Ticks</label>
                                <input type="number" id="y_n_ticks" value="4" min="2" max="10" step="1">
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>Font Size (pt)</label>
                                <input type="number" id="y_tick_fontsize" value="7" min="4" max="16" step="1">
                            </div>
                            <div class="field">
                                <label>Direction</label>
                                <select id="y_tick_direction">
                                    <option value="out">Out</option>
                                    <option value="in">In</option>
                                    <option value="inout">Both</option>
                                </select>
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>Length (mm)</label>
                                <input type="number" id="y_tick_length" value="0.8" min="0.1" max="3" step="0.1">
                            </div>
                            <div class="field">
                                <label>Width (mm)</label>
                                <input type="number" id="y_tick_width" value="0.2" min="0.05" max="1" step="0.05">
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label class="checkbox-field">
                                    <input type="checkbox" id="hide_left_spine" unchecked>
                                    <span>Hide Spine</span>
                                </label>
                            </div>
                        </div>

                        <div class="subsection-header" style="margin-top: 12px;">Right (Secondary)</div>
                        <div class="field-row">
                            <div class="field">
                                <label class="checkbox-field">
                                    <input type="checkbox" id="show_y_right" unchecked>
                                    <span>Show</span>
                                </label>
                            </div>
                            <div class="field">
                                <label class="checkbox-field">
                                    <input type="checkbox" id="y_right_mirror">
                                    <span>Mirror</span>
                                </label>
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label class="checkbox-field">
                                    <input type="checkbox" id="hide_right_spine" checked>
                                    <span>Hide Spine</span>
                                </label>
                            </div>
                        </div>
                    </div>

                    <!-- Z Axis Panel (for 3D plots) -->
                    <div class="axis-panel" id="axis-panel-z" style="display: none;">
                        <div class="subsection-header">Z Axis (3D)</div>
                        <div class="field-row">
                            <div class="field">
                                <label class="checkbox-field">
                                    <input type="checkbox" id="hide_z_ticks">
                                    <span>Hide</span>
                                </label>
                            </div>
                            <div class="field">
                                <label>N Ticks</label>
                                <input type="number" id="z_n_ticks" value="4" min="2" max="10" step="1">
                            </div>
                        </div>
                        <div class="field-row">
                            <div class="field">
                                <label>Font Size (pt)</label>
                                <input type="number" id="z_tick_fontsize" value="7" min="4" max="16" step="1">
                            </div>
                            <div class="field">
                                <label>Direction</label>
                                <select id="z_tick_direction">
                                    <option value="out">Out</option>
                                    <option value="in">In</option>
                                    <option value="inout">Both</option>
                                </select>
                            </div>
                        </div>
                        <p class="section-hint">Z axis settings apply to 3D plots only.</p>
                    </div>

                    <!-- Common Settings -->
                    <div class="subsection-header" style="margin-top: 12px;">Common</div>
                    <div class="field">
                        <label>Spine Width (mm)</label>
                        <input type="number" id="axis_width" value="0.2" min="0.05" max="1" step="0.05">
                    </div>
                </div>
            </div>

            <!-- Traces Section (Simplified for beta) -->
            <div class="section" id="section-traces">
                <div class="section-header section-toggle collapsed" onclick="toggleSection(this)">Traces</div>
                <div class="section-content collapsed">
                    <p class="section-hint">Click on a trace in the preview to edit its properties.</p>
                    <div class="traces-list" id="traces-list">
                        <!-- Dynamically populated -->
                    </div>
                </div>
            </div>

            <!-- Legend Section (Enhanced) -->
            <div class="section" id="section-legend">
                <div class="section-header section-toggle collapsed" onclick="toggleSection(this)">Legend</div>
                <div class="section-content collapsed">
                    <label class="checkbox-field">
                        <input type="checkbox" id="legend_visible" checked>
                        <span>Show Legend</span>
                    </label>
                    <div class="field-row">
                        <div class="field">
                            <label>Position</label>
                            <select id="legend_loc" onchange="toggleCustomLegendPosition()">
                                <option value="best">Best (auto)</option>
                                <option value="upper right">Upper Right</option>
                                <option value="upper left">Upper Left</option>
                                <option value="lower right">Lower Right</option>
                                <option value="lower left">Lower Left</option>
                                <option value="center right">Center Right</option>
                                <option value="center left">Center Left</option>
                                <option value="upper center">Upper Center</option>
                                <option value="lower center">Lower Center</option>
                                <option value="center">Center</option>
                                <option value="custom">Custom...</option>
                            </select>
                        </div>
                        <div class="field">
                            <label>Columns</label>
                            <input type="number" id="legend_ncols" value="1" min="1" max="5" step="1" title="Number of columns">
                        </div>
                    </div>
                    <div id="custom-legend-coords" class="field-row" style="display: none; margin-top: 8px;">
                        <div class="field">
                            <label>X (0-1)</label>
                            <input type="number" id="legend_x" value="0.95" min="0" max="1.5" step="0.01" title="X coordinate in axes fraction">
                        </div>
                        <div class="field">
                            <label>Y (0-1)</label>
                            <input type="number" id="legend_y" value="0.95" min="-0.5" max="1.5" step="0.01" title="Y coordinate in axes fraction">
                        </div>
                    </div>
                    <div class="field-row" style="margin-top: 8px;">
                        <div class="field">
                            <label class="checkbox-field">
                                <input type="checkbox" id="legend_frameon">
                                <span>Show Frame</span>
                            </label>
                        </div>
                        <div class="field">
                            <label>Font Size (pt)</label>
                            <input type="number" id="legend_fontsize" value="6" min="4" max="16" step="1">
                        </div>
                    </div>
                </div>
            </div>

            <!-- Statistics Section -->
            <div class="section" id="section-statistics">
                <div class="section-header section-toggle" onclick="toggleSection(this)">Statistics</div>
                <div class="section-content">
                    <div id="stats-container">
                        <div class="stats-loading">Loading statistics...</div>
                    </div>
                    <button class="btn btn-secondary" onclick="refreshStats()" style="margin-top: 8px;">Refresh Stats</button>
                </div>
            </div>

            <!-- Annotations Section -->
            <div class="section" id="section-annotations">
                <div class="section-header section-toggle collapsed" onclick="toggleSection(this)">Annotations</div>
                <div class="section-content collapsed">
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
                <div class="field-row" style="align-items: center; margin-bottom: 8px;">
                    <div class="field" style="flex: 1;">
                        <label>Auto-Update</label>
                        <select id="auto_update_interval" onchange="setAutoUpdateInterval()">
                            <option value="0">Off</option>
                            <option value="500">Hot (0.5s)</option>
                            <option value="1000">Fast (1s)</option>
                            <option value="2000" selected>Normal (2s)</option>
                            <option value="5000">Slow (5s)</option>
                        </select>
                    </div>
                    <button class="btn btn-cta" onclick="updatePreview(true)" style="flex: 0; margin-left: 8px;">Update Now</button>
                </div>
                <button class="btn btn-primary" onclick="saveManual()" title="Ctrl+S">Save</button>
                <button class="btn btn-secondary" onclick="resetOverrides()" title="Reset to original values">Reset</button>
            </div>

            <div class="status-bar" id="status">Ready</div>
        </div>
    </div>
</div>
"""


# EOF
