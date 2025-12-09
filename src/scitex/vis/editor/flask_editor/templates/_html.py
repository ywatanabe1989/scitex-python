#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/templates/html.py
"""HTML structure for the Flask editor UI."""

HTML_BODY = """
<div class="container">
    <div class="preview">
        <div class="preview-wrapper">
            <div class="preview-container" id="preview-container">
                <img id="preview-img" src="" alt="Figure Preview">
                <svg id="hover-overlay" class="hover-overlay"></svg>
            </div>
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
            <div class="section" id="section-labels">
                <div class="section-header section-toggle collapsed" onclick="toggleSection(this)">Labels</div>
                <div class="section-content collapsed">
                    <div class="field">
                        <label>Title</label>
                        <input type="text" id="title" placeholder="Figure title">
                    </div>
                    <div class="field-row">
                        <div class="field">
                            <label>X Label (Left)</label>
                            <input type="text" id="xlabel" placeholder="X axis label">
                        </div>
                        <div class="field">
                            <label>Y Label (Right)</label>
                            <input type="text" id="ylabel" placeholder="Y axis label">
                        </div>
                    </div>
                </div>
            </div>

            <!-- Axis Limits Section -->
            <div class="section" id="section-axis-limits">
                <div class="section-header section-toggle collapsed" onclick="toggleSection(this)">Axis Limits</div>
                <div class="section-content collapsed">
                    <div class="field-row">
                        <div class="field">
                            <label>X (Left)</label>
                            <div class="field-row" style="gap: 4px; margin-top: 4px;">
                                <input type="number" id="xmin" step="any" placeholder="Min">
                                <input type="number" id="xmax" step="any" placeholder="Max">
                            </div>
                        </div>
                        <div class="field">
                            <label>Y (Right)</label>
                            <div class="field-row" style="gap: 4px; margin-top: 4px;">
                                <input type="number" id="ymin" step="any" placeholder="Min">
                                <input type="number" id="ymax" step="any" placeholder="Max">
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Traces Section -->
            <div class="section" id="section-traces">
                <div class="section-header section-toggle collapsed" onclick="toggleSection(this)">Traces</div>
                <div class="section-content collapsed">
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
            <div class="section" id="section-legend">
                <div class="section-header section-toggle collapsed" onclick="toggleSection(this)">Legend</div>
                <div class="section-content collapsed">
                    <label class="checkbox-field">
                        <input type="checkbox" id="legend_visible" checked>
                        <span>Show Legend</span>
                    </label>
                    <div class="field">
                        <label>Position</label>
                        <select id="legend_loc" onchange="toggleCustomLegendPosition()">
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
                            <option value="custom">Custom (Drag)</option>
                        </select>
                    </div>
                    <div id="custom-legend-coords" class="field-row" style="display: none;">
                        <div class="field">
                            <label>X (0-1)</label>
                            <input type="number" id="legend_x" value="0.5" min="0" max="1" step="0.01">
                        </div>
                        <div class="field">
                            <label>Y (0-1)</label>
                            <input type="number" id="legend_y" value="0.5" min="0" max="1" step="0.01">
                        </div>
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
            <div class="section" id="section-ticks">
                <div class="section-header section-toggle collapsed" onclick="toggleSection(this)">Ticks</div>
                <div class="section-content collapsed">
                    <div class="field-row">
                        <div class="field">
                            <label>X (Left)</label>
                            <label class="checkbox-field" style="margin-top: 4px;">
                                <input type="checkbox" id="hide_x_ticks">
                                <span>Hide</span>
                            </label>
                            <input type="number" id="x_n_ticks" value="4" min="2" max="10" step="1" placeholder="N Ticks" title="Number of ticks">
                        </div>
                        <div class="field">
                            <label>Y (Right)</label>
                            <label class="checkbox-field" style="margin-top: 4px;">
                                <input type="checkbox" id="hide_y_ticks">
                                <span>Hide</span>
                            </label>
                            <input type="number" id="y_n_ticks" value="4" min="2" max="10" step="1" placeholder="N Ticks" title="Number of ticks">
                        </div>
                    </div>
                    <div class="field-row">
                        <div class="field">
                            <label>Font Size (pt)</label>
                            <input type="number" id="tick_fontsize" value="7" min="4" max="16" step="1">
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
                </div>
            </div>

            <!-- Style Section -->
            <div class="section" id="section-style">
                <div class="section-header section-toggle collapsed" onclick="toggleSection(this)">Style</div>
                <div class="section-content collapsed">
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
            <div class="section" id="section-dimensions">
                <div class="section-header section-toggle collapsed" onclick="toggleSection(this)">Dimensions</div>
                <div class="section-content collapsed">
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
                            <option value="2000">Normal (2s)</option>
                            <option value="5000" selected>Slow (5s)</option>
                        </select>
                    </div>
                    <button class="btn btn-cta" onclick="updatePreview()" style="flex: 0; margin-left: 8px;">Update Now</button>
                </div>
                <button class="btn btn-primary" onclick="saveManual()" title="Ctrl+S">Save to .manual.json</button>
                <button class="btn btn-secondary" onclick="resetOverrides()">Reset to Original</button>
            </div>

            <div class="status-bar" id="status">Ready</div>
        </div>
    </div>
</div>
"""


# EOF
