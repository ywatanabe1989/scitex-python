#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/templates/styles.py
"""CSS styles for the Flask editor UI.

DEPRECATED: This inline CSS module is kept for fallback compatibility only.
The CSS has been modularized into static/css/ directory:
- static/css/index.css (main entry point with @imports)
- static/css/base/ (variables, reset, typography)
- static/css/layout/ (container, preview, controls)
- static/css/components/ (buttons, forms, sections, dropdown, modal, context-menu)
- static/css/features/ (canvas, panel-grid, selection, overlay, loading, statistics)

To use static files (recommended):
    Set USE_STATIC_FILES = True in templates/__init__.py

To use this inline version (fallback):
    Set USE_STATIC_FILES = False in templates/__init__.py
"""

CSS_STYLES = """
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

/* Hover overlay for interactive selection */
.preview-container {
    position: relative;
    display: inline-block;
}

.hover-overlay {
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
}

/* Color-neutral hover/selection: white outline with glow works with any element color */
.hover-rect {
    fill: none;
    stroke: rgba(255, 255, 255, 0.9);
    stroke-width: 2;
    pointer-events: none;
    filter: drop-shadow(0 0 3px rgba(0, 0, 0, 0.5)) drop-shadow(0 0 1px rgba(255, 255, 255, 0.8));
}

.selected-rect {
    fill: none;
    stroke: rgba(255, 255, 255, 1);
    stroke-width: 2;
    stroke-dasharray: 4 2;
    pointer-events: none;
    filter: drop-shadow(0 0 4px rgba(0, 0, 0, 0.7)) drop-shadow(0 0 2px rgba(255, 255, 255, 1));
}

.hover-label {
    font-size: 10px;
    fill: rgba(255, 255, 255, 0.9);
    pointer-events: none;
    filter: drop-shadow(0 0 2px rgba(0, 0, 0, 0.8));
}

.selected-label {
    font-size: 10px;
    fill: rgba(255, 255, 255, 1);
    pointer-events: none;
    filter: drop-shadow(0 0 3px rgba(0, 0, 0, 0.9));
}

.hover-path {
    fill: none;
    stroke: rgba(255, 255, 255, 0.9);
    stroke-width: 6;
    stroke-linecap: round;
    stroke-linejoin: round;
    pointer-events: none;
    filter: drop-shadow(0 0 3px rgba(0, 0, 0, 0.6));
}

.selected-path {
    fill: none;
    stroke: rgba(255, 255, 255, 1);
    stroke-width: 6;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-dasharray: 8 4;
    pointer-events: none;
    filter: drop-shadow(0 0 4px rgba(0, 0, 0, 0.7));
}

/* Debug mode - show all hit areas */
.debug-rect {
    fill: rgba(255, 0, 0, 0.15);
    stroke: rgba(255, 0, 0, 0.6);
    stroke-width: 1;
    stroke-dasharray: 4 2;
    pointer-events: none;
}

.debug-rect-trace {
    fill: rgba(0, 255, 0, 0.15);
    stroke: rgba(0, 200, 0, 0.8);
    stroke-width: 1;
    stroke-dasharray: 3 2;
    pointer-events: none;
}

.debug-rect-legend {
    fill: rgba(255, 165, 0, 0.2);
    stroke: rgba(255, 140, 0, 0.8);
    stroke-width: 1;
    stroke-dasharray: 3 2;
    pointer-events: none;
}

.debug-label {
    font-size: 8px;
    fill: rgba(255, 0, 0, 0.9);
    pointer-events: none;
    font-family: monospace;
}

.debug-path {
    fill: none;
    stroke: rgba(0, 200, 0, 0.7);
    stroke-width: 2;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-dasharray: 5 3;
    pointer-events: none;
}

.debug-toggle {
    position: fixed;
    bottom: 10px;
    right: 10px;
    z-index: 1000;
    padding: 8px 12px;
    background: #333;
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
}

.debug-toggle:hover {
    background: #555;
}

.debug-toggle.active {
    background: #c00;
}

/* Hover scatter: White ring around points */
.hover-scatter {
    fill: none;
    stroke: rgba(255, 255, 255, 0.9);
    stroke-width: 3;
    pointer-events: none;
    filter: drop-shadow(0 0 2px rgba(0, 0, 0, 0.6));
}

/* Selected scatter: Dashed white ring */
.selected-scatter {
    fill: none;
    stroke: rgba(255, 255, 255, 1);
    stroke-width: 3;
    stroke-dasharray: 3 2;
    pointer-events: none;
    filter: drop-shadow(0 0 3px rgba(0, 0, 0, 0.7));
}

/* =============================================================================
 * Selected Element Panel
 * ============================================================================= */
#section-selected {
    border: 2px solid var(--accent-muted);
    background: var(--bg-secondary);
}

#section-selected .section-header {
    background: var(--accent-muted);
    color: var(--text-primary);
    font-weight: 600;
}

.selected-info {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    padding: 8px;
    background: var(--bg-primary);
    border-radius: 4px;
}

.element-type-badge {
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}

.element-type-badge.trace { background: #3498db; color: white; }
.element-type-badge.scatter { background: #e74c3c; color: white; }
.element-type-badge.fill { background: #9b59b6; color: white; }
.element-type-badge.bar { background: #f39c12; color: white; }
.element-type-badge.label { background: #2ecc71; color: white; }
.element-type-badge.panel { background: #34495e; color: white; }
.element-type-badge.legend { background: #1abc9c; color: white; }

.element-axis-info {
    font-size: 12px;
    color: var(--text-secondary);
}

.element-props {
    border-top: 1px solid var(--border-color);
    padding-top: 12px;
    margin-top: 8px;
}

/* Range slider styling */
input[type="range"] {
    width: 100%;
    height: 6px;
    border-radius: 3px;
    background: var(--bg-tertiary);
    outline: none;
    -webkit-appearance: none;
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
}

input[type="range"]::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    border: none;
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
    padding: 12px 16px;
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
    font-size: 1.0em;
    font-weight: 600;
    color: var(--status-success);
}

.controls-body {
    padding: 0 14px 14px;
    flex: 1;
}

.filename {
    font-size: 0.8em;
    color: var(--text-muted);
    margin-top: 4px;
    word-break: break-all;
}

.panel-path {
    font-size: 0.75em;
    color: var(--text-muted);
    margin-top: 2px;
    opacity: 0.8;
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
    margin-top: 10px;
}

.section-header {
    font-size: 0.72em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-inverse);
    background: var(--status-success);
    padding: 6px 10px;
    border-radius: 4px;
    margin-bottom: 8px;
}

/* =============================================================================
 * Form Fields
 * ============================================================================= */
.field { margin-bottom: 8px; }

.field label {
    display: block;
    font-size: 0.78em;
    font-weight: 500;
    margin-bottom: 3px;
    color: var(--text-secondary);
}

.field input[type="text"],
.field input[type="number"],
.field select {
    width: 100%;
    padding: 5px 8px;
    border: 1px solid var(--border-muted);
    border-radius: 4px;
    background: var(--bg-surface);
    color: var(--text-primary);
    font-size: 0.82em;
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
    gap: 8px;
}

.field-row .field { flex: 1; margin-bottom: 0; }

/* Checkbox styling */
.checkbox-field {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    padding: 4px 0;
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

.trace-item-highlight {
    background: var(--status-success);
    color: var(--text-inverse);
    animation: traceHighlightFade 1.5s ease-out forwards;
}

@keyframes traceHighlightFade {
    0% { background: var(--status-success); }
    100% { background: var(--bg-muted); }
}

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
    padding: 7px 12px;
    margin-top: 6px;
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
 * Unit Toggle
 * ============================================================================= */
.unit-toggle {
    display: flex;
    gap: 0;
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid var(--border-default);
}

.unit-btn {
    padding: 4px 12px;
    font-size: 0.8em;
    font-weight: 500;
    border: none;
    background: var(--bg-muted);
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
}

.unit-btn:first-child {
    border-right: 1px solid var(--border-default);
}

.unit-btn:hover {
    background: var(--bg-surface);
    color: var(--text-secondary);
}

.unit-btn.active {
    background: var(--color-cta);
    color: white;
}

/* =============================================================================
 * Background Type Toggle
 * ============================================================================= */
.bg-toggle {
    display: flex;
    gap: 6px;
}

.bg-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 6px 10px;
    font-size: 0.75em;
    border: 1px solid var(--border-default);
    border-radius: 4px;
    background: var(--bg-muted);
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
    flex: 1;
}

.bg-btn:hover {
    background: var(--bg-surface);
    color: var(--text-secondary);
}

.bg-btn.active {
    border-color: var(--color-cta);
    background: var(--bg-surface);
    color: var(--text-primary);
    box-shadow: 0 0 0 1px var(--color-cta);
}

.bg-preview {
    width: 20px;
    height: 14px;
    border-radius: 2px;
    border: 1px solid var(--border-default);
}

.bg-preview.white {
    background: #ffffff;
}

.bg-preview.transparent {
    background: linear-gradient(45deg, #ccc 25%, transparent 25%),
                linear-gradient(-45deg, #ccc 25%, transparent 25%),
                linear-gradient(45deg, transparent 75%, #ccc 75%),
                linear-gradient(-45deg, transparent 75%, #ccc 75%);
    background-size: 8px 8px;
    background-position: 0 0, 0 4px, 4px -4px, -4px 0px;
    background-color: #fff;
}

.bg-preview.black {
    background: #000000;
}

/* =============================================================================
 * Section Hints
 * ============================================================================= */
.section-hint {
    font-size: 0.75em;
    color: var(--text-muted);
    font-style: italic;
    margin-bottom: 10px;
    padding: 6px 8px;
    background: var(--bg-muted);
    border-radius: 4px;
    border-left: 2px solid var(--border-default);
}

/* =============================================================================
 * Element Statistics
 * ============================================================================= */
.element-stats {
    margin-top: 12px;
    padding: 10px;
    background: var(--bg-muted);
    border-radius: 4px;
    border: 1px solid var(--border-muted);
}

.stats-header {
    font-size: 0.8em;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}

.stat-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.stat-label {
    font-size: 0.7em;
    color: var(--text-muted);
    text-transform: uppercase;
}

.stat-value {
    font-size: 0.85em;
    font-weight: 500;
    color: var(--text-primary);
    font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
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

/* =============================================================================
 * Statistics Section
 * ============================================================================= */
#stats-container {
    font-family: 'JetBrains Mono', 'Fira Code', 'Monaco', monospace;
    font-size: 11px;
    line-height: 1.5;
}

.stats-loading {
    color: var(--text-muted);
    font-style: italic;
    padding: 10px;
    text-align: center;
}

.stats-empty {
    color: var(--text-muted);
    padding: 10px;
    text-align: center;
    background: var(--bg-muted);
    border-radius: 4px;
}

.stats-card {
    background: var(--bg-surface);
    border: 1px solid var(--workspace-border-subtle);
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 10px;
}

.stats-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--workspace-border-subtle);
}

.stats-card-title {
    font-weight: 600;
    font-size: 12px;
    color: var(--text-primary);
}

.stats-significance {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 11px;
}

.stats-significance.sig-high {
    background: #dcfce7;
    color: #166534;
}

.stats-significance.sig-medium {
    background: #fef9c3;
    color: #854d0e;
}

.stats-significance.sig-low {
    background: #fee2e2;
    color: #991b1b;
}

.stats-significance.sig-ns {
    background: var(--bg-muted);
    color: var(--text-muted);
}

.stats-row {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
}

.stats-label {
    color: var(--text-secondary);
}

.stats-value {
    font-weight: 500;
    color: var(--text-primary);
}

.stats-groups {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px dashed var(--workspace-border-subtle);
}

.stats-group {
    background: var(--bg-muted);
    padding: 8px;
    border-radius: 4px;
    font-size: 10px;
}

.stats-group-name {
    font-weight: 600;
    margin-bottom: 4px;
    color: var(--text-primary);
}

.stats-correction-badge {
    display: inline-block;
    background: var(--color-cta);
    color: white;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 9px;
    margin-left: 6px;
}

.stats-summary-header {
    background: var(--scitex-01);
    color: var(--text-inverse);
    padding: 10px 12px;
    border-radius: 6px 6px 0 0;
    font-weight: 600;
    margin-bottom: 0;
}

.stats-summary-body {
    background: var(--bg-surface);
    border: 1px solid var(--scitex-01);
    border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 12px;
}

/* =============================================================================
 * Subsection Headers (for grouped controls like X/Y axis settings)
 * ============================================================================= */
.subsection-header {
    font-size: 0.68em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    color: var(--text-secondary);
    padding: 4px 6px;
    margin-bottom: 6px;
    margin-top: 8px;
    background: var(--bg-muted);
    border-left: 2px solid var(--status-success);
    border-radius: 0 4px 4px 0;
}

.subsection-header:first-child {
    margin-top: 0;
}

/* =============================================================================
 * Axis Tabs (for X/Y/Z axis switching)
 * ============================================================================= */
.axis-tabs {
    display: flex;
    gap: 3px;
    margin-bottom: 8px;
    background: var(--bg-muted);
    padding: 3px;
    border-radius: 6px;
}

.axis-tab {
    flex: 1;
    padding: 4px 8px;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    font-size: 0.8em;
    font-weight: 600;
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.15s ease;
}

.axis-tab:hover {
    background: var(--bg-surface);
    color: var(--text-primary);
}

.axis-tab.active {
    background: var(--scitex-01);
    color: var(--bg-primary);
}

.axis-panel {
    animation: fadeIn 0.15s ease;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* =============================================================================
 * Loading Spinner Overlay
 * ============================================================================= */
.loading-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
    border-radius: 8px;
}

/* Global loading overlay - non-intrusive top bar style */
.global-loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 36px;
    background: var(--workspace-bg-elevated);
    border-bottom: 1px solid var(--workspace-border-default);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    gap: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.global-loading-overlay .loading-text {
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 500;
}

.spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--scitex-06);
    border-top: 2px solid var(--status-success);
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
}

.spinner-small {
    width: 14px;
    height: 14px;
    border-width: 2px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* =============================================================================
 * Panel Grid View (for multi-panel figure bundles)
 * ============================================================================= */
.panel-grid-section {
    width: 100%;
    margin-bottom: 20px;
    background: var(--workspace-bg-elevated);
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.panel-grid-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.panel-grid-header h3 {
    font-size: 1em;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
}

.panel-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
}

.panel-card {
    background: var(--preview-bg);
    background-size: 16px 16px;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0px;
    border-radius: 6px;
    overflow: hidden;
    cursor: pointer;
    transition: all 0.2s ease;
    border: 2px solid transparent;
    position: relative;
}

.panel-card:hover {
    border-color: var(--color-cta);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.panel-card.active {
    border-color: var(--status-success);
    box-shadow: 0 0 0 2px var(--status-success);
}

.panel-card img {
    width: 100%;
    height: auto;
    display: block;
}

/* Dark mode: invert plot colors for visibility */
[data-theme="dark"] .panel-card img,
[data-theme="dark"] .panel-canvas-item img,
[data-theme="dark"] .preview-wrapper img {
    filter: invert(0.88) hue-rotate(180deg);
}

.panel-card-container {
    position: relative;
    display: inline-block;
    width: 100%;
}

.panel-card-overlay {
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
    z-index: 5;
}

.panel-card-label {
    position: absolute;
    top: 6px;
    left: 6px;
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.75em;
    font-weight: 600;
}

.panel-card-loading {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 120px;
    color: var(--text-muted);
    font-size: 0.85em;
}

/* Panel Navigation Header */
.preview-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: var(--workspace-bg-elevated);
    border-radius: 8px 8px 0 0;
    border-bottom: 1px solid var(--workspace-border-subtle);
}

#current-panel-name {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.9em;
}

.panel-nav {
    display: flex;
    align-items: center;
    gap: 8px;
}

#panel-indicator {
    font-size: 0.8em;
    color: var(--text-muted);
    min-width: 50px;
    text-align: center;
}

.btn-sm {
    padding: 4px 10px;
    font-size: 0.8em;
    width: auto;
    margin: 0;
}

/* Preview area adjustments when panel header is visible */
.preview-wrapper:has(.preview-header:not([style*="display: none"])) {
    border-radius: 0 0 8px 8px;
}

.preview {
    flex-direction: column;
    align-items: stretch;
    padding: 20px;
    overflow-y: auto;
}

.preview > .preview-wrapper {
    flex-shrink: 0;
}

/* =============================================================================
 * Unified Panel Canvas (matches figure export layout)
 * Exactly matches SciTeX Cloud's .plot-preview-area styling
 * ============================================================================= */
.panel-canvas {
    position: relative;
    min-height: 400px;
    /* White background with grid pattern - exact match to SciTeX Cloud */
    background: #ffffff;
    background-image:
        linear-gradient(0deg, transparent 23px, #e5e5e5 23px),
        linear-gradient(90deg, transparent 23px, #e5e5e5 23px);
    background-size: 24px 24px;
    border-radius: 8px;
    overflow: visible;  /* Allow panels to extend slightly if needed */
}

/* Dark mode: dark canvas background for eye comfort */
[data-theme="dark"] .panel-canvas,
.dark-mode .panel-canvas {
    background: #1a1a1a;
    background-image:
        linear-gradient(0deg, transparent 23px, #2a2a2a 23px),
        linear-gradient(90deg, transparent 23px, #2a2a2a 23px);
    background-size: 24px 24px;
}

/* Hide grid when toggled off (G key) */
.panel-canvas.hide-grid {
    background-image: none !important;
}

/* Unified canvas - panels appear as single figure matching figure export */
.panel-canvas-item {
    position: absolute;
    user-select: none;
    background: transparent;
    overflow: visible;
    border: none;
    cursor: grab;  /* Indicate draggability */
}

.panel-canvas-item:hover {
    z-index: 10;
}

.panel-canvas-item:active {
    cursor: grabbing;
}

.panel-canvas-item.active {
    outline: 2px solid var(--status-success);
    outline-offset: 2px;
}

.panel-canvas-item .panel-card-container {
    width: 100%;
    height: 100%;
    position: relative;
}

.panel-canvas-item img {
    width: 100%;
    height: 100%;
    object-fit: contain;  /* Preserve aspect ratio - never distort figures */
    pointer-events: none;
    display: block;
}

.panel-canvas-label {
    position: absolute;
    top: 4px;
    left: 4px;
    background: rgba(0,0,0,0.6);
    color: white;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.65em;
    font-weight: 600;
    cursor: move;
    opacity: 1;  /* Show by default */
    transition: opacity 0.2s, background 0.2s;
    z-index: 5;
}

.panel-canvas-item:hover .panel-canvas-label {
    opacity: 1;
    background: rgba(0,0,0,0.8);  /* Slightly darker on hover */
}

/* Drag handle for panel repositioning */
.panel-drag-handle {
    position: absolute;
    top: 4px;
    right: 4px;
    background: rgba(0,0,0,0.5);
    color: white;
    padding: 2px 4px;
    border-radius: 3px;
    font-size: 0.7em;
    cursor: move;
    opacity: 0;
    transition: opacity 0.2s, background 0.2s;
    z-index: 5;
    user-select: none;
}

.panel-canvas-item:hover .panel-drag-handle {
    opacity: 1;
}

.panel-drag-handle:hover {
    background: rgba(0,0,0,0.8);
}

.panel-drag-handle:active {
    background: var(--accent-primary);
}

/* Position indicator while dragging */
.panel-position-indicator {
    position: absolute;
    bottom: 4px;
    left: 4px;
    background: rgba(0,0,0,0.7);
    color: #4fc3f7;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.6em;
    font-family: monospace;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.3s;
    z-index: 5;
}

/* Dragging state */
.panel-canvas-item.dragging {
    opacity: 0.85;
    z-index: 100;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    outline: 2px dashed var(--accent-primary);
    outline-offset: 2px;
    cursor: grabbing;
}

.panel-canvas-item.dragging .panel-position-indicator {
    opacity: 1;
}

.canvas-controls {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
    align-items: center;
}

.canvas-controls button {
    padding: 4px 12px;
    font-size: 0.8em;
}

/* Toolbar separator */
.toolbar-separator {
    width: 1px;
    height: 24px;
    background: #555;
    margin: 0 4px;
}

/* Download dropdown in toolbar */
.download-dropdown {
    position: relative;
    display: inline-block;
}

.download-dropdown #download-btn {
    background: #4a90d9;
    border: none;
    color: white;
    padding: 4px 12px;
    font-size: 0.8em;
    border-radius: 3px;
    cursor: pointer;
}

.download-dropdown #download-btn:hover {
    background: #5a9fe9;
}

#download-menu {
    display: none;
    position: absolute;
    top: 100%;
    left: 0;
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 4px;
    min-width: 160px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    z-index: 1000;
    margin-top: 4px;
    padding: 4px 0;
}

.download-item {
    display: block;
    padding: 8px 12px;
    color: #ddd;
    text-decoration: none;
    font-size: 0.85em;
    transition: background 0.15s;
    white-space: nowrap;
}

.download-item:hover {
    background: #3a3a3a;
    color: #fff;
}

.download-divider {
    height: 1px;
    background: #444;
    margin: 4px 0;
}

/* =============================================================================
   Context Menu (Right-Click)
   ============================================================================= */
.context-menu {
    position: fixed;
    background: var(--bg-secondary, #2a2a2a);
    border: 1px solid var(--border-color, #444);
    border-radius: 6px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    min-width: 200px;
    z-index: 10000;
    padding: 4px 0;
    font-size: 13px;
}

.context-menu-item {
    display: flex;
    align-items: center;
    padding: 8px 12px;
    cursor: pointer;
    color: var(--text-primary, #ddd);
    transition: background 0.1s;
    position: relative;
}

.context-menu-item:hover:not(.disabled) {
    background: var(--accent-primary, #4a90d9);
    color: #fff;
}

.context-menu-item.disabled {
    color: var(--text-muted, #666);
    cursor: not-allowed;
}

.context-menu-icon {
    width: 20px;
    margin-right: 8px;
    text-align: center;
    font-size: 14px;
}

.context-menu-shortcut {
    margin-left: auto;
    font-size: 11px;
    color: var(--text-muted, #888);
    padding-left: 16px;
}

.context-menu-item:hover:not(.disabled) .context-menu-shortcut {
    color: rgba(255, 255, 255, 0.7);
}

.context-menu-arrow {
    margin-left: auto;
    font-size: 10px;
    color: var(--text-muted, #888);
}

.context-menu-divider {
    height: 1px;
    background: var(--border-color, #444);
    margin: 4px 0;
}

/* Submenu */
.context-menu-submenu {
    position: relative;
}

.context-submenu {
    display: none;
    position: absolute;
    left: 100%;
    top: 0;
    background: var(--bg-secondary, #2a2a2a);
    border: 1px solid var(--border-color, #444);
    border-radius: 6px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    min-width: 150px;
    padding: 4px 0;
}

.context-menu-submenu:hover > .context-submenu {
    display: block;
}

.context-submenu .context-menu-item {
    padding: 6px 12px;
}

/* Panel selection styling */
.panel-canvas-item.active {
    outline: 3px solid var(--accent-primary, #4a90d9);
    outline-offset: 2px;
    z-index: 100;
}

.panel-canvas-item:hover:not(.active) {
    outline: 2px dashed var(--accent-secondary, #6ab04c);
    outline-offset: 2px;
}
"""


# EOF
