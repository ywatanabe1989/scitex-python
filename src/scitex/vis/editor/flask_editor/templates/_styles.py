#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/templates/styles.py
"""CSS styles for the Flask editor UI."""

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

.hover-rect {
    fill: none;
    stroke: rgba(100, 180, 255, 0.6);
    stroke-width: 1;
    pointer-events: none;
}

.selected-rect {
    fill: none;
    stroke: rgba(255, 200, 80, 0.8);
    stroke-width: 2;
    pointer-events: none;
}

.hover-label {
    font-size: 10px;
    fill: rgba(100, 180, 255, 0.9);
    pointer-events: none;
}

.selected-label {
    font-size: 10px;
    fill: rgba(255, 200, 80, 0.9);
    pointer-events: none;
}

.hover-path {
    fill: none;
    stroke: rgba(100, 180, 255, 0.9);
    stroke-width: 4;
    stroke-linecap: round;
    stroke-linejoin: round;
    pointer-events: none;
}

.selected-path {
    fill: none;
    stroke: rgba(255, 200, 80, 0.9);
    stroke-width: 5;
    stroke-linecap: round;
    stroke-linejoin: round;
    pointer-events: none;
}

.hover-scatter {
    fill: rgba(100, 180, 255, 0.7);
    stroke: rgba(100, 180, 255, 0.9);
    stroke-width: 1;
    pointer-events: none;
}

.selected-scatter {
    fill: rgba(255, 200, 80, 0.7);
    stroke: rgba(255, 200, 80, 0.9);
    stroke-width: 2;
    pointer-events: none;
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
"""


# EOF
