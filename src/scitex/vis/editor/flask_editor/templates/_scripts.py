#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/templates/scripts.py
"""JavaScript for the Flask editor UI."""

JS_SCRIPTS = """
let overrides = {{ overrides|safe }};
let traces = overrides.traces || [];
let elementBboxes = {};
let imgSize = {width: 0, height: 0};
let hoveredElement = null;
let selectedElement = null;

// Cycle selection state for overlapping elements
let elementsAtCursor = [];  // All elements at current cursor position
let currentCycleIndex = 0;  // Current index in cycle

// Hover system - client-side hit testing
function initHoverSystem() {
    const container = document.getElementById('preview-container');
    const img = document.getElementById('preview-img');

    img.addEventListener('mousemove', (e) => {
        if (imgSize.width === 0 || imgSize.height === 0) return;

        const rect = img.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const scaleX = imgSize.width / rect.width;
        const scaleY = imgSize.height / rect.height;
        const imgX = x * scaleX;
        const imgY = y * scaleY;

        const element = findElementAt(imgX, imgY);
        if (element !== hoveredElement) {
            hoveredElement = element;
            updateOverlay();
        }
    });

    img.addEventListener('mouseleave', () => {
        hoveredElement = null;
        updateOverlay();
    });

    img.addEventListener('click', (e) => {
        const rect = img.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const scaleX = imgSize.width / rect.width;
        const scaleY = imgSize.height / rect.height;
        const imgX = x * scaleX;
        const imgY = y * scaleY;

        // Alt+click or find all overlapping elements
        if (e.altKey) {
            // Cycle through overlapping elements
            const allElements = findAllElementsAt(imgX, imgY);
            if (allElements.length > 0) {
                // If cursor moved to different location, reset cycle
                if (JSON.stringify(allElements) !== JSON.stringify(elementsAtCursor)) {
                    elementsAtCursor = allElements;
                    currentCycleIndex = 0;
                } else {
                    // Cycle to next element
                    currentCycleIndex = (currentCycleIndex + 1) % elementsAtCursor.length;
                }
                selectedElement = elementsAtCursor[currentCycleIndex];
                updateOverlay();
                scrollToSection(selectedElement);

                // Show cycle indicator in status
                const total = elementsAtCursor.length;
                const current = currentCycleIndex + 1;
                console.log(`Cycle selection: ${current}/${total} - ${selectedElement}`);
            }
        } else if (hoveredElement) {
            // Normal click - select hovered element
            selectedElement = hoveredElement;
            elementsAtCursor = [];  // Reset cycle
            currentCycleIndex = 0;
            updateOverlay();
            scrollToSection(selectedElement);
        }
    });

    // Right-click for cycle selection menu
    img.addEventListener('contextmenu', (e) => {
        e.preventDefault();

        const rect = img.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const scaleX = imgSize.width / rect.width;
        const scaleY = imgSize.height / rect.height;
        const imgX = x * scaleX;
        const imgY = y * scaleY;

        const allElements = findAllElementsAt(imgX, imgY);
        if (allElements.length > 1) {
            // Cycle to next element
            if (JSON.stringify(allElements) !== JSON.stringify(elementsAtCursor)) {
                elementsAtCursor = allElements;
                currentCycleIndex = 0;
            } else {
                currentCycleIndex = (currentCycleIndex + 1) % elementsAtCursor.length;
            }
            selectedElement = elementsAtCursor[currentCycleIndex];
            updateOverlay();
            scrollToSection(selectedElement);

            const total = elementsAtCursor.length;
            const current = currentCycleIndex + 1;
            console.log(`Right-click cycle: ${current}/${total} - ${selectedElement}`);
        } else if (allElements.length === 1) {
            selectedElement = allElements[0];
            updateOverlay();
            scrollToSection(selectedElement);
        }
    });

    img.addEventListener('load', () => {
        updateOverlay();
    });
}

function findElementAt(x, y) {
    // Multi-panel aware hit detection with specificity hierarchy:
    // 1. Data elements with points (lines, scatter) - proximity detection
    // 2. Small elements (labels, ticks, legends, bars, fills)
    // 3. Panel bboxes - lowest priority (fallback)

    const PROXIMITY_THRESHOLD = 15;
    const SCATTER_THRESHOLD = 20;  // Larger threshold for scatter points

    // First: Check for data elements with points (lines, scatter)
    let closestDataElement = null;
    let minDistance = Infinity;

    for (const [name, bbox] of Object.entries(elementBboxes)) {
        if (bbox.points && bbox.points.length > 0) {
            // Check if cursor is within general bbox area first
            if (x >= bbox.x0 - SCATTER_THRESHOLD && x <= bbox.x1 + SCATTER_THRESHOLD &&
                y >= bbox.y0 - SCATTER_THRESHOLD && y <= bbox.y1 + SCATTER_THRESHOLD) {

                const elementType = bbox.element_type || 'line';
                let dist;

                if (elementType === 'scatter') {
                    // For scatter, find distance to nearest point
                    dist = distanceToNearestPoint(x, y, bbox.points);
                } else {
                    // For lines, find distance to line segments
                    dist = distanceToLine(x, y, bbox.points);
                }

                if (dist < minDistance) {
                    minDistance = dist;
                    closestDataElement = name;
                }
            }
        }
    }

    // Use appropriate threshold based on element type
    if (closestDataElement) {
        const bbox = elementBboxes[closestDataElement];
        const threshold = (bbox.element_type === 'scatter') ? SCATTER_THRESHOLD : PROXIMITY_THRESHOLD;
        if (minDistance <= threshold) {
            return closestDataElement;
        }
    }

    // Second: Collect all bbox matches, excluding panels and data elements with points
    const elementMatches = [];
    const panelMatches = [];

    for (const [name, bbox] of Object.entries(elementBboxes)) {
        if (x >= bbox.x0 && x <= bbox.x1 && y >= bbox.y0 && y <= bbox.y1) {
            const area = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0);
            const isPanel = bbox.is_panel || name.endsWith('_panel');
            const hasPoints = bbox.points && bbox.points.length > 0;

            if (hasPoints) {
                // Already handled above with proximity
                continue;
            } else if (isPanel) {
                panelMatches.push({name, area, bbox});
            } else {
                elementMatches.push({name, area, bbox});
            }
        }
    }

    // Return smallest non-panel element if any
    if (elementMatches.length > 0) {
        elementMatches.sort((a, b) => a.area - b.area);
        return elementMatches[0].name;
    }

    // Fallback to panel selection (useful for multi-panel figures)
    if (panelMatches.length > 0) {
        panelMatches.sort((a, b) => a.area - b.area);
        return panelMatches[0].name;
    }

    return null;
}

function distanceToNearestPoint(px, py, points) {
    // Find distance to nearest point in scatter
    let minDist = Infinity;
    for (const [x, y] of points) {
        const dist = Math.sqrt((px - x) ** 2 + (py - y) ** 2);
        if (dist < minDist) minDist = dist;
    }
    return minDist;
}

function distanceToLine(px, py, points) {
    let minDist = Infinity;
    for (let i = 0; i < points.length - 1; i++) {
        const [x1, y1] = points[i];
        const [x2, y2] = points[i + 1];
        const dist = distanceToSegment(px, py, x1, y1, x2, y2);
        if (dist < minDist) minDist = dist;
    }
    return minDist;
}

function distanceToSegment(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lenSq = dx * dx + dy * dy;

    if (lenSq === 0) {
        return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2);
    }

    let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));

    const projX = x1 + t * dx;
    const projY = y1 + t * dy;

    return Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);
}

function findAllElementsAt(x, y) {
    // Find all elements at cursor position (for cycle selection)
    // Returns array sorted by specificity (most specific first)
    const PROXIMITY_THRESHOLD = 15;
    const SCATTER_THRESHOLD = 20;

    const results = [];

    for (const [name, bbox] of Object.entries(elementBboxes)) {
        let match = false;
        let distance = Infinity;
        let priority = 0;  // Lower = more specific

        const hasPoints = bbox.points && bbox.points.length > 0;
        const elementType = bbox.element_type || '';
        const isPanel = bbox.is_panel || name.endsWith('_panel');

        // Check data elements with points (lines, scatter)
        if (hasPoints) {
            if (x >= bbox.x0 - SCATTER_THRESHOLD && x <= bbox.x1 + SCATTER_THRESHOLD &&
                y >= bbox.y0 - SCATTER_THRESHOLD && y <= bbox.y1 + SCATTER_THRESHOLD) {

                if (elementType === 'scatter') {
                    distance = distanceToNearestPoint(x, y, bbox.points);
                    if (distance <= SCATTER_THRESHOLD) {
                        match = true;
                        priority = 1;  // Scatter points = high priority
                    }
                } else {
                    distance = distanceToLine(x, y, bbox.points);
                    if (distance <= PROXIMITY_THRESHOLD) {
                        match = true;
                        priority = 2;  // Lines = high priority
                    }
                }
            }
        }

        // Check bbox containment
        if (x >= bbox.x0 && x <= bbox.x1 && y >= bbox.y0 && y <= bbox.y1) {
            const area = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0);

            if (!match) {
                match = true;
                distance = 0;
            }

            if (isPanel) {
                priority = 100;  // Panels = lowest priority
            } else if (!hasPoints) {
                // Small elements like labels, ticks - use area for priority
                priority = 10 + Math.min(area / 10000, 50);
            }
        }

        if (match) {
            results.push({ name, distance, priority, bbox });
        }
    }

    // Sort by priority (lower first), then by distance
    results.sort((a, b) => {
        if (a.priority !== b.priority) return a.priority - b.priority;
        return a.distance - b.distance;
    });

    return results.map(r => r.name);
}

function drawTracePath(bbox, scaleX, scaleY, type) {
    if (!bbox.points || bbox.points.length < 2) return '';

    const points = bbox.points;
    let pathD = `M ${points[0][0] * scaleX} ${points[0][1] * scaleY}`;
    for (let i = 1; i < points.length; i++) {
        pathD += ` L ${points[i][0] * scaleX} ${points[i][1] * scaleY}`;
    }

    const className = type === 'hover' ? 'hover-path' : 'selected-path';
    const labelX = points[0][0] * scaleX;
    const labelY = points[0][1] * scaleY - 8;
    const labelClass = type === 'hover' ? 'hover-label' : 'selected-label';

    return `<path class="${className}" d="${pathD}"/>` +
           `<text class="${labelClass}" x="${labelX}" y="${labelY}">${bbox.label}</text>`;
}

function drawScatterPoints(bbox, scaleX, scaleY, type) {
    // Draw scatter points as circles
    if (!bbox.points || bbox.points.length === 0) return '';

    const className = type === 'hover' ? 'hover-scatter' : 'selected-scatter';
    const labelClass = type === 'hover' ? 'hover-label' : 'selected-label';
    const radius = 4;

    let svg = '';
    for (const [x, y] of bbox.points) {
        svg += `<circle class="${className}" cx="${x * scaleX}" cy="${y * scaleY}" r="${radius}"/>`;
    }

    // Add label near first point
    if (bbox.points.length > 0) {
        const labelX = bbox.points[0][0] * scaleX;
        const labelY = bbox.points[0][1] * scaleY - 10;
        svg += `<text class="${labelClass}" x="${labelX}" y="${labelY}">${bbox.label}</text>`;
    }

    return svg;
}

function updateOverlay() {
    const overlay = document.getElementById('hover-overlay');
    const img = document.getElementById('preview-img');
    const rect = img.getBoundingClientRect();

    overlay.setAttribute('width', rect.width);
    overlay.setAttribute('height', rect.height);

    const scaleX = rect.width / imgSize.width;
    const scaleY = rect.height / imgSize.height;

    let svg = '';

    function drawElement(elementName, type) {
        const bbox = elementBboxes[elementName];
        if (!bbox) return '';

        const elementType = bbox.element_type || '';
        const hasPoints = bbox.points && bbox.points.length > 0;

        // Lines - draw as path
        if ((elementType === 'line' || elementName.includes('trace_')) && hasPoints) {
            return drawTracePath(bbox, scaleX, scaleY, type);
        }
        // Scatter - draw as circles
        else if (elementType === 'scatter' && hasPoints) {
            return drawScatterPoints(bbox, scaleX, scaleY, type);
        }
        // Default - draw bbox rectangle
        else {
            const rectClass = type === 'hover' ? 'hover-rect' : 'selected-rect';
            const labelClass = type === 'hover' ? 'hover-label' : 'selected-label';
            const x = bbox.x0 * scaleX - 2;
            const y = bbox.y0 * scaleY - 2;
            const w = (bbox.x1 - bbox.x0) * scaleX + 4;
            const h = (bbox.y1 - bbox.y0) * scaleY + 4;
            return `<rect class="${rectClass}" x="${x}" y="${y}" width="${w}" height="${h}" rx="2"/>` +
                   `<text class="${labelClass}" x="${x}" y="${y - 4}">${bbox.label}</text>`;
        }
    }

    if (hoveredElement && hoveredElement !== selectedElement) {
        svg += drawElement(hoveredElement, 'hover');
    }

    if (selectedElement) {
        svg += drawElement(selectedElement, 'selected');
    }

    overlay.innerHTML = svg;
}

function expandSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
        const header = section.querySelector('.section-header');
        const content = section.querySelector('.section-content');
        if (section.id === sectionId) {
            header?.classList.remove('collapsed');
            content?.classList.remove('collapsed');
        } else if (header?.classList.contains('section-toggle')) {
            header?.classList.add('collapsed');
            content?.classList.add('collapsed');
        }
    });
}

function scrollToSection(elementName) {
    const elementToSection = {
        'title': 'section-labels',
        'xlabel': 'section-labels',
        'ylabel': 'section-labels',
        'xaxis_ticks': 'section-ticks',
        'yaxis_ticks': 'section-ticks',
        'legend': 'section-legend'
    };

    const fieldMap = {
        'title': 'title',
        'xlabel': 'xlabel',
        'ylabel': 'ylabel',
        'xaxis_ticks': 'x_n_ticks',
        'yaxis_ticks': 'y_n_ticks',
        'legend': 'legend_visible'
    };

    if (elementName.startsWith('trace_')) {
        expandSection('section-traces');
        const traceIdx = elementBboxes[elementName]?.trace_idx;
        if (traceIdx !== undefined) {
            const traceColors = document.querySelectorAll('.trace-color');
            if (traceColors[traceIdx]) {
                setTimeout(() => {
                    traceColors[traceIdx].scrollIntoView({behavior: 'smooth', block: 'center'});
                    traceColors[traceIdx].click();
                }, 100);
            }
        }
        return;
    }

    const sectionId = elementToSection[elementName];
    if (sectionId) {
        expandSection(sectionId);
    }

    const fieldId = fieldMap[elementName];
    if (fieldId) {
        const field = document.getElementById(fieldId);
        if (field) {
            setTimeout(() => {
                field.scrollIntoView({behavior: 'smooth', block: 'center'});
                field.focus();
            }, 100);
        }
    }
}

// Theme management
function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    document.getElementById('theme-icon').innerHTML = next === 'dark' ? '&#9790;' : '&#9788;';
    localStorage.setItem('scitex-editor-theme', next);
}

// Collapsible sections
function toggleSection(header) {
    header.classList.toggle('collapsed');
    const content = header.nextElementSibling;
    content.classList.toggle('collapsed');
}

function toggleCustomLegendPosition() {
    const legendLoc = document.getElementById('legend_loc').value;
    const customCoordsDiv = document.getElementById('custom-legend-coords');
    customCoordsDiv.style.display = legendLoc === 'custom' ? 'flex' : 'none';
}

// Initialize fields
document.addEventListener('DOMContentLoaded', () => {
    // Load saved theme
    const savedTheme = localStorage.getItem('scitex-editor-theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
        document.getElementById('theme-icon').innerHTML = savedTheme === 'dark' ? '&#9790;' : '&#9788;';
    }

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
    document.getElementById('legend_x').value = overrides.legend_x !== undefined ? overrides.legend_x : 0.5;
    document.getElementById('legend_y').value = overrides.legend_y !== undefined ? overrides.legend_y : 0.5;
    toggleCustomLegendPosition();

    // Ticks
    document.getElementById('x_n_ticks').value = overrides.x_n_ticks || overrides.n_ticks || 4;
    document.getElementById('y_n_ticks').value = overrides.y_n_ticks || overrides.n_ticks || 4;
    document.getElementById('hide_x_ticks').checked = overrides.hide_x_ticks || false;
    document.getElementById('hide_y_ticks').checked = overrides.hide_y_ticks || false;
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
        document.getElementById('fig_width').value = Math.round(overrides.fig_size[0] * 100) / 100;
        document.getElementById('fig_height').value = Math.round(overrides.fig_size[1] * 100) / 100;
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
    initHoverSystem();
    setAutoUpdateInterval();
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
        scheduleUpdate();
    }
}

function updateTraceStyle(idx, style) {
    if (traces[idx]) {
        traces[idx].linestyle = style;
        scheduleUpdate();
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
    o.legend_x = parseFloat(document.getElementById('legend_x').value) || 0.5;
    o.legend_y = parseFloat(document.getElementById('legend_y').value) || 0.5;

    // Ticks
    o.x_n_ticks = parseInt(document.getElementById('x_n_ticks').value) || 4;
    o.y_n_ticks = parseInt(document.getElementById('y_n_ticks').value) || 4;
    o.hide_x_ticks = document.getElementById('hide_x_ticks').checked;
    o.hide_y_ticks = document.getElementById('hide_y_ticks').checked;
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

        if (data.bboxes) {
            elementBboxes = data.bboxes;
        }
        if (data.img_size) {
            imgSize = data.img_size;
        }

        selectedElement = null;
        hoveredElement = null;
        updateOverlay();

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

// Debounced auto-update
let updateTimer = null;
const DEBOUNCE_DELAY = 500;

function scheduleUpdate() {
    if (updateTimer) clearTimeout(updateTimer);
    updateTimer = setTimeout(() => {
        updatePreview();
    }, DEBOUNCE_DELAY);
}

// Auto-update on input changes
document.querySelectorAll('input[type="text"], input[type="number"]').forEach(el => {
    el.addEventListener('input', scheduleUpdate);
    el.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            if (updateTimer) clearTimeout(updateTimer);
            updatePreview();
        }
    });
});

document.querySelectorAll('input[type="checkbox"], select').forEach(el => {
    el.addEventListener('change', () => {
        if (updateTimer) clearTimeout(updateTimer);
        updatePreview();
    });
});

document.querySelectorAll('input[type="color"]').forEach(el => {
    el.addEventListener('change', () => {
        if (updateTimer) clearTimeout(updateTimer);
        updatePreview();
    });
});

// Ctrl+S keyboard shortcut to save
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        saveManual();
    }
});

// Auto-update interval system
let autoUpdateIntervalId = null;

function setAutoUpdateInterval() {
    if (autoUpdateIntervalId) {
        clearInterval(autoUpdateIntervalId);
        autoUpdateIntervalId = null;
    }

    const intervalMs = parseInt(document.getElementById('auto_update_interval').value);
    if (intervalMs > 0) {
        autoUpdateIntervalId = setInterval(() => {
            updatePreview();
        }, intervalMs);
    }
}
"""


# EOF
