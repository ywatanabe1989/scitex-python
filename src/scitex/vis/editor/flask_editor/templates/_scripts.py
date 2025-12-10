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

// Schema v0.3 metadata for axes-local coordinate transforms
let schemaMeta = null;

// Cycle selection state for overlapping elements
let elementsAtCursor = [];  // All elements at current cursor position
let currentCycleIndex = 0;  // Current index in cycle

// Unit system state (default: mm)
let dimensionUnit = 'mm';
const MM_TO_INCH = 1 / 25.4;
const INCH_TO_MM = 25.4;

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

// Convert axes-local pixel coordinates to image coordinates
function axesLocalToImage(axLocalX, axLocalY, axesBbox) {
    // axesBbox has: x, y, width, height in figure pixel coordinates
    // The local editor uses tight layout which shifts coordinates
    // For now we use the existing image coordinates from bboxes
    return [axLocalX + axesBbox.x, axLocalY + axesBbox.y];
}

// Get geometry_px points converted to image coordinates
function getGeometryPoints(bbox) {
    const geom = bbox.geometry_px;
    if (!geom) return null;

    // For scatter: use points array directly
    if (geom.points && geom.points.length > 0) {
        return {
            type: 'scatter',
            points: geom.points,
            hitRadius: geom.hit_radius_px || 5
        };
    }

    // For lines: use path_simplified
    if (geom.path_simplified && geom.path_simplified.length > 0) {
        return {
            type: 'line',
            points: geom.path_simplified,
            linewidth: geom.linewidth_px || 1
        };
    }

    // For fills/polygons: use polygon
    if (geom.polygon && geom.polygon.length > 0) {
        return {
            type: 'polygon',
            points: geom.polygon
        };
    }

    return null;
}

function findElementAt(x, y) {
    // Multi-panel aware hit detection with specificity hierarchy:
    // 1. Data elements with legacy points - proximity detection (correct saved-image coords)
    // 2. Small elements (labels, ticks, legends, bars, fills)
    // 3. Panel bboxes - lowest priority (fallback)
    // Note: geometry_px (v0.3) uses axes-local coords which need coordinate transformation

    const PROXIMITY_THRESHOLD = 15;
    const SCATTER_THRESHOLD = 20;  // Larger threshold for scatter points

    // First: Check for data elements using legacy points (in saved-image coordinates)
    let closestDataElement = null;
    let minDistance = Infinity;

    for (const [name, bbox] of Object.entries(elementBboxes)) {
        if (name === '_meta') continue;  // Skip metadata entry

        // Prioritize legacy points array (already in correct saved-image coordinates)
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
    if (!Array.isArray(points) || points.length === 0) return Infinity;
    let minDist = Infinity;
    for (const pt of points) {
        if (!Array.isArray(pt) || pt.length < 2) continue;
        const [x, y] = pt;
        const dist = Math.sqrt((px - x) ** 2 + (py - y) ** 2);
        if (dist < minDist) minDist = dist;
    }
    return minDist;
}

function distanceToLine(px, py, points) {
    if (!Array.isArray(points) || points.length < 2) return Infinity;
    let minDist = Infinity;
    for (let i = 0; i < points.length - 1; i++) {
        const pt1 = points[i];
        const pt2 = points[i + 1];
        if (!Array.isArray(pt1) || pt1.length < 2) continue;
        if (!Array.isArray(pt2) || pt2.length < 2) continue;
        const [x1, y1] = pt1;
        const [x2, y2] = pt2;
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

// Point-in-polygon test using ray casting algorithm
function pointInPolygon(px, py, polygon) {
    if (!Array.isArray(polygon) || polygon.length < 3) return false;

    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const ptI = polygon[i];
        const ptJ = polygon[j];
        if (!Array.isArray(ptI) || ptI.length < 2) continue;
        if (!Array.isArray(ptJ) || ptJ.length < 2) continue;
        const [xi, yi] = ptI;
        const [xj, yj] = ptJ;

        if (((yi > py) !== (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
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
    if (!Array.isArray(bbox.points) || bbox.points.length < 2) return '';

    const points = bbox.points.filter(pt => Array.isArray(pt) && pt.length >= 2);
    if (points.length < 2) return '';

    let pathD = `M ${points[0][0] * scaleX} ${points[0][1] * scaleY}`;
    for (let i = 1; i < points.length; i++) {
        pathD += ` L ${points[i][0] * scaleX} ${points[i][1] * scaleY}`;
    }

    const className = type === 'hover' ? 'hover-path' : 'selected-path';
    const labelX = points[0][0] * scaleX;
    const labelY = points[0][1] * scaleY - 8;
    const labelClass = type === 'hover' ? 'hover-label' : 'selected-label';

    return `<path class="${className}" d="${pathD}"/>` +
           `<text class="${labelClass}" x="${labelX}" y="${labelY}">${bbox.label || ''}</text>`;
}

function drawScatterPoints(bbox, scaleX, scaleY, type) {
    // Draw scatter points as circles
    if (!Array.isArray(bbox.points) || bbox.points.length === 0) return '';

    const className = type === 'hover' ? 'hover-scatter' : 'selected-scatter';
    const labelClass = type === 'hover' ? 'hover-label' : 'selected-label';
    const radius = 4;

    let svg = '';
    for (const pt of bbox.points) {
        if (!Array.isArray(pt) || pt.length < 2) continue;
        const [x, y] = pt;
        svg += `<circle class="${className}" cx="${x * scaleX}" cy="${y * scaleY}" r="${radius}"/>`;
    }

    // Add label near first point
    const validPoints = bbox.points.filter(pt => Array.isArray(pt) && pt.length >= 2);
    if (validPoints.length > 0) {
        const labelX = validPoints[0][0] * scaleX;
        const labelY = validPoints[0][1] * scaleY - 10;
        svg += `<text class="${labelClass}" x="${labelX}" y="${labelY}">${bbox.label || ''}</text>`;
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

    // Always show selected element panel
    showSelectedElementPanel(elementName);
}

// Selected element panel management
function showSelectedElementPanel(elementName) {
    const section = document.getElementById('section-selected');
    const titleEl = document.getElementById('selected-element-title');
    const typeBadge = document.getElementById('element-type-badge');
    const axisInfo = document.getElementById('element-axis-info');

    // Hide all property sections first
    document.querySelectorAll('.element-props').forEach(el => el.style.display = 'none');

    if (!elementName) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';

    // Parse element name to extract type and info
    const elementInfo = parseElementName(elementName);
    const bbox = elementBboxes[elementName] || {};

    // Update title
    titleEl.textContent = `Selected: ${elementInfo.displayName}`;

    // Update type badge
    typeBadge.className = `element-type-badge ${elementInfo.type}`;
    typeBadge.textContent = elementInfo.type;

    // Update axis info
    if (elementInfo.axisId) {
        const row = elementInfo.axisId.match(/ax_(\\d)(\\d)/);
        if (row) {
            axisInfo.textContent = `Panel: Row ${parseInt(row[1])+1}, Col ${parseInt(row[2])+1}`;
        } else {
            axisInfo.textContent = `Axis: ${elementInfo.axisId}`;
        }
    } else {
        axisInfo.textContent = '';
    }

    // Show appropriate property panel and populate with current values
    showPropertiesForElement(elementInfo, bbox);
}

function parseElementName(name) {
    // Parse names like: ax_00_scatter_0, ax_11_trace_1, ax_01_xlabel, trace_0, xlabel, ax_00_xaxis, etc.
    const result = {
        original: name,
        type: 'unknown',
        displayName: name,
        axisId: null,
        index: null
    };

    // Check for axis prefix (ax_XX_)
    const axisMatch = name.match(/^(ax_\\d+)_(.+)$/);
    if (axisMatch) {
        result.axisId = axisMatch[1];
        name = axisMatch[2];  // Rest of the name
    }

    // Determine element type
    if (name.includes('scatter')) {
        result.type = 'scatter';
        const idx = name.match(/scatter_(\\d+)/);
        result.index = idx ? parseInt(idx[1]) : 0;
        result.displayName = `Scatter ${result.index + 1}`;
    } else if (name.includes('trace')) {
        result.type = 'trace';
        const idx = name.match(/trace_(\\d+)/);
        result.index = idx ? parseInt(idx[1]) : 0;
        result.displayName = `Line ${result.index + 1}`;
    } else if (name.includes('fill')) {
        result.type = 'fill';
        const idx = name.match(/fill_(\\d+)/);
        result.index = idx ? parseInt(idx[1]) : 0;
        result.displayName = `Fill Area ${result.index + 1}`;
    } else if (name.includes('bar')) {
        result.type = 'bar';
        const idx = name.match(/bar_(\\d+)/);
        result.index = idx ? parseInt(idx[1]) : 0;
        result.displayName = `Bar ${result.index + 1}`;
    } else if (name === 'xlabel' || name === 'ylabel' || name === 'title') {
        result.type = 'label';
        result.displayName = name.charAt(0).toUpperCase() + name.slice(1);
    } else if (name === 'legend') {
        result.type = 'legend';
        result.displayName = 'Legend';
    } else if (name === 'xaxis') {
        result.type = 'xaxis';
        result.displayName = 'X-Axis';
    } else if (name === 'yaxis') {
        result.type = 'yaxis';
        result.displayName = 'Y-Axis';
    } else if (name.includes('panel')) {
        result.type = 'panel';
        result.displayName = 'Panel';
    }

    return result;
}

function showPropertiesForElement(elementInfo, bbox) {
    const type = elementInfo.type;

    if (type === 'trace') {
        const props = document.getElementById('selected-trace-props');
        props.style.display = 'block';

        // Try to get current values from overrides
        const traceOverrides = getTraceOverrides(elementInfo);
        if (traceOverrides) {
            document.getElementById('sel-trace-label').value = traceOverrides.label || '';
            document.getElementById('sel-trace-color').value = traceOverrides.color || '#1f77b4';
            document.getElementById('sel-trace-color-text').value = traceOverrides.color || '#1f77b4';
            document.getElementById('sel-trace-linewidth').value = traceOverrides.linewidth || 1.0;
            document.getElementById('sel-trace-linestyle').value = traceOverrides.linestyle || '-';
            document.getElementById('sel-trace-marker').value = traceOverrides.marker || '';
            document.getElementById('sel-trace-markersize').value = traceOverrides.markersize || 4;
            document.getElementById('sel-trace-alpha').value = traceOverrides.alpha || 1;
        }
    } else if (type === 'scatter') {
        const props = document.getElementById('selected-scatter-props');
        props.style.display = 'block';

        const scatterOverrides = getScatterOverrides(elementInfo);
        if (scatterOverrides) {
            document.getElementById('sel-scatter-color').value = scatterOverrides.color || '#1f77b4';
            document.getElementById('sel-scatter-color-text').value = scatterOverrides.color || '#1f77b4';
            document.getElementById('sel-scatter-size').value = scatterOverrides.size || 20;
            document.getElementById('sel-scatter-marker').value = scatterOverrides.marker || 'o';
            document.getElementById('sel-scatter-alpha').value = scatterOverrides.alpha || 0.7;
            document.getElementById('sel-scatter-edgecolor').value = scatterOverrides.edgecolor || '#000000';
            document.getElementById('sel-scatter-edgecolor-text').value = scatterOverrides.edgecolor || '#000000';
        }
    } else if (type === 'fill') {
        const props = document.getElementById('selected-fill-props');
        props.style.display = 'block';

        const fillOverrides = getFillOverrides(elementInfo);
        if (fillOverrides) {
            document.getElementById('sel-fill-color').value = fillOverrides.color || '#1f77b4';
            document.getElementById('sel-fill-color-text').value = fillOverrides.color || '#1f77b4';
            document.getElementById('sel-fill-alpha').value = fillOverrides.alpha || 0.3;
        }
    } else if (type === 'bar') {
        const props = document.getElementById('selected-bar-props');
        props.style.display = 'block';
    } else if (type === 'label') {
        const props = document.getElementById('selected-label-props');
        props.style.display = 'block';

        // Get label text from global overrides
        const labelName = elementInfo.displayName.toLowerCase();
        document.getElementById('sel-label-text').value = overrides[labelName] || '';
        document.getElementById('sel-label-fontsize').value = overrides.axis_fontsize || 7;
    } else if (type === 'panel') {
        const props = document.getElementById('selected-panel-props');
        props.style.display = 'block';

        // Load existing panel overrides, fall back to actual bbox values
        const panelOverrides = getPanelOverrides(elementInfo);
        const panelBbox = elementBboxes[selectedElement] || {};
        document.getElementById('sel-panel-title').value = panelOverrides.title || panelBbox.title || '';
        document.getElementById('sel-panel-xlabel').value = panelOverrides.xlabel || panelBbox.xlabel || '';
        document.getElementById('sel-panel-ylabel').value = panelOverrides.ylabel || panelBbox.ylabel || '';
    } else if (type === 'legend') {
        // For legend, expand the legend section instead
        expandSection('section-legend');
    } else if (type === 'xaxis') {
        const props = document.getElementById('selected-xaxis-props');
        props.style.display = 'block';

        // Load existing xaxis overrides
        const xaxisOverrides = getAxisOverrides(elementInfo, 'xaxis');
        document.getElementById('sel-xaxis-fontsize').value = xaxisOverrides.tick_fontsize || overrides.tick_fontsize || 7;
        document.getElementById('sel-xaxis-label-fontsize').value = xaxisOverrides.label_fontsize || overrides.axis_fontsize || 7;
        document.getElementById('sel-xaxis-direction').value = xaxisOverrides.tick_direction || overrides.tick_direction || 'out';
        document.getElementById('sel-xaxis-nticks').value = xaxisOverrides.n_ticks || overrides.x_n_ticks || 4;
        document.getElementById('sel-xaxis-hide-ticks').checked = xaxisOverrides.hide_ticks || false;
        document.getElementById('sel-xaxis-hide-label').checked = xaxisOverrides.hide_label || false;
        document.getElementById('sel-xaxis-hide-spine').checked = xaxisOverrides.hide_spine || false;
    } else if (type === 'yaxis') {
        const props = document.getElementById('selected-yaxis-props');
        props.style.display = 'block';

        // Load existing yaxis overrides
        const yaxisOverrides = getAxisOverrides(elementInfo, 'yaxis');
        document.getElementById('sel-yaxis-fontsize').value = yaxisOverrides.tick_fontsize || overrides.tick_fontsize || 7;
        document.getElementById('sel-yaxis-label-fontsize').value = yaxisOverrides.label_fontsize || overrides.axis_fontsize || 7;
        document.getElementById('sel-yaxis-direction').value = yaxisOverrides.tick_direction || overrides.tick_direction || 'out';
        document.getElementById('sel-yaxis-nticks').value = yaxisOverrides.n_ticks || overrides.y_n_ticks || 4;
        document.getElementById('sel-yaxis-hide-ticks').checked = yaxisOverrides.hide_ticks || false;
        document.getElementById('sel-yaxis-hide-label').checked = yaxisOverrides.hide_label || false;
        document.getElementById('sel-yaxis-hide-spine').checked = yaxisOverrides.hide_spine || false;
    }

    // Show statistics for data elements (trace, scatter, fill, bar)
    if (['trace', 'scatter', 'fill', 'bar'].includes(type)) {
        showElementStatistics(bbox);
    } else {
        hideElementStatistics();
    }
}

function showElementStatistics(bbox) {
    const statsDiv = document.getElementById('selected-stats');
    if (!bbox || !bbox.points || bbox.points.length === 0) {
        statsDiv.style.display = 'none';
        return;
    }

    statsDiv.style.display = 'block';

    // Extract Y values from points (format: [[x,y], [x,y], ...])
    const yValues = bbox.points.map(pt => pt[1]).filter(v => isFinite(v));
    const xValues = bbox.points.map(pt => pt[0]).filter(v => isFinite(v));

    if (yValues.length === 0) {
        statsDiv.style.display = 'none';
        return;
    }

    // Calculate statistics
    const n = yValues.length;
    const mean = yValues.reduce((a, b) => a + b, 0) / n;
    const variance = yValues.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / n;
    const std = Math.sqrt(variance);
    const min = Math.min(...yValues);
    const max = Math.max(...yValues);
    const range = max - min;

    // Format numbers appropriately
    const fmt = (v) => {
        if (Math.abs(v) < 0.01 || Math.abs(v) >= 10000) {
            return v.toExponential(2);
        }
        return v.toFixed(2);
    };

    // Update display
    document.getElementById('stat-n').textContent = n;
    document.getElementById('stat-mean').textContent = fmt(mean);
    document.getElementById('stat-std').textContent = fmt(std);
    document.getElementById('stat-min').textContent = fmt(min);
    document.getElementById('stat-max').textContent = fmt(max);
    document.getElementById('stat-range').textContent = fmt(range);
}

function hideElementStatistics() {
    document.getElementById('selected-stats').style.display = 'none';
}

function getTraceOverrides(elementInfo) {
    // Initialize element overrides storage if not exists
    if (!overrides.element_overrides) {
        overrides.element_overrides = {};
    }

    const key = elementInfo.original;
    if (!overrides.element_overrides[key]) {
        // Try to get from traces array
        if (traces[elementInfo.index]) {
            overrides.element_overrides[key] = { ...traces[elementInfo.index] };
        } else {
            overrides.element_overrides[key] = {};
        }
    }
    return overrides.element_overrides[key];
}

function getScatterOverrides(elementInfo) {
    if (!overrides.element_overrides) {
        overrides.element_overrides = {};
    }
    const key = elementInfo.original;
    if (!overrides.element_overrides[key]) {
        overrides.element_overrides[key] = {};
    }
    return overrides.element_overrides[key];
}

function getFillOverrides(elementInfo) {
    if (!overrides.element_overrides) {
        overrides.element_overrides = {};
    }
    const key = elementInfo.original;
    if (!overrides.element_overrides[key]) {
        overrides.element_overrides[key] = {};
    }
    return overrides.element_overrides[key];
}

function getPanelOverrides(elementInfo) {
    if (!overrides.element_overrides) {
        overrides.element_overrides = {};
    }
    const key = elementInfo.original;
    if (!overrides.element_overrides[key]) {
        overrides.element_overrides[key] = {};
    }
    return overrides.element_overrides[key];
}

function getAxisOverrides(elementInfo, axisType) {
    // Get overrides for xaxis or yaxis element
    if (!overrides.element_overrides) {
        overrides.element_overrides = {};
    }
    const key = elementInfo.original;
    if (!overrides.element_overrides[key]) {
        overrides.element_overrides[key] = {};
    }
    return overrides.element_overrides[key];
}

function applySelectedElementChanges() {
    if (!selectedElement) return;

    const elementInfo = parseElementName(selectedElement);
    const type = elementInfo.type;

    if (!overrides.element_overrides) {
        overrides.element_overrides = {};
    }

    if (type === 'trace') {
        overrides.element_overrides[selectedElement] = {
            label: document.getElementById('sel-trace-label').value,
            color: document.getElementById('sel-trace-color').value,
            linewidth: parseFloat(document.getElementById('sel-trace-linewidth').value),
            linestyle: document.getElementById('sel-trace-linestyle').value,
            marker: document.getElementById('sel-trace-marker').value,
            markersize: parseFloat(document.getElementById('sel-trace-markersize').value),
            alpha: parseFloat(document.getElementById('sel-trace-alpha').value)
        };
    } else if (type === 'scatter') {
        overrides.element_overrides[selectedElement] = {
            color: document.getElementById('sel-scatter-color').value,
            size: parseFloat(document.getElementById('sel-scatter-size').value),
            marker: document.getElementById('sel-scatter-marker').value,
            alpha: parseFloat(document.getElementById('sel-scatter-alpha').value),
            edgecolor: document.getElementById('sel-scatter-edgecolor').value
        };
    } else if (type === 'fill') {
        overrides.element_overrides[selectedElement] = {
            color: document.getElementById('sel-fill-color').value,
            alpha: parseFloat(document.getElementById('sel-fill-alpha').value)
        };
    } else if (type === 'label') {
        const labelName = elementInfo.displayName.toLowerCase();
        overrides[labelName] = document.getElementById('sel-label-text').value;
        overrides.axis_fontsize = parseFloat(document.getElementById('sel-label-fontsize').value);
    } else if (type === 'bar') {
        overrides.element_overrides[selectedElement] = {
            facecolor: document.getElementById('sel-bar-facecolor').value,
            edgecolor: document.getElementById('sel-bar-edgecolor').value,
            alpha: parseFloat(document.getElementById('sel-bar-alpha').value)
        };
    } else if (type === 'panel') {
        // Panel-specific overrides (per-axis) including title, xlabel, ylabel
        overrides.element_overrides[selectedElement] = {
            title: document.getElementById('sel-panel-title').value,
            xlabel: document.getElementById('sel-panel-xlabel').value,
            ylabel: document.getElementById('sel-panel-ylabel').value,
            facecolor: document.getElementById('sel-panel-facecolor').value,
            transparent: document.getElementById('sel-panel-transparent').checked,
            grid: document.getElementById('sel-panel-grid').checked
        };
    } else if (type === 'xaxis') {
        // X-Axis specific overrides
        overrides.element_overrides[selectedElement] = {
            tick_fontsize: parseFloat(document.getElementById('sel-xaxis-fontsize').value),
            label_fontsize: parseFloat(document.getElementById('sel-xaxis-label-fontsize').value),
            tick_direction: document.getElementById('sel-xaxis-direction').value,
            n_ticks: parseInt(document.getElementById('sel-xaxis-nticks').value),
            hide_ticks: document.getElementById('sel-xaxis-hide-ticks').checked,
            hide_label: document.getElementById('sel-xaxis-hide-label').checked,
            hide_spine: document.getElementById('sel-xaxis-hide-spine').checked
        };
    } else if (type === 'yaxis') {
        // Y-Axis specific overrides
        overrides.element_overrides[selectedElement] = {
            tick_fontsize: parseFloat(document.getElementById('sel-yaxis-fontsize').value),
            label_fontsize: parseFloat(document.getElementById('sel-yaxis-label-fontsize').value),
            tick_direction: document.getElementById('sel-yaxis-direction').value,
            n_ticks: parseInt(document.getElementById('sel-yaxis-nticks').value),
            hide_ticks: document.getElementById('sel-yaxis-hide-ticks').checked,
            hide_label: document.getElementById('sel-yaxis-hide-label').checked,
            hide_spine: document.getElementById('sel-yaxis-hide-spine').checked
        };
    }

    // Trigger update
    updatePreview();
    document.getElementById('status').textContent = `Applied changes to ${elementInfo.displayName}`;
}

// Sync color inputs
function setupColorSync(colorId, textId) {
    const colorInput = document.getElementById(colorId);
    const textInput = document.getElementById(textId);
    if (colorInput && textInput) {
        colorInput.addEventListener('input', () => {
            textInput.value = colorInput.value;
        });
        textInput.addEventListener('input', () => {
            if (/^#[0-9A-Fa-f]{6}$/.test(textInput.value)) {
                colorInput.value = textInput.value;
            }
        });
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

// Dimension unit toggle
function setDimensionUnit(unit) {
    if (unit === dimensionUnit) return;

    const widthInput = document.getElementById('fig_width');
    const heightInput = document.getElementById('fig_height');
    const widthLabel = document.getElementById('fig_width_label');
    const heightLabel = document.getElementById('fig_height_label');
    const mmBtn = document.getElementById('unit-mm');
    const inchBtn = document.getElementById('unit-inch');

    // Get current values
    let width = parseFloat(widthInput.value) || 0;
    let height = parseFloat(heightInput.value) || 0;

    // Convert values
    if (unit === 'mm' && dimensionUnit === 'inch') {
        // inch to mm
        width = Math.round(width * INCH_TO_MM * 10) / 10;
        height = Math.round(height * INCH_TO_MM * 10) / 10;
        widthInput.min = 10;
        widthInput.max = 300;
        widthInput.step = 1;
        heightInput.min = 10;
        heightInput.max = 300;
        heightInput.step = 1;
    } else if (unit === 'inch' && dimensionUnit === 'mm') {
        // mm to inch
        width = Math.round(width * MM_TO_INCH * 100) / 100;
        height = Math.round(height * MM_TO_INCH * 100) / 100;
        widthInput.min = 0.5;
        widthInput.max = 12;
        widthInput.step = 0.05;
        heightInput.min = 0.5;
        heightInput.max = 12;
        heightInput.step = 0.05;
    }

    // Update values and labels
    widthInput.value = width;
    heightInput.value = height;
    widthLabel.textContent = `Width (${unit})`;
    heightLabel.textContent = `Height (${unit})`;

    // Update button states
    if (unit === 'mm') {
        mmBtn.classList.add('active');
        inchBtn.classList.remove('active');
    } else {
        mmBtn.classList.remove('active');
        inchBtn.classList.add('active');
    }

    dimensionUnit = unit;
}

// Background type management
let backgroundType = 'transparent';
let initializingBackground = true;  // Flag to prevent updates during init

function setBackgroundType(type) {
    backgroundType = type;

    // Update hidden inputs for collectOverrides
    const facecolorInput = document.getElementById('facecolor');
    const transparentInput = document.getElementById('transparent');

    if (type === 'white') {
        facecolorInput.value = '#ffffff';
        transparentInput.value = 'false';
    } else if (type === 'black') {
        facecolorInput.value = '#000000';
        transparentInput.value = 'false';
    } else {
        // transparent
        facecolorInput.value = '#ffffff';
        transparentInput.value = 'true';
    }

    // Update button states
    document.querySelectorAll('.bg-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`bg-${type}`).classList.add('active');

    // Trigger update only after initialization
    if (!initializingBackground) {
        scheduleUpdate();
    }
}

// Get figure dimensions in inches (for matplotlib)
function getFigSizeInches() {
    let width = parseFloat(document.getElementById('fig_width').value) || 80;
    let height = parseFloat(document.getElementById('fig_height').value) || 68;

    if (dimensionUnit === 'mm') {
        width = width * MM_TO_INCH;
        height = height * MM_TO_INCH;
    }

    return [Math.round(width * 100) / 100, Math.round(height * 100) / 100];
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
    // Initialize background type from overrides
    const isTransparent = overrides.transparent !== false;
    const facecolor = overrides.facecolor || '#ffffff';
    document.getElementById('facecolor').value = facecolor;

    if (isTransparent) {
        setBackgroundType('transparent');
    } else if (facecolor === '#000000') {
        setBackgroundType('black');
    } else {
        setBackgroundType('white');
    }

    // Dimensions (convert from inches in metadata to mm by default)
    if (overrides.fig_size) {
        // fig_size is in inches in the JSON - convert to mm for default display
        const widthMm = Math.round(overrides.fig_size[0] * INCH_TO_MM);
        const heightMm = Math.round(overrides.fig_size[1] * INCH_TO_MM);
        document.getElementById('fig_width').value = widthMm;
        document.getElementById('fig_height').value = heightMm;
    }
    document.getElementById('dpi').value = overrides.dpi || 300;
    // Default unit is mm, which is already set in HTML and JS state

    // Note: facecolor is now managed by background toggle buttons (white/transparent/black)
    // No text input sync needed

    updateAnnotationsList();
    updatePreview();
    initHoverSystem();
    setAutoUpdateInterval();

    // Setup color sync for selected element property inputs
    setupColorSync('sel-trace-color', 'sel-trace-color-text');
    setupColorSync('sel-scatter-color', 'sel-scatter-color-text');
    setupColorSync('sel-scatter-edgecolor', 'sel-scatter-edgecolor-text');
    setupColorSync('sel-fill-color', 'sel-fill-color-text');
    setupColorSync('sel-bar-facecolor', 'sel-bar-facecolor-text');
    setupColorSync('sel-bar-edgecolor', 'sel-bar-edgecolor-text');
    setupColorSync('sel-panel-facecolor', 'sel-panel-facecolor-text');

    // Mark initialization complete - now background changes will trigger updates
    initializingBackground = false;
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
    o.transparent = document.getElementById('transparent').value === 'true';

    // Dimensions (always in inches for matplotlib)
    o.fig_size = getFigSizeInches();
    o.dpi = parseInt(document.getElementById('dpi').value) || 300;

    // Annotations
    o.annotations = overrides.annotations || [];

    // Element-specific overrides (per-element styles)
    if (overrides.element_overrides) {
        o.element_overrides = overrides.element_overrides;
    }

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
            // Store schema v0.3 metadata if available
            if (data.bboxes._meta) {
                schemaMeta = data.bboxes._meta;
                console.log('Schema v0.3 geometry available:', schemaMeta.schema_version);
            }
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
