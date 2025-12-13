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
let debugMode = false;  // Debug mode to show all hit areas
let isShowingOriginalPreview = false;  // True when showing existing SVG/PNG from bundle
let originalBboxes = null;  // Store original bboxes from /preview
let originalImgSize = null;  // Store original img size from /preview

// Schema v0.3 metadata for axes-local coordinate transforms
let schemaMeta = null;

// Multi-panel state
let panelData = null;  // Panel info from /preview
let currentPanelIndex = 0;
let showingPanelGrid = false;
let panelBboxesCache = {};  // Cache bboxes per panel {panelName: {bboxes, imgSize}}
let activePanelCard = null;  // Currently active panel card for hover/click
let panelHoveredElement = null;  // Hovered element in panel grid
let panelDebugMode = false;  // Show hit regions in panel grid

// Cycle selection state for overlapping elements
let elementsAtCursor = [];  // All elements at current cursor position
let currentCycleIndex = 0;  // Current index in cycle

// Unit system state (default: mm)
let dimensionUnit = 'mm';
const MM_TO_INCH = 1 / 25.4;
const INCH_TO_MM = 25.4;

// Dark mode detection
function isDarkMode() {
    return document.documentElement.getAttribute('data-theme') === 'dark';
}

// Hitmap-based element detection
let hitmapCanvas = null;
let hitmapCtx = null;
let hitmapColorMap = {};  // Maps RGB string -> element info
let hitmapLoaded = false;
let hitmapImgSize = {width: 0, height: 0};

// Load hitmap for pixel-based element detection
async function loadHitmap() {
    try {
        const resp = await fetch('/hitmap');
        const data = await resp.json();

        if (data.error) {
            console.log('Hitmap not available:', data.error);
            return;
        }

        // Create hidden canvas for hitmap sampling
        hitmapCanvas = document.createElement('canvas');
        hitmapCtx = hitmapCanvas.getContext('2d', { willReadFrequently: true });

        // Load hitmap image
        const img = new Image();
        img.onload = () => {
            hitmapCanvas.width = img.width;
            hitmapCanvas.height = img.height;
            hitmapImgSize = {width: img.width, height: img.height};
            hitmapCtx.drawImage(img, 0, 0);
            hitmapLoaded = true;
            console.log('Hitmap loaded:', img.width, 'x', img.height);
        };
        img.src = 'data:image/png;base64,' + data.image;

        // Build color map from response
        if (data.color_map) {
            for (const [key, info] of Object.entries(data.color_map)) {
                if (info.rgb) {
                    const rgbKey = `${info.rgb[0]},${info.rgb[1]},${info.rgb[2]}`;
                    hitmapColorMap[rgbKey] = {
                        id: info.id,
                        type: info.type,
                        label: info.label,
                        axes_index: info.axes_index,
                    };
                }
            }
            console.log('Hitmap color map:', Object.keys(hitmapColorMap).length, 'elements');
        }
    } catch (e) {
        console.error('Error loading hitmap:', e);
    }
}

// Get element at position using hitmap
function getElementFromHitmap(imgX, imgY) {
    if (!hitmapLoaded || !hitmapCtx) return null;

    // Scale coordinates from display image to hitmap
    const scaleX = hitmapImgSize.width / imgSize.width;
    const scaleY = hitmapImgSize.height / imgSize.height;
    const hitmapX = Math.floor(imgX * scaleX);
    const hitmapY = Math.floor(imgY * scaleY);

    // Bounds check
    if (hitmapX < 0 || hitmapX >= hitmapImgSize.width ||
        hitmapY < 0 || hitmapY >= hitmapImgSize.height) {
        return null;
    }

    // Sample pixel from hitmap
    const pixel = hitmapCtx.getImageData(hitmapX, hitmapY, 1, 1).data;
    const r = pixel[0], g = pixel[1], b = pixel[2], a = pixel[3];

    // Skip transparent or white pixels
    if (a < 128 || (r > 250 && g > 250 && b > 250)) {
        return null;
    }

    // Look up in color map
    const rgbKey = `${r},${g},${b}`;
    const element = hitmapColorMap[rgbKey];

    if (element) {
        return `trace_${element.id}`;  // Return element key for selection
    }

    return null;
}

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
    // Find the visible preview element (SVG wrapper or img)
    const svgWrapper = document.getElementById('preview-svg-wrapper');
    const imgEl = document.getElementById('preview-img');

    let targetEl = null;
    if (svgWrapper) {
        targetEl = svgWrapper.querySelector('svg') || svgWrapper;
    } else if (imgEl && imgEl.offsetWidth > 0) {
        targetEl = imgEl;
    }

    if (!targetEl) {
        console.log('updateOverlay: No visible target element');
        return;
    }

    const rect = targetEl.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
        console.log('updateOverlay: Target element has zero dimensions');
        return;
    }

    // Guard against zero imgSize (can cause Infinity scale)
    if (!imgSize || !imgSize.width || !imgSize.height || imgSize.width === 0 || imgSize.height === 0) {
        console.log('updateOverlay: imgSize not set or zero', imgSize);
        return;
    }

    overlay.setAttribute('width', rect.width);
    overlay.setAttribute('height', rect.height);
    overlay.style.width = rect.width + 'px';
    overlay.style.height = rect.height + 'px';

    // Position overlay over the target element
    const containerRect = document.getElementById('preview-container').getBoundingClientRect();
    overlay.style.left = (rect.left - containerRect.left) + 'px';
    overlay.style.top = (rect.top - containerRect.top) + 'px';

    const scaleX = rect.width / imgSize.width;
    const scaleY = rect.height / imgSize.height;

    console.log('updateOverlay: rect=', rect.width, 'x', rect.height, 'imgSize=', imgSize, 'scale=', scaleX, scaleY);

    let svg = '';

    // Debug mode: draw ALL bboxes
    if (debugMode) {
        svg += drawDebugBboxes(scaleX, scaleY);
    }

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
                   `<text class="${labelClass}" x="${x}" y="${y - 4}">${bbox.label || elementName}</text>`;
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

// Draw all bboxes for debugging
function drawDebugBboxes(scaleX, scaleY) {
    let svg = '';
    let count = 0;

    console.log('=== DEBUG BBOXES ===');
    console.log('imgSize:', imgSize);
    console.log('scale:', scaleX, scaleY);

    for (const [name, bbox] of Object.entries(elementBboxes)) {
        if (name === '_meta') continue;

        count++;
        const hasPoints = bbox.points && bbox.points.length > 0;
        const elementType = bbox.element_type || '';

        // Choose color based on element type
        let rectClass = 'debug-rect';
        if (name.includes('trace_') || elementType === 'line') {
            rectClass = 'debug-rect-trace';
        } else if (name.includes('legend')) {
            rectClass = 'debug-rect-legend';
        }

        // Draw bbox rectangle
        const x = bbox.x0 * scaleX;
        const y = bbox.y0 * scaleY;
        const w = (bbox.x1 - bbox.x0) * scaleX;
        const h = (bbox.y1 - bbox.y0) * scaleY;

        svg += `<rect class="${rectClass}" x="${x}" y="${y}" width="${w}" height="${h}"/>`;
        svg += `<text class="debug-label" x="${x + 2}" y="${y + 10}">${name}</text>`;

        // Draw path points if available
        if (hasPoints && bbox.points.length > 1) {
            let pathD = `M ${bbox.points[0][0] * scaleX} ${bbox.points[0][1] * scaleY}`;
            for (let i = 1; i < bbox.points.length; i++) {
                const pt = bbox.points[i];
                if (pt && pt.length >= 2) {
                    pathD += ` L ${pt[0] * scaleX} ${pt[1] * scaleY}`;
                }
            }
            svg += `<path class="debug-path" d="${pathD}"/>`;
        }

        console.log(`  ${name}: (${bbox.x0?.toFixed(1)}, ${bbox.y0?.toFixed(1)}) - (${bbox.x1?.toFixed(1)}, ${bbox.y1?.toFixed(1)}), points: ${bbox.points?.length || 0}`);
    }

    console.log(`Total elements: ${count}`);
    return svg;
}

// Toggle debug mode
function toggleDebugMode() {
    debugMode = !debugMode;
    const btn = document.getElementById('debug-toggle-btn');
    if (btn) {
        btn.classList.toggle('active', debugMode);
        btn.textContent = debugMode ? 'Hide Hit Areas' : 'Show Hit Areas';
    }
    updateOverlay();
    console.log('Debug mode:', debugMode ? 'ON' : 'OFF');
}

function expandSection(sectionId) {
    console.log('expandSection called with:', sectionId);
    let foundSection = null;
    document.querySelectorAll('.section').forEach(section => {
        const header = section.querySelector('.section-header');
        const content = section.querySelector('.section-content');
        if (section.id === sectionId) {
            foundSection = section;
            console.log('expandSection: Found section', sectionId, 'header:', header, 'content:', content);
            header?.classList.remove('collapsed');
            content?.classList.remove('collapsed');
        } else if (header?.classList.contains('section-toggle')) {
            header?.classList.add('collapsed');
            content?.classList.add('collapsed');
        }
    });
    if (!foundSection) {
        console.warn('expandSection: Section not found:', sectionId);
    } else {
        // Scroll the section into view
        setTimeout(() => {
            foundSection.scrollIntoView({behavior: 'smooth', block: 'start'});
        }, 50);
    }
}

function scrollToSection(elementName) {
    console.log('scrollToSection called with:', elementName);

    // Map element names to their corresponding sections
    // Elements with a mapping here will NOT show the "Selected" panel
    const elementToSection = {
        'title': 'section-labels',
        'xlabel': 'section-labels',
        'ylabel': 'section-labels',
        'caption': 'section-labels',
        'xaxis': 'section-ticks',
        'yaxis': 'section-ticks',
        'xaxis_ticks': 'section-ticks',
        'yaxis_ticks': 'section-ticks',
        'xaxis_spine': 'section-ticks',
        'yaxis_spine': 'section-ticks',
        'legend': 'section-legend'
    };

    const fieldMap = {
        'title': 'title',
        'xlabel': 'xlabel',
        'ylabel': 'ylabel',
        'caption': 'caption',
        'xaxis': 'xmin',
        'yaxis': 'ymin',
        'xaxis_ticks': 'x_tick_fontsize',
        'yaxis_ticks': 'y_tick_fontsize',
        'xaxis_spine': 'axis_width',
        'yaxis_spine': 'axis_width',
        'legend': 'legend_visible'
    };

    if (elementName.startsWith('trace_')) {
        expandSection('section-traces');
        const traceIdx = elementBboxes[elementName]?.trace_idx;
        if (traceIdx !== undefined) {
            const traceItems = document.querySelectorAll('.trace-item');
            if (traceItems[traceIdx]) {
                setTimeout(() => {
                    // Scroll into view and highlight the trace item
                    traceItems[traceIdx].scrollIntoView({behavior: 'smooth', block: 'center'});
                    // Add temporary highlight effect
                    traceItems[traceIdx].classList.add('trace-item-highlight');
                    setTimeout(() => {
                        traceItems[traceIdx].classList.remove('trace-item-highlight');
                    }, 1500);
                }, 100);
            }
        }
        return;
    }

    // Extract base element name from prefixed names like "ax_00_yaxis_spine" or "ax0_title"
    let baseElementName = elementName;
    const match = elementName.match(/ax_?\d+_(.+)/);
    if (match) {
        baseElementName = match[1];
    }
    console.log('scrollToSection: baseElementName=', baseElementName, 'from', elementName);

    const sectionId = elementToSection[baseElementName];
    const fieldId = fieldMap[baseElementName];
    console.log('scrollToSection: sectionId=', sectionId, 'fieldId=', fieldId);

    if (sectionId) {
        console.log('scrollToSection: expanding section', sectionId);
        // Element has a corresponding section - expand it, don't show "Selected" panel
        expandSection(sectionId);

        if (fieldId) {
            const field = document.getElementById(fieldId);
            if (field) {
                setTimeout(() => {
                    field.scrollIntoView({behavior: 'smooth', block: 'center'});
                    field.focus();
                }, 100);
            }
        }

        // Hide the "Selected" panel since we're using existing section
        const selectedSection = document.getElementById('section-selected');
        if (selectedSection) {
            selectedSection.style.display = 'none';
        }
    } else {
        // No corresponding section - show the "Selected" panel for this element
        showSelectedElementPanel(elementName);
    }
}

// Field to element synchronization - highlight element when field is focused
function setupFieldToElementSync() {
    // Map field IDs to element names
    const fieldToElement = {
        // Title, Labels & Caption section
        'title': 'title',
        'title_fontsize': 'title',
        'show_title': 'title',
        'xlabel': 'xlabel',
        'ylabel': 'ylabel',
        'caption': 'caption',
        'caption_fontsize': 'caption',
        'show_caption': 'caption',
        // Axis & Ticks section
        'xmin': 'xaxis',
        'xmax': 'xaxis',
        'ymin': 'yaxis',
        'ymax': 'yaxis',
        'x_n_ticks': 'xaxis_ticks',
        'hide_x_ticks': 'xaxis_ticks',
        'x_tick_fontsize': 'xaxis_ticks',
        'x_tick_direction': 'xaxis_ticks',
        'x_tick_length': 'xaxis_ticks',
        'x_tick_width': 'xaxis_ticks',
        'y_n_ticks': 'yaxis_ticks',
        'hide_y_ticks': 'yaxis_ticks',
        'y_tick_fontsize': 'yaxis_ticks',
        'y_tick_direction': 'yaxis_ticks',
        'y_tick_length': 'yaxis_ticks',
        'y_tick_width': 'yaxis_ticks',
        // Legend section
        'legend_visible': 'legend',
        'legend_loc': 'legend',
        'legend_frameon': 'legend',
        'legend_fontsize': 'legend',
        'legend_ncols': 'legend',
        'legend_x': 'legend',
        'legend_y': 'legend'
    };

    // Add focus listeners to all mapped fields
    Object.entries(fieldToElement).forEach(([fieldId, elementName]) => {
        const field = document.getElementById(fieldId);
        if (field) {
            field.addEventListener('focus', () => {
                // Find the element in bboxes - for multi-panel, check ax_00 first
                let targetElement = null;
                if (elementBboxes[elementName]) {
                    targetElement = elementName;
                } else {
                    // Try to find with axis prefix (e.g., ax_00_title)
                    for (const key of Object.keys(elementBboxes)) {
                        if (key.endsWith('_' + elementName) || key === elementName) {
                            targetElement = key;
                            break;
                        }
                    }
                }

                if (targetElement) {
                    selectedElement = targetElement;
                    updateOverlay();
                    setStatus(`Highlighting: ${targetElement}`, false);
                }
            });

            // Also handle mouseenter for hover feedback
            field.addEventListener('mouseenter', () => {
                let targetElement = null;
                if (elementBboxes[elementName]) {
                    targetElement = elementName;
                } else {
                    for (const key of Object.keys(elementBboxes)) {
                        if (key.endsWith('_' + elementName) || key === elementName) {
                            targetElement = key;
                            break;
                        }
                    }
                }

                if (targetElement && targetElement !== selectedElement) {
                    hoveredElement = targetElement;
                    updateOverlay();
                }
            });

            field.addEventListener('mouseleave', () => {
                if (hoveredElement && hoveredElement !== selectedElement) {
                    hoveredElement = null;
                    updateOverlay();
                }
            });
        }
    });
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

        // Get values from: 1) element_overrides, 2) traces array (pltz metadata), 3) bbox data
        const traceOverrides = getTraceOverrides(elementInfo);
        const traceIdx = elementInfo.index || 0;
        const traceFromMeta = traces[traceIdx] || {};

        // Label: prefer user override, then pltz metadata, then bbox label
        const label = traceOverrides.label || traceFromMeta.label || bbox.label?.replace(/.*:\s*/, '') || '';
        const color = traceOverrides.color || traceFromMeta.color || '#1f77b4';
        const linewidth = traceOverrides.linewidth || traceFromMeta.linewidth || 1.0;
        const linestyle = traceOverrides.linestyle || traceFromMeta.linestyle || '-';
        const marker = traceOverrides.marker || traceFromMeta.marker || '';
        const markersize = traceOverrides.markersize || traceFromMeta.markersize || 4;
        const alpha = traceOverrides.alpha || traceFromMeta.alpha || 1;

        document.getElementById('sel-trace-label').value = label;
        document.getElementById('sel-trace-color').value = color;
        document.getElementById('sel-trace-color-text').value = color;
        document.getElementById('sel-trace-linewidth').value = linewidth;
        document.getElementById('sel-trace-linestyle').value = linestyle;
        document.getElementById('sel-trace-marker').value = marker;
        document.getElementById('sel-trace-markersize').value = markersize;
        document.getElementById('sel-trace-alpha').value = alpha;
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

function switchAxisTab(axis) {
    // Update tab buttons
    document.querySelectorAll('.axis-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.getElementById('axis-tab-' + axis).classList.add('active');

    // Update panels
    document.querySelectorAll('.axis-panel').forEach(panel => {
        panel.style.display = 'none';
    });
    document.getElementById('axis-panel-' + axis).style.display = 'block';
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

    // Labels - Title
    if (overrides.title) document.getElementById('title').value = overrides.title;
    document.getElementById('show_title').checked = overrides.show_title !== false;
    document.getElementById('title_fontsize').value = overrides.title_fontsize || 8;
    // Labels - Caption
    if (overrides.caption) document.getElementById('caption').value = overrides.caption;
    document.getElementById('show_caption').checked = overrides.show_caption || false;
    document.getElementById('caption_fontsize').value = overrides.caption_fontsize || 7;
    // Labels - Axis
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
    updateTracesList();

    // Legend
    document.getElementById('legend_visible').checked = overrides.legend_visible !== false;
    document.getElementById('legend_loc').value = overrides.legend_loc || 'best';
    document.getElementById('legend_frameon').checked = overrides.legend_frameon || false;
    document.getElementById('legend_fontsize').value = overrides.legend_fontsize || 6;
    document.getElementById('legend_ncols').value = overrides.legend_ncols || 1;
    document.getElementById('legend_x').value = overrides.legend_x !== undefined ? overrides.legend_x : 0.95;
    document.getElementById('legend_y').value = overrides.legend_y !== undefined ? overrides.legend_y : 0.95;
    toggleCustomLegendPosition();

    // Axis and Ticks - X Axis (Bottom)
    document.getElementById('x_n_ticks').value = overrides.x_n_ticks || overrides.n_ticks || 4;
    document.getElementById('hide_x_ticks').checked = overrides.hide_x_ticks || false;
    document.getElementById('x_tick_fontsize').value = overrides.x_tick_fontsize || overrides.tick_fontsize || 7;
    document.getElementById('x_tick_direction').value = overrides.x_tick_direction || overrides.tick_direction || 'out';
    document.getElementById('x_tick_length').value = overrides.x_tick_length || overrides.tick_length || 0.8;
    document.getElementById('x_tick_width').value = overrides.x_tick_width || overrides.tick_width || 0.2;
    // X Axis (Top)
    document.getElementById('show_x_top').checked = overrides.show_x_top || false;
    document.getElementById('x_top_mirror').checked = overrides.x_top_mirror || false;
    // Y Axis (Left)
    document.getElementById('y_n_ticks').value = overrides.y_n_ticks || overrides.n_ticks || 4;
    document.getElementById('hide_y_ticks').checked = overrides.hide_y_ticks || false;
    document.getElementById('y_tick_fontsize').value = overrides.y_tick_fontsize || overrides.tick_fontsize || 7;
    document.getElementById('y_tick_direction').value = overrides.y_tick_direction || overrides.tick_direction || 'out';
    document.getElementById('y_tick_length').value = overrides.y_tick_length || overrides.tick_length || 0.8;
    document.getElementById('y_tick_width').value = overrides.y_tick_width || overrides.tick_width || 0.2;
    // Y Axis (Right)
    document.getElementById('show_y_right').checked = overrides.show_y_right || false;
    document.getElementById('y_right_mirror').checked = overrides.y_right_mirror || false;
    // Spines
    document.getElementById('hide_bottom_spine').checked = overrides.hide_bottom_spine || false;
    document.getElementById('hide_left_spine').checked = overrides.hide_left_spine || false;
    // Z Axis (3D)
    document.getElementById('hide_z_ticks').checked = overrides.hide_z_ticks || false;
    document.getElementById('z_n_ticks').value = overrides.z_n_ticks || 4;
    document.getElementById('z_tick_fontsize').value = overrides.z_tick_fontsize || 7;
    document.getElementById('z_tick_direction').value = overrides.z_tick_direction || 'out';

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
    // NOTE: Don't call updatePreview() here - we want to use existing PNG/SVG
    // loadInitialPreview() will load the original file and then start auto-update
    initHoverSystem();
    refreshStats();  // Load statistical test results

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

    // Setup field-to-element synchronization (highlight element when field is focused)
    setupFieldToElementSync();

    // Load initial preview from existing PNG/SVG (no re-render)
    loadInitialPreview();

    // Add resize handler to update overlay when window/image size changes
    window.addEventListener('resize', () => {
        updateOverlay();
    });

    // Use ResizeObserver to detect when the preview container changes size
    const previewContainer = document.getElementById('preview-container');
    if (previewContainer && typeof ResizeObserver !== 'undefined') {
        const resizeObserver = new ResizeObserver(() => {
            updateOverlay();
        });
        resizeObserver.observe(previewContainer);
    }
});

// =============================================================================
// Loading Helpers
// =============================================================================
function showLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.style.display = 'flex';
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.style.display = 'none';
}

// Update form controls from overrides (used when switching panels)
function updateControlsFromOverrides() {
    console.log('updateControlsFromOverrides called');
    console.log('overrides.traces:', overrides.traces);
    console.log('traces variable:', traces);

    // Update title - try both id and name selectors
    const titleInput = document.getElementById('title') || document.querySelector('input[name="title"]');
    if (titleInput) {
        // Try overrides.title first, then axes[0].title
        let title = overrides.title;
        if (title === undefined && overrides.axes && overrides.axes[0]) {
            title = overrides.axes[0].title || '';
        }
        titleInput.value = title || '';
    }

    // Update figure size
    const widthInput = document.getElementById('fig-width');
    const heightInput = document.getElementById('fig-height');
    if (widthInput && overrides.figure_width !== undefined) {
        widthInput.value = overrides.figure_width;
    }
    if (heightInput && overrides.figure_height !== undefined) {
        heightInput.value = overrides.figure_height;
    }

    // Update xlabel, ylabel - try overrides first, then axes[0]
    const xlabelInput = document.getElementById('xlabel');
    const ylabelInput = document.getElementById('ylabel');
    let xlabel = overrides.xlabel;
    let ylabel = overrides.ylabel;
    if (overrides.axes && overrides.axes[0]) {
        xlabel = xlabel || overrides.axes[0].xlabel || '';
        ylabel = ylabel || overrides.axes[0].ylabel || '';
    }
    if (xlabelInput) xlabelInput.value = xlabel || '';
    if (ylabelInput) ylabelInput.value = ylabel || '';

    // Update traces list
    updateTracesList();
}

// Load preview from existing file without re-rendering
async function loadInitialPreview() {
    setStatus('Loading preview...', false);
    try {
        const darkMode = isDarkMode();
        const resp = await fetch(`/preview?dark_mode=${darkMode}`);
        const data = await resp.json();

        console.log('=== PREVIEW DATA RECEIVED ===');
        console.log('format:', data.format);
        console.log('img_size:', data.img_size);
        console.log('bboxes keys:', Object.keys(data.bboxes || {}));
        console.log('bboxes:', JSON.stringify(data.bboxes, null, 2));

        const previewContainer = document.getElementById('preview-container');
        const img = document.getElementById('preview-img');

        if (data.format === 'svg' && data.svg) {
            // Handle SVG: replace img with inline SVG
            const svgWrapper = document.createElement('div');
            svgWrapper.id = 'preview-svg-wrapper';
            svgWrapper.innerHTML = data.svg;
            svgWrapper.style.width = '100%';
            svgWrapper.style.maxHeight = '70vh';

            // Find the SVG element and set styles
            const svgEl = svgWrapper.querySelector('svg');
            if (svgEl) {
                svgEl.style.width = '100%';
                svgEl.style.height = 'auto';
                svgEl.style.maxHeight = '70vh';
                svgEl.id = 'preview-img';  // Keep same ID for event handlers
            }

            img.style.display = 'none';
            const existingWrapper = document.getElementById('preview-svg-wrapper');
            if (existingWrapper) existingWrapper.remove();
            previewContainer.appendChild(svgWrapper);
        } else if (data.image) {
            // Handle PNG: show as base64 image
            img.src = 'data:image/png;base64,' + data.image;
            img.style.display = 'block';
            const existingWrapper = document.getElementById('preview-svg-wrapper');
            if (existingWrapper) existingWrapper.remove();
        }

        if (data.bboxes) {
            elementBboxes = data.bboxes;
            originalBboxes = JSON.parse(JSON.stringify(data.bboxes));  // Deep copy
            if (data.bboxes._meta) {
                schemaMeta = data.bboxes._meta;
            }
            console.log('Loaded bboxes:', Object.keys(elementBboxes).filter(k => k !== '_meta'));
        }
        if (data.img_size) {
            imgSize = data.img_size;
            originalImgSize = {...data.img_size};  // Copy
            console.log('Loaded imgSize:', imgSize);
        }

        isShowingOriginalPreview = true;
        updateOverlay();
        setStatus('Preview loaded', false);

        // Initialize hover system for the SVG if needed
        if (data.format === 'svg') {
            const svgWrapper = document.getElementById('preview-svg-wrapper');
            if (svgWrapper) {
                initHoverSystemForElement(svgWrapper.querySelector('svg'));
            }
        }

        // Draw debug bboxes if debug mode is on
        if (debugMode) {
            drawDebugBboxes();
        }

        // Handle multi-panel figz bundles
        if (data.panel_info && data.panel_info.panels) {
            panelData = data.panel_info;
            currentPanelIndex = data.panel_info.current_index || 0;
            console.log('Multi-panel figz detected:', panelData.panels.length, 'panels');
            loadPanelGrid();
        }

        // Start auto-update AFTER initial preview is loaded
        setAutoUpdateInterval();
    } catch (e) {
        setStatus('Error loading preview: ' + e.message, true);
        console.error('Preview load error:', e);
        // Start auto-update even on error so the editor works
        setAutoUpdateInterval();
    }
}

// =============================================================================
// Multi-Panel Navigation
// =============================================================================
async function loadPanelGrid() {
    if (!panelData || panelData.panels.length <= 1) {
        // Not a multi-panel bundle or only one panel
        document.getElementById('panel-grid-section').style.display = 'none';
        document.getElementById('preview-header').style.display = 'none';
        return;
    }

    console.log('Loading panel canvas for', panelData.panels.length, 'panels');

    // Show panel header
    document.getElementById('preview-header').style.display = 'flex';

    // Fetch all panel images with bboxes
    try {
        const resp = await fetch('/panels');
        const data = await resp.json();

        if (data.error) {
            console.error('Panel canvas error:', data.error);
            return;
        }

        const canvasEl = document.getElementById('panel-canvas');
        canvasEl.innerHTML = '';

        // Calculate layout - arrange panels in a grid-like canvas
        const numPanels = data.panels.length;
        const cols = Math.ceil(Math.sqrt(numPanels));
        const baseWidth = 220;
        const baseHeight = 180;
        const padding = 15;

        data.panels.forEach((panel, idx) => {
            // Store bboxes and imgSize in cache for interactive hover/click
            if (panel.bboxes && panel.img_size) {
                panelBboxesCache[panel.name] = {
                    bboxes: panel.bboxes,
                    imgSize: panel.img_size
                };
                console.log(`Panel ${panel.name}: ${Object.keys(panel.bboxes).filter(k => k !== '_meta').length} bboxes, img: ${panel.img_size.width}x${panel.img_size.height}`);
            } else {
                console.warn(`Panel ${panel.name}: missing bboxes or img_size`, {bboxes: !!panel.bboxes, img_size: !!panel.img_size});
            }

            // Calculate position
            const col = idx % cols;
            const row = Math.floor(idx / cols);
            if (!panelPositions[panel.name]) {
                panelPositions[panel.name] = {
                    x: padding + col * (baseWidth + padding),
                    y: padding + row * (baseHeight + padding),
                    width: baseWidth,
                    height: baseHeight,
                };
            }
            const pos = panelPositions[panel.name];

            const item = document.createElement('div');
            item.className = 'panel-canvas-item' + (idx === currentPanelIndex ? ' active' : '');
            item.dataset.panelIndex = idx;
            item.dataset.panelName = panel.name;
            item.style.left = pos.x + 'px';
            item.style.top = pos.y + 'px';
            item.style.width = pos.width + 'px';
            item.style.height = pos.height + 'px';

            if (panel.image) {
                item.innerHTML = `
                    <span class="panel-canvas-label">Panel ${panel.name}</span>
                    <div class="panel-card-container">
                        <img src="data:image/png;base64,${panel.image}" alt="Panel ${panel.name}">
                        <svg class="panel-card-overlay" id="panel-overlay-${idx}"></svg>
                    </div>
                    <div class="panel-canvas-resize"></div>
                `;
            } else {
                item.innerHTML = `
                    <span class="panel-canvas-label">Panel ${panel.name}</span>
                    <div style="padding: 20px; color: var(--text-muted);">No preview</div>
                `;
            }

            // Add interactive event handlers
            initCanvasItemInteraction(item, idx, panel.name);

            canvasEl.appendChild(item);
        });

        // Update canvas height to fit all panels
        const maxY = Math.max(...Object.values(panelPositions).map(p => p.y + p.height)) + padding;
        canvasEl.style.minHeight = Math.max(400, maxY) + 'px';

        // Update panel indicator
        updatePanelIndicator();

        // Show canvas for multi-panel figures
        if (data.panels.length > 1) {
            showingPanelGrid = true;
            document.getElementById('panel-grid-section').style.display = 'block';
            // Hide single-panel preview for multi-panel bundles
            const previewWrapper = document.querySelector('.preview-wrapper');
            if (previewWrapper) {
                previewWrapper.style.display = 'none';
            }
        }
    } catch (e) {
        console.error('Error loading panels:', e);
    }
}

// Initialize interactive handlers for canvas panel items
function initCanvasItemInteraction(item, panelIdx, panelName) {
    const container = item.querySelector('.panel-card-container');
    if (!container) return;

    const img = container.querySelector('img');
    const overlay = container.querySelector('svg');
    if (!img || !overlay) return;

    // Wait for image to load to get dimensions
    img.addEventListener('load', () => {
        overlay.setAttribute('width', img.offsetWidth);
        overlay.setAttribute('height', img.offsetHeight);
        overlay.style.width = img.offsetWidth + 'px';
        overlay.style.height = img.offsetHeight + 'px';
    });

    // Mousemove for hover detection
    container.addEventListener('mousemove', (e) => {
        const panelCache = panelBboxesCache[panelName];
        if (!panelCache) return;

        const rect = img.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const scaleX = panelCache.imgSize.width / rect.width;
        const scaleY = panelCache.imgSize.height / rect.height;
        const imgX = x * scaleX;
        const imgY = y * scaleY;

        const element = findElementInPanelAt(imgX, imgY, panelCache.bboxes);
        if (element !== panelHoveredElement || activePanelCard !== item) {
            panelHoveredElement = element;
            activePanelCard = item;
            updatePanelOverlay(overlay, panelCache.bboxes, panelCache.imgSize, rect.width, rect.height, panelHoveredElement, null);
        }
    });

    // Mouseleave to clear hover
    container.addEventListener('mouseleave', () => {
        panelHoveredElement = null;
        if (activePanelCard === item) {
            updatePanelOverlay(overlay, {}, {width: 0, height: 0}, 0, 0, null, null);
        }
    });

    // Click to select element
    container.addEventListener('click', (e) => {
        e.stopPropagation();

        // Recalculate element at click position (in case hover didn't detect it)
        const panelCache = panelBboxesCache[panelName];
        let clickedElement = panelHoveredElement;

        if (panelCache && img) {
            const rect = img.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const scaleX = panelCache.imgSize.width / rect.width;
            const scaleY = panelCache.imgSize.height / rect.height;
            const imgX = x * scaleX;
            const imgY = y * scaleY;

            clickedElement = findElementInPanelAt(imgX, imgY, panelCache.bboxes);
            console.log(`Click at (${imgX.toFixed(0)}, ${imgY.toFixed(0)}) -> element: ${clickedElement}`);
        }

        if (clickedElement) {
            document.querySelectorAll('.panel-canvas-item').forEach((c, i) => {
                c.classList.toggle('active', i === panelIdx);
            });

            // If already on this panel, just update selection without server call
            if (currentPanelIndex === panelIdx && panelCache) {
                console.log(`Same panel (${panelIdx}), updating selection to: ${clickedElement}`);
                selectedElement = clickedElement;
                // Sync elementBboxes with panel cache bboxes
                elementBboxes = panelCache.bboxes || {};
                imgSize = panelCache.imgSize || imgSize;
                console.log('elementBboxes keys:', Object.keys(elementBboxes));
                updateOverlay();
                console.log('Calling scrollToSection with:', selectedElement);
                scrollToSection(selectedElement);
                setStatus(`Selected: ${clickedElement}`, false);
            } else {
                currentPanelIndex = panelIdx;
                loadPanelForEditing(panelIdx, panelName, clickedElement);
            }
        } else {
            console.log(`No element found, selecting panel ${panelName}`);
            selectPanel(panelIdx);
        }
    });

    // Drag support for repositioning
    item.addEventListener('mousedown', (e) => {
        if (e.target.classList.contains('panel-canvas-resize')) {
            startResize(e, item, panelName);
        } else if (!e.target.closest('.panel-card-container')) {
            startDrag(e, item, panelName);
        }
    });
}

// Initialize interactive hover/click handlers for a panel card
function initPanelCardInteraction(card, panelIdx, panelName) {
    const container = card.querySelector('.panel-card-container');
    if (!container) return;

    const img = container.querySelector('img');
    const overlay = container.querySelector('svg');
    if (!img || !overlay) return;

    // Wait for image to load to get dimensions
    img.addEventListener('load', () => {
        overlay.setAttribute('width', img.offsetWidth);
        overlay.setAttribute('height', img.offsetHeight);
        overlay.style.width = img.offsetWidth + 'px';
        overlay.style.height = img.offsetHeight + 'px';
    });

    // Mousemove for hover detection
    container.addEventListener('mousemove', (e) => {
        const panelCache = panelBboxesCache[panelName];
        if (!panelCache) return;

        const rect = img.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const scaleX = panelCache.imgSize.width / rect.width;
        const scaleY = panelCache.imgSize.height / rect.height;
        const imgX = x * scaleX;
        const imgY = y * scaleY;

        // Find element at cursor using panel's bboxes
        const element = findElementInPanelAt(imgX, imgY, panelCache.bboxes);
        if (element !== panelHoveredElement || activePanelCard !== card) {
            panelHoveredElement = element;
            activePanelCard = card;
            updatePanelOverlay(overlay, panelCache.bboxes, panelCache.imgSize, rect.width, rect.height, panelHoveredElement, null);
        }
    });

    // Mouseleave to clear hover
    container.addEventListener('mouseleave', () => {
        panelHoveredElement = null;
        if (activePanelCard === card) {
            updatePanelOverlay(overlay, {}, {width: 0, height: 0}, 0, 0, null, null);
        }
    });

    // Click to select element
    container.addEventListener('click', (e) => {
        e.stopPropagation();  // Prevent card click from triggering selectPanel

        if (panelHoveredElement) {
            // Set this panel as current and select the element
            currentPanelIndex = panelIdx;

            // Update active state in grid
            document.querySelectorAll('.panel-card').forEach((c, i) => {
                c.classList.toggle('active', i === panelIdx);
            });

            // Load this panel's data into the main editor
            loadPanelForEditing(panelIdx, panelName, panelHoveredElement);
        } else {
            // No element hovered, select the panel itself
            selectPanel(panelIdx);
        }
    });
}

// Find element at position within a panel's bboxes
function findElementInPanelAt(x, y, bboxes) {
    const PROXIMITY_THRESHOLD = 15;
    const SCATTER_THRESHOLD = 20;

    let closestDataElement = null;
    let minDistance = Infinity;

    for (const [name, bbox] of Object.entries(bboxes)) {
        if (name === '_meta') continue;

        // Check data elements with points
        if (bbox.points && bbox.points.length > 0) {
            if (x >= bbox.x0 - SCATTER_THRESHOLD && x <= bbox.x1 + SCATTER_THRESHOLD &&
                y >= bbox.y0 - SCATTER_THRESHOLD && y <= bbox.y1 + SCATTER_THRESHOLD) {

                const elementType = bbox.element_type || 'line';
                let dist;

                if (elementType === 'scatter') {
                    dist = distanceToNearestPoint(x, y, bbox.points);
                } else {
                    dist = distanceToLine(x, y, bbox.points);
                }

                if (dist < minDistance) {
                    minDistance = dist;
                    closestDataElement = name;
                }
            }
        }
    }

    if (closestDataElement) {
        const bbox = bboxes[closestDataElement];
        const threshold = (bbox.element_type === 'scatter') ? SCATTER_THRESHOLD : PROXIMITY_THRESHOLD;
        if (minDistance <= threshold) {
            return closestDataElement;
        }
    }

    // Check bbox containment for other elements
    const elementMatches = [];
    const panelMatches = [];

    for (const [name, bbox] of Object.entries(bboxes)) {
        if (name === '_meta') continue;
        if (x >= bbox.x0 && x <= bbox.x1 && y >= bbox.y0 && y <= bbox.y1) {
            const area = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0);
            const isPanel = bbox.is_panel || name.endsWith('_panel');
            const hasPoints = bbox.points && bbox.points.length > 0;

            if (hasPoints) continue;
            else if (isPanel) panelMatches.push({name, area, bbox});
            else elementMatches.push({name, area, bbox});
        }
    }

    if (elementMatches.length > 0) {
        elementMatches.sort((a, b) => a.area - b.area);
        return elementMatches[0].name;
    }

    if (panelMatches.length > 0) {
        panelMatches.sort((a, b) => a.area - b.area);
        return panelMatches[0].name;
    }

    return null;
}

// Toggle debug mode for panel grid
function togglePanelDebugMode() {
    panelDebugMode = !panelDebugMode;
    const btn = document.getElementById('panel-debug-btn');
    if (btn) {
        btn.classList.toggle('active', panelDebugMode);
        btn.textContent = panelDebugMode ? 'Hide Hit Regions' : 'Show Hit Regions';
    }
    console.log('Panel debug mode:', panelDebugMode ? 'ON' : 'OFF');

    // Redraw all panel overlays
    redrawAllPanelOverlays();
}

// Redraw all panel overlays (useful for debug mode toggle)
function redrawAllPanelOverlays() {
    document.querySelectorAll('.panel-canvas-item').forEach((item) => {
        const panelName = item.dataset.panelName;
        const panelCache = panelBboxesCache[panelName];
        if (!panelCache) {
            console.log('No cache for panel:', panelName);
            return;
        }

        const container = item.querySelector('.panel-card-container');
        if (!container) return;

        const img = container.querySelector('img');
        const overlay = container.querySelector('svg');
        if (!img || !overlay) return;

        const rect = img.getBoundingClientRect();
        console.log(`Redraw panel ${panelName}: rect=${rect.width}x${rect.height}, bboxes=${Object.keys(panelCache.bboxes).length}`);
        if (rect.width > 0 && rect.height > 0) {
            updatePanelOverlay(overlay, panelCache.bboxes, panelCache.imgSize, rect.width, rect.height, null, null);
        }
    });
}

// Update SVG overlay for a panel card
function updatePanelOverlay(overlay, bboxes, imgSizePanel, displayWidth, displayHeight, hovered, selected) {
    if (!overlay || displayWidth === 0 || displayHeight === 0 || !imgSizePanel || imgSizePanel.width === 0) {
        if (overlay) overlay.innerHTML = '';
        return;
    }

    overlay.setAttribute('width', displayWidth);
    overlay.setAttribute('height', displayHeight);

    const scaleX = displayWidth / imgSizePanel.width;
    const scaleY = displayHeight / imgSizePanel.height;

    let svg = '';

    // Debug mode: draw all bboxes
    if (panelDebugMode && bboxes) {
        svg += drawPanelDebugBboxes(bboxes, scaleX, scaleY);
    }

    function drawPanelElement(elementName, type) {
        const bbox = bboxes[elementName];
        if (!bbox) return '';

        const elementType = bbox.element_type || '';
        const hasPoints = bbox.points && bbox.points.length > 0;

        // Lines - draw as path
        if ((elementType === 'line' || elementName.includes('trace_')) && hasPoints) {
            if (bbox.points.length < 2) return '';
            const points = bbox.points.filter(pt => Array.isArray(pt) && pt.length >= 2);
            if (points.length < 2) return '';

            let pathD = `M ${points[0][0] * scaleX} ${points[0][1] * scaleY}`;
            for (let i = 1; i < points.length; i++) {
                pathD += ` L ${points[i][0] * scaleX} ${points[i][1] * scaleY}`;
            }

            const className = type === 'hover' ? 'hover-path' : 'selected-path';
            return `<path class="${className}" d="${pathD}"/>`;
        }
        // Scatter - draw as circles
        else if (elementType === 'scatter' && hasPoints) {
            const className = type === 'hover' ? 'hover-scatter' : 'selected-scatter';
            let result = '';
            for (const pt of bbox.points) {
                if (!Array.isArray(pt) || pt.length < 2) continue;
                result += `<circle class="${className}" cx="${pt[0] * scaleX}" cy="${pt[1] * scaleY}" r="3"/>`;
            }
            return result;
        }
        // Default - draw bbox rectangle
        else {
            const rectClass = type === 'hover' ? 'hover-rect' : 'selected-rect';
            const x = bbox.x0 * scaleX - 1;
            const y = bbox.y0 * scaleY - 1;
            const w = (bbox.x1 - bbox.x0) * scaleX + 2;
            const h = (bbox.y1 - bbox.y0) * scaleY + 2;
            return `<rect class="${rectClass}" x="${x}" y="${y}" width="${w}" height="${h}" rx="2"/>`;
        }
    }

    if (hovered && hovered !== selected) {
        svg += drawPanelElement(hovered, 'hover');
    }

    if (selected) {
        svg += drawPanelElement(selected, 'selected');
    }

    overlay.innerHTML = svg;
}

// Draw all bboxes for a panel in debug mode
function drawPanelDebugBboxes(bboxes, scaleX, scaleY) {
    let svg = '';
    let count = 0;

    for (const [name, bbox] of Object.entries(bboxes)) {
        if (name === '_meta') continue;
        if (bbox.x0 === undefined || bbox.y0 === undefined) continue;

        count++;
        const hasPoints = bbox.points && bbox.points.length > 0;
        const elementType = bbox.element_type || '';

        // Choose color based on element type
        let rectClass = 'debug-rect';
        if (name.includes('trace_') || elementType === 'line') {
            rectClass = 'debug-rect-trace';
        } else if (name.includes('legend')) {
            rectClass = 'debug-rect-legend';
        } else if (elementType === 'scatter') {
            rectClass = 'debug-rect-trace';
        }

        // Draw bbox rectangle
        const x = bbox.x0 * scaleX;
        const y = bbox.y0 * scaleY;
        const w = (bbox.x1 - bbox.x0) * scaleX;
        const h = (bbox.y1 - bbox.y0) * scaleY;

        svg += `<rect class="${rectClass}" x="${x}" y="${y}" width="${w}" height="${h}"/>`;

        // Draw short label (truncated for small panels)
        const shortName = name.length > 10 ? name.substring(0, 8) + '..' : name;
        svg += `<text class="debug-label" x="${x + 1}" y="${y + 8}" style="font-size: 6px;">${shortName}</text>`;

        // Draw path points if available
        if (hasPoints && bbox.points.length > 1) {
            let pathD = `M ${bbox.points[0][0] * scaleX} ${bbox.points[0][1] * scaleY}`;
            for (let i = 1; i < bbox.points.length; i++) {
                const pt = bbox.points[i];
                if (pt && pt.length >= 2) {
                    pathD += ` L ${pt[0] * scaleX} ${pt[1] * scaleY}`;
                }
            }
            svg += `<path class="debug-path" d="${pathD}"/>`;
        }
    }

    console.log(`Panel debug: ${count} elements with bboxes`);
    return svg;
}

// Load panel data and switch to it for editing with a pre-selected element
async function loadPanelForEditing(panelIdx, panelName, elementToSelect) {
    showLoading();
    setStatus(`Loading Panel ${panelName}...`, false);

    try {
        const resp = await fetch(`/switch_panel/${panelIdx}`);
        const data = await resp.json();

        if (data.error) {
            console.error('switch_panel error:', data.error);
            if (data.traceback) {
                console.error('Traceback:', data.traceback);
            }
            setStatus('Error: ' + data.error, true);
            hideLoading();
            return;
        }

        // Update panel state
        currentPanelIndex = panelIdx;
        panelData.current_index = panelIdx;
        updatePanelIndicator();

        // Update preview image
        const img = document.getElementById('preview-img');
        img.src = 'data:image/png;base64,' + data.image;

        // Update bboxes and overlays
        elementBboxes = data.bboxes || {};
        if (data.img_size) {
            imgSize = data.img_size;
        }

        // Update overrides
        if (data.overrides) {
            overrides = data.overrides;
            traces = overrides.traces || [];
            updateControlsFromOverrides();
        }

        // Select the element that was clicked
        selectedElement = elementToSelect;
        updateOverlay();

        // Scroll to section and show properties
        scrollToSection(selectedElement);

        // Show single-panel preview when element selected
        const previewWrapper = document.querySelector('.preview-wrapper');
        if (previewWrapper) {
            previewWrapper.style.display = 'block';
        }

        // Update panel path display in right panel header
        const panelPathEl = document.getElementById('panel-path-display');
        if (panelPathEl) {
            panelPathEl.textContent = `Panel: ${panelName}.pltz.d/spec.json`;
        }

        setStatus(`Selected: ${elementToSelect} in Panel ${panelName}`, false);
    } catch (e) {
        setStatus('Error: ' + e.message, true);
        console.error('Panel load error:', e);
    } finally {
        hideLoading();
    }
}

function togglePanelGrid() {
    showingPanelGrid = !showingPanelGrid;
    const gridSection = document.getElementById('panel-grid-section');
    const showBtn = document.getElementById('show-grid-btn');

    if (showingPanelGrid) {
        gridSection.style.display = 'block';
        showBtn.textContent = 'Hide All';
    } else {
        gridSection.style.display = 'none';
        showBtn.textContent = 'Show All';
    }
}

async function selectPanel(idx) {
    if (!panelData || idx < 0 || idx >= panelData.panels.length) return;

    // Show loading state
    showLoading();
    setStatus('Switching panel...', false);

    try {
        const resp = await fetch(`/switch_panel/${idx}`);
        const data = await resp.json();

        if (data.error) {
            setStatus('Error: ' + data.error, true);
            hideLoading();
            return;
        }

        // Update panel state
        currentPanelIndex = idx;
        panelData.current_index = idx;
        updatePanelIndicator();

        // Update active state in grid
        document.querySelectorAll('.panel-card').forEach((card, i) => {
            card.classList.toggle('active', i === idx);
        });

        // Update preview image
        const img = document.getElementById('preview-img');
        img.src = 'data:image/png;base64,' + data.image;

        // Update bboxes and overlays
        elementBboxes = data.bboxes || {};
        if (data.img_size) {
            imgSize = data.img_size;
        }

        // Update overrides
        if (data.overrides) {
            overrides = data.overrides;
            traces = overrides.traces || [];
            updateControlsFromOverrides();
        }

        updateOverlay();
        if (debugMode) {
            drawDebugBboxes();
        }

        // Update panel path display in right panel header
        const panelPathEl = document.getElementById('panel-path-display');
        if (panelPathEl && data.panel_name) {
            panelPathEl.textContent = `Panel: ${data.panel_name}/spec.json`;
        }

        setStatus(`Switched to Panel ${data.panel_name.replace('.pltz.d', '')}`, false);
    } catch (e) {
        setStatus('Error switching panel: ' + e.message, true);
        console.error('Panel switch error:', e);
    } finally {
        hideLoading();
    }
}

function prevPanel() {
    if (panelData && currentPanelIndex > 0) {
        selectPanel(currentPanelIndex - 1);
    }
}

function nextPanel() {
    if (panelData && currentPanelIndex < panelData.panels.length - 1) {
        selectPanel(currentPanelIndex + 1);
    }
}

function updatePanelIndicator() {
    if (!panelData) return;

    const total = panelData.panels.length;
    const current = currentPanelIndex + 1;
    const panelName = panelData.panels[currentPanelIndex];

    document.getElementById('panel-indicator').textContent = `${current} / ${total}`;
    document.getElementById('current-panel-name').textContent = `Panel ${panelName.replace('.pltz.d', '')}`;

    // Update prev/next button states
    document.getElementById('prev-panel-btn').disabled = currentPanelIndex === 0;
    document.getElementById('next-panel-btn').disabled = currentPanelIndex === total - 1;
}

// =============================================================================
// Canvas Mode (Draggable Panel Layout)
// =============================================================================
let canvasMode = 'grid';  // 'grid' or 'canvas'
let panelPositions = {};  // Store panel positions {name: {x, y, width, height}}
let draggedPanel = null;
let dragOffset = {x: 0, y: 0};

function setCanvasMode(mode) {
    canvasMode = mode;
    const gridEl = document.getElementById('panel-grid');
    const canvasEl = document.getElementById('panel-canvas');
    const gridBtn = document.getElementById('view-grid-btn');
    const canvasBtn = document.getElementById('view-canvas-btn');

    if (mode === 'grid') {
        gridEl.style.display = 'grid';
        canvasEl.style.display = 'none';
        gridBtn.classList.remove('btn-secondary');
        gridBtn.classList.add('btn-primary');
        canvasBtn.classList.add('btn-secondary');
        canvasBtn.classList.remove('btn-primary');
    } else {
        gridEl.style.display = 'none';
        canvasEl.style.display = 'block';
        canvasBtn.classList.remove('btn-secondary');
        canvasBtn.classList.add('btn-primary');
        gridBtn.classList.add('btn-secondary');
        gridBtn.classList.remove('btn-primary');
        renderPanelCanvas();
    }
}

async function renderPanelCanvas() {
    const canvasEl = document.getElementById('panel-canvas');
    if (!panelData || !canvasEl) return;

    // Fetch panels if not cached
    try {
        const resp = await fetch('/panels');
        const data = await resp.json();
        if (data.error) return;

        canvasEl.innerHTML = '';

        // Calculate canvas size based on number of panels
        const numPanels = data.panels.length;
        const cols = Math.ceil(Math.sqrt(numPanels));
        const baseWidth = 200;
        const baseHeight = 150;
        const padding = 20;

        data.panels.forEach((panel, idx) => {
            const name = panel.name;

            // Initialize position if not set
            if (!panelPositions[name]) {
                const col = idx % cols;
                const row = Math.floor(idx / cols);
                panelPositions[name] = {
                    x: padding + col * (baseWidth + padding),
                    y: padding + row * (baseHeight + padding),
                    width: baseWidth,
                    height: baseHeight,
                };
            }

            const pos = panelPositions[name];
            const item = document.createElement('div');
            item.className = 'panel-canvas-item' + (idx === currentPanelIndex ? ' active' : '');
            item.dataset.panelIndex = idx;
            item.dataset.panelName = name;
            item.style.left = pos.x + 'px';
            item.style.top = pos.y + 'px';
            item.style.width = pos.width + 'px';
            item.style.height = pos.height + 'px';

            item.innerHTML = `
                <span class="panel-canvas-label">Panel ${name}</span>
                ${panel.image ? `<img src="data:image/png;base64,${panel.image}" alt="Panel ${name}">` : '<div style="padding: 20px; color: var(--text-muted);">No preview</div>'}
                <div class="panel-canvas-resize"></div>
            `;

            // Double-click to edit
            item.addEventListener('dblclick', () => selectPanel(idx));

            // Drag start
            item.addEventListener('mousedown', (e) => {
                if (e.target.classList.contains('panel-canvas-resize')) {
                    startResize(e, item, name);
                } else {
                    startDrag(e, item, name);
                }
            });

            canvasEl.appendChild(item);
        });

        // Update canvas height to fit all panels
        const maxY = Math.max(...Object.values(panelPositions).map(p => p.y + p.height)) + padding;
        canvasEl.style.minHeight = Math.max(400, maxY) + 'px';

    } catch (e) {
        console.error('Error rendering canvas:', e);
    }
}

function startDrag(e, item, name) {
    e.preventDefault();
    draggedPanel = {item, name};
    dragOffset.x = e.clientX - item.offsetLeft;
    dragOffset.y = e.clientY - item.offsetTop;
    item.classList.add('dragging');

    document.addEventListener('mousemove', onDrag);
    document.addEventListener('mouseup', stopDrag);
}

function onDrag(e) {
    if (!draggedPanel) return;
    const canvasEl = document.getElementById('panel-canvas');
    const rect = canvasEl.getBoundingClientRect();

    let newX = e.clientX - dragOffset.x;
    let newY = e.clientY - dragOffset.y;

    // Constrain to canvas bounds
    newX = Math.max(0, Math.min(newX, canvasEl.offsetWidth - draggedPanel.item.offsetWidth));
    newY = Math.max(0, newY);

    draggedPanel.item.style.left = newX + 'px';
    draggedPanel.item.style.top = newY + 'px';

    panelPositions[draggedPanel.name].x = newX;
    panelPositions[draggedPanel.name].y = newY;
}

function stopDrag() {
    if (draggedPanel) {
        draggedPanel.item.classList.remove('dragging');
        draggedPanel = null;
    }
    document.removeEventListener('mousemove', onDrag);
    document.removeEventListener('mouseup', stopDrag);
}

let resizingPanel = null;

function startResize(e, item, name) {
    e.preventDefault();
    e.stopPropagation();
    resizingPanel = {item, name, startX: e.clientX, startY: e.clientY, startW: item.offsetWidth, startH: item.offsetHeight};

    document.addEventListener('mousemove', onResize);
    document.addEventListener('mouseup', stopResize);
}

function onResize(e) {
    if (!resizingPanel) return;
    const newW = Math.max(100, resizingPanel.startW + (e.clientX - resizingPanel.startX));
    const newH = Math.max(80, resizingPanel.startH + (e.clientY - resizingPanel.startY));

    resizingPanel.item.style.width = newW + 'px';
    resizingPanel.item.style.height = newH + 'px';

    panelPositions[resizingPanel.name].width = newW;
    panelPositions[resizingPanel.name].height = newH;
}

function stopResize() {
    resizingPanel = null;
    document.removeEventListener('mousemove', onResize);
    document.removeEventListener('mouseup', stopResize);
}

// Initialize hover system for a specific element (img or svg)
function initHoverSystemForElement(el) {
    if (!el) return;

    el.addEventListener('mousemove', (e) => {
        if (imgSize.width === 0 || imgSize.height === 0) return;

        const rect = el.getBoundingClientRect();
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

    el.addEventListener('mouseleave', () => {
        hoveredElement = null;
        updateOverlay();
    });

    el.addEventListener('click', (e) => {
        if (hoveredElement) {
            selectedElement = hoveredElement;
            updateOverlay();
            scrollToSection(selectedElement);
        }
    });
}

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

    // Labels - Title
    const title = document.getElementById('title').value;
    if (title) o.title = title;
    o.show_title = document.getElementById('show_title').checked;
    o.title_fontsize = parseInt(document.getElementById('title_fontsize').value) || 8;
    // Labels - Caption
    const caption = document.getElementById('caption').value;
    if (caption) o.caption = caption;
    o.show_caption = document.getElementById('show_caption').checked;
    o.caption_fontsize = parseInt(document.getElementById('caption_fontsize').value) || 7;
    // Labels - Axis
    const xlabel = document.getElementById('xlabel').value;
    const ylabel = document.getElementById('ylabel').value;
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
    o.traces = traces;

    // Legend
    o.legend_visible = document.getElementById('legend_visible').checked;
    o.legend_loc = document.getElementById('legend_loc').value;
    o.legend_frameon = document.getElementById('legend_frameon').checked;
    o.legend_fontsize = parseInt(document.getElementById('legend_fontsize').value) || 6;
    o.legend_ncols = parseInt(document.getElementById('legend_ncols').value) || 1;
    o.legend_x = parseFloat(document.getElementById('legend_x').value) || 0.95;
    o.legend_y = parseFloat(document.getElementById('legend_y').value) || 0.95;

    // Axis and Ticks - X Axis (Bottom)
    o.x_n_ticks = parseInt(document.getElementById('x_n_ticks').value) || 4;
    o.hide_x_ticks = document.getElementById('hide_x_ticks').checked;
    o.x_tick_fontsize = parseInt(document.getElementById('x_tick_fontsize').value) || 7;
    o.x_tick_direction = document.getElementById('x_tick_direction').value;
    o.x_tick_length = parseFloat(document.getElementById('x_tick_length').value) || 0.8;
    o.x_tick_width = parseFloat(document.getElementById('x_tick_width').value) || 0.2;
    // X Axis (Top)
    o.show_x_top = document.getElementById('show_x_top').checked;
    o.x_top_mirror = document.getElementById('x_top_mirror').checked;
    // Y Axis (Left)
    o.y_n_ticks = parseInt(document.getElementById('y_n_ticks').value) || 4;
    o.hide_y_ticks = document.getElementById('hide_y_ticks').checked;
    o.y_tick_fontsize = parseInt(document.getElementById('y_tick_fontsize').value) || 7;
    o.y_tick_direction = document.getElementById('y_tick_direction').value;
    o.y_tick_length = parseFloat(document.getElementById('y_tick_length').value) || 0.8;
    o.y_tick_width = parseFloat(document.getElementById('y_tick_width').value) || 0.2;
    // Y Axis (Right)
    o.show_y_right = document.getElementById('show_y_right').checked;
    o.y_right_mirror = document.getElementById('y_right_mirror').checked;
    // Spines
    o.hide_bottom_spine = document.getElementById('hide_bottom_spine').checked;
    o.hide_left_spine = document.getElementById('hide_left_spine').checked;
    // Z Axis (3D)
    o.hide_z_ticks = document.getElementById('hide_z_ticks').checked;
    o.z_n_ticks = parseInt(document.getElementById('z_n_ticks').value) || 4;
    o.z_tick_fontsize = parseInt(document.getElementById('z_tick_fontsize').value) || 7;
    o.z_tick_direction = document.getElementById('z_tick_direction').value;

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

async function updatePreview(forceUpdate = false) {
    // Skip auto-update if showing original preview (user hasn't explicitly requested update)
    if (isShowingOriginalPreview && !forceUpdate) {
        console.log('Skipping auto-update: showing original preview');
        return;
    }

    setStatus('Updating...', false);
    overrides = collectOverrides();

    // Preserve current selection to restore after update
    const previousSelection = selectedElement;

    try {
        const darkMode = isDarkMode();
        const resp = await fetch('/update', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({overrides, dark_mode: darkMode})
        });
        const data = await resp.json();

        // Remove SVG wrapper if exists, show img element for re-rendered preview
        const existingSvgWrapper = document.getElementById('preview-svg-wrapper');
        if (existingSvgWrapper) {
            existingSvgWrapper.remove();
        }
        const imgEl = document.getElementById('preview-img');
        imgEl.style.display = 'block';
        imgEl.src = 'data:image/png;base64,' + data.image;

        // Mark that we're no longer showing original preview
        isShowingOriginalPreview = false;

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

        // Restore selection if the element still exists in the new bboxes
        if (previousSelection && elementBboxes[previousSelection]) {
            selectedElement = previousSelection;
        } else {
            selectedElement = null;
        }
        hoveredElement = null;
        updateOverlay();

        setStatus('Preview updated', false);
    } catch (e) {
        setStatus('Error: ' + e.message, true);
    }
}

// Restore original preview (SVG/PNG from bundle)
async function restoreOriginalPreview() {
    if (!originalBboxes || !originalImgSize) {
        console.log('No original preview to restore');
        return;
    }

    setStatus('Restoring original preview...', false);
    await loadInitialPreview();
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

// =============================================================================
// Statistics Display
// =============================================================================
async function refreshStats() {
    const container = document.getElementById('stats-container');
    container.innerHTML = '<div class="stats-loading">Loading statistics...</div>';

    try {
        const resp = await fetch('/stats');
        const data = await resp.json();

        if (!data.has_stats) {
            container.innerHTML = '<div class="stats-empty">No statistical tests in this figure</div>';
            return;
        }

        let html = '';

        // Show summary if available
        if (data.stats_summary) {
            const summary = data.stats_summary;
            html += `
                <div class="stats-summary-header">
                    ${summary.test_type.replace('_', '-')}
                    <span class="stats-correction-badge">${summary.correction_method}</span>
                </div>
                <div class="stats-summary-body">
                    <div class="stats-row">
                        <span class="stats-label">Comparisons:</span>
                        <span class="stats-value">${summary.n_comparisons}</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">α (original):</span>
                        <span class="stats-value">${summary.alpha}</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">α (corrected):</span>
                        <span class="stats-value">${summary.corrected_alpha.toFixed(4)}</span>
                    </div>
                </div>
            `;
        }

        // Show individual test results
        data.stats.forEach((stat, idx) => {
            const sigClass = getSigClass(stat.stars);
            const samples = stat.samples || {};
            const correction = stat.correction || {};

            html += `
                <div class="stats-card">
                    <div class="stats-card-header">
                        <span class="stats-card-title">
                            ${samples.group1?.name || 'Group 1'} vs ${samples.group2?.name || 'Group 2'}
                        </span>
                        <span class="stats-significance ${sigClass}">${stat.stars}</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">${stat.statistic?.name || 'Stat'}:</span>
                        <span class="stats-value">${(stat.statistic?.value || 0).toFixed(3)}</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">p (raw):</span>
                        <span class="stats-value">${stat.p_value.toFixed(4)}</span>
                    </div>
                    ${correction.corrected_p ? `
                    <div class="stats-row">
                        <span class="stats-label">p (corrected):</span>
                        <span class="stats-value">${correction.corrected_p.toFixed(4)}</span>
                    </div>` : ''}
                    <div class="stats-groups">
                        ${samples.group1 ? renderGroupStats(samples.group1) : ''}
                        ${samples.group2 ? renderGroupStats(samples.group2) : ''}
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = `<div class="stats-empty">Error loading stats: ${e.message}</div>`;
    }
}

function getSigClass(stars) {
    if (stars === '***') return 'sig-high';
    if (stars === '**') return 'sig-medium';
    if (stars === '*') return 'sig-low';
    return 'sig-ns';
}

function renderGroupStats(group) {
    return `
        <div class="stats-group">
            <div class="stats-group-name">${group.name || 'Group'}</div>
            <div>n = ${group.n}</div>
            <div>μ = ${group.mean?.toFixed(2) || '-'}</div>
            <div>σ = ${group.std?.toFixed(2) || '-'}</div>
        </div>
    `;
}

function setStatus(msg, isError = false) {
    const el = document.getElementById('status');
    const loadingOverlay = document.getElementById('loading-overlay');

    // Show/hide spinner for loading states
    if (msg === 'Updating...' || msg === 'Loading preview...') {
        loadingOverlay.style.display = 'flex';
        el.textContent = '';  // Clear status text during loading
    } else {
        loadingOverlay.style.display = 'none';
        el.textContent = msg;
    }
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
document.querySelectorAll('input[type="text"], input[type="number"], textarea').forEach(el => {
    el.addEventListener('input', scheduleUpdate);
    el.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && el.tagName !== 'TEXTAREA') {
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
