#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/templates/scripts.py
"""JavaScript for the Flask editor UI.

DEPRECATED: This inline JavaScript module is kept for fallback compatibility only.
The JavaScript has been modularized into static/js/ directory:
- static/js/main.js (main entry point)
- static/js/core/ (state, api, utils)
- static/js/canvas/ (canvas, dragging, resize, selection)
- static/js/editor/ (preview, overlay, bbox, element-drag)
- static/js/alignment/ (basic, axis, distribute)
- static/js/shortcuts/ (keyboard, context-menu)
- static/js/ui/ (controls, download, help, theme)

To use static files (recommended):
    Set USE_STATIC_FILES = True in templates/__init__.py

To use this inline version (fallback):
    Set USE_STATIC_FILES = False in templates/__init__.py
"""

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

// Calculate actual rendered image dimensions when using object-fit: contain
// Returns: {renderedWidth, renderedHeight, offsetX, offsetY, containerWidth, containerHeight}
function getObjectFitContainDimensions(img) {
    const containerWidth = img.offsetWidth;
    const containerHeight = img.offsetHeight;
    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;

    // Handle edge cases
    if (!naturalWidth || !naturalHeight || !containerWidth || !containerHeight) {
        return {
            renderedWidth: containerWidth,
            renderedHeight: containerHeight,
            offsetX: 0,
            offsetY: 0,
            containerWidth,
            containerHeight
        };
    }

    // Calculate scale factor for object-fit: contain
    const containerRatio = containerWidth / containerHeight;
    const imageRatio = naturalWidth / naturalHeight;

    let renderedWidth, renderedHeight, offsetX, offsetY;

    if (imageRatio > containerRatio) {
        // Image is wider than container - fit to width, letterbox top/bottom
        renderedWidth = containerWidth;
        renderedHeight = containerWidth / imageRatio;
        offsetX = 0;
        offsetY = (containerHeight - renderedHeight) / 2;
    } else {
        // Image is taller than container - fit to height, letterbox left/right
        renderedHeight = containerHeight;
        renderedWidth = containerHeight * imageRatio;
        offsetX = (containerWidth - renderedWidth) / 2;
        offsetY = 0;
    }

    return {
        renderedWidth,
        renderedHeight,
        offsetX,
        offsetY,
        containerWidth,
        containerHeight
    };
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

        // Get values from: 1) element_overrides, 2) traces array (plot metadata), 3) bbox data
        const traceOverrides = getTraceOverrides(elementInfo);
        const traceIdx = elementInfo.index || 0;
        const traceFromMeta = traces[traceIdx] || {};

        // Label: prefer user override, then plot metadata, then bbox label
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
    // Re-render single panel preview with dark/light mode colors (if visible)
    updatePreview(true);
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
function showLoading(message = 'Loading...') {
    const globalOverlay = document.getElementById('global-loading-overlay');
    const localOverlay = document.getElementById('loading-overlay');
    if (globalOverlay) {
        globalOverlay.style.display = 'flex';
        const loadingText = globalOverlay.querySelector('.loading-text');
        if (loadingText) loadingText.textContent = message;
    }
    if (localOverlay) localOverlay.style.display = 'flex';
}

function hideLoading() {
    const globalOverlay = document.getElementById('global-loading-overlay');
    const localOverlay = document.getElementById('loading-overlay');
    if (globalOverlay) globalOverlay.style.display = 'none';
    if (localOverlay) localOverlay.style.display = 'none';
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

        // Handle multi-panel figure bundles
        if (data.panel_info && data.panel_info.panels) {
            panelData = data.panel_info;
            currentPanelIndex = data.panel_info.current_index || 0;
            console.log('Multi-panel figure detected:', panelData.panels.length, 'panels');
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

    // Hide single-panel preview completely for multi-panel bundles (unified canvas only)
    document.getElementById('preview-header').style.display = 'none';
    const previewWrapper = document.querySelector('.preview-wrapper');
    if (previewWrapper) {
        previewWrapper.style.display = 'none';
    }

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

        // Use figure layout to position panels as unified canvas (matching export)
        const hasLayout = data.layout && Object.keys(data.layout).length > 0;

        // Calculate scale factor: convert mm to pixels
        // Find total figure dimensions from layout
        let maxX = 0, maxY = 0;
        if (hasLayout) {
            Object.values(data.layout).forEach(l => {
                const right = (l.position?.x_mm || 0) + (l.size?.width_mm || 80);
                const bottom = (l.position?.y_mm || 0) + (l.size?.height_mm || 50);
                maxX = Math.max(maxX, right);
                maxY = Math.max(maxY, bottom);
            });
        }

        // Scale to fit canvas (max width ~700px for good display)
        const canvasMaxWidth = 700;
        const scale = hasLayout && maxX > 0 ? canvasMaxWidth / maxX : 3;  // ~3px per mm fallback
        canvasScale = scale;  // Store globally for drag conversions

        // Reset layout tracking
        panelLayoutMm = {};
        layoutModified = false;

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

            // Use figure layout for positioning (unified canvas like export)
            let pos, posMm;
            if (panel.layout && panel.layout.position && panel.layout.size) {
                const x_mm = panel.layout.position.x_mm || 0;
                const y_mm = panel.layout.position.y_mm || 0;
                const width_mm = panel.layout.size.width_mm || 80;
                const height_mm = panel.layout.size.height_mm || 50;
                pos = {
                    x: x_mm * scale,
                    y: y_mm * scale,
                    width: width_mm * scale,
                    height: height_mm * scale,
                };
                posMm = { x_mm, y_mm, width_mm, height_mm };
            } else {
                // Fallback grid layout if no figure layout
                const cols = Math.ceil(Math.sqrt(data.panels.length));
                const baseWidth = 220, baseHeight = 180, padding = 15;
                const col = idx % cols;
                const row = Math.floor(idx / cols);
                pos = {
                    x: padding + col * (baseWidth + padding),
                    y: padding + row * (baseHeight + padding),
                    width: baseWidth,
                    height: baseHeight,
                };
                // Convert to mm for fallback
                posMm = {
                    x_mm: pos.x / scale,
                    y_mm: pos.y / scale,
                    width_mm: pos.width / scale,
                    height_mm: pos.height / scale,
                };
            }
            panelPositions[panel.name] = pos;
            panelLayoutMm[panel.name] = posMm;

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
                    <span class="panel-canvas-label">${panel.name}</span>
                    <span class="panel-position-indicator" id="pos-${panel.name}"></span>
                    <div class="panel-drag-handle" title="Drag to move panel">⋮⋮</div>
                    <div class="panel-card-container">
                        <img src="data:image/png;base64,${panel.image}" alt="Panel ${panel.name}">
                        <svg class="panel-card-overlay" id="panel-overlay-${idx}"></svg>
                    </div>
                `;
            } else {
                item.innerHTML = `
                    <span class="panel-canvas-label">${panel.name}</span>
                    <span class="panel-position-indicator" id="pos-${panel.name}"></span>
                    <div class="panel-drag-handle" title="Drag to move panel">⋮⋮</div>
                    <div style="padding: 20px; color: var(--text-muted);">No preview</div>
                `;
            }

            // Add interactive event handlers (hover, click for element selection)
            initCanvasItemInteraction(item, idx, panel.name);

            // Add drag handler for panel repositioning
            initPanelDrag(item, panel.name);

            canvasEl.appendChild(item);
        });

        // Update canvas size to fit all panels (unified canvas)
        const canvasHeight = Math.max(...Object.values(panelPositions).map(p => p.y + p.height)) + 10;
        const canvasWidth = Math.max(...Object.values(panelPositions).map(p => p.x + p.width)) + 10;
        canvasEl.style.minHeight = Math.max(400, canvasHeight) + 'px';
        canvasEl.style.minWidth = canvasWidth + 'px';

        // Update panel indicator
        updatePanelIndicator();

        // Show unified canvas for multi-panel figures
        showingPanelGrid = true;
        document.getElementById('panel-grid-section').style.display = 'block';
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

    // Mousemove for hover detection (accounting for object-fit:contain letterboxing)
    container.addEventListener('mousemove', (e) => {
        const panelCache = panelBboxesCache[panelName];
        if (!panelCache) return;

        const rect = img.getBoundingClientRect();
        const dims = getObjectFitContainDimensions(img);

        // Mouse position relative to container
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Adjust for letterbox offset to get position relative to actual rendered image
        const imgRelX = x - dims.offsetX;
        const imgRelY = y - dims.offsetY;

        // Check if click is within rendered image bounds
        if (imgRelX < 0 || imgRelY < 0 || imgRelX > dims.renderedWidth || imgRelY > dims.renderedHeight) {
            // Outside rendered image area (in letterbox region)
            if (panelHoveredElement !== null) {
                panelHoveredElement = null;
                updatePanelOverlay(overlay, panelCache.bboxes, panelCache.imgSize, rect.width, rect.height, null, null, img);
            }
            return;
        }

        // Scale to original image coordinates
        const scaleX = panelCache.imgSize.width / dims.renderedWidth;
        const scaleY = panelCache.imgSize.height / dims.renderedHeight;
        const imgX = imgRelX * scaleX;
        const imgY = imgRelY * scaleY;

        const element = findElementInPanelAt(imgX, imgY, panelCache.bboxes);
        if (element !== panelHoveredElement || activePanelCard !== item) {
            panelHoveredElement = element;
            activePanelCard = item;
            updatePanelOverlay(overlay, panelCache.bboxes, panelCache.imgSize, rect.width, rect.height, panelHoveredElement, null, img);
        }
    });

    // Mouseleave to clear hover
    container.addEventListener('mouseleave', () => {
        panelHoveredElement = null;
        if (activePanelCard === item) {
            updatePanelOverlay(overlay, {}, {width: 0, height: 0}, 0, 0, null, null, null);
        }
    });

    // Mousedown to start element drag (ONLY for legends and panel letters)
    container.addEventListener('mousedown', (e) => {
        const panelCache = panelBboxesCache[panelName];
        if (!panelCache || !panelHoveredElement) return;

        // Only allow dragging of legends and panel letters (scientific rigor)
        if (isDraggableElement(panelHoveredElement, panelCache.bboxes)) {
            startElementDrag(e, panelHoveredElement, panelName, img, panelCache.bboxes);
        }
    });

    // Click to select element (accounting for object-fit:contain letterboxing)
    container.addEventListener('click', (e) => {
        e.stopPropagation();

        // Recalculate element at click position (in case hover didn't detect it)
        const panelCache = panelBboxesCache[panelName];
        let clickedElement = panelHoveredElement;

        if (panelCache && img) {
            const rect = img.getBoundingClientRect();
            const dims = getObjectFitContainDimensions(img);

            // Mouse position relative to container
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Adjust for letterbox offset
            const imgRelX = x - dims.offsetX;
            const imgRelY = y - dims.offsetY;

            // Check if click is within rendered image bounds
            if (imgRelX >= 0 && imgRelY >= 0 && imgRelX <= dims.renderedWidth && imgRelY <= dims.renderedHeight) {
                // Scale to original image coordinates
                const scaleX = panelCache.imgSize.width / dims.renderedWidth;
                const scaleY = panelCache.imgSize.height / dims.renderedHeight;
                const imgX = imgRelX * scaleX;
                const imgY = imgRelY * scaleY;

                clickedElement = findElementInPanelAt(imgX, imgY, panelCache.bboxes);
                console.log(`Click at (${imgX.toFixed(0)}, ${imgY.toFixed(0)}) -> element: ${clickedElement}`);
            } else {
                clickedElement = null;
                console.log('Click outside rendered image bounds (in letterbox area)');
            }
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

    // Mousemove for hover detection (accounting for object-fit:contain letterboxing)
    container.addEventListener('mousemove', (e) => {
        const panelCache = panelBboxesCache[panelName];
        if (!panelCache) return;

        const rect = img.getBoundingClientRect();
        const dims = getObjectFitContainDimensions(img);

        // Mouse position relative to container
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Adjust for letterbox offset to get position relative to actual rendered image
        const imgRelX = x - dims.offsetX;
        const imgRelY = y - dims.offsetY;

        // Check if mouse is within rendered image bounds
        if (imgRelX < 0 || imgRelY < 0 || imgRelX > dims.renderedWidth || imgRelY > dims.renderedHeight) {
            // Outside rendered image area (in letterbox region)
            if (panelHoveredElement !== null) {
                panelHoveredElement = null;
                updatePanelOverlay(overlay, panelCache.bboxes, panelCache.imgSize, rect.width, rect.height, null, null, img);
            }
            return;
        }

        // Scale to original image coordinates
        const scaleX = panelCache.imgSize.width / dims.renderedWidth;
        const scaleY = panelCache.imgSize.height / dims.renderedHeight;
        const imgX = imgRelX * scaleX;
        const imgY = imgRelY * scaleY;

        // Find element at cursor using panel's bboxes
        const element = findElementInPanelAt(imgX, imgY, panelCache.bboxes);
        if (element !== panelHoveredElement || activePanelCard !== card) {
            panelHoveredElement = element;
            activePanelCard = card;
            updatePanelOverlay(overlay, panelCache.bboxes, panelCache.imgSize, rect.width, rect.height, panelHoveredElement, null, img);
        }
    });

    // Mouseleave to clear hover
    container.addEventListener('mouseleave', () => {
        panelHoveredElement = null;
        if (activePanelCard === card) {
            updatePanelOverlay(overlay, {}, {width: 0, height: 0}, 0, 0, null, null, null);
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
            updatePanelOverlay(overlay, panelCache.bboxes, panelCache.imgSize, rect.width, rect.height, null, null, img);
        }
    });
}

// Update SVG overlay for a panel card
// img: the img element (to calculate object-fit:contain dimensions)
// OR pass null with displayWidth/displayHeight for backward compatibility
function updatePanelOverlay(overlay, bboxes, imgSizePanel, displayWidth, displayHeight, hovered, selected, img) {
    if (!overlay || !imgSizePanel || imgSizePanel.width === 0) {
        if (overlay) overlay.innerHTML = '';
        return;
    }

    // Calculate actual rendered dimensions accounting for object-fit: contain
    let renderedWidth, renderedHeight, offsetX, offsetY;
    if (img) {
        const dims = getObjectFitContainDimensions(img);
        renderedWidth = dims.renderedWidth;
        renderedHeight = dims.renderedHeight;
        offsetX = dims.offsetX;
        offsetY = dims.offsetY;
        // Use container dimensions for the overlay size
        overlay.setAttribute('width', dims.containerWidth);
        overlay.setAttribute('height', dims.containerHeight);
        overlay.style.width = dims.containerWidth + 'px';
        overlay.style.height = dims.containerHeight + 'px';
    } else {
        // Fallback for backward compatibility
        if (displayWidth === 0 || displayHeight === 0) {
            if (overlay) overlay.innerHTML = '';
            return;
        }
        renderedWidth = displayWidth;
        renderedHeight = displayHeight;
        offsetX = 0;
        offsetY = 0;
        overlay.setAttribute('width', displayWidth);
        overlay.setAttribute('height', displayHeight);
    }

    const scaleX = renderedWidth / imgSizePanel.width;
    const scaleY = renderedHeight / imgSizePanel.height;

    let svg = '';

    // Debug mode: draw all bboxes (with offset for object-fit:contain letterboxing)
    if (panelDebugMode && bboxes) {
        svg += drawPanelDebugBboxes(bboxes, scaleX, scaleY, offsetX, offsetY);
    }

    function drawPanelElement(elementName, type) {
        const bbox = bboxes[elementName];
        if (!bbox) return '';

        const elementType = bbox.element_type || '';
        const hasPoints = bbox.points && bbox.points.length > 0;

        // Lines - draw as path (with offset)
        if ((elementType === 'line' || elementName.includes('trace_')) && hasPoints) {
            if (bbox.points.length < 2) return '';
            const points = bbox.points.filter(pt => Array.isArray(pt) && pt.length >= 2);
            if (points.length < 2) return '';

            let pathD = `M ${points[0][0] * scaleX + offsetX} ${points[0][1] * scaleY + offsetY}`;
            for (let i = 1; i < points.length; i++) {
                pathD += ` L ${points[i][0] * scaleX + offsetX} ${points[i][1] * scaleY + offsetY}`;
            }

            const className = type === 'hover' ? 'hover-path' : 'selected-path';
            return `<path class="${className}" d="${pathD}"/>`;
        }
        // Scatter - draw as circles (with offset)
        else if (elementType === 'scatter' && hasPoints) {
            const className = type === 'hover' ? 'hover-scatter' : 'selected-scatter';
            let result = '';
            for (const pt of bbox.points) {
                if (!Array.isArray(pt) || pt.length < 2) continue;
                result += `<circle class="${className}" cx="${pt[0] * scaleX + offsetX}" cy="${pt[1] * scaleY + offsetY}" r="3"/>`;
            }
            return result;
        }
        // Default - draw bbox rectangle (with offset)
        else {
            const rectClass = type === 'hover' ? 'hover-rect' : 'selected-rect';
            const x = bbox.x0 * scaleX + offsetX - 1;
            const y = bbox.y0 * scaleY + offsetY - 1;
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

// Draw all bboxes for a panel in debug mode (with offset for object-fit:contain)
function drawPanelDebugBboxes(bboxes, scaleX, scaleY, offsetX, offsetY) {
    let svg = '';
    let count = 0;
    // Default offset to 0 if not provided
    offsetX = offsetX || 0;
    offsetY = offsetY || 0;

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

        // Draw bbox rectangle (with offset for object-fit:contain letterboxing)
        const x = bbox.x0 * scaleX + offsetX;
        const y = bbox.y0 * scaleY + offsetY;
        const w = (bbox.x1 - bbox.x0) * scaleX;
        const h = (bbox.y1 - bbox.y0) * scaleY;

        svg += `<rect class="${rectClass}" x="${x}" y="${y}" width="${w}" height="${h}"/>`;

        // Draw short label (truncated for small panels)
        const shortName = name.length > 10 ? name.substring(0, 8) + '..' : name;
        svg += `<text class="debug-label" x="${x + 1}" y="${y + 8}" style="font-size: 6px;">${shortName}</text>`;

        // Draw path points if available (with offset)
        if (hasPoints && bbox.points.length > 1) {
            let pathD = `M ${bbox.points[0][0] * scaleX + offsetX} ${bbox.points[0][1] * scaleY + offsetY}`;
            for (let i = 1; i < bbox.points.length; i++) {
                const pt = bbox.points[i];
                if (pt && pt.length >= 2) {
                    pathD += ` L ${pt[0] * scaleX + offsetX} ${pt[1] * scaleY + offsetY}`;
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

        // Keep unified canvas view only - don't show single-panel preview

        // Update panel path display in right panel header
        const panelPathEl = document.getElementById('panel-path-display');
        if (panelPathEl) {
            panelPathEl.textContent = `Panel: ${panelName}.plot/spec.json`;
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
    if (gridSection) {
        gridSection.style.display = showingPanelGrid ? 'block' : 'none';
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

        setStatus(`Switched to Panel ${data.panel_name.replace('.plot', '')}`, false);
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

    // Update indicator text (if elements exist)
    const indicatorEl = document.getElementById('panel-indicator');
    if (indicatorEl) indicatorEl.textContent = `${current} / ${total}`;

    const nameEl = document.getElementById('current-panel-name');
    if (nameEl) nameEl.textContent = `Panel ${panelName.replace('.plot', '')}`;
}

// =============================================================================
// Canvas Mode (Draggable Panel Layout)
// =============================================================================
let canvasMode = 'grid';  // 'grid' or 'canvas'
let panelPositions = {};  // Store panel positions {name: {x, y, width, height}} in pixels
let panelLayoutMm = {};   // Store panel positions in mm for saving {name: {x_mm, y_mm, width_mm, height_mm}}
let canvasScale = 3;      // Scale factor: pixels per mm (updated in loadPanelGrid)
let draggedPanel = null;
let dragOffset = {x: 0, y: 0};
let layoutModified = false;  // Track if layout has been modified

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

// Check if an element is interactive (should not initiate drag)
function isInteractiveElement(target) {
    // SVG paths with hover-path class are interactive elements
    if (target.classList && target.classList.contains('hover-path')) return true;
    if (target.classList && target.classList.contains('hit-path')) return true;

    // Check parent elements for hover-path (click might be on child)
    let el = target;
    while (el && el !== document.body) {
        if (el.tagName === 'path' || el.tagName === 'PATH') {
            // Path elements in SVG overlay are interactive
            const svg = el.closest('svg');
            if (svg && svg.classList.contains('panel-card-overlay')) {
                return true;
            }
        }
        el = el.parentElement;
    }
    return false;
}

// =============================================================================
// Element Dragging (Legends, Panel Letters)
// =============================================================================
let elementDragState = null;  // {element, panelName, startPos, elementType, axId}

// Snap positions for draggable elements (normalized axes coordinates 0-1)
const SNAP_POSITIONS = {
    'upper left':    {x: 0.02, y: 0.98},
    'upper center':  {x: 0.50, y: 0.98},
    'upper right':   {x: 0.98, y: 0.98},
    'center left':   {x: 0.02, y: 0.50},
    'center':        {x: 0.50, y: 0.50},
    'center right':  {x: 0.98, y: 0.50},
    'lower left':    {x: 0.02, y: 0.02},
    'lower center':  {x: 0.50, y: 0.02},
    'lower right':   {x: 0.98, y: 0.02},
};

// Check if an element is draggable
// ONLY panel letters and legends are draggable to maintain scientific rigor
// Data elements (lines, scatter, bars, etc.) must NOT be movable
function isDraggableElement(elementName, bboxes) {
    if (!elementName || !bboxes) return false;

    // Whitelist: ONLY these element types are draggable
    const DRAGGABLE_TYPES = ['legend', 'panel_letter'];

    // Check by element_type in bbox info
    const info = bboxes[elementName];
    if (info && DRAGGABLE_TYPES.includes(info.element_type)) {
        return true;
    }

    // Check by naming convention (strict match)
    if (elementName.match(/_legend$/)) return true;
    if (elementName.match(/_panel_letter_[A-Z]$/)) return true;

    // Everything else is NOT draggable (data integrity)
    return false;
}

// Start element drag (for legends and panel letters)
function startElementDrag(e, elementName, panelName, img, bboxes) {
    const info = bboxes[elementName] || {};
    const elementType = info.element_type || (elementName.includes('legend') ? 'legend' : 'panel_letter');

    // Extract ax_id from element name (e.g., "ax_00_legend" -> "ax_00")
    const axId = elementName.split('_').slice(0, 2).join('_');

    // Get axes bbox for constraining drag
    const axesBbox = bboxes[`${axId}_panel`] || null;

    elementDragState = {
        element: elementName,
        panelName: panelName,
        elementType: elementType,
        axId: axId,
        axesBbox: axesBbox,
        bboxes: bboxes,
        img: img,
        startMouseX: e.clientX,
        startMouseY: e.clientY,
        startBbox: {...info},
    };

    // Show snap guide overlay
    showSnapGuides(img, axesBbox, bboxes);

    document.addEventListener('mousemove', onElementDrag);
    document.addEventListener('mouseup', stopElementDrag);

    e.preventDefault();
    e.stopPropagation();
}

// Handle element drag movement
function onElementDrag(e) {
    if (!elementDragState) return;

    const {img, bboxes, element, axId, axesBbox, startBbox, startMouseX, startMouseY} = elementDragState;
    if (!img) return;

    const rect = img.getBoundingClientRect();
    const dims = getObjectFitContainDimensions(img);

    // Calculate delta in image coordinates
    const deltaX = e.clientX - startMouseX;
    const deltaY = e.clientY - startMouseY;

    // Convert to image pixel coordinates
    const scaleX = dims.renderedWidth / rect.width;
    const scaleY = dims.renderedHeight / rect.height;
    const imgDeltaX = deltaX * scaleX * (bboxes._meta?.imgSize?.width || 1) / dims.renderedWidth;
    const imgDeltaY = deltaY * scaleY * (bboxes._meta?.imgSize?.height || 1) / dims.renderedHeight;

    // Update bbox position (for visual feedback)
    if (bboxes[element]) {
        const newX0 = startBbox.x0 + imgDeltaX;
        const newY0 = startBbox.y0 + imgDeltaY;
        bboxes[element].x0 = newX0;
        bboxes[element].y0 = newY0;
        bboxes[element].x1 = newX0 + (startBbox.x1 - startBbox.x0);
        bboxes[element].y1 = newY0 + (startBbox.y1 - startBbox.y0);
    }

    // Calculate normalized axes position (0-1)
    if (axesBbox) {
        const axesWidth = axesBbox.x1 - axesBbox.x0;
        const axesHeight = axesBbox.y1 - axesBbox.y0;
        const elemCenterX = (bboxes[element].x0 + bboxes[element].x1) / 2;
        const elemCenterY = (bboxes[element].y0 + bboxes[element].y1) / 2;
        const normX = (elemCenterX - axesBbox.x0) / axesWidth;
        const normY = 1 - (elemCenterY - axesBbox.y0) / axesHeight;  // Flip Y

        // Update snap guide highlighting
        updateSnapHighlight(normX, normY);

        // Show position indicator
        showElementPositionIndicator(element, normX, normY);
    }

    // Redraw overlay
    const overlay = img.parentElement?.querySelector('svg.panel-card-overlay');
    if (overlay) {
        const panelCache = panelBboxesCache[elementDragState.panelName];
        if (panelCache) {
            updatePanelOverlay(overlay, bboxes, panelCache.imgSize, rect.width, rect.height, element, element, img);
        }
    }
}

// Stop element drag and save position
function stopElementDrag() {
    if (!elementDragState) return;

    const {element, panelName, elementType, axId, bboxes, axesBbox} = elementDragState;

    // Calculate final normalized position
    let finalPosition = null;
    let snapName = null;

    if (axesBbox && bboxes[element]) {
        const axesWidth = axesBbox.x1 - axesBbox.x0;
        const axesHeight = axesBbox.y1 - axesBbox.y0;
        const elemCenterX = (bboxes[element].x0 + bboxes[element].x1) / 2;
        const elemCenterY = (bboxes[element].y0 + bboxes[element].y1) / 2;
        const normX = (elemCenterX - axesBbox.x0) / axesWidth;
        const normY = 1 - (elemCenterY - axesBbox.y0) / axesHeight;

        // Check for snap to named position
        snapName = findNearestSnapPosition(normX, normY);
        finalPosition = snapName ? SNAP_POSITIONS[snapName] : {x: normX, y: normY};
    }

    // Hide snap guides
    hideSnapGuides();
    hideElementPositionIndicator();

    // Save position to server
    if (finalPosition) {
        saveElementPosition(element, panelName, elementType, finalPosition, snapName);
    }

    document.removeEventListener('mousemove', onElementDrag);
    document.removeEventListener('mouseup', stopElementDrag);
    elementDragState = null;
}

// Find nearest snap position if within threshold
function findNearestSnapPosition(normX, normY, threshold = 0.08) {
    let nearest = null;
    let minDist = Infinity;

    for (const [name, pos] of Object.entries(SNAP_POSITIONS)) {
        const dist = Math.sqrt(Math.pow(normX - pos.x, 2) + Math.pow(normY - pos.y, 2));
        if (dist < threshold && dist < minDist) {
            minDist = dist;
            nearest = name;
        }
    }
    return nearest;
}

// Show snap guide overlay on axes
function showSnapGuides(img, axesBbox, bboxes) {
    if (!img || !axesBbox) return;

    const container = img.parentElement;
    if (!container) return;

    // Remove existing guides
    container.querySelectorAll('.snap-guide').forEach(el => el.remove());

    const rect = img.getBoundingClientRect();
    const dims = getObjectFitContainDimensions(img);
    const imgSize = bboxes._meta?.imgSize || {width: dims.renderedWidth, height: dims.renderedHeight};

    // Scale factors
    const scaleX = dims.renderedWidth / imgSize.width;
    const scaleY = dims.renderedHeight / imgSize.height;

    // Create snap points
    for (const [name, pos] of Object.entries(SNAP_POSITIONS)) {
        const axesWidth = axesBbox.x1 - axesBbox.x0;
        const axesHeight = axesBbox.y1 - axesBbox.y0;

        // Calculate pixel position
        const imgX = axesBbox.x0 + pos.x * axesWidth;
        const imgY = axesBbox.y0 + (1 - pos.y) * axesHeight;

        const displayX = dims.offsetX + imgX * scaleX;
        const displayY = dims.offsetY + imgY * scaleY;

        const guide = document.createElement('div');
        guide.className = 'snap-guide';
        guide.dataset.snapName = name;
        guide.style.cssText = `
            position: absolute;
            left: ${displayX - 6}px;
            top: ${displayY - 6}px;
            width: 12px;
            height: 12px;
            border: 2px dashed rgba(100, 149, 237, 0.6);
            border-radius: 50%;
            pointer-events: none;
            z-index: 50;
            transition: all 0.15s ease;
        `;
        container.appendChild(guide);
    }
}

// Highlight snap position when near
function updateSnapHighlight(normX, normY) {
    const threshold = 0.08;
    document.querySelectorAll('.snap-guide').forEach(guide => {
        const name = guide.dataset.snapName;
        const pos = SNAP_POSITIONS[name];
        const dist = Math.sqrt(Math.pow(normX - pos.x, 2) + Math.pow(normY - pos.y, 2));
        if (dist < threshold) {
            guide.style.borderColor = 'rgba(76, 175, 80, 0.9)';
            guide.style.backgroundColor = 'rgba(76, 175, 80, 0.3)';
            guide.style.transform = 'scale(1.5)';
        } else {
            guide.style.borderColor = 'rgba(100, 149, 237, 0.6)';
            guide.style.backgroundColor = 'transparent';
            guide.style.transform = 'scale(1)';
        }
    });
}

// Hide snap guides
function hideSnapGuides() {
    document.querySelectorAll('.snap-guide').forEach(el => el.remove());
}

// Show position indicator while dragging element
function showElementPositionIndicator(element, normX, normY) {
    let indicator = document.getElementById('element-pos-indicator');
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = 'element-pos-indicator';
        indicator.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: #4fc3f7;
            padding: 8px 12px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            z-index: 1000;
        `;
        document.body.appendChild(indicator);
    }
    const snapName = findNearestSnapPosition(normX, normY);
    if (snapName) {
        indicator.innerHTML = `Position: <b>${snapName}</b>`;
        indicator.style.color = '#4caf50';
    } else {
        indicator.innerHTML = `Position: (${normX.toFixed(2)}, ${normY.toFixed(2)})`;
        indicator.style.color = '#4fc3f7';
    }
    indicator.style.display = 'block';
}

// Hide position indicator
function hideElementPositionIndicator() {
    const indicator = document.getElementById('element-pos-indicator');
    if (indicator) indicator.style.display = 'none';
}

// Save element position to server
async function saveElementPosition(element, panelName, elementType, position, snapName) {
    try {
        const response = await fetch('/save_element_position', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                element: element,
                panel: panelName,
                element_type: elementType,
                position: position,
                snap_name: snapName,
            }),
        });
        const data = await response.json();
        if (data.success) {
            setStatus(`Saved ${elementType} position: ${snapName || `(${position.x.toFixed(2)}, ${position.y.toFixed(2)})`}`, false);
        } else {
            setStatus(`Failed to save position: ${data.error}`, true);
        }
    } catch (err) {
        console.error('Error saving element position:', err);
        setStatus('Error saving position', true);
    }
}

// Initialize drag handler for a panel item
function initPanelDrag(item, panelName) {
    const dragHandle = item.querySelector('.panel-drag-handle');

    // Drag from handle (always works)
    if (dragHandle) {
        dragHandle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            startPanelDrag(e, item, panelName);
        });
    }

    // Also allow dragging from panel label
    const label = item.querySelector('.panel-canvas-label');
    if (label) {
        label.style.cursor = 'move';
        label.addEventListener('mousedown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            startPanelDrag(e, item, panelName);
        });
    }

    // Allow dragging from anywhere on the panel (except interactive elements)
    // This enables intuitive drag behavior while preserving element selection
    item.addEventListener('mousedown', (e) => {
        // Skip if clicking on interactive elements (legends, text paths, etc.)
        if (isInteractiveElement(e.target)) return;

        // Skip if clicking on drag handle or label (already handled above)
        if (e.target.closest('.panel-drag-handle')) return;
        if (e.target.closest('.panel-canvas-label')) return;
        if (e.target.closest('.panel-position-indicator')) return;

        // Start drag from anywhere else on the panel
        e.preventDefault();
        startPanelDrag(e, item, panelName);
    });

    // Set cursor to indicate draggability
    item.style.cursor = 'grab';
}

function startPanelDrag(e, item, name) {
    e.preventDefault();

    // Handle selection based on Ctrl key
    const isCtrlPressed = e.ctrlKey || e.metaKey;
    const wasAlreadySelected = item.classList.contains('active');

    if (isCtrlPressed) {
        // Ctrl+Click: toggle this panel's selection
        item.classList.toggle('active');
    } else if (!wasAlreadySelected) {
        // Regular click on unselected panel: select only this one
        deselectAllPanels();
        item.classList.add('active');
    }
    // If clicking on already-selected panel without Ctrl:
    // Don't change selection yet - could be start of multi-panel drag
    // Selection will be finalized in stopPanelDrag based on hasMoved

    // Collect all selected panels for group dragging
    const selectedPanels = Array.from(document.querySelectorAll('.panel-canvas-item.active'));
    if (selectedPanels.length === 0) {
        // If somehow nothing selected, select the clicked item
        item.classList.add('active');
        selectedPanels.push(item);
    }

    // Store drag state for all selected panels
    draggedPanel = {
        item,
        name,
        hasMoved: false,  // Track if actual drag occurred
        wasAlreadySelected,  // Track initial selection state for click handling
        isCtrlPressed,  // Track if Ctrl was pressed
        selectedPanels: selectedPanels.map(p => ({
            item: p,
            name: p.dataset.panelName,
            startLeft: parseFloat(p.style.left) || 0,
            startTop: parseFloat(p.style.top) || 0
        }))
    };
    dragOffset.x = e.clientX;
    dragOffset.y = e.clientY;

    selectedPanels.forEach(p => {
        p.classList.add('dragging');
        p.style.cursor = 'grabbing';
    });

    // Show position indicator for primary panel
    updatePositionIndicator(name, item.offsetLeft, item.offsetTop);

    document.addEventListener('mousemove', onPanelDrag);
    document.addEventListener('mouseup', stopPanelDrag);
}

function onPanelDrag(e) {
    if (!draggedPanel || !draggedPanel.selectedPanels) return;
    const canvasEl = document.getElementById('panel-canvas');

    // Calculate delta from drag start
    let deltaX = e.clientX - dragOffset.x;
    let deltaY = e.clientY - dragOffset.y;

    // Mark as moved if we've actually dragged (threshold: 3px)
    if (Math.abs(deltaX) > 3 || Math.abs(deltaY) > 3) {
        draggedPanel.hasMoved = true;
    }

    // Snap to grid (optional: 5mm grid)
    const gridSnap = 5 * canvasScale;  // 5mm in pixels
    if (e.shiftKey) {
        deltaX = Math.round(deltaX / gridSnap) * gridSnap;
        deltaY = Math.round(deltaY / gridSnap) * gridSnap;
    }

    // Move all selected panels by the same delta
    for (const panelInfo of draggedPanel.selectedPanels) {
        let newX = panelInfo.startLeft + deltaX;
        let newY = panelInfo.startTop + deltaY;

        // Constrain to canvas bounds (allow slight negative for edge alignment)
        newX = Math.max(-5, Math.min(newX, canvasEl.offsetWidth - panelInfo.item.offsetWidth + 5));
        newY = Math.max(-5, newY);

        panelInfo.item.style.left = newX + 'px';
        panelInfo.item.style.top = newY + 'px';

        // Update pixel positions
        if (panelPositions[panelInfo.name]) {
            panelPositions[panelInfo.name].x = newX;
            panelPositions[panelInfo.name].y = newY;
        }

        // Update mm positions
        if (panelLayoutMm[panelInfo.name]) {
            panelLayoutMm[panelInfo.name].x_mm = newX / canvasScale;
            panelLayoutMm[panelInfo.name].y_mm = newY / canvasScale;
        }
    }

    // Show position indicator for primary panel
    const primaryNewX = draggedPanel.selectedPanels[0].startLeft + deltaX;
    const primaryNewY = draggedPanel.selectedPanels[0].startTop + deltaY;
    updatePositionIndicator(draggedPanel.name, primaryNewX, primaryNewY);

    // Mark layout as modified
    layoutModified = true;
}

function stopPanelDrag() {
    if (draggedPanel) {
        // Handle click (no movement) on already-selected panel without Ctrl:
        // Finalize selection to only the clicked panel
        if (!draggedPanel.hasMoved && draggedPanel.wasAlreadySelected && !draggedPanel.isCtrlPressed) {
            // This was a simple click on an already-selected panel
            // Deselect all others, keep only the clicked panel selected
            deselectAllPanels();
            draggedPanel.item.classList.add('active');
        }

        // Reset cursor for all selected panels
        if (draggedPanel.selectedPanels) {
            draggedPanel.selectedPanels.forEach(p => {
                p.item.classList.remove('dragging');
                p.item.style.cursor = 'grab';
            });
        } else {
            draggedPanel.item.classList.remove('dragging');
            draggedPanel.item.style.cursor = 'grab';
        }

        // Update canvas size if panel moved outside
        updateCanvasSize();

        // Hide position indicator after a delay
        const name = draggedPanel.name;
        setTimeout(() => {
            const indicator = document.getElementById(`pos-${name}`);
            if (indicator) indicator.style.opacity = '0';
        }, 1500);

        // Auto-save layout
        if (layoutModified) {
            autoSaveLayout();
        }

        draggedPanel = null;
    }
    document.removeEventListener('mousemove', onPanelDrag);
    document.removeEventListener('mouseup', stopPanelDrag);
}

// Update position indicator showing mm coordinates
function updatePositionIndicator(panelName, x, y) {
    const indicator = document.getElementById(`pos-${panelName}`);
    if (!indicator) return;

    const x_mm = (x / canvasScale).toFixed(1);
    const y_mm = (y / canvasScale).toFixed(1);
    indicator.textContent = `${x_mm}, ${y_mm} mm`;
    indicator.style.opacity = '1';
}

// Update canvas size to fit all panels after drag
function updateCanvasSize() {
    const canvasEl = document.getElementById('panel-canvas');
    if (!canvasEl) return;

    const maxY = Math.max(...Object.values(panelPositions).map(p => p.y + p.height)) + 20;
    const maxX = Math.max(...Object.values(panelPositions).map(p => p.x + p.width)) + 20;
    canvasEl.style.minHeight = Math.max(400, maxY) + 'px';
    canvasEl.style.minWidth = Math.max(700, maxX) + 'px';
}

// Auto-save layout to server
async function autoSaveLayout() {
    if (!layoutModified) return;

    try {
        const resp = await fetch('/save_layout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ layout: panelLayoutMm })
        });
        const data = await resp.json();

        if (data.success) {
            layoutModified = false;
            setStatus('Layout saved', false);
            console.log('Layout auto-saved:', panelLayoutMm);
        } else {
            console.error('Layout save failed:', data.error);
            setStatus('Layout save failed: ' + data.error, true);
        }
    } catch (e) {
        console.error('Error saving layout:', e);
        setStatus('Error saving layout', true);
    }
}

// Manual save layout button handler
function saveLayoutManually() {
    layoutModified = true;  // Force save
    autoSaveLayout();
}

// Legacy drag functions (kept for backward compatibility with canvas mode)
function startDrag(e, item, name) {
    startPanelDrag(e, item, name);
}

function onDrag(e) {
    onPanelDrag(e);
}

function stopDrag() {
    stopPanelDrag();
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

            // Also export to bundle (png and svg)
            try {
                await fetch('/export', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({formats: ['png', 'svg']})
                });
                setStatus('Saved and exported to bundle', false);
            } catch (exportErr) {
                console.warn('Export failed:', exportErr);
            }
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

// Download menu toggle
function toggleDownloadMenu() {
    const menu = document.getElementById('download-menu');
    if (menu.style.display === 'none') {
        menu.style.display = 'block';
        // Close when clicking outside
        setTimeout(() => {
            document.addEventListener('click', closeDownloadMenuOnClickOutside);
        }, 10);
    } else {
        menu.style.display = 'none';
    }
}

function closeDownloadMenuOnClickOutside(e) {
    const menu = document.getElementById('download-menu');
    const btn = document.getElementById('download-btn');
    if (!menu.contains(e.target) && !btn.contains(e.target)) {
        menu.style.display = 'none';
        document.removeEventListener('click', closeDownloadMenuOnClickOutside);
    }
}

// Export figure to bundle and trigger download
async function exportAndDownload(format) {
    setStatus(`Exporting ${format.toUpperCase()}...`, false);
    try {
        // First export to bundle
        await fetch('/export', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({formats: [format]})
        });
        // Then trigger download
        window.location.href = `/download/${format}`;
        setStatus(`Downloaded ${format.toUpperCase()}`, false);
    } catch (e) {
        setStatus('Export error: ' + e.message, true);
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
    const globalOverlay = document.getElementById('global-loading-overlay');
    const localOverlay = document.getElementById('loading-overlay');

    // Show/hide spinner for loading states
    if (msg === 'Updating...' || msg === 'Loading preview...') {
        // Show global overlay (visible for both single and multi-panel views)
        if (globalOverlay) {
            globalOverlay.style.display = 'flex';
            const loadingText = globalOverlay.querySelector('.loading-text');
            if (loadingText) loadingText.textContent = msg;
        }
        // Also show local overlay if visible
        if (localOverlay) localOverlay.style.display = 'flex';
        el.textContent = '';  // Clear status text during loading
    } else {
        // Hide both overlays
        if (globalOverlay) globalOverlay.style.display = 'none';
        if (localOverlay) localOverlay.style.display = 'none';
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

// =============================================================================
// Keyboard Shortcuts (matching SciTeX Cloud vis app)
// =============================================================================
let shortcutMode = null;  // For multi-key shortcuts like Alt+A → L

document.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase();
    const isCtrl = e.ctrlKey || e.metaKey;
    const isShift = e.shiftKey;
    const isAlt = e.altKey;

    // Don't capture shortcuts when typing in inputs
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
        return;
    }

    // =========================================================================
    // Multi-key shortcut mode (Alt+A → alignment, Alt+Shift+A → axis alignment)
    // =========================================================================
    if (shortcutMode === 'align') {
        e.preventDefault();
        handleAlignShortcut(key, isShift);
        shortcutMode = null;
        return;
    }

    if (shortcutMode === 'alignByAxis') {
        e.preventDefault();
        handleAlignByAxisShortcut(key);
        shortcutMode = null;
        return;
    }

    // =========================================================================
    // Basic Operations
    // =========================================================================

    // Ctrl+S: Save
    if (isCtrl && key === 's') {
        e.preventDefault();
        saveManual();
        return;
    }

    // Ctrl+Z: Undo
    if (isCtrl && !isShift && key === 'z') {
        e.preventDefault();
        undoLastChange();
        return;
    }

    // Ctrl+Y or Ctrl+Shift+Z: Redo
    if ((isCtrl && key === 'y') || (isCtrl && isShift && key === 'z')) {
        e.preventDefault();
        redoLastChange();
        return;
    }

    // Delete: Remove selected element override
    if (key === 'delete' || key === 'backspace') {
        if (selectedElement && !isCtrl) {
            e.preventDefault();
            deleteSelectedOverride();
            return;
        }
    }

    // =========================================================================
    // Panel/Element Movement (Arrow keys)
    // =========================================================================

    // Arrow keys: Move selected panel by 1mm (or 5mm with Shift)
    if (['arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(key)) {
        e.preventDefault();
        const amount = isShift ? 5 : 1;  // 5mm or 1mm
        moveSelectedPanel(key.replace('arrow', ''), amount);
        return;
    }

    // =========================================================================
    // View Controls
    // =========================================================================

    // + or =: Zoom in
    if ((key === '+' || key === '=') && !isCtrl) {
        e.preventDefault();
        zoomCanvas(1.1);
        return;
    }

    // -: Zoom out
    if (key === '-' && !isCtrl) {
        e.preventDefault();
        zoomCanvas(0.9);
        return;
    }

    // 0: Fit to window
    if (key === '0' && !isCtrl) {
        e.preventDefault();
        fitCanvasToWindow();
        return;
    }

    // Ctrl++ : Increase canvas size
    if (isCtrl && (key === '+' || key === '=')) {
        e.preventDefault();
        resizeCanvas(1.1);
        return;
    }

    // Ctrl+- : Decrease canvas size
    if (isCtrl && key === '-') {
        e.preventDefault();
        resizeCanvas(0.9);
        return;
    }

    // =========================================================================
    // Alignment Modes (Alt+A → basic, Alt+Shift+A → by axis)
    // =========================================================================
    if (isAlt && isShift && key === 'a') {
        // Alt+Shift+A: Align by Axis (scientific alignment based on plot axes)
        e.preventDefault();
        shortcutMode = 'alignByAxis';
        setStatus('Align by Axis: L=Y-Axis(left) R=Right T=Top B=X-Axis(bottom) C=Center-H M=Center-V S=Stack', false);
        setTimeout(() => {
            if (shortcutMode === 'alignByAxis') {
                shortcutMode = null;
                setStatus('Ready', false);
            }
        }, 3000);
        return;
    }
    if (isAlt && !isShift && key === 'a') {
        // Alt+A: Basic alignment (by bounding box)
        e.preventDefault();
        shortcutMode = 'align';
        setStatus('Alignment mode: L=Left R=Right T=Top B=Bottom C=Center H=DistH V=DistV', false);
        setTimeout(() => {
            if (shortcutMode === 'align') {
                shortcutMode = null;
                setStatus('Ready', false);
            }
        }, 3000);
        return;
    }

    // =========================================================================
    // Arrange (Alt+F, Alt+B)
    // =========================================================================
    if (isAlt && key === 'f') {
        e.preventDefault();
        bringPanelToFront();
        return;
    }
    if (isAlt && key === 'b') {
        e.preventDefault();
        sendPanelToBack();
        return;
    }

    // =========================================================================
    // Escape: Deselect/Cancel mode
    // =========================================================================
    if (key === 'escape') {
        e.preventDefault();
        shortcutMode = null;
        deselectAllPanels();
        setStatus('Ready', false);
        return;
    }

    // =========================================================================
    // G: Toggle grid visibility
    // =========================================================================
    if (key === 'g' && !isCtrl && !isAlt) {
        e.preventDefault();
        toggleGridVisibility();
        return;
    }

    // =========================================================================
    // Ctrl+A: Select all panels
    // =========================================================================
    if (isCtrl && key === 'a') {
        e.preventDefault();
        selectAllPanels();
        return;
    }

    // =========================================================================
    // Help (? or F1)
    // =========================================================================
    if (key === '?' || key === 'f1') {
        e.preventDefault();
        showShortcutHelp();
        return;
    }
});

// Handle alignment sub-shortcuts (basic bounding box alignment)
function handleAlignShortcut(key, isShift) {
    const panels = document.querySelectorAll('.panel-canvas-item');
    if (panels.length < 2) {
        setStatus('Need multiple panels for alignment', true);
        return;
    }

    switch(key) {
        case 'l': alignPanels('left'); break;
        case 'r': alignPanels('right'); break;
        case 't': alignPanels('top'); break;
        case 'b': alignPanels('bottom'); break;
        case 'c': alignPanels('center-h'); break;
        case 'm': alignPanels('center-v'); break;
        case 'h': distributePanels('horizontal'); break;
        case 'v': distributePanels('vertical'); break;
        default:
            setStatus('Unknown alignment key: ' + key, true);
    }
}

// Handle axis-based alignment sub-shortcuts (scientific plot alignment)
function handleAlignByAxisShortcut(key) {
    const panels = document.querySelectorAll('.panel-canvas-item');
    if (panels.length < 2) {
        setStatus('Need multiple panels for axis alignment', true);
        return;
    }

    const dirNames = {
        'l': 'Y-axis (left edge)',
        'r': 'Right edge',
        't': 'Top edge',
        'b': 'X-axis (bottom edge)',
        'c': 'Center horizontal',
        'm': 'Center vertical',
        's': 'Stacked vertically'
    };

    switch(key) {
        case 'l': alignPanelsByAxis('left'); break;      // Y-axis left
        case 'r': alignPanelsByAxis('right'); break;     // Right edge
        case 't': alignPanelsByAxis('top'); break;       // Top edge
        case 'b': alignPanelsByAxis('bottom'); break;    // X-axis bottom
        case 'c': alignPanelsByAxis('center-h'); break;  // Horizontal center
        case 'm': alignPanelsByAxis('center-v'); break;  // Vertical center
        case 's': stackPanelsVertically(); break;        // Stack with Y-axis alignment
        default:
            setStatus('Unknown axis key: ' + key + '. Use L/R/T/B/C/M/S', true);
            return;
    }
    if (dirNames[key]) {
        setStatus(`Aligned by axis: ${dirNames[key]}`, false);
    }
}

// Get axes bounding box from panel's cached bboxes
// Returns {x0, y0, x1, y1} in image pixels, or null if not found
function getAxesBboxForPanel(panelName) {
    const cache = panelBboxesCache[panelName];
    if (!cache || !cache.bboxes) return null;

    // Look for ax_00_panel, ax_01_panel, etc.
    const bboxes = cache.bboxes;
    for (const key of Object.keys(bboxes)) {
        if (key.endsWith('_panel') && key.startsWith('ax_')) {
            const bbox = bboxes[key];
            if (bbox && bbox.x0 !== undefined) {
                return {
                    x0: bbox.x0,
                    y0: bbox.y0,
                    x1: bbox.x1,
                    y1: bbox.y1,
                    key: key
                };
            }
        }
    }

    // Fallback: check _meta.axes_bbox_px for single-axes plots
    if (bboxes._meta && bboxes._meta.axes_bbox_px) {
        const axBbox = bboxes._meta.axes_bbox_px;
        return {
            x0: axBbox.x0 || axBbox.x,
            y0: axBbox.y0 || axBbox.y,
            x1: axBbox.x1 || (axBbox.x + axBbox.width),
            y1: axBbox.y1 || (axBbox.y + axBbox.height),
            key: '_meta.axes_bbox_px'
        };
    }

    return null;
}

// Calculate panel offset to align by axis edge
// Returns the axis edge position in canvas pixels relative to panel's top-left
function getAxisEdgeOffset(panel, axesBbox, edge, imgSize) {
    if (!axesBbox || !imgSize) return 0;

    // Scale factor from image pixels to displayed panel pixels
    const panelEl = panel;
    const displayWidth = panelEl.offsetWidth;
    const displayHeight = panelEl.offsetHeight;
    const scaleX = displayWidth / imgSize.width;
    const scaleY = displayHeight / imgSize.height;

    switch(edge) {
        case 'left':
            // Y-axis left edge
            return axesBbox.x0 * scaleX;
        case 'right':
            // Right edge of axes
            return axesBbox.x1 * scaleX;
        case 'top':
            // Top edge of axes
            return axesBbox.y0 * scaleY;
        case 'bottom':
            // X-axis bottom edge
            return axesBbox.y1 * scaleY;
        case 'center-h':
            // Horizontal center of axes
            return ((axesBbox.x0 + axesBbox.x1) / 2) * scaleX;
        case 'center-v':
            // Vertical center of axes
            return ((axesBbox.y0 + axesBbox.y1) / 2) * scaleY;
        default:
            return 0;
    }
}

// Align panels by axis edges (scientific alignment for plots)
function alignPanelsByAxis(edge) {
    const panels = Array.from(document.querySelectorAll('.panel-canvas-item'));
    if (panels.length < 2) {
        setStatus('Need multiple panels for axis alignment', true);
        return;
    }

    // Collect panel info with axes bboxes
    const panelInfos = [];
    for (const panel of panels) {
        const panelName = panel.dataset.panelName;
        const cache = panelBboxesCache[panelName];
        const axesBbox = getAxesBboxForPanel(panelName);
        const imgSize = cache ? cache.imgSize : null;

        if (!axesBbox || !imgSize) {
            console.warn(`Panel ${panelName} has no axes bbox data`);
            continue;
        }

        panelInfos.push({
            el: panel,
            name: panelName,
            left: parseFloat(panel.style.left) || 0,
            top: parseFloat(panel.style.top) || 0,
            width: panel.offsetWidth,
            height: panel.offsetHeight,
            axesBbox: axesBbox,
            imgSize: imgSize,
            axisOffset: getAxisEdgeOffset(panel, axesBbox, edge, imgSize)
        });
    }

    if (panelInfos.length < 2) {
        setStatus('Need at least 2 panels with axis data for alignment', true);
        return;
    }

    // Calculate target position - use the first panel's axis position as reference
    const isHorizontal = ['left', 'right', 'center-h'].includes(edge);

    if (isHorizontal) {
        // Align horizontally (match X positions of axis edges)
        // Target = first panel's axis X position in canvas coords
        const refPanel = panelInfos[0];
        const targetAxisX = refPanel.left + refPanel.axisOffset;

        for (const info of panelInfos) {
            const newLeft = targetAxisX - info.axisOffset;
            info.el.style.left = newLeft + 'px';
        }
    } else {
        // Align vertically (match Y positions of axis edges)
        // Target = first panel's axis Y position in canvas coords
        const refPanel = panelInfos[0];
        const targetAxisY = refPanel.top + refPanel.axisOffset;

        for (const info of panelInfos) {
            const newTop = targetAxisY - info.axisOffset;
            info.el.style.top = newTop + 'px';
        }
    }

    // Update layout data
    updatePanelLayoutFromDOM();
    console.log(`Aligned ${panelInfos.length} panels by axis: ${edge}`);
}

// Stack panels vertically with Y-axis alignment
function stackPanelsVertically() {
    const panels = Array.from(document.querySelectorAll('.panel-canvas-item'));
    if (panels.length < 2) {
        setStatus('Need multiple panels for stacking', true);
        return;
    }

    // Collect panel info with axes bboxes
    const panelInfos = [];
    for (const panel of panels) {
        const panelName = panel.dataset.panelName;
        const cache = panelBboxesCache[panelName];
        const axesBbox = getAxesBboxForPanel(panelName);
        const imgSize = cache ? cache.imgSize : null;

        if (!axesBbox || !imgSize) {
            console.warn(`Panel ${panelName} has no axes bbox data`);
            continue;
        }

        panelInfos.push({
            el: panel,
            name: panelName,
            left: parseFloat(panel.style.left) || 0,
            top: parseFloat(panel.style.top) || 0,
            width: panel.offsetWidth,
            height: panel.offsetHeight,
            axesBbox: axesBbox,
            imgSize: imgSize,
            yAxisOffset: getAxisEdgeOffset(panel, axesBbox, 'left', imgSize)
        });
    }

    if (panelInfos.length < 2) {
        setStatus('Need at least 2 panels with axis data for stacking', true);
        return;
    }

    // Sort by current vertical position
    panelInfos.sort((a, b) => a.top - b.top);

    // Use first panel as reference for Y-axis alignment
    const refPanel = panelInfos[0];
    const targetAxisX = refPanel.left + refPanel.yAxisOffset;

    // Stack panels vertically with small gap, aligned by Y-axis
    const gap = 10; // pixels gap between panels
    let currentY = refPanel.top;

    for (let i = 0; i < panelInfos.length; i++) {
        const info = panelInfos[i];

        // Align Y-axis (left edge of axes)
        const newLeft = targetAxisX - info.yAxisOffset;
        info.el.style.left = newLeft + 'px';

        // Stack vertically
        info.el.style.top = currentY + 'px';
        currentY += info.height + gap;
    }

    // Update layout data
    updatePanelLayoutFromDOM();
    setStatus(`Stacked ${panelInfos.length} panels with Y-axis alignment`, false);
}

// Move selected panel(s) by delta in mm
function moveSelectedPanel(direction, amountMm) {
    const selectedPanels = document.querySelectorAll('.panel-canvas-item.active');
    if (selectedPanels.length === 0) {
        setStatus('No panel selected', true);
        return;
    }

    const deltaX = direction === 'left' ? -amountMm : (direction === 'right' ? amountMm : 0);
    const deltaY = direction === 'up' ? -amountMm : (direction === 'down' ? amountMm : 0);

    selectedPanels.forEach(panel => {
        const panelName = panel.dataset.panelName;

        // Update position in pixels (canvasScale = px/mm)
        const currentLeft = parseFloat(panel.style.left) || 0;
        const currentTop = parseFloat(panel.style.top) || 0;

        panel.style.left = (currentLeft + deltaX * canvasScale) + 'px';
        panel.style.top = (currentTop + deltaY * canvasScale) + 'px';

        // Update layout data
        if (panelLayoutMm[panelName]) {
            panelLayoutMm[panelName].x_mm += deltaX;
            panelLayoutMm[panelName].y_mm += deltaY;
            layoutModified = true;
        }
    });

    const count = selectedPanels.length;
    const panelText = count === 1 ? selectedPanels[0].dataset.panelName : `${count} panels`;
    setStatus(`Moved ${panelText} by ${amountMm}mm ${direction}`, false);
}

// Zoom canvas view
let canvasZoom = 1.0;
function zoomCanvas(factor) {
    canvasZoom *= factor;
    canvasZoom = Math.max(0.25, Math.min(4, canvasZoom));  // Limit 25%-400%
    const canvas = document.getElementById('panel-canvas');
    if (canvas) {
        canvas.style.transform = `scale(${canvasZoom})`;
        canvas.style.transformOrigin = 'top left';
    }
    setStatus(`Zoom: ${Math.round(canvasZoom * 100)}%`, false);
}

// Fit canvas to window
function fitCanvasToWindow() {
    canvasZoom = 1.0;
    const canvas = document.getElementById('panel-canvas');
    if (canvas) {
        canvas.style.transform = 'scale(1)';
    }
    setStatus('Fit to window', false);
}

// Resize canvas (actual size, not view)
function resizeCanvas(factor) {
    const canvas = document.getElementById('panel-canvas');
    if (!canvas) return;
    const currentWidth = canvas.offsetWidth;
    const currentHeight = canvas.offsetHeight;
    canvas.style.width = (currentWidth * factor) + 'px';
    canvas.style.minHeight = (currentHeight * factor) + 'px';
    setStatus(`Canvas: ${Math.round(currentWidth * factor)}x${Math.round(currentHeight * factor)}px`, false);
}

// Align panels
function alignPanels(mode) {
    const panels = Array.from(document.querySelectorAll('.panel-canvas-item'));
    if (panels.length < 2) return;

    // Get bounds
    const bounds = panels.map(p => ({
        el: p,
        left: parseFloat(p.style.left) || 0,
        top: parseFloat(p.style.top) || 0,
        width: p.offsetWidth,
        height: p.offsetHeight
    }));

    let targetValue;
    switch(mode) {
        case 'left':
            targetValue = Math.min(...bounds.map(b => b.left));
            bounds.forEach(b => { b.el.style.left = targetValue + 'px'; });
            break;
        case 'right':
            targetValue = Math.max(...bounds.map(b => b.left + b.width));
            bounds.forEach(b => { b.el.style.left = (targetValue - b.width) + 'px'; });
            break;
        case 'top':
            targetValue = Math.min(...bounds.map(b => b.top));
            bounds.forEach(b => { b.el.style.top = targetValue + 'px'; });
            break;
        case 'bottom':
            targetValue = Math.max(...bounds.map(b => b.top + b.height));
            bounds.forEach(b => { b.el.style.top = (targetValue - b.height) + 'px'; });
            break;
        case 'center-h':
            targetValue = bounds.reduce((sum, b) => sum + b.left + b.width/2, 0) / bounds.length;
            bounds.forEach(b => { b.el.style.left = (targetValue - b.width/2) + 'px'; });
            break;
        case 'center-v':
            targetValue = bounds.reduce((sum, b) => sum + b.top + b.height/2, 0) / bounds.length;
            bounds.forEach(b => { b.el.style.top = (targetValue - b.height/2) + 'px'; });
            break;
    }

    // Update layout data
    updatePanelLayoutFromDOM();
    setStatus(`Aligned panels: ${mode}`, false);
}

// Distribute panels evenly
function distributePanels(direction) {
    const panels = Array.from(document.querySelectorAll('.panel-canvas-item'));
    if (panels.length < 3) {
        setStatus('Need at least 3 panels to distribute', true);
        return;
    }

    const bounds = panels.map(p => ({
        el: p,
        left: parseFloat(p.style.left) || 0,
        top: parseFloat(p.style.top) || 0,
        width: p.offsetWidth,
        height: p.offsetHeight
    }));

    if (direction === 'horizontal') {
        bounds.sort((a, b) => a.left - b.left);
        const totalWidth = bounds.reduce((sum, b) => sum + b.width, 0);
        const start = bounds[0].left;
        const end = bounds[bounds.length - 1].left + bounds[bounds.length - 1].width;
        const gap = (end - start - totalWidth) / (bounds.length - 1);

        let currentX = start;
        bounds.forEach(b => {
            b.el.style.left = currentX + 'px';
            currentX += b.width + gap;
        });
    } else {
        bounds.sort((a, b) => a.top - b.top);
        const totalHeight = bounds.reduce((sum, b) => sum + b.height, 0);
        const start = bounds[0].top;
        const end = bounds[bounds.length - 1].top + bounds[bounds.length - 1].height;
        const gap = (end - start - totalHeight) / (bounds.length - 1);

        let currentY = start;
        bounds.forEach(b => {
            b.el.style.top = currentY + 'px';
            currentY += b.height + gap;
        });
    }

    updatePanelLayoutFromDOM();
    setStatus(`Distributed panels: ${direction}`, false);
}

// Update layout data from DOM positions
function updatePanelLayoutFromDOM() {
    document.querySelectorAll('.panel-canvas-item').forEach(panel => {
        const name = panel.dataset.panelName;
        if (panelLayoutMm[name]) {
            panelLayoutMm[name].x_mm = parseFloat(panel.style.left) / canvasScale;
            panelLayoutMm[name].y_mm = parseFloat(panel.style.top) / canvasScale;
        }
    });
    layoutModified = true;
    autoSaveLayout();
}

// Bring selected panel(s) to front
function bringPanelToFront() {
    const selectedPanels = document.querySelectorAll('.panel-canvas-item.active');
    if (selectedPanels.length === 0) return;
    const maxZ = Math.max(...Array.from(document.querySelectorAll('.panel-canvas-item')).map(p => parseInt(p.style.zIndex) || 0));
    selectedPanels.forEach((panel, i) => {
        panel.style.zIndex = maxZ + 1 + i;
    });
    setStatus(`Brought ${selectedPanels.length > 1 ? selectedPanels.length + ' panels' : 'panel'} to front`, false);
}

// Send selected panel(s) to back
function sendPanelToBack() {
    const selectedPanels = document.querySelectorAll('.panel-canvas-item.active');
    if (selectedPanels.length === 0) return;
    const minZ = Math.min(...Array.from(document.querySelectorAll('.panel-canvas-item')).map(p => parseInt(p.style.zIndex) || 0));
    selectedPanels.forEach((panel, i) => {
        panel.style.zIndex = minZ - selectedPanels.length + i;
    });
    setStatus(`Sent ${selectedPanels.length > 1 ? selectedPanels.length + ' panels' : 'panel'} to back`, false);
}

// Deselect all panels
function deselectAllPanels() {
    document.querySelectorAll('.panel-canvas-item.active').forEach(p => {
        p.classList.remove('active');
    });
    // Also clear element selection in single-panel view
    if (typeof selectedElement !== 'undefined') {
        selectedElement = null;
    }
}

// Select all panels
function selectAllPanels() {
    const panels = document.querySelectorAll('.panel-canvas-item');
    panels.forEach(p => p.classList.add('active'));
    setStatus(`Selected ${panels.length} panels`, false);
}

// Toggle grid visibility
let gridVisible = true;
function toggleGridVisibility() {
    gridVisible = !gridVisible;
    const gridElements = document.querySelectorAll('.canvas-grid, .grid-lines, .ruler-marks');
    gridElements.forEach(el => {
        el.style.opacity = gridVisible ? '1' : '0';
    });
    // Also toggle the canvas background grid if using CSS grid
    const canvasContainer = document.querySelector('.panel-canvas, #canvas-container');
    if (canvasContainer) {
        if (gridVisible) {
            canvasContainer.classList.remove('hide-grid');
        } else {
            canvasContainer.classList.add('hide-grid');
        }
    }
    setStatus(gridVisible ? 'Grid visible' : 'Grid hidden', false);
}

// Undo/Redo stacks
let undoStack = [];
let redoStack = [];

function undoLastChange() {
    if (undoStack.length === 0) {
        setStatus('Nothing to undo', true);
        return;
    }
    const state = undoStack.pop();
    redoStack.push(JSON.stringify(overrides));
    overrides = JSON.parse(state);
    updatePreview();
    setStatus('Undo', false);
}

function redoLastChange() {
    if (redoStack.length === 0) {
        setStatus('Nothing to redo', true);
        return;
    }
    const state = redoStack.pop();
    undoStack.push(JSON.stringify(overrides));
    overrides = JSON.parse(state);
    updatePreview();
    setStatus('Redo', false);
}

// Save state for undo before changes
function saveUndoState() {
    undoStack.push(JSON.stringify(overrides));
    if (undoStack.length > 50) undoStack.shift();  // Limit stack size
    redoStack = [];  // Clear redo on new change
}

// Delete selected element override
function deleteSelectedOverride() {
    if (!selectedElement) return;
    saveUndoState();
    if (overrides.element_overrides && overrides.element_overrides[selectedElement]) {
        delete overrides.element_overrides[selectedElement];
        updatePreview();
        setStatus(`Deleted override for ${selectedElement}`, false);
    }
}

// Show keyboard shortcuts help
function showShortcutHelp() {
    const helpHtml = `
    <div id="shortcut-modal" style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.8); z-index: 10000; display: flex; align-items: center; justify-content: center;" onclick="this.remove()">
        <div style="background: var(--bg-secondary); padding: 24px; border-radius: 8px; max-width: 700px; max-height: 80vh; overflow-y: auto; color: var(--text-primary);" onclick="event.stopPropagation()">
            <h2 style="margin-top: 0; border-bottom: 1px solid var(--border-color); padding-bottom: 8px;">⌨️ Keyboard Shortcuts</h2>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h4 style="color: var(--accent-primary);">Basic</h4>
                    <div><kbd>Ctrl+S</kbd> Save</div>
                    <div><kbd>Ctrl+Z</kbd> Undo</div>
                    <div><kbd>Ctrl+Y</kbd> Redo</div>
                    <div><kbd>Del</kbd> Delete override</div>
                    <div><kbd>Esc</kbd> Deselect / Cancel</div>
                    <div><kbd>Ctrl+A</kbd> Select all panels</div>

                    <h4 style="color: var(--accent-primary); margin-top: 16px;">Selection</h4>
                    <div><kbd>Click</kbd> Select panel</div>
                    <div><kbd>Ctrl+Click</kbd> Multi-select</div>
                    <div><kbd>Right-Click</kbd> Context menu</div>

                    <h4 style="color: var(--accent-primary); margin-top: 16px;">Movement</h4>
                    <div><kbd>↑↓←→</kbd> Move panel 1mm</div>
                    <div><kbd>Shift+↑↓←→</kbd> Move panel 5mm</div>
                </div>

                <div>
                    <h4 style="color: var(--accent-primary);">View</h4>
                    <div><kbd>+</kbd> Zoom in</div>
                    <div><kbd>-</kbd> Zoom out</div>
                    <div><kbd>0</kbd> Fit to window</div>
                    <div><kbd>G</kbd> Toggle grid</div>
                    <div><kbd>Ctrl++</kbd> Increase canvas</div>
                    <div><kbd>Ctrl+-</kbd> Decrease canvas</div>

                    <h4 style="color: var(--accent-primary); margin-top: 16px;">Arrange</h4>
                    <div><kbd>Alt+F</kbd> Bring to front</div>
                    <div><kbd>Alt+B</kbd> Send to back</div>
                </div>
            </div>

            <h4 style="color: var(--accent-primary); margin-top: 16px;">Alignment (Alt+A → ...)</h4>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;">
                <div><kbd>L</kbd> Left</div>
                <div><kbd>R</kbd> Right</div>
                <div><kbd>T</kbd> Top</div>
                <div><kbd>B</kbd> Bottom</div>
                <div><kbd>C</kbd> Center H</div>
                <div><kbd>M</kbd> Center V</div>
                <div><kbd>H</kbd> Distribute H</div>
                <div><kbd>V</kbd> Distribute V</div>
            </div>

            <h4 style="color: var(--accent-primary); margin-top: 16px;">Axis Alignment (Alt+Shift+A → ...)</h4>
            <p style="font-size: 0.85em; color: var(--text-muted); margin-top: 4px;">Aligns panels by plot axis edges, not bounding boxes</p>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;">
                <div><kbd>L</kbd> Y-axis (left)</div>
                <div><kbd>R</kbd> Right edge</div>
                <div><kbd>T</kbd> Top edge</div>
                <div><kbd>B</kbd> X-axis (bottom)</div>
                <div><kbd>C</kbd> Axes center H</div>
                <div><kbd>M</kbd> Axes center V</div>
                <div><kbd>S</kbd> Stack vertically</div>
            </div>

            <div style="margin-top: 20px; text-align: center; color: var(--text-muted);">
                Press <kbd>?</kbd> or <kbd>F1</kbd> anytime to show this help
            </div>
        </div>
    </div>`;
    document.body.insertAdjacentHTML('beforeend', helpHtml);
}

// Add kbd styling
const kbdStyle = document.createElement('style');
kbdStyle.textContent = `
    kbd {
        background: var(--bg-tertiary, #333);
        border: 1px solid var(--border-color, #555);
        border-radius: 3px;
        padding: 2px 6px;
        font-family: monospace;
        font-size: 0.85em;
        margin-right: 8px;
    }
`;
document.head.appendChild(kbdStyle);

// =============================================================================
// Right-Click Context Menu
// =============================================================================
let contextMenu = null;

function showContextMenu(e, panelName) {
    e.preventDefault();
    hideContextMenu();

    const selectedCount = document.querySelectorAll('.panel-canvas-item.active').length;
    const hasSelection = selectedCount > 0;

    const menu = document.createElement('div');
    menu.id = 'canvas-context-menu';
    menu.className = 'context-menu';
    menu.innerHTML = `
        <div class="context-menu-item" onclick="selectAllPanels(); hideContextMenu();">
            <span class="context-menu-icon">⬚</span> Select All <span class="context-menu-shortcut">Ctrl+A</span>
        </div>
        <div class="context-menu-item ${!hasSelection ? 'disabled' : ''}" onclick="${hasSelection ? 'deselectAllPanels(); hideContextMenu();' : ''}">
            <span class="context-menu-icon">○</span> Deselect All <span class="context-menu-shortcut">Esc</span>
        </div>
        <div class="context-menu-divider"></div>
        <div class="context-menu-item ${!hasSelection ? 'disabled' : ''}" onclick="${hasSelection ? 'bringPanelToFront(); hideContextMenu();' : ''}">
            <span class="context-menu-icon">↑</span> Bring to Front <span class="context-menu-shortcut">Alt+F</span>
        </div>
        <div class="context-menu-item ${!hasSelection ? 'disabled' : ''}" onclick="${hasSelection ? 'sendPanelToBack(); hideContextMenu();' : ''}">
            <span class="context-menu-icon">↓</span> Send to Back <span class="context-menu-shortcut">Alt+B</span>
        </div>
        <div class="context-menu-divider"></div>
        <div class="context-menu-submenu">
            <div class="context-menu-item">
                <span class="context-menu-icon">≡</span> Align <span class="context-menu-arrow">▶</span>
            </div>
            <div class="context-submenu">
                <div class="context-menu-item ${selectedCount < 2 ? 'disabled' : ''}" onclick="${selectedCount >= 2 ? "alignPanels('left'); hideContextMenu();" : ''}">Left</div>
                <div class="context-menu-item ${selectedCount < 2 ? 'disabled' : ''}" onclick="${selectedCount >= 2 ? "alignPanels('right'); hideContextMenu();" : ''}">Right</div>
                <div class="context-menu-item ${selectedCount < 2 ? 'disabled' : ''}" onclick="${selectedCount >= 2 ? "alignPanels('top'); hideContextMenu();" : ''}">Top</div>
                <div class="context-menu-item ${selectedCount < 2 ? 'disabled' : ''}" onclick="${selectedCount >= 2 ? "alignPanels('bottom'); hideContextMenu();" : ''}">Bottom</div>
                <div class="context-menu-divider"></div>
                <div class="context-menu-item ${selectedCount < 2 ? 'disabled' : ''}" onclick="${selectedCount >= 2 ? "alignPanels('center-h'); hideContextMenu();" : ''}">Center H</div>
                <div class="context-menu-item ${selectedCount < 2 ? 'disabled' : ''}" onclick="${selectedCount >= 2 ? "alignPanels('center-v'); hideContextMenu();" : ''}">Center V</div>
            </div>
        </div>
        <div class="context-menu-submenu">
            <div class="context-menu-item">
                <span class="context-menu-icon">⊞</span> Align by Axis <span class="context-menu-arrow">▶</span>
            </div>
            <div class="context-submenu">
                <div class="context-menu-item ${selectedCount < 2 ? 'disabled' : ''}" onclick="${selectedCount >= 2 ? "alignPanelsByAxis('left'); hideContextMenu();" : ''}">Y-Axis (Left)</div>
                <div class="context-menu-item ${selectedCount < 2 ? 'disabled' : ''}" onclick="${selectedCount >= 2 ? "alignPanelsByAxis('bottom'); hideContextMenu();" : ''}">X-Axis (Bottom)</div>
                <div class="context-menu-divider"></div>
                <div class="context-menu-item ${selectedCount < 2 ? 'disabled' : ''}" onclick="${selectedCount >= 2 ? "stackPanelsVertically(); hideContextMenu();" : ''}">Stack Vertically</div>
            </div>
        </div>
        <div class="context-menu-submenu">
            <div class="context-menu-item">
                <span class="context-menu-icon">⇔</span> Distribute <span class="context-menu-arrow">▶</span>
            </div>
            <div class="context-submenu">
                <div class="context-menu-item ${selectedCount < 2 ? 'disabled' : ''}" onclick="${selectedCount >= 2 ? "distributePanels('horizontal'); hideContextMenu();" : ''}">Horizontal</div>
                <div class="context-menu-item ${selectedCount < 2 ? 'disabled' : ''}" onclick="${selectedCount >= 2 ? "distributePanels('vertical'); hideContextMenu();" : ''}">Vertical</div>
            </div>
        </div>
        <div class="context-menu-divider"></div>
        <div class="context-menu-item" onclick="toggleGridVisibility(); hideContextMenu();">
            <span class="context-menu-icon">⊞</span> Toggle Grid <span class="context-menu-shortcut">G</span>
        </div>
        <div class="context-menu-divider"></div>
        <div class="context-menu-item" onclick="showShortcutHelp(); hideContextMenu();">
            <span class="context-menu-icon">⌨</span> Keyboard Shortcuts <span class="context-menu-shortcut">?</span>
        </div>
    `;

    // Position menu at cursor
    menu.style.left = e.clientX + 'px';
    menu.style.top = e.clientY + 'px';

    document.body.appendChild(menu);
    contextMenu = menu;

    // Adjust position if menu goes off screen
    const rect = menu.getBoundingClientRect();
    if (rect.right > window.innerWidth) {
        menu.style.left = (window.innerWidth - rect.width - 5) + 'px';
    }
    if (rect.bottom > window.innerHeight) {
        menu.style.top = (window.innerHeight - rect.height - 5) + 'px';
    }
}

function hideContextMenu() {
    if (contextMenu) {
        contextMenu.remove();
        contextMenu = null;
    }
}

// Close context menu on click outside
document.addEventListener('click', (e) => {
    if (contextMenu && !contextMenu.contains(e.target)) {
        hideContextMenu();
    }
});

// Close context menu on Escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && contextMenu) {
        hideContextMenu();
    }
});

// Attach context menu to canvas
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('panel-canvas');
    if (canvas) {
        canvas.addEventListener('contextmenu', (e) => {
            // Check if right-click is on a panel
            const panel = e.target.closest('.panel-canvas-item');
            const panelName = panel ? panel.dataset.panelName : null;

            // If clicking on a panel that's not selected, select it
            if (panel && !panel.classList.contains('active')) {
                if (!e.ctrlKey && !e.metaKey) {
                    deselectAllPanels();
                }
                panel.classList.add('active');
            }

            showContextMenu(e, panelName);
        });
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
