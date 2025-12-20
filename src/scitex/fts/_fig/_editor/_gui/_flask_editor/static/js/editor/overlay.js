/**
 * Overlay Rendering
 * Handles SVG overlay for element highlighting and debug visualization
 */

// ============================================================================
// Main Overlay Update
// ============================================================================
function updateOverlay() {
    const overlay = document.getElementById('hover-overlay');
    // Find the visible preview element (SVG wrapper or img)
    const svgWrapper = document.getElementById('preview-svg-wrapper');
    const imgEl = document.getElementById('preview');

    let targetEl = null;
    if (svgWrapper) {
        targetEl = svgWrapper.querySelector('svg') || svgWrapper;
    } else if (imgEl && imgEl.offsetWidth > 0) {
        targetEl = imgEl;
    }

    if (!targetEl) return;

    const rect = targetEl.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    // Guard against zero imgSize (can cause Infinity scale)
    if (!imgSize || !imgSize.width || !imgSize.height || imgSize.width === 0 || imgSize.height === 0) {
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

// ============================================================================
// Draw Path/Line Elements
// ============================================================================
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

// ============================================================================
// Draw Scatter Points
// ============================================================================
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

// ============================================================================
// Debug Visualization
// ============================================================================
function drawDebugBboxes(scaleX, scaleY) {
    let svg = '';
    let count = 0;

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
    }

    return svg;
}

function toggleDebugMode() {
    debugMode = !debugMode;
    const btn = document.getElementById('debug-toggle-btn');
    if (btn) {
        btn.classList.toggle('active', debugMode);
        btn.textContent = debugMode ? 'Hide Hit Areas' : 'Show Hit Areas';
    }
    updateOverlay();
}

// ============================================================================
// Panel Overlay Update (for multi-panel canvas)
// ============================================================================
function updatePanelOverlay(overlay, bboxes, imgSizePanel, displayWidth, displayHeight, hovered, selected, img) {
    if (!overlay) return;

    overlay.innerHTML = '';

    // Calculate actual rendered dimensions accounting for object-fit: contain
    let scaleX, scaleY, offsetX = 0, offsetY = 0;

    if (img && img.naturalWidth && img.naturalHeight) {
        const dims = getObjectFitContainDimensions(img);
        scaleX = dims.displayWidth / imgSizePanel.width;
        scaleY = dims.displayHeight / imgSizePanel.height;
        offsetX = dims.offsetX;
        offsetY = dims.offsetY;

        // Use container dimensions for the overlay size
        overlay.setAttribute('width', img.clientWidth);
        overlay.setAttribute('height', img.clientHeight);
    } else {
        // Fallback for backward compatibility
        scaleX = displayWidth / imgSizePanel.width;
        scaleY = displayHeight / imgSizePanel.height;
        overlay.setAttribute('width', displayWidth);
        overlay.setAttribute('height', displayHeight);
    }

    let svg = '';

    // Debug mode: draw all bboxes (with offset for object-fit:contain letterboxing)
    if (panelDebugMode) {
        svg += drawPanelDebugBboxes(bboxes, scaleX, scaleY, offsetX, offsetY);
    }

    function drawPanelElement(elementName, type) {
        const bbox = bboxes[elementName];
        if (!bbox) return '';

        const elementType = bbox.element_type || '';
        const hasPoints = bbox.points && bbox.points.length > 0;

        // Lines - draw as path (with offset)
        if ((elementType === 'line' || elementName.includes('trace_')) && hasPoints) {
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
            let circles = '';
            for (const pt of bbox.points) {
                if (!Array.isArray(pt) || pt.length < 2) continue;
                const [x, y] = pt;
                circles += `<circle class="${className}" cx="${x * scaleX + offsetX}" cy="${y * scaleY + offsetY}" r="3"/>`;
            }
            return circles;
        }
        // Default - draw bbox rectangle (with offset)
        else {
            const rectClass = type === 'hover' ? 'hover-rect' : 'selected-rect';
            const x = bbox.x0 * scaleX + offsetX;
            const y = bbox.y0 * scaleY + offsetY;
            const w = (bbox.x1 - bbox.x0) * scaleX;
            const h = (bbox.y1 - bbox.y0) * scaleY;
            return `<rect class="${rectClass}" x="${x}" y="${y}" width="${w}" height="${h}" rx="1"/>`;
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

function drawPanelDebugBboxes(bboxes, scaleX, scaleY, offsetX, offsetY) {
    let svg = '';

    // Default offset to 0 if not provided
    offsetX = offsetX || 0;
    offsetY = offsetY || 0;

    for (const [name, bbox] of Object.entries(bboxes)) {
        if (name === '_meta') continue;

        const hasPoints = bbox.points && bbox.points.length > 0;
        const elementType = bbox.element_type || '';

        // Choose color based on element type
        let rectClass = 'debug-rect';
        if (name.includes('trace_') || elementType === 'line') {
            rectClass = 'debug-rect-trace';
        } else if (name.includes('legend')) {
            rectClass = 'debug-rect-legend';
        }

        // Draw bbox rectangle (with offset for object-fit:contain letterboxing)
        const x = bbox.x0 * scaleX + offsetX;
        const y = bbox.y0 * scaleY + offsetY;
        const w = (bbox.x1 - bbox.x0) * scaleX;
        const h = (bbox.y1 - bbox.y0) * scaleY;

        svg += `<rect class="${rectClass}" x="${x}" y="${y}" width="${w}" height="${h}"/>`;

        // Draw short label (truncated for small panels)
        const shortName = name.length > 15 ? name.substring(0, 12) + '...' : name;
        svg += `<text class="debug-label" x="${x + 2}" y="${y + 8}" font-size="8">${shortName}</text>`;

        // Draw path points if available (with offset)
        if (hasPoints && bbox.points.length > 1) {
            let pathD = `M ${bbox.points[0][0] * scaleX + offsetX} ${bbox.points[0][1] * scaleY + offsetY}`;
            for (let i = 1; i < bbox.points.length; i++) {
                const pt = bbox.points[i];
                if (pt && pt.length >= 2) {
                    pathD += ` L ${pt[0] * scaleX + offsetX} ${pt[1] * scaleY + offsetY}`;
                }
            }
            svg += `<path class="debug-path" d="${pathD}" stroke-width="0.5"/>`;
        }
    }

    return svg;
}

function togglePanelDebugMode() {
    panelDebugMode = !panelDebugMode;
    const btn = document.getElementById('panel-debug-toggle-btn');
    if (btn) {
        btn.classList.toggle('active', panelDebugMode);
        btn.textContent = panelDebugMode ? 'Hide Panel Hit Areas' : 'Show Panel Hit Areas';
    }

    // Redraw all panel overlays
    redrawAllPanelOverlays();
}

function redrawAllPanelOverlays() {
    document.querySelectorAll('.panel-canvas-item').forEach((item, idx) => {
        const panelName = item.dataset.panelName;
        const overlay = item.querySelector('.panel-card-overlay');
        const img = item.querySelector('img');
        const panelCache = panelBboxesCache[panelName];

        if (overlay && img && panelCache) {
            const rect = img.getBoundingClientRect();
            updatePanelOverlay(
                overlay,
                panelCache.bboxes,
                panelCache.imgSize,
                rect.width,
                rect.height,
                panelHoveredElement,
                selectedElement,
                img
            );
        }
    });
}
