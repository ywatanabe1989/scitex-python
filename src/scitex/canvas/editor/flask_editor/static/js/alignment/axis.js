/**
 * Axis-Based Alignment
 * Scientific plot alignment using axes bounding boxes
 */

// ============================================================================
// Get Axes Bbox for Panel
// ============================================================================
function getAxesBboxForPanel(panelName) {
    const cache = panelBboxesCache[panelName];
    if (!cache || !cache.bboxes) return null;

    const bboxes = cache.bboxes;

    // Method 1: Look for ax_00_panel, ax_01_panel, etc.
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

    // Method 2: Calculate axes bbox from spine bboxes (xaxis_spine + yaxis_spine)
    // This is the common case for matplotlib figures
    let xSpine = null, ySpine = null;
    for (const key of Object.keys(bboxes)) {
        if (key.endsWith('_xaxis_spine') && key.startsWith('ax_')) {
            xSpine = bboxes[key];
        }
        if (key.endsWith('_yaxis_spine') && key.startsWith('ax_')) {
            ySpine = bboxes[key];
        }
    }

    if (xSpine && ySpine) {
        // Combine spine bboxes to get axes area
        // Y-spine defines left edge, X-spine defines bottom edge
        const x0 = ySpine.x0 !== undefined ? ySpine.x0 : ySpine.x;
        const y0 = ySpine.y0 !== undefined ? ySpine.y0 : ySpine.y;
        const x1 = xSpine.x1 !== undefined ? xSpine.x1 : (xSpine.x + xSpine.width);
        const y1 = xSpine.y1 !== undefined ? xSpine.y1 : (xSpine.y + xSpine.height);

        return {
            x0: Math.min(x0, xSpine.x0 || xSpine.x || x0),
            y0: Math.min(y0, xSpine.y0 || xSpine.y || y0),
            x1: Math.max(x1, ySpine.x1 || (ySpine.x + ySpine.width) || x1),
            y1: Math.max(y1, ySpine.y1 || (ySpine.y + ySpine.height) || y1),
            key: 'derived_from_spines'
        };
    }

    // Method 3: Fallback to _meta.axes_bbox_px for single-axes plots
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

// ============================================================================
// Calculate Axis Edge Offset
// ============================================================================
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

// ============================================================================
// Align Panels by Axis
// ============================================================================
function alignPanelsByAxis(edge) {
    // Use selected panels only
    const panels = Array.from(document.querySelectorAll('.panel-canvas-item.selected'));
    if (panels.length < 2) {
        setStatus('Select at least 2 panels for axis alignment', true);
        return;
    }

    // Collect panel info with axes bboxes
    const panelInfos = [];
    for (const panel of panels) {
        const panelName = panel.dataset.panelName;
        const cache = panelBboxesCache[panelName];
        const axesBbox = getAxesBboxForPanel(panelName);

        if (!axesBbox || !cache) {
            console.warn(`Panel ${panelName}: no axes bbox found`);
            continue;
        }

        const currentPos = panelPositions[panelName];
        const axisOffset = getAxisEdgeOffset(panel, axesBbox, edge, cache.imgSize);

        panelInfos.push({
            panel,
            panelName,
            axesBbox,
            imgSize: cache.imgSize,
            currentPos,
            axisOffset
        });
    }

    if (panelInfos.length < 2) {
        setStatus('Not enough panels with axes info', true);
        return;
    }

    // Calculate target position - use the first panel's axis position as reference
    const referenceInfo = panelInfos[0];
    const referenceAxisPos = referenceInfo.currentPos.x + (edge.includes('h') || edge === 'left' || edge === 'right' ? referenceInfo.axisOffset : 0);
    const referenceAxisPosY = referenceInfo.currentPos.y + (edge.includes('v') || edge === 'top' || edge === 'bottom' ? referenceInfo.axisOffset : 0);

    if (edge === 'left' || edge === 'right' || edge === 'center-h') {
        // Align horizontally (match X positions of axis edges)
        // Target = first panel's axis X position in canvas coords
        for (const info of panelInfos) {
            const newX = referenceAxisPos - info.axisOffset;
            info.currentPos.x = newX;
            info.panel.style.left = newX + 'px';
            panelLayoutMm[info.panelName].x_mm = newX / canvasScale;
        }
    } else {
        // Align vertically (match Y positions of axis edges)
        // Target = first panel's axis Y position in canvas coords
        for (const info of panelInfos) {
            const newY = referenceAxisPosY - info.axisOffset;
            info.currentPos.y = newY;
            info.panel.style.top = newY + 'px';
            panelLayoutMm[info.panelName].y_mm = newY / canvasScale;
        }
    }

    // Update layout data
    updatePanelLayoutFromDOM();
    layoutModified = true;
}

// ============================================================================
// Stack Panels Vertically
// ============================================================================
function stackPanelsVertically() {
    // Use selected panels only
    const panels = Array.from(document.querySelectorAll('.panel-canvas-item.selected'));
    if (panels.length < 2) {
        setStatus('Select at least 2 panels for stacking', true);
        return;
    }

    // Collect panel info with axes bboxes
    const panelInfos = [];
    for (const panel of panels) {
        const panelName = panel.dataset.panelName;
        const cache = panelBboxesCache[panelName];
        const axesBbox = getAxesBboxForPanel(panelName);

        if (!axesBbox || !cache) continue;

        const currentPos = panelPositions[panelName];
        const axisOffsetLeft = getAxisEdgeOffset(panel, axesBbox, 'left', cache.imgSize);

        panelInfos.push({
            panel,
            panelName,
            axesBbox,
            imgSize: cache.imgSize,
            currentPos,
            axisOffsetLeft,
            height: panel.offsetHeight
        });
    }

    if (panelInfos.length < 2) return;

    // Sort by current vertical position
    panelInfos.sort((a, b) => a.currentPos.y - b.currentPos.y);

    // Use first panel as reference for Y-axis alignment
    const referenceAxisX = panelInfos[0].currentPos.x + panelInfos[0].axisOffsetLeft;

    // Stack panels vertically with small gap, aligned by Y-axis
    const gap = 10;  // pixels
    let currentY = panelInfos[0].currentPos.y;

    for (const info of panelInfos) {
        // Align Y-axis (left edge of axes)
        const newX = referenceAxisX - info.axisOffsetLeft;
        info.currentPos.x = newX;
        info.panel.style.left = newX + 'px';

        // Stack vertically
        info.currentPos.y = currentY;
        info.panel.style.top = currentY + 'px';
        currentY += info.height + gap;
    }

    // Update layout data
    updatePanelLayoutFromDOM();
    layoutModified = true;
}

// ============================================================================
// Axis Alignment Shortcut Handler
// ============================================================================
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

// ============================================================================
// Panel Movement (Arrow Keys)
// ============================================================================
function moveSelectedPanel(direction, amountMm) {
    const selected = document.querySelector('.panel-canvas-item.selected');
    if (!selected) return;

    const panelName = selected.dataset.panelName;
    const pos = panelPositions[panelName];
    if (!pos) return;

    switch(direction) {
        case 'left': pos.x -= amountMm * canvasScale; break;
        case 'right': pos.x += amountMm * canvasScale; break;
        case 'up': pos.y -= amountMm * canvasScale; break;
        case 'down': pos.y += amountMm * canvasScale; break;
    }

    // Update position in pixels (canvasScale = px/mm)
    selected.style.left = pos.x + 'px';
    selected.style.top = pos.y + 'px';

    // Update layout data
    panelLayoutMm[panelName] = {
        ...panelLayoutMm[panelName],
        x_mm: pos.x / canvasScale,
        y_mm: pos.y / canvasScale
    };

    layoutModified = true;
    setStatus(`Moved ${panelName} ${direction} by ${amountMm}mm`);
}
