/**
 * Element Dragging
 * Handles dragging of legends and panel letters (maintains scientific rigor)
 */

// ============================================================================
// Draggable Element Check
// ============================================================================
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

// ============================================================================
// Element Drag Start
// ============================================================================
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

// ============================================================================
// Element Drag Move
// ============================================================================
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
    const scaleX = dims.displayWidth / rect.width;
    const scaleY = dims.displayHeight / rect.height;
    const imgSize = bboxes._meta?.imgSize || {width: dims.displayWidth, height: dims.displayHeight};
    const imgDeltaX = deltaX * scaleX * imgSize.width / dims.displayWidth;
    const imgDeltaY = deltaY * scaleY * imgSize.height / dims.displayHeight;

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

// ============================================================================
// Element Drag Stop
// ============================================================================
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
        updateElementPosition(panelName, element, finalPosition);
    }

    document.removeEventListener('mousemove', onElementDrag);
    document.removeEventListener('mouseup', stopElementDrag);
    elementDragState = null;
}

// ============================================================================
// Snap Position Finding
// ============================================================================
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

// ============================================================================
// Snap Guides UI
// ============================================================================
function showSnapGuides(img, axesBbox, bboxes) {
    if (!img || !axesBbox) return;

    const container = img.parentElement;
    if (!container) return;

    // Remove existing guides
    container.querySelectorAll('.snap-guide').forEach(el => el.remove());

    const rect = img.getBoundingClientRect();
    const dims = getObjectFitContainDimensions(img);
    const imgSize = bboxes._meta?.imgSize || {width: dims.displayWidth, height: dims.displayHeight};

    // Scale factors
    const scaleX = dims.displayWidth / imgSize.width;
    const scaleY = dims.displayHeight / imgSize.height;

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

function hideSnapGuides() {
    document.querySelectorAll('.snap-guide').forEach(el => el.remove());
}

// ============================================================================
// Position Indicator
// ============================================================================
function showElementPositionIndicator(element, normX, normY) {
    let indicator = document.getElementById('element-position-indicator');
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = 'element-position-indicator';
        indicator.style.cssText = `
            position: fixed;
            bottom: 60px;
            right: 20px;
            background: rgba(0, 0, 0, 0.85);
            color: white;
            padding: 6px 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 11px;
            z-index: 10001;
            pointer-events: none;
        `;
        document.body.appendChild(indicator);
    }

    indicator.textContent = `${element}: (${normX.toFixed(2)}, ${normY.toFixed(2)})`;
    indicator.style.display = 'block';
}

function hideElementPositionIndicator() {
    const indicator = document.getElementById('element-position-indicator');
    if (indicator) {
        indicator.style.display = 'none';
    }
}
