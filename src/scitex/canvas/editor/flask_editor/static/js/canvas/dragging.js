/**
 * Panel Dragging
 * Handles drag-and-drop for panel repositioning in canvas view
 */

// ============================================================================
// Panel Drag Initialization
// ============================================================================
function initPanelDrag(item, panelName) {
    const handle = item.querySelector('.panel-drag-handle');
    const label = item.querySelector('.panel-canvas-label');

    // Drag from handle (always works)
    if (handle) {
        handle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            startPanelDrag(e, item, panelName);
        });
    }

    // Also allow dragging from panel label
    if (label) {
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
        if (e.target === handle || e.target === label) return;

        // Start drag from anywhere else on the panel
        startPanelDrag(e, item, panelName);
    });

    // Set cursor to indicate draggability
    item.style.cursor = 'move';
}

// ============================================================================
// Panel Drag Start
// ============================================================================
function startPanelDrag(e, item, name) {
    e.preventDefault();

    // Handle selection based on Ctrl key
    const isCtrlPressed = e.ctrlKey || e.metaKey;
    const wasAlreadySelected = item.classList.contains('selected');

    if (isCtrlPressed) {
        // Ctrl+Click: toggle this panel's selection (add/remove from multi-select)
        item.classList.toggle('selected');
    } else if (!wasAlreadySelected) {
        // Regular click on unselected panel: select only this one
        deselectAllPanels();
        item.classList.add('selected');
    }
    // If clicking on already-selected panel without Ctrl:
    // Don't change selection yet - could be start of multi-panel drag
    // Selection will be finalized in stopPanelDrag based on hasMoved

    // Collect all selected panels for group dragging
    const selectedItems = Array.from(document.querySelectorAll('.panel-canvas-item.selected'));
    if (selectedItems.length === 0) {
        // If somehow nothing selected, select the clicked item
        item.classList.add('selected');
        selectedItems.push(item);
    }

    // Store drag state for all selected panels
    draggedPanel = {
        primaryItem: item,
        primaryName: name,
        selectedItems: selectedItems,
        startX: e.clientX,
        startY: e.clientY,
        hasMoved: false,  // Track if actual drag occurred
        wasAlreadySelected,  // Track initial selection state for click handling
        isCtrlPressed,  // Track if Ctrl was pressed
        startPositions: selectedItems.map(el => ({
            item: el,
            name: el.dataset.panelName,
            x: parseFloat(el.style.left) || 0,
            y: parseFloat(el.style.top) || 0
        }))
    };

    document.addEventListener('mousemove', onPanelDrag);
    document.addEventListener('mouseup', stopPanelDrag);

    // Show position indicator for primary panel
    updatePositionIndicator(name,
        panelPositions[name]?.x || 0,
        panelPositions[name]?.y || 0
    );
}

// ============================================================================
// Panel Drag Move
// ============================================================================
function onPanelDrag(e) {
    if (!draggedPanel) return;

    // Calculate delta from drag start
    const dx = e.clientX - draggedPanel.startX;
    const dy = e.clientY - draggedPanel.startY;

    // Mark as moved if we've actually dragged (threshold: 3px)
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
        draggedPanel.hasMoved = true;
    }

    // Snap to grid (optional: 5mm grid)
    const gridSize = 5 * canvasScale;  // 5mm grid
    const snappedDx = Math.round(dx / gridSize) * gridSize;
    const snappedDy = Math.round(dy / gridSize) * gridSize;

    // Move all selected panels by the same delta
    draggedPanel.startPositions.forEach(({item, name, x, y}) => {
        const newX = x + snappedDx;
        const newY = y + snappedDy;

        // Constrain to canvas bounds (allow slight negative for edge alignment)
        const constrainedX = Math.max(-10, newX);
        const constrainedY = Math.max(-10, newY);

        // Update pixel positions
        item.style.left = constrainedX + 'px';
        item.style.top = constrainedY + 'px';

        panelPositions[name] = {
            ...panelPositions[name],
            x: constrainedX,
            y: constrainedY
        };

        // Update mm positions
        panelLayoutMm[name] = {
            ...panelLayoutMm[name],
            x_mm: constrainedX / canvasScale,
            y_mm: constrainedY / canvasScale
        };
    });

    // Show position indicator for primary panel
    const primaryPos = panelPositions[draggedPanel.primaryName];
    if (primaryPos) {
        updatePositionIndicator(draggedPanel.primaryName, primaryPos.x, primaryPos.y);
    }

    // Mark layout as modified
    layoutModified = true;
}

// ============================================================================
// Panel Drag Stop
// ============================================================================
function stopPanelDrag() {
    if (draggedPanel) {
        // Handle click (no movement) on already-selected panel without Ctrl:
        // Finalize selection to only the clicked panel
        if (!draggedPanel.hasMoved && draggedPanel.wasAlreadySelected && !draggedPanel.isCtrlPressed) {
            // This was a simple click on an already-selected panel
            // Deselect all others, keep only the clicked panel selected
            deselectAllPanels();
            draggedPanel.primaryItem.classList.add('selected');
        }

        // Reset cursor for all selected panels
        draggedPanel.selectedItems.forEach(item => {
            item.style.cursor = 'move';
        });

        draggedPanel = null;
        document.removeEventListener('mousemove', onPanelDrag);
        document.removeEventListener('mouseup', stopPanelDrag);

        // Update canvas size if panel moved outside
        updateCanvasSize();

        // Hide position indicator after a delay
        setTimeout(() => {
            const indicator = document.getElementById('position-indicator');
            if (indicator) {
                indicator.style.display = 'none';
            }
        }, 1000);

        // Auto-save layout
        if (layoutModified) {
            saveLayout();
        }
    }
}

// ============================================================================
// Position Indicator
// ============================================================================
function updatePositionIndicator(panelName, x, y) {
    let indicator = document.getElementById('position-indicator');
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = 'position-indicator';
        indicator.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            z-index: 10000;
            pointer-events: none;
        `;
        document.body.appendChild(indicator);
    }

    const xMm = (x / canvasScale).toFixed(1);
    const yMm = (y / canvasScale).toFixed(1);
    indicator.textContent = `${panelName}: x=${xMm}mm, y=${yMm}mm`;
    indicator.style.display = 'block';
}

// ============================================================================
// Manual Layout Save
// ============================================================================
function saveLayoutManually() {
    if (layoutModified) {
        saveLayout();
    } else {
        setStatus('No changes to save');
    }
}

// ============================================================================
// Legacy Drag Functions (for backward compatibility)
// ============================================================================
function startDrag(e, item, name) {
    startPanelDrag(e, item, name);
}

function onDrag(e) {
    onPanelDrag(e);
}

function stopDrag() {
    stopPanelDrag();
}
