/**
 * Panel Selection
 * Handles panel selection and multi-selection in canvas view
 */

// ============================================================================
// Select All Panels
// ============================================================================
function selectAllPanels() {
    document.querySelectorAll('.panel-canvas-item').forEach(item => {
        item.classList.add('selected');
    });
}

// ============================================================================
// Deselect All Panels
// ============================================================================
function deselectAllPanels() {
    document.querySelectorAll('.panel-canvas-item').forEach(item => {
        item.classList.remove('selected');
    });

    // Also clear element selection in single-panel view
    selectedElement = null;
    hoveredElement = null;
    updateOverlay();
}

// ============================================================================
// Get Selected Panels
// ============================================================================
function getSelectedPanels() {
    const selectedItems = document.querySelectorAll('.panel-canvas-item.selected');
    return Array.from(selectedItems).map(item => ({
        item: item,
        name: item.dataset.panelName,
        pos: panelPositions[item.dataset.panelName]
    }));
}

// ============================================================================
// Toggle Panel Selection
// ============================================================================
function togglePanelSelection(panelName) {
    const item = document.querySelector(`.panel-canvas-item[data-panel-name="${panelName}"]`);
    if (item) {
        item.classList.toggle('selected');
    }
}

// ============================================================================
// Select Single Panel
// ============================================================================
function selectPanel(panelName, clearOthers = true) {
    if (clearOthers) {
        deselectAllPanels();
    }

    const item = document.querySelector(`.panel-canvas-item[data-panel-name="${panelName}"]`);
    if (item) {
        item.classList.add('selected');
    }
}

// ============================================================================
// Check if Panel is Selected
// ============================================================================
function isPanelSelected(panelName) {
    const item = document.querySelector(`.panel-canvas-item[data-panel-name="${panelName}"]`);
    return item ? item.classList.contains('selected') : false;
}
