/**
 * Keyboard Shortcuts
 * Handles all keyboard shortcuts for the editor
 */

// ============================================================================
// Main Keyboard Event Handler
// ============================================================================
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
        saveToBundle();
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
        deselectAllPanels();
        shortcutMode = null;
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

// ============================================================================
// Undo/Redo Functions
// ============================================================================
function undoLastChange() {
    if (undoStack.length === 0) {
        setStatus('Nothing to undo', true);
        return;
    }

    const prevState = undoStack.pop();
    redoStack.push({...overrides});
    overrides = prevState;
    updatePreview();
    setStatus('Undone');
}

function redoLastChange() {
    if (redoStack.length === 0) {
        setStatus('Nothing to redo', true);
        return;
    }

    const nextState = redoStack.pop();
    undoStack.push({...overrides});
    overrides = nextState;
    updatePreview();
    setStatus('Redone');
}

function saveUndoState() {
    undoStack.push({...overrides});
    redoStack = [];  // Clear redo stack on new change
}

// ============================================================================
// Delete Override
// ============================================================================
function deleteSelectedOverride() {
    if (!selectedElement) return;

    // Remove from element_overrides
    if (overrides.element_overrides && overrides.element_overrides[selectedElement]) {
        delete overrides.element_overrides[selectedElement];
        setStatus(`Deleted override for ${selectedElement}`);
        updatePreview();
    }
}

// ============================================================================
// Grid Visibility Toggle
// ============================================================================
function toggleGridVisibility() {
    gridVisible = !gridVisible;
    const canvas = document.getElementById('panel-canvas');
    if (canvas) {
        canvas.style.backgroundImage = gridVisible ?
            'linear-gradient(rgba(128,128,128,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(128,128,128,0.1) 1px, transparent 1px)' :
            'none';
    }
    setStatus(`Grid ${gridVisible ? 'visible' : 'hidden'}`);
}
