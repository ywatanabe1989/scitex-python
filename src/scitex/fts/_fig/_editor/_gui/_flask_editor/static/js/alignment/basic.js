/**
 * Basic Panel Alignment
 * Bounding box-based alignment (non-scientific)
 */

// ============================================================================
// Basic Alignment (by bounding box)
// ============================================================================
function alignPanels(mode) {
    const selectedPanels = getSelectedPanels();
    if (selectedPanels.length < 2) return;

    // Get bounds
    let targetValue;
    switch(mode) {
        case 'left':
            targetValue = Math.min(...selectedPanels.map(p => p.pos.x));
            selectedPanels.forEach(p => {
                p.pos.x = targetValue;
                p.item.style.left = targetValue + 'px';
            });
            break;
        case 'right':
            targetValue = Math.max(...selectedPanels.map(p => p.pos.x + p.pos.width));
            selectedPanels.forEach(p => {
                p.pos.x = targetValue - p.pos.width;
                p.item.style.left = p.pos.x + 'px';
            });
            break;
        case 'top':
            targetValue = Math.min(...selectedPanels.map(p => p.pos.y));
            selectedPanels.forEach(p => {
                p.pos.y = targetValue;
                p.item.style.top = targetValue + 'px';
            });
            break;
        case 'bottom':
            targetValue = Math.max(...selectedPanels.map(p => p.pos.y + p.pos.height));
            selectedPanels.forEach(p => {
                p.pos.y = targetValue - p.pos.height;
                p.item.style.top = p.pos.y + 'px';
            });
            break;
        case 'center-h':
            const avgX = selectedPanels.reduce((sum, p) => sum + p.pos.x + p.pos.width/2, 0) / selectedPanels.length;
            selectedPanels.forEach(p => {
                p.pos.x = avgX - p.pos.width/2;
                p.item.style.left = p.pos.x + 'px';
            });
            break;
        case 'center-v':
            const avgY = selectedPanels.reduce((sum, p) => sum + p.pos.y + p.pos.height/2, 0) / selectedPanels.length;
            selectedPanels.forEach(p => {
                p.pos.y = avgY - p.pos.height/2;
                p.item.style.top = p.pos.y + 'px';
            });
            break;
    }

    // Update layout data
    updatePanelLayoutFromDOM();
    layoutModified = true;
    setStatus(`Aligned panels: ${mode}`);
}

// ============================================================================
// Alignment Shortcut Handler
// ============================================================================
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

// ============================================================================
// Z-Order Management
// ============================================================================
function bringPanelToFront() {
    const selected = document.querySelector('.panel-canvas-item.selected');
    if (selected) {
        selected.style.zIndex = (parseInt(selected.style.zIndex || 0) + 1).toString();
        setStatus('Brought panel to front');
    }
}

function sendPanelToBack() {
    const selected = document.querySelector('.panel-canvas-item.selected');
    if (selected) {
        selected.style.zIndex = (parseInt(selected.style.zIndex || 0) - 1).toString();
        setStatus('Sent panel to back');
    }
}
