/**
 * Panel Resizing
 * Handles panel resize operations in canvas view
 */

// ============================================================================
// Resize Start
// ============================================================================
function startResize(e, item, name) {
    e.preventDefault();
    e.stopPropagation();

    resizingPanel = {item, name, startX: e.clientX, startY: e.clientY};
    document.addEventListener('mousemove', onResize);
    document.addEventListener('mouseup', stopResize);
}

// ============================================================================
// Resize Move
// ============================================================================
function onResize(e) {
    if (!resizingPanel) return;

    const dx = e.clientX - resizingPanel.startX;
    const dy = e.clientY - resizingPanel.startY;

    const pos = panelPositions[resizingPanel.name];
    pos.width = Math.max(100, pos.width + dx);
    pos.height = Math.max(100, pos.height + dy);

    resizingPanel.item.style.width = pos.width + 'px';
    resizingPanel.item.style.height = pos.height + 'px';

    resizingPanel.startX = e.clientX;
    resizingPanel.startY = e.clientY;
}

// ============================================================================
// Resize Stop
// ============================================================================
function stopResize() {
    if (resizingPanel) {
        resizingPanel = null;
        document.removeEventListener('mousemove', onResize);
        document.removeEventListener('mouseup', stopResize);
        layoutModified = true;
    }
}
