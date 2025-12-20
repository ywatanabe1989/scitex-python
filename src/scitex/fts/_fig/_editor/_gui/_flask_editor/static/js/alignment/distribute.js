/**
 * Panel Distribution
 * Evenly distribute panels horizontally or vertically
 */

// ============================================================================
// Distribute Panels
// ============================================================================
function distributePanels(direction) {
    const selectedPanels = getSelectedPanels();
    if (selectedPanels.length < 3) {
        setStatus('Need at least 3 panels for distribution', true);
        return;
    }

    if (direction === 'horizontal') {
        // Sort by X position
        selectedPanels.sort((a, b) => a.pos.x - b.pos.x);

        const first = selectedPanels[0];
        const last = selectedPanels[selectedPanels.length - 1];
        const totalSpace = (last.pos.x + last.pos.width) - first.pos.x;
        const totalPanelWidth = selectedPanels.reduce((sum, p) => sum + p.pos.width, 0);
        const gap = (totalSpace - totalPanelWidth) / (selectedPanels.length - 1);

        let currentX = first.pos.x;
        selectedPanels.forEach(p => {
            p.pos.x = currentX;
            p.item.style.left = currentX + 'px';
            currentX += p.pos.width + gap;
        });
    } else {
        // Sort by Y position
        selectedPanels.sort((a, b) => a.pos.y - b.pos.y);

        const first = selectedPanels[0];
        const last = selectedPanels[selectedPanels.length - 1];
        const totalSpace = (last.pos.y + last.pos.height) - first.pos.y;
        const totalPanelHeight = selectedPanels.reduce((sum, p) => sum + p.pos.height, 0);
        const gap = (totalSpace - totalPanelHeight) / (selectedPanels.length - 1);

        let currentY = first.pos.y;
        selectedPanels.forEach(p => {
            p.pos.y = currentY;
            p.item.style.top = currentY + 'px';
            currentY += p.pos.height + gap;
        });
    }

    // Update layout data
    updatePanelLayoutFromDOM();
    layoutModified = true;
    setStatus(`Distributed panels ${direction}ly`);
}
