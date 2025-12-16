/**
 * Context Menu
 * Right-click context menu for panel operations
 */

// ============================================================================
// Show Context Menu
// ============================================================================
function showContextMenu(e, panelName) {
    e.preventDefault();
    hideContextMenu();

    const selectedCount = document.querySelectorAll('.panel-canvas-item.selected').length;
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
                <div class="context-menu-item ${selectedCount < 3 ? 'disabled' : ''}" onclick="${selectedCount >= 3 ? "distributePanels('horizontal'); hideContextMenu();" : ''}">Horizontal</div>
                <div class="context-menu-item ${selectedCount < 3 ? 'disabled' : ''}" onclick="${selectedCount >= 3 ? "distributePanels('vertical'); hideContextMenu();" : ''}">Vertical</div>
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

// ============================================================================
// Hide Context Menu
// ============================================================================
function hideContextMenu() {
    if (contextMenu) {
        contextMenu.remove();
        contextMenu = null;
    }
}

// ============================================================================
// Event Listeners for Context Menu
// ============================================================================

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
            if (panel && !panel.classList.contains('selected')) {
                if (!e.ctrlKey && !e.metaKey) {
                    deselectAllPanels();
                }
                panel.classList.add('selected');
            }

            showContextMenu(e, panelName);
        });

        // Click on empty canvas space deselects all panels
        canvas.addEventListener('click', (e) => {
            // Only deselect if clicking directly on canvas background, not on a panel
            const panel = e.target.closest('.panel-canvas-item');
            if (!panel && !e.ctrlKey && !e.metaKey) {
                deselectAllPanels();
            }
        });
    }
});
