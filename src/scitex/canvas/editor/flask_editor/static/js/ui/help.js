/**
 * Help System
 * Keyboard shortcut help dialog
 */

// ============================================================================
// Show Shortcut Help
// ============================================================================
function showShortcutHelp() {
    const helpHtml = `
        <div id="shortcut-help-overlay" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 10000; display: flex; justify-content: center; align-items: center;" onclick="this.remove();">
            <div style="background: var(--bg-primary); padding: 30px; border-radius: 8px; max-width: 600px; max-height: 80vh; overflow-y: auto;" onclick="event.stopPropagation();">
                <h2 style="margin-top: 0;">Keyboard Shortcuts</h2>

                <h3>Basic Operations</h3>
                <table style="width: 100%; margin-bottom: 20px;">
                    <tr><td><kbd>Ctrl</kbd>+<kbd>S</kbd></td><td>Save changes</td></tr>
                    <tr><td><kbd>Ctrl</kbd>+<kbd>Z</kbd></td><td>Undo</td></tr>
                    <tr><td><kbd>Ctrl</kbd>+<kbd>Y</kbd></td><td>Redo</td></tr>
                    <tr><td><kbd>Delete</kbd></td><td>Remove selected override</td></tr>
                    <tr><td><kbd>Esc</kbd></td><td>Deselect all</td></tr>
                </table>

                <h3>Panel Movement</h3>
                <table style="width: 100%; margin-bottom: 20px;">
                    <tr><td><kbd>Arrow Keys</kbd></td><td>Move panel by 1mm</td></tr>
                    <tr><td><kbd>Shift</kbd>+<kbd>Arrows</kbd></td><td>Move panel by 5mm</td></tr>
                </table>

                <h3>View Controls</h3>
                <table style="width: 100%; margin-bottom: 20px;">
                    <tr><td><kbd>+</kbd> / <kbd>=</kbd></td><td>Zoom in</td></tr>
                    <tr><td><kbd>-</kbd></td><td>Zoom out</td></tr>
                    <tr><td><kbd>0</kbd></td><td>Fit to window</td></tr>
                    <tr><td><kbd>G</kbd></td><td>Toggle grid</td></tr>
                </table>

                <h3>Alignment (Basic)</h3>
                <table style="width: 100%; margin-bottom: 20px;">
                    <tr><td><kbd>Alt</kbd>+<kbd>A</kbd> then <kbd>L</kbd></td><td>Align left</td></tr>
                    <tr><td><kbd>Alt</kbd>+<kbd>A</kbd> then <kbd>R</kbd></td><td>Align right</td></tr>
                    <tr><td><kbd>Alt</kbd>+<kbd>A</kbd> then <kbd>T</kbd></td><td>Align top</td></tr>
                    <tr><td><kbd>Alt</kbd>+<kbd>A</kbd> then <kbd>B</kbd></td><td>Align bottom</td></tr>
                    <tr><td><kbd>Alt</kbd>+<kbd>A</kbd> then <kbd>C</kbd></td><td>Center horizontally</td></tr>
                    <tr><td><kbd>Alt</kbd>+<kbd>A</kbd> then <kbd>M</kbd></td><td>Center vertically</td></tr>
                </table>

                <h3>Alignment (By Axis)</h3>
                <table style="width: 100%; margin-bottom: 20px;">
                    <tr><td><kbd>Alt</kbd>+<kbd>Shift</kbd>+<kbd>A</kbd> then <kbd>L</kbd></td><td>Align by Y-axis (left)</td></tr>
                    <tr><td><kbd>Alt</kbd>+<kbd>Shift</kbd>+<kbd>A</kbd> then <kbd>B</kbd></td><td>Align by X-axis (bottom)</td></tr>
                    <tr><td><kbd>Alt</kbd>+<kbd>Shift</kbd>+<kbd>A</kbd> then <kbd>S</kbd></td><td>Stack vertically</td></tr>
                </table>

                <h3>Selection</h3>
                <table style="width: 100%; margin-bottom: 20px;">
                    <tr><td><kbd>Ctrl</kbd>+<kbd>A</kbd></td><td>Select all panels</td></tr>
                    <tr><td><kbd>Ctrl</kbd>+<kbd>Click</kbd></td><td>Toggle panel selection</td></tr>
                </table>

                <h3>Arrange</h3>
                <table style="width: 100%; margin-bottom: 20px;">
                    <tr><td><kbd>Alt</kbd>+<kbd>F</kbd></td><td>Bring to front</td></tr>
                    <tr><td><kbd>Alt</kbd>+<kbd>B</kbd></td><td>Send to back</td></tr>
                </table>

                <h3>Developer Tools</h3>
                <table style="width: 100%; margin-bottom: 20px;">
                    <tr><td><kbd>Alt</kbd>+<kbd>I</kbd></td><td>Toggle element inspector</td></tr>
                    <tr><td><kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>I</kbd></td><td>Rectangle selection mode</td></tr>
                    <tr><td><kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>I</kbd></td><td>Copy console logs</td></tr>
                </table>

                <button onclick="this.closest('#shortcut-help-overlay').remove()" style="margin-top: 20px;">Close</button>
            </div>
        </div>
    `;

    // Remove existing help if any
    const existing = document.getElementById('shortcut-help-overlay');
    if (existing) existing.remove();

    // Add to DOM
    document.body.insertAdjacentHTML('beforeend', helpHtml);
}

// Add kbd styling
const kbdStyle = document.createElement('style');
kbdStyle.textContent = `
    kbd {
        background: var(--bg-tertiary, #333);
        border: 1px solid var(--border-color, #555);
        border-radius: 3px;
        padding: 2px 6px;
        font-family: monospace;
        font-size: 0.85em;
        margin-right: 8px;
    }
`;
document.head.appendChild(kbdStyle);
