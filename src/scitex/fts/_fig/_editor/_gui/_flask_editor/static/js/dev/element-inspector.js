/**
 * Element Inspector - Visual Debugging Tool
 * Shows all HTML elements with colored rectangles and labels.
 *
 * Shortcuts:
 *   Alt+I: Toggle inspector overlay
 *   Ctrl+Alt+I: Rectangle selection mode
 *   Ctrl+Shift+I: Debug snapshot (console logs + page info)
 *   Escape: Deactivate inspector / Cancel selection
 */

(function() {
    'use strict';

    // ============================================================================
    // State
    // ============================================================================
    var isActive = false;
    var overlayContainer = null;
    var elementBoxMap = new Map();
    var currentlyHoveredBox = null;
    var currentlyHoveredElement = null;
    var consoleLogs = [];
    var maxLogs = 500;
    var onCopyCallback = null;

    // Selection mode state
    var selectionMode = false;
    var selectionStart = null;
    var selectionRect = null;
    var selectionOverlay = null;
    var currentlySelectedElements = new Set();

    // ============================================================================
    // Console Capture (start immediately)
    // ============================================================================
    var originalConsole = {
        log: console.log.bind(console),
        warn: console.warn.bind(console),
        error: console.error.bind(console),
        info: console.info.bind(console)
    };

    function captureLog(type, args) {
        var entry = {
            type: type,
            timestamp: new Date().toISOString(),
            args: args.map(function(arg) { return stringify(arg); }),
            source: getCallSource()
        };
        consoleLogs.push(entry);
        if (consoleLogs.length > maxLogs) {
            consoleLogs.shift();
        }
    }

    function getCallSource() {
        try {
            var stack = new Error().stack;
            if (!stack) return '';
            var lines = stack.split('\n');
            for (var i = 4; i < lines.length; i++) {
                var line = lines[i];
                var match = line.match(/(?:at\s+)?(?:.*?\s+\()?([^\s()]+):(\d+):(\d+)\)?$/);
                if (match) {
                    var file = match[1];
                    var lineNum = match[2];
                    var fileName = file.split('/').pop() || file;
                    if (fileName.includes('element-inspector')) continue;
                    return fileName + ':' + lineNum;
                }
            }
        } catch (e) {}
        return '';
    }

    function stringify(obj) {
        if (obj === null) return 'null';
        if (obj === undefined) return 'undefined';
        if (typeof obj === 'string') return obj;
        if (typeof obj === 'number' || typeof obj === 'boolean') return String(obj);
        if (obj instanceof Error) {
            return obj.name + ': ' + obj.message + '\n' + (obj.stack || '');
        }
        try {
            return JSON.stringify(obj, null, 2);
        } catch (e) {
            return String(obj);
        }
    }

    // Override console methods
    console.log = function() {
        captureLog('log', Array.from(arguments));
        originalConsole.log.apply(console, arguments);
    };
    console.warn = function() {
        captureLog('warn', Array.from(arguments));
        originalConsole.warn.apply(console, arguments);
    };
    console.error = function() {
        captureLog('error', Array.from(arguments));
        originalConsole.error.apply(console, arguments);
    };
    console.info = function() {
        captureLog('info', Array.from(arguments));
        originalConsole.info.apply(console, arguments);
    };

    // ============================================================================
    // Notification Manager
    // ============================================================================
    function showNotification(message, type) {
        var notification = document.createElement('div');
        notification.className = 'element-inspector-notification ' + type;
        notification.textContent = message;
        document.body.appendChild(notification);

        requestAnimationFrame(function() {
            notification.classList.add('visible');
        });

        setTimeout(function() {
            notification.classList.remove('visible');
            setTimeout(function() { notification.remove(); }, 200);
        }, 1000);
    }

    function showCameraFlash() {
        var flash = document.createElement('div');
        flash.className = 'camera-flash';
        document.body.appendChild(flash);
        setTimeout(function() { flash.remove(); }, 300);
    }

    function triggerCopyCallback() {
        if (onCopyCallback) {
            setTimeout(function() {
                onCopyCallback();
            }, 400);
        }
    }

    // ============================================================================
    // Debug Info Collector
    // ============================================================================
    function buildCSSSelector(element) {
        var tag = element.tagName.toLowerCase();
        var id = element.id;
        var classes = element.className;

        var selector = tag;
        if (id) selector += '#' + id;
        if (classes && typeof classes === 'string') {
            var classList = classes.split(/\s+/).filter(function(c) { return c; });
            if (classList.length > 0) {
                selector += '.' + classList.join('.');
            }
        }
        return selector;
    }

    function getXPath(element) {
        if (element.id) {
            return '//*[@id="' + element.id + '"]';
        }

        var parts = [];
        var current = element;

        while (current && current.nodeType === Node.ELEMENT_NODE) {
            var index = 0;
            var sibling = current.previousSibling;

            while (sibling) {
                if (sibling.nodeType === Node.ELEMENT_NODE && sibling.nodeName === current.nodeName) {
                    index++;
                }
                sibling = sibling.previousSibling;
            }

            var tagName = current.nodeName.toLowerCase();
            var pathIndex = index > 0 ? '[' + (index + 1) + ']' : '';
            parts.unshift(tagName + pathIndex);
            current = current.parentElement;
        }

        return '/' + parts.join('/');
    }

    function getParentChain(element) {
        var chain = [];
        var current = element.parentElement;
        var depth = 0;

        while (current && depth < 5) {
            chain.push(buildCSSSelector(current));
            current = current.parentElement;
            depth++;
        }

        return chain;
    }

    function gatherElementDebugInfo(element) {
        var info = {};

        info.url = window.location.href;
        info.timestamp = new Date().toISOString();

        var className = typeof element.className === 'string' ? element.className : '';
        info.element = {
            tag: element.tagName.toLowerCase(),
            id: element.id || null,
            classes: className ? className.split(/\s+/).filter(function(c) { return c; }) : [],
            selector: buildCSSSelector(element),
            xpath: getXPath(element)
        };

        info.attributes = {};
        for (var i = 0; i < element.attributes.length; i++) {
            var attr = element.attributes[i];
            info.attributes[attr.name] = attr.value;
        }

        if (element instanceof HTMLElement) {
            var computed = window.getComputedStyle(element);
            info.styles = {
                display: computed.display,
                position: computed.position,
                width: computed.width,
                height: computed.height,
                backgroundColor: computed.backgroundColor,
                color: computed.color,
                fontSize: computed.fontSize
            };

            var rect = element.getBoundingClientRect();
            info.dimensions = {
                width: rect.width,
                height: rect.height,
                top: rect.top,
                left: rect.left
            };

            info.content = {
                textContent: (element.textContent || '').substring(0, 200)
            };
        }

        info.parentChain = getParentChain(element);

        return formatDebugInfoForAI(info);
    }

    function formatDebugInfoForAI(info) {
        return '# Element Debug Information\n\n' +
            '## Page Context\n' +
            '- URL: ' + info.url + '\n' +
            '- Timestamp: ' + info.timestamp + '\n\n' +
            '## Element Identification\n' +
            '- Tag: <' + info.element.tag + '>\n' +
            '- ID: ' + (info.element.id || 'none') + '\n' +
            '- Classes: ' + (info.element.classes.join(', ') || 'none') + '\n' +
            '- CSS Selector: ' + info.element.selector + '\n' +
            '- XPath: ' + info.element.xpath + '\n\n' +
            '## Attributes\n' +
            Object.entries(info.attributes).map(function(entry) {
                return '- ' + entry[0] + ': ' + entry[1];
            }).join('\n') + '\n\n' +
            '## Computed Styles\n' +
            Object.entries(info.styles || {}).map(function(entry) {
                return '- ' + entry[0] + ': ' + entry[1];
            }).join('\n') + '\n\n' +
            '## Dimensions & Position\n' +
            '- Width: ' + (info.dimensions ? info.dimensions.width : 0) + 'px\n' +
            '- Height: ' + (info.dimensions ? info.dimensions.height : 0) + 'px\n' +
            '- Top: ' + (info.dimensions ? info.dimensions.top : 0) + 'px\n' +
            '- Left: ' + (info.dimensions ? info.dimensions.left : 0) + 'px\n\n' +
            '## Content (truncated)\n' +
            (info.content ? info.content.textContent : 'none') + '\n\n' +
            '## Parent Chain\n' +
            info.parentChain.map(function(p, i) { return (i + 1) + '. ' + p; }).join('\n') + '\n\n' +
            '---\nGenerated by Element Inspector';
    }

    // ============================================================================
    // Element Scanner
    // ============================================================================
    function getDepth(element) {
        var depth = 0;
        var current = element;

        while (current && current !== document.body) {
            depth++;
            current = current.parentElement;
        }

        return depth;
    }

    function getColorForDepth(depth) {
        var colors = [
            '#3B82F6', // Blue (depth 0-2)
            '#10B981', // Green (depth 3-5)
            '#F59E0B', // Yellow (depth 6-8)
            '#EF4444', // Red (depth 9-11)
            '#EC4899'  // Pink (depth 12+)
        ];

        var index = Math.min(Math.floor(depth / 3), colors.length - 1);
        return colors[index];
    }

    function shouldShowLabel(element, rect, depth) {
        if (element.id) {
            return rect.width > 20 && rect.height > 20;
        }
        if (rect.width > 100 || rect.height > 100) {
            return true;
        }
        var importantTags = ['header', 'nav', 'main', 'section', 'article', 'aside', 'footer', 'form', 'table'];
        if (importantTags.indexOf(element.tagName.toLowerCase()) !== -1 && (rect.width > 50 || rect.height > 50)) {
            return true;
        }
        var interactiveTags = ['button', 'a', 'input', 'select', 'textarea'];
        if (interactiveTags.indexOf(element.tagName.toLowerCase()) !== -1 && (rect.width > 30 || rect.height > 30)) {
            return true;
        }
        if (depth > 8 && rect.width < 100 && rect.height < 100) {
            return false;
        }
        return false;
    }

    function scanElements(container) {
        var allElements = document.querySelectorAll('*');
        var count = 0;
        var occupiedPositions = [];

        allElements.forEach(function(element) {
            if (element.closest('#element-inspector-overlay')) return;

            if (element instanceof HTMLElement) {
                var computed = window.getComputedStyle(element);
                if (computed.display === 'none' || computed.visibility === 'hidden') return;
            }

            var depth = getDepth(element);
            var color = getColorForDepth(depth);

            var rect = element.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return;

            var box = document.createElement('div');
            box.className = 'element-inspector-box';
            box.style.cssText =
                'top: ' + (rect.top + window.scrollY) + 'px;' +
                'left: ' + (rect.left + window.scrollX) + 'px;' +
                'width: ' + rect.width + 'px;' +
                'height: ' + rect.height + 'px;' +
                'border-color: ' + color + ';';

            var tag = element.tagName.toLowerCase();
            var id = element.id ? '#' + element.id : '';
            box.title = 'Click to copy debug info: ' + tag + id;

            elementBoxMap.set(box, element);

            box.addEventListener('mouseenter', function() {
                currentlyHoveredBox = box;
                currentlyHoveredElement = element;
            });

            box.addEventListener('mouseleave', function() {
                if (currentlyHoveredBox === box) {
                    currentlyHoveredBox = null;
                    currentlyHoveredElement = null;
                }
            });

            box.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();

                var selectedElement = currentlyHoveredElement || element;
                var selectedBox = currentlyHoveredBox || box;

                selectedBox.classList.add('highlighted');

                var debugInfo = gatherElementDebugInfo(selectedElement);
                navigator.clipboard.writeText(debugInfo).then(function() {
                    showNotification('✓ Copied!', 'success');
                    console.log('[ElementInspector] Copied:', debugInfo);
                    triggerCopyCallback();
                }).catch(function(err) {
                    console.error('[ElementInspector] Copy failed:', err);
                    showNotification('✗ Copy Failed', 'error');
                    selectedBox.classList.remove('highlighted');
                });
            });

            if (shouldShowLabel(element, rect, depth)) {
                var label = createLabel(element, depth);
                if (label) {
                    var labelPos = findLabelPosition(rect, occupiedPositions);
                    if (labelPos.isValid) {
                        label.style.top = labelPos.top + 'px';
                        label.style.left = labelPos.left + 'px';
                        addCopyToClipboard(label, element);
                        addHoverHighlight(label, box, element);

                        occupiedPositions.push({
                            top: labelPos.top - 8,
                            left: labelPos.left - 8,
                            bottom: labelPos.top + 20 + 8,
                            right: labelPos.left + 250 + 8
                        });

                        container.appendChild(label);
                    }
                }
            }

            container.appendChild(box);
            count++;
        });

        console.log('[ElementInspector] Visualized ' + count + ' elements');
    }

    function createLabel(element, depth) {
        var tag = element.tagName.toLowerCase();
        var id = element.id;
        var classes = element.className;

        var labelText = '<span class="element-inspector-label-tag">' + tag + '</span>';

        if (id) {
            labelText += ' <span class="element-inspector-label-id">#' + id + '</span>';
        }

        if (classes && typeof classes === 'string') {
            var classList = classes.split(/\s+/).filter(function(c) { return c.length > 0; });
            if (classList.length > 0) {
                var classPreview = classList.slice(0, 2).join('.');
                labelText += ' <span class="element-inspector-label-class">.' + classPreview + '</span>';
                if (classList.length > 2) {
                    labelText += '<span class="element-inspector-label-class">+' + (classList.length - 2) + '</span>';
                }
            }
        }

        if (depth > 5) {
            labelText += ' <span style="color: #999; font-size: 9px;">d' + depth + '</span>';
        }

        var label = document.createElement('div');
        label.className = 'element-inspector-label';
        label.innerHTML = labelText;
        label.title = 'Click to copy comprehensive debug info for AI';

        return label;
    }

    function findLabelPosition(rect, occupiedPositions) {
        var scrollY = window.scrollY;
        var scrollX = window.scrollX;

        var positions = [
            { top: rect.top + scrollY - 24, left: rect.left + scrollX },
            { top: rect.top + scrollY - 24, left: rect.right + scrollX - 200 },
            { top: rect.top + scrollY + 4, left: rect.left + scrollX + 4 },
            { top: rect.bottom + scrollY + 4, left: rect.left + scrollX }
        ];

        for (var i = 0; i < positions.length; i++) {
            if (!isPositionOccupied(positions[i], occupiedPositions)) {
                return { top: positions[i].top, left: positions[i].left, isValid: true };
            }
        }

        return { top: 0, left: 0, isValid: false };
    }

    function isPositionOccupied(pos, occupiedPositions) {
        var labelWidth = 250;
        var labelHeight = 20;

        for (var i = 0; i < occupiedPositions.length; i++) {
            var occupied = occupiedPositions[i];
            if (!(pos.left + labelWidth < occupied.left ||
                  pos.left > occupied.right ||
                  pos.top + labelHeight < occupied.top ||
                  pos.top > occupied.bottom)) {
                return true;
            }
        }
        return false;
    }

    function addHoverHighlight(label, box, element) {
        label.addEventListener('mouseenter', function() {
            currentlyHoveredBox = box;
            currentlyHoveredElement = element;
            box.classList.add('highlighted');
            if (element instanceof HTMLElement) {
                element.style.outline = '3px solid rgba(59, 130, 246, 0.8)';
                element.style.outlineOffset = '2px';
            }
        });

        label.addEventListener('mouseleave', function() {
            currentlyHoveredBox = null;
            currentlyHoveredElement = null;
            box.classList.remove('highlighted');
            if (element instanceof HTMLElement) {
                element.style.outline = '';
                element.style.outlineOffset = '';
            }
        });
    }

    function addCopyToClipboard(label, element) {
        label.addEventListener('click', function(e) {
            e.stopPropagation();
            e.preventDefault();

            var debugInfo = gatherElementDebugInfo(element);

            navigator.clipboard.writeText(debugInfo).then(function() {
                showNotification('✓ Copied!', 'success');
                console.log('[ElementInspector] Copied debug info to clipboard');
                triggerCopyCallback();
            }).catch(function(err) {
                console.error('[ElementInspector] Failed to copy:', err);
                showNotification('✗ Copy Failed', 'error');
            });
        });
    }

    // ============================================================================
    // Selection Mode
    // ============================================================================
    function startSelectionMode() {
        if (!isActive) {
            activate();
        }

        selectionMode = true;
        document.body.classList.add('element-inspector-selection-mode');

        selectionOverlay = document.createElement('div');
        selectionOverlay.className = 'selection-overlay';
        document.body.appendChild(selectionOverlay);

        showNotification('Drag to select area', 'success');

        document.addEventListener('mousedown', onSelectionMouseDown);
        document.addEventListener('mousemove', onSelectionMouseMove);
        document.addEventListener('mouseup', onSelectionMouseUp);
    }

    function cancelSelectionMode() {
        selectionMode = false;
        document.body.classList.remove('element-inspector-selection-mode');

        if (selectionOverlay) {
            selectionOverlay.remove();
            selectionOverlay = null;
        }

        if (selectionRect) {
            selectionRect.remove();
            selectionRect = null;
        }

        document.removeEventListener('mousedown', onSelectionMouseDown);
        document.removeEventListener('mousemove', onSelectionMouseMove);
        document.removeEventListener('mouseup', onSelectionMouseUp);

        selectionStart = null;
        currentlySelectedElements.clear();
    }

    function onSelectionMouseDown(e) {
        if (!selectionMode) return;

        e.preventDefault();
        selectionStart = { x: e.clientX, y: e.clientY };

        selectionRect = document.createElement('div');
        selectionRect.className = 'selection-rectangle';
        selectionRect.style.left = e.clientX + 'px';
        selectionRect.style.top = e.clientY + 'px';
        selectionRect.style.width = '0px';
        selectionRect.style.height = '0px';

        document.body.appendChild(selectionRect);
    }

    function onSelectionMouseMove(e) {
        if (!selectionMode || !selectionStart || !selectionRect) return;

        e.preventDefault();

        var left = Math.min(selectionStart.x, e.clientX);
        var top = Math.min(selectionStart.y, e.clientY);
        var width = Math.abs(e.clientX - selectionStart.x);
        var height = Math.abs(e.clientY - selectionStart.y);

        selectionRect.style.left = left + 'px';
        selectionRect.style.top = top + 'px';
        selectionRect.style.width = width + 'px';
        selectionRect.style.height = height + 'px';
    }

    function onSelectionMouseUp(e) {
        if (!selectionMode || !selectionStart || !selectionRect) return;

        e.preventDefault();

        var left = Math.min(selectionStart.x, e.clientX);
        var top = Math.min(selectionStart.y, e.clientY);
        var width = Math.abs(e.clientX - selectionStart.x);
        var height = Math.abs(e.clientY - selectionStart.y);

        if (width < 5 || height < 5) {
            cancelSelectionMode();
            showNotification('Selection too small', 'error');
            return;
        }

        var selectedElements = findElementsInRect({ left: left, top: top, width: width, height: height });

        console.log('[ElementInspector] Found ' + selectedElements.length + ' elements in selection');

        var selectionInfo = gatherSelectionInfo(selectedElements, { left: left, top: top, width: width, height: height });

        navigator.clipboard.writeText(selectionInfo).then(function() {
            showNotification('✓ ' + selectedElements.length + ' elements copied!', 'success');
            triggerCopyCallback();
        }).catch(function(err) {
            console.error('[ElementInspector] Failed to copy:', err);
            showNotification('✗ Copy Failed', 'error');
        });

        cancelSelectionMode();
    }

    function findElementsInRect(rect) {
        var selectedElements = [];
        var allElements = document.querySelectorAll('*');

        var selectionRect = {
            left: rect.left,
            top: rect.top,
            right: rect.left + rect.width,
            bottom: rect.top + rect.height
        };

        allElements.forEach(function(element) {
            if (element.closest('#element-inspector-overlay') ||
                element.classList.contains('selection-rectangle') ||
                element.classList.contains('selection-overlay')) {
                return;
            }

            if (element instanceof HTMLElement) {
                var computed = window.getComputedStyle(element);
                if (computed.display === 'none' || computed.visibility === 'hidden') return;
            }

            var elementRect = element.getBoundingClientRect();
            var intersects = !(
                elementRect.right < selectionRect.left ||
                elementRect.left > selectionRect.right ||
                elementRect.bottom < selectionRect.top ||
                elementRect.top > selectionRect.bottom
            );

            if (intersects) {
                selectedElements.push(element);
            }
        });

        return selectedElements;
    }

    function gatherSelectionInfo(elements, rect) {
        var info = '# Rectangle Selection Debug Information\n\n' +
            '## Selection Area\n' +
            '- Position: (' + Math.round(rect.left) + ', ' + Math.round(rect.top) + ')\n' +
            '- Size: ' + Math.round(rect.width) + 'x' + Math.round(rect.height) + 'px\n' +
            '- URL: ' + window.location.href + '\n' +
            '- Elements Found: ' + elements.length + '\n\n---\n\n';

        var elementTypes = {};
        elements.forEach(function(el) {
            var tag = el.tagName.toLowerCase();
            elementTypes[tag] = (elementTypes[tag] || 0) + 1;
        });

        info += '## Element Type Summary\n';
        Object.entries(elementTypes)
            .sort(function(a, b) { return b[1] - a[1]; })
            .forEach(function(entry) {
                info += '- ' + entry[0] + ': ' + entry[1] + '\n';
            });

        info += '\n---\n\n## Detailed Element Information (first 10 elements)\n\n';

        elements.slice(0, 10).forEach(function(element, index) {
            info += '### Element ' + (index + 1) + '\n';
            info += gatherElementDebugInfo(element);
            info += '\n---\n\n';
        });

        return info;
    }

    // ============================================================================
    // Debug Snapshot (Screenshot + Console Logs)
    // ============================================================================
    function captureDebugSnapshot() {
        showCameraFlash();

        // Run screenshot and console log capture in parallel
        Promise.all([
            captureScreenshot(),
            captureConsoleLogs()
        ]).then(function(results) {
            var screenshotOk = results[0];
            var logsOk = results[1];

            if (screenshotOk && logsOk) {
                showNotification('✓ Screenshot + logs copied', 'success');
            } else if (screenshotOk) {
                showNotification('✓ Screenshot copied', 'success');
            } else if (logsOk) {
                showNotification('✓ Console logs copied', 'success');
            } else {
                showNotification('✗ Capture failed', 'error');
            }
        });
    }

    function captureScreenshot() {
        return navigator.mediaDevices.getDisplayMedia({
            video: {
                displaySurface: 'browser'
            },
            preferCurrentTab: true,
            selfBrowserSurface: 'include',
            systemAudio: 'exclude'
        }).then(function(stream) {
            var video = document.createElement('video');
            video.srcObject = stream;
            video.muted = true;

            return new Promise(function(resolve) {
                video.onloadedmetadata = function() {
                    video.play().then(function() {
                        // Small delay to ensure frame is rendered
                        setTimeout(function() {
                            var canvas = document.createElement('canvas');
                            canvas.width = video.videoWidth;
                            canvas.height = video.videoHeight;
                            var ctx = canvas.getContext('2d');

                            if (!ctx) {
                                stream.getTracks().forEach(function(t) { t.stop(); });
                                resolve(false);
                                return;
                            }

                            ctx.drawImage(video, 0, 0);
                            stream.getTracks().forEach(function(t) { t.stop(); });

                            // Convert to blob and copy to clipboard
                            canvas.toBlob(function(blob) {
                                if (!blob) {
                                    resolve(false);
                                    return;
                                }

                                navigator.clipboard.write([
                                    new ClipboardItem({ 'image/png': blob })
                                ]).then(function() {
                                    console.log('[ElementInspector] Screenshot copied to clipboard');
                                    resolve(true);
                                }).catch(function(err) {
                                    console.error('[ElementInspector] Screenshot clipboard write failed:', err);
                                    resolve(false);
                                });
                            }, 'image/png');
                        }, 100);
                    }).catch(function() {
                        stream.getTracks().forEach(function(t) { t.stop(); });
                        resolve(false);
                    });
                };

                video.onerror = function() {
                    stream.getTracks().forEach(function(t) { t.stop(); });
                    resolve(false);
                };

                // Timeout fallback
                setTimeout(function() {
                    stream.getTracks().forEach(function(t) { t.stop(); });
                    resolve(false);
                }, 5000);
            });
        }).catch(function(err) {
            // User cancelled or permission denied - this is normal
            if (err.name !== 'NotAllowedError') {
                console.error('[ElementInspector] Screenshot capture failed:', err);
            }
            return false;
        });
    }

    function captureConsoleLogs() {
        var logs = getConsoleLogs();
        return navigator.clipboard.writeText(logs).then(function() {
            console.log('[ElementInspector] Console logs copied to clipboard');
            return true;
        }).catch(function(err) {
            console.error('[ElementInspector] Failed to copy logs:', err);
            return false;
        });
    }

    function getConsoleLogs() {
        if (consoleLogs.length === 0) {
            return 'No console logs captured.';
        }

        var output = '# Console Logs\n\n';
        output += 'URL: ' + window.location.href + '\n';
        output += 'Timestamp: ' + new Date().toISOString() + '\n';
        output += 'Total Logs: ' + consoleLogs.length + '\n\n---\n\n';

        consoleLogs.forEach(function(entry) {
            var icon = entry.type === 'error' ? '❌' : entry.type === 'warn' ? '⚠️' : '📝';
            var source = entry.source ? entry.source + ' ' : '';
            output += icon + ' ' + source + entry.args.join(' ') + '\n';
        });

        return output;
    }

    // ============================================================================
    // Main Functions
    // ============================================================================
    function activate() {
        console.log('[ElementInspector] Activating...');
        isActive = true;

        // Create overlay container
        overlayContainer = document.createElement('div');
        overlayContainer.id = 'element-inspector-overlay';

        var docHeight = Math.max(
            document.body.scrollHeight,
            document.documentElement.scrollHeight,
            document.body.offsetHeight,
            document.documentElement.offsetHeight
        );

        overlayContainer.style.cssText =
            'position: absolute;' +
            'top: 0;' +
            'left: 0;' +
            'width: 100%;' +
            'height: ' + docHeight + 'px;' +
            'pointer-events: none;' +
            'z-index: 999999;';

        document.body.appendChild(overlayContainer);

        // Scan all elements
        scanElements(overlayContainer);

        console.log('[ElementInspector] Active - Press Alt+I to deactivate');
    }

    function deactivate() {
        console.log('[ElementInspector] Deactivating...');
        isActive = false;

        elementBoxMap.clear();
        currentlyHoveredBox = null;
        currentlyHoveredElement = null;

        if (overlayContainer) {
            overlayContainer.remove();
            overlayContainer = null;
        }
    }

    function toggle() {
        if (isActive) {
            deactivate();
        } else {
            activate();
        }
    }

    function refresh() {
        if (isActive) {
            deactivate();
            activate();
        }
    }

    // ============================================================================
    // Keyboard Shortcuts
    // ============================================================================
    document.addEventListener('keydown', function(e) {
        var key = e.key.toLowerCase();

        // Ctrl+Shift+I: Debug snapshot
        if (e.ctrlKey && e.shiftKey && !e.altKey && key === 'i') {
            e.preventDefault();
            e.stopPropagation();
            console.log('[ElementInspector] Ctrl+Shift+I pressed - capturing debug snapshot');
            captureDebugSnapshot();
            return;
        }

        // Ctrl+Alt+I: Start rectangle selection mode
        if (e.ctrlKey && e.altKey && !e.shiftKey && key === 'i') {
            e.preventDefault();
            startSelectionMode();
            return;
        }

        // Alt+I: Toggle inspector
        if (e.altKey && !e.shiftKey && !e.ctrlKey && key === 'i') {
            e.preventDefault();
            toggle();
            return;
        }

        // Escape: Deactivate
        if (e.key === 'Escape') {
            e.preventDefault();
            if (selectionMode) {
                cancelSelectionMode();
                deactivate();
            } else if (isActive) {
                deactivate();
            }
            return;
        }
    });

    // Set up auto-dismiss callback
    onCopyCallback = function() {
        deactivate();
    };

    // Auto-refresh on window resize
    var resizeTimeout;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(function() {
            if (isActive) {
                refresh();
            }
        }, 500);
    });

    // Export to window for manual control
    window.elementInspector = {
        toggle: toggle,
        activate: activate,
        deactivate: deactivate,
        refresh: refresh,
        captureDebugSnapshot: captureDebugSnapshot,
        getConsoleLogs: getConsoleLogs
    };

    console.log('[ElementInspector] Initialized');
    console.log('  Alt+I: Toggle inspector overlay');
    console.log('  Ctrl+Alt+I: Rectangle selection mode');
    console.log('  Ctrl+Shift+I: Debug snapshot (console logs)');
    console.log('  Escape: Deactivate inspector');

})();
