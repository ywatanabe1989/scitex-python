#!/usr/bin/env python3
# Timestamp: 2026-02-02
# File: scitex/_dev/_dashboard/_scripts.py

"""JavaScript for the dashboard."""


def get_javascript() -> str:
    """Return dashboard JavaScript."""
    return """
let cachedData = { packages: {}, hosts: {}, remotes: {}, rtd: {} };

async function fetchVersions() {
    showLoading(true);
    cachedData = { packages: {}, hosts: {}, remotes: {}, rtd: {} };
    renderFilters();
    renderData();

    // Load packages first (fast)
    fetchPackages();
    // Load hosts, remotes, and RTD in parallel (slower)
    fetchHosts();
    fetchRemotes();
    fetchRtd();
}

async function fetchPackages() {
    setSectionLoading('package', true);
    try {
        const response = await fetch('/api/packages');
        cachedData.packages = await response.json();
        renderFilters();
        renderData();
        updateTimestamp();
        setSectionUpdated('package');
    } catch (err) {
        console.error('Failed to fetch packages:', err);
    } finally {
        showLoading(false);
        setSectionLoading('package', false);
    }
}

async function fetchHosts() {
    setSectionLoading('host', true);
    try {
        const response = await fetch('/api/hosts');
        cachedData.hosts = await response.json();
        renderFilters();
        renderData();
        setSectionUpdated('host');
    } catch (err) {
        console.error('Failed to fetch hosts:', err);
        cachedData.hosts = { error: err.message };
    } finally {
        setSectionLoading('host', false);
    }
}

async function fetchRemotes() {
    setSectionLoading('remote', true);
    try {
        const response = await fetch('/api/remotes');
        cachedData.remotes = await response.json();
        renderFilters();
        renderData();
        setSectionUpdated('remote');
    } catch (err) {
        console.error('Failed to fetch remotes:', err);
        cachedData.remotes = { error: err.message };
    } finally {
        setSectionLoading('remote', false);
    }
}

async function fetchRtd() {
    setSectionLoading('rtd', true);
    try {
        const response = await fetch('/api/rtd');
        cachedData.rtd = await response.json();
        renderFilters();
        renderData();
        setSectionUpdated('rtd');
    } catch (err) {
        console.error('Failed to fetch RTD status:', err);
        cachedData.rtd = { error: err.message };
    } finally {
        setSectionLoading('rtd', false);
    }
}

function setSectionLoading(section, loading) {
    const el = document.getElementById(section + 'Filters');
    if (el) {
        if (loading) {
            el.classList.add('loading-section');
        } else {
            el.classList.remove('loading-section');
        }
    }
}

function setSectionUpdated(section) {
    const el = document.getElementById(section + 'Filters');
    if (el) {
        el.classList.add('just-updated');
        setTimeout(() => el.classList.remove('just-updated'), 1000);
    }
}

function updateTimestamp() {
    document.getElementById('lastUpdated').textContent =
        'Last updated: ' + new Date().toLocaleTimeString();
}

// Auto-refresh settings
let autoRefreshInterval = null;
let autoRefreshSeconds = 0;

function toggleAutoRefresh(seconds) {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        autoRefreshSeconds = 0;
        document.getElementById('autoRefreshBtn').textContent = 'Auto: Off';
        return;
    }
    autoRefreshSeconds = seconds;
    document.getElementById('autoRefreshBtn').textContent = `Auto: ${seconds}s`;
    autoRefreshInterval = setInterval(() => {
        fetchPackages();
        fetchHosts();
        fetchRemotes();
    }, seconds * 1000);
}

function cycleAutoRefresh() {
    const options = [0, 30, 60, 120];
    const current = options.indexOf(autoRefreshSeconds);
    const next = options[(current + 1) % options.length];
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
    }
    if (next > 0) {
        toggleAutoRefresh(next);
    } else {
        autoRefreshSeconds = 0;
        document.getElementById('autoRefreshBtn').textContent = 'Auto: Off';
    }
}

function showLoading(show) {
    document.getElementById('loading').classList.toggle('active', show);
    document.getElementById('overlay').classList.toggle('active', show);
}

function renderFilters() {
    if (!cachedData) return;

    const packageFilters = document.getElementById('packageFilters');
    const packages = Object.keys(cachedData.packages || {});
    packageFilters.innerHTML = packages.map(pkg =>
        `<label><input type="checkbox" value="${pkg}" checked onchange="renderData()"> ${pkg}</label>`
    ).join('');

    const hostFilters = document.getElementById('hostFilters');
    const hosts = Object.keys(cachedData.hosts || {});
    if (hosts.length > 0) {
        hostFilters.innerHTML = hosts.map(host =>
            `<label><input type="checkbox" value="${host}" checked onchange="renderData()"> ${host}</label>`
        ).join('');
    } else {
        hostFilters.innerHTML = '<span style="color: var(--text-secondary)">No hosts configured</span>';
    }

    const remoteFilters = document.getElementById('remoteFilters');
    const remotes = Object.keys(cachedData.remotes || {});
    if (remotes.length > 0) {
        remoteFilters.innerHTML = remotes.map(remote =>
            `<label><input type="checkbox" value="${remote}" checked onchange="renderData()"> ${remote}</label>`
        ).join('');
    } else {
        remoteFilters.innerHTML = '<span style="color: var(--text-secondary)">No remotes configured</span>';
    }

    const rtdFilters = document.getElementById('rtdFilters');
    const rtdVersions = Object.keys(cachedData.rtd || {});
    if (rtdVersions.length > 0) {
        rtdFilters.innerHTML = rtdVersions.map(v =>
            `<label><input type="checkbox" value="${v}" checked onchange="renderData()"> ${v}</label>`
        ).join('');
    } else {
        rtdFilters.innerHTML = '<span style="color: var(--text-secondary)">Loading...</span>';
    }

    document.querySelectorAll('#statusFilters input').forEach(input => {
        input.onchange = renderData;
    });
}

function getSelectedFilters() {
    const getChecked = (containerId) =>
        [...document.querySelectorAll(`#${containerId} input:checked`)].map(el => el.value);
    return {
        packages: getChecked('packageFilters'),
        statuses: getChecked('statusFilters'),
        hosts: getChecked('hostFilters'),
        remotes: getChecked('remoteFilters')
    };
}

function getEffectiveStatus(name, info) {
    // Check RTD status and downgrade if needed
    const rtdData = cachedData.rtd || {};
    let status = info.status;
    if (status === 'ok') {
        const rtdLatest = rtdData['latest'] && rtdData['latest'][name];
        const rtdStable = rtdData['stable'] && rtdData['stable'][name];
        if ((rtdLatest && rtdLatest.status === 'failing') ||
            (rtdStable && rtdStable.status === 'failing') ||
            (rtdLatest && rtdLatest.status === 'not_found')) {
            status = 'mismatch';
        }
    }
    return status;
}

function renderData() {
    if (!cachedData) return;
    const filters = getSelectedFilters();
    const packages = cachedData.packages || {};

    const filteredPackages = Object.entries(packages)
        .filter(([name, info]) => {
            if (!filters.packages.includes(name)) return false;
            const effectiveStatus = getEffectiveStatus(name, info);
            if (!filters.statuses.includes(effectiveStatus)) return false;
            return true;
        });

    const summary = {
        total: filteredPackages.length,
        ok: filteredPackages.filter(([n, i]) => getEffectiveStatus(n, i) === 'ok').length,
        unreleased: filteredPackages.filter(([n, i]) => getEffectiveStatus(n, i) === 'unreleased').length,
        mismatch: filteredPackages.filter(([n, i]) => getEffectiveStatus(n, i) === 'mismatch').length,
        outdated: filteredPackages.filter(([n, i]) => getEffectiveStatus(n, i) === 'outdated').length
    };

    document.getElementById('summary').innerHTML = `
        <div class="summary-card total"><div class="number">${summary.total}</div><div class="label">Total</div></div>
        <div class="summary-card ok"><div class="number">${summary.ok}</div><div class="label">OK</div></div>
        <div class="summary-card unreleased"><div class="number">${summary.unreleased}</div><div class="label">Unreleased</div></div>
        <div class="summary-card mismatch"><div class="number">${summary.mismatch}</div><div class="label">Mismatch</div></div>
    `;

    document.getElementById('packages').innerHTML = filteredPackages.map(([name, info]) => {
        const local = info.local || {};
        const git = info.git || {};
        const remote = info.remote || {};
        const hostData = cachedData.hosts || {};
        const remoteData = cachedData.remotes || {};
        const rtdData = cachedData.rtd || {};

        const hostVersions = Object.entries(hostData)
            .filter(([h]) => !h.startsWith('_') && filters.hosts.includes(h))
            .map(([hostName, hostInfo]) => ({ name: hostName, ...(hostInfo[name] || {}) }));

        const remoteVersions = Object.entries(remoteData)
            .filter(([r]) => !r.startsWith('_') && filters.remotes.includes(r))
            .map(([remoteName, remoteInfo]) => ({ name: remoteName, ...(remoteInfo[name] || {}) }));

        // Get RTD status for this package (latest and stable)
        const rtdStatus = {};
        Object.entries(rtdData).forEach(([version, pkgData]) => {
            if (pkgData[name]) {
                rtdStatus[version] = pkgData[name];
            }
        });

        return renderPackageCard(name, info, local, git, remote, hostVersions, remoteVersions, rtdStatus);
    }).join('');
}

function renderPackageCard(name, info, local, git, remote, hostVersions, remoteVersions, rtdStatus) {
    const pypiUrl = `https://pypi.org/project/${name}/`;
    const githubUrl = `https://github.com/ywatanabe1989/${name}`;
    const rtdUrl = `https://${name === 'scitex' ? 'scitex-python' : name}.readthedocs.io/`;

    // Collect all issues for tooltip
    let allIssues = [...(info.issues || [])];

    // Re-evaluate status based on RTD
    let effectiveStatus = info.status;
    if (rtdStatus && Object.keys(rtdStatus).length > 0) {
        const rtdLatest = rtdStatus['latest'];
        const rtdStable = rtdStatus['stable'];
        if (rtdLatest && rtdLatest.status === 'failing') {
            allIssues.push('RTD latest build failing');
            if (effectiveStatus === 'ok') effectiveStatus = 'mismatch';
        }
        if (rtdStable && rtdStable.status === 'failing') {
            allIssues.push('RTD stable build failing');
            if (effectiveStatus === 'ok') effectiveStatus = 'mismatch';
        }
        if (rtdLatest && rtdLatest.status === 'not_found') {
            allIssues.push('RTD project not found');
            if (effectiveStatus === 'ok') effectiveStatus = 'mismatch';
        }
    }

    // Create tooltip text from issues (using &#10; for newlines in title attribute)
    const tooltipText = allIssues.length > 0 ? allIssues.join('&#10;') : '';
    const tooltipAttr = tooltipText ? `title="${tooltipText}"` : '';

    let html = `
        <div class="package-card collapsed">
            <div class="package-header" onclick="toggleCard(this)">
                <span class="fold-icon">▶</span>
                <a href="${githubUrl}" target="_blank" class="package-name" onclick="event.stopPropagation()">${name}</a>
                <span class="status-badge status-${effectiveStatus}" ${tooltipAttr}>${effectiveStatus}</span>
                <span class="quick-links">
                    <a href="${pypiUrl}" target="_blank" title="PyPI" onclick="event.stopPropagation()">📦</a>
                    <a href="${githubUrl}" target="_blank" title="GitHub" onclick="event.stopPropagation()">🐙</a>
                    <a href="${rtdUrl}" target="_blank" title="Docs" onclick="event.stopPropagation()">📖</a>
                </span>
            </div>
            <div class="package-body">
                <div class="version-grid">
                    <div class="version-section">
                        <h4>LOCAL</h4>
                        <div class="version-item"><span class="key">toml</span><span class="value">${local.pyproject_toml || '-'}</span></div>
                        <div class="version-item"><span class="key">installed</span><span class="value">${local.installed || '-'}</span></div>
                    </div>
                    <div class="version-section">
                        <h4>GIT</h4>
                        <div class="version-item"><span class="key">tag</span><span class="value">${git.latest_tag || '-'}</span></div>
                        <div class="version-item"><span class="key">branch</span><span class="value">${git.branch || '-'}</span></div>
                    </div>
                    <div class="version-section">
                        <h4><a href="https://pypi.org/project/${name}/" target="_blank">PYPI</a></h4>
                        <div class="version-item"><span class="key">published</span><span class="value">${remote.pypi || '-'}</span></div>
                    </div>`;

    if (hostVersions.length > 0) {
        hostVersions.forEach(h => {
            html += `<div class="version-section"><h4>${h.name.toUpperCase()}</h4>`;
            html += `<div class="version-item"><span class="key">toml</span><span class="value">${h.toml || '-'}</span></div>`;
            html += `<div class="version-item"><span class="key">installed</span><span class="value">${h.installed || h.error || '-'}</span></div>`;
            html += `</div>`;
        });
    }

    if (remoteVersions.length > 0) {
        html += `<div class="version-section"><h4><a href="${githubUrl}" target="_blank">GITHUB</a></h4>`;
        remoteVersions.forEach(r => {
            html += `<div class="version-item"><span class="key">${r.name}</span><span class="value">${r.latest_tag || r.error || '-'}</span></div>`;
        });
        html += `</div>`;
    }

    if (rtdStatus && Object.keys(rtdStatus).length > 0) {
        html += `<div class="version-section"><h4><a href="${rtdUrl}" target="_blank">RTD</a></h4>`;
        Object.entries(rtdStatus).forEach(([version, data]) => {
            const statusClass = data.status === 'passing' ? 'rtd-passing' : (data.status === 'failing' ? 'rtd-failing' : 'rtd-unknown');
            const statusIcon = data.status === 'passing' ? '✓' : (data.status === 'failing' ? '✗' : '?');
            const link = data.url ? `<a href="${data.url}" target="_blank">${statusIcon}</a>` : statusIcon;
            html += `<div class="version-item"><span class="key">${version}</span><span class="value ${statusClass}">${link} ${data.status || '-'}</span></div>`;
        });
        html += `</div>`;
    }

    html += `</div>`;

    if (allIssues.length > 0) {
        html += `<div class="issues"><h4>Issues</h4><ul>`;
        allIssues.forEach(i => { html += `<li>${i}</li>`; });
        html += `</ul></div>`;
    }

    html += `</div></div>`;
    return html;
}

async function refreshData() { await fetchVersions(); }

function toggleCard(header) {
    const card = header.parentElement;
    card.classList.toggle('collapsed');
}

function toggleAllCards(expand) {
    document.querySelectorAll('.package-card').forEach(card => {
        if (expand) {
            card.classList.remove('collapsed');
        } else {
            card.classList.add('collapsed');
        }
    });
}

function toggleFilters() {
    const filters = document.querySelector('.filters');
    filters.classList.toggle('collapsed');
}

function exportJSON() {
    if (!cachedData) return;
    const blob = new Blob([JSON.stringify(cachedData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'scitex-versions.json';
    a.click();
    URL.revokeObjectURL(url);
}

fetchVersions();
"""


# EOF
