#!/usr/bin/env python3
# Timestamp: 2026-02-02
# File: scitex/_dev/_dashboard/_scripts.py

"""JavaScript for the dashboard."""


def get_javascript() -> str:
    """Return dashboard JavaScript."""
    return """
let cachedData = { packages: {}, hosts: {}, remotes: {} };

async function fetchVersions() {
    showLoading(true);
    cachedData = { packages: {}, hosts: {}, remotes: {} };
    renderFilters();
    renderData();

    // Load packages first (fast)
    fetchPackages();
    // Load hosts and remotes in parallel (slower)
    fetchHosts();
    fetchRemotes();
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

function renderData() {
    if (!cachedData) return;
    const filters = getSelectedFilters();
    const packages = cachedData.packages || {};

    const filteredPackages = Object.entries(packages)
        .filter(([name, info]) => {
            if (!filters.packages.includes(name)) return false;
            if (!filters.statuses.includes(info.status)) return false;
            return true;
        });

    const summary = {
        total: filteredPackages.length,
        ok: filteredPackages.filter(([, i]) => i.status === 'ok').length,
        unreleased: filteredPackages.filter(([, i]) => i.status === 'unreleased').length,
        mismatch: filteredPackages.filter(([, i]) => i.status === 'mismatch').length,
        outdated: filteredPackages.filter(([, i]) => i.status === 'outdated').length
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

        const hostVersions = Object.entries(hostData)
            .filter(([h]) => !h.startsWith('_') && filters.hosts.includes(h))
            .map(([hostName, hostInfo]) => ({ name: hostName, ...(hostInfo[name] || {}) }));

        const remoteVersions = Object.entries(remoteData)
            .filter(([r]) => !r.startsWith('_') && filters.remotes.includes(r))
            .map(([remoteName, remoteInfo]) => ({ name: remoteName, ...(remoteInfo[name] || {}) }));

        return renderPackageCard(name, info, local, git, remote, hostVersions, remoteVersions);
    }).join('');
}

function renderPackageCard(name, info, local, git, remote, hostVersions, remoteVersions) {
    let html = `
        <div class="package-card">
            <div class="package-header">
                <span class="package-name">${name}</span>
                <span class="status-badge status-${info.status}">${info.status}</span>
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
                        <h4>PYPI</h4>
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
        html += `<div class="version-section"><h4>GITHUB</h4>`;
        remoteVersions.forEach(r => {
            html += `<div class="version-item"><span class="key">${r.name}</span><span class="value">${r.latest_tag || r.error || '-'}</span></div>`;
        });
        html += `</div>`;
    }

    html += `</div>`;

    if (info.issues && info.issues.length > 0) {
        html += `<div class="issues"><h4>Issues</h4><ul>`;
        info.issues.forEach(i => { html += `<li>${i}</li>`; });
        html += `</ul></div>`;
    }

    html += `</div></div>`;
    return html;
}

async function refreshData() { await fetchVersions(); }

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
