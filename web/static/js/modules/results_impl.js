/* implementations for result rendering and modal helpers */
(function(){
    window.resultsImpl = window.resultsImpl || {};
    const RECORDS_STORAGE_KEY = 'detection_records';
    const MAX_RECORDS_PER_KIND = 500;
    const PERSIST_DEBOUNCE_MS = 300; // debounce writes to localStorage
    const PERSIST_BACKOFF_MS = 60 * 1000; // backoff on quota error

    window.resultsImpl.toPersistableEntry = function(entry) {
        const safe = entry && typeof entry === 'object' ? entry : {};
        const data = Object.assign({}, (safe.data && typeof safe.data === 'object') ? safe.data : {});
        // Avoid oversized localStorage payloads; records list does not require full visualization blobs.
        delete data.imageData;
        delete data.image;
        delete data.visualization;
        delete data.visualization_data;
        return {
            name: safe.name || '',
            size: safe.size || '-',
            time: safe.time || '',
            data,
            options: Object.assign({}, (safe.options && typeof safe.options === 'object') ? safe.options : {})
        };
    };

    window.resultsImpl.persistRecordStores = function() {
        try {
            window.resultsImpl = window.resultsImpl || {};
            // avoid concurrent persist runs
            if (window.resultsImpl._persistInFlight) return;
            window.resultsImpl._persistInFlight = true;
            // respect backoff after quota error
            const now = Date.now();
            if (window.resultsImpl._persistBackoffUntil && now < window.resultsImpl._persistBackoffUntil) { window.resultsImpl._persistInFlight = false; return; }

            window.recordStores = window.recordStores || { live: [], upload: [] };
            const buildPayload = (maxEntries) => ({
                upload: (Array.isArray(window.recordStores.upload) ? window.recordStores.upload : [])
                    .slice(-maxEntries)
                    .map(window.resultsImpl.toPersistableEntry),
                live: (Array.isArray(window.recordStores.live) ? window.recordStores.live : [])
                    .slice(-maxEntries)
                    .map(window.resultsImpl.toPersistableEntry)
            });

            if (!(window.localStorage && window.localStorage.setItem)) return;

            // first try full size
            let payload = buildPayload(MAX_RECORDS_PER_KIND);
            try {
                window.localStorage.setItem(RECORDS_STORAGE_KEY, JSON.stringify(payload));
                // clear any prior backoff
                window.resultsImpl._persistBackoffUntil = 0;
                return;
            } catch (e) {
                // if quota exceeded, attempt to trim payload and retry a few times
                const isQuota = e && (e.name === 'QuotaExceededError' || e.code === 22 || /quota/i.test(String(e.message || '')));
                if (!isQuota) {
                    // warn but throttle repeated messages
                    const last = window.resultsImpl._lastPersistWarnTs || 0;
                    if (Date.now() - last > 5000) {
                        console.warn('resultsImpl.persistRecordStores failed', e);
                        window.resultsImpl._lastPersistWarnTs = Date.now();
                    }
                    window.resultsImpl._persistInFlight = false;
                    return;
                }
                // set a short backoff immediately to avoid many concurrent retries
                window.resultsImpl._persistBackoffUntil = Date.now() + PERSIST_BACKOFF_MS;
                // attempt with progressively smaller payloads (best-effort)
                const sizes = [200, 100, 50, 20, 5];
                let success = false;
                for (const s of sizes) {
                    try {
                        payload = buildPayload(s);
                        window.localStorage.setItem(RECORDS_STORAGE_KEY, JSON.stringify(payload));
                        success = true; break;
                    } catch (ee) { /* try next */ }
                }
                if (!success) {
                    // give up for a while to avoid repeated console spam
                    const last = window.resultsImpl._lastPersistWarnTs || 0;
                    if (Date.now() - last > 5000) {
                        console.warn('resultsImpl.persistRecordStores failing due to storage quota — backing off for', PERSIST_BACKOFF_MS, 'ms');
                        window.resultsImpl._lastPersistWarnTs = Date.now();
                    }
                    window.resultsImpl._persistInFlight = false;
                    return;
                }
                // success: clear backoff so future writes may resume
                window.resultsImpl._persistBackoffUntil = 0;
            }
        } catch (e) {
            console.warn('resultsImpl.persistRecordStores failed', e);
        }
        finally {
            try { window.resultsImpl._persistInFlight = false; } catch(_){}
        }
    };

    window.resultsImpl.restoreRecordStores = function() {
        try {
            window.recordStores = window.recordStores || { live: [], upload: [] };
            if (!(window.localStorage && window.localStorage.getItem)) return;
            const raw = window.localStorage.getItem(RECORDS_STORAGE_KEY);
            if (!raw) return;
            const data = JSON.parse(raw);
            const restoredUpload = Array.isArray(data && data.upload) ? data.upload : [];
            const restoredLive = Array.isArray(data && data.live) ? data.live : [];
            window.recordStores.upload = restoredUpload.slice(-MAX_RECORDS_PER_KIND);
            window.recordStores.live = restoredLive.slice(-MAX_RECORDS_PER_KIND);
        } catch (e) {
            console.warn('resultsImpl.restoreRecordStores failed', e);
        }
    };

    window.resultsImpl.normalizeDetailPayload = function(data) {
        const d = data && typeof data === 'object' ? data : {};
        return { name: d.name || '', statusText: d.statusText || d.status || d.confidence_label || d.result || '', conf: d.conf || d.confidence || '0', execTime: d.execution_time || d.execTime || d.duration_ms || d.time_ms || '-', captureTime: d.capture_time || d.captureTime || d.timestamp || '', imageData: d.imageData || d.image_data || d.image || d.visualization || d.visualization_data || '', payloadDist: d.payloadDist || d.payload_dist || d.payload_distribution || d.histogram || null, flows: Array.isArray(d.flows) ? d.flows : [], sessionParseReport: d.sessionParseReport || d.session_parse_report || d.parse_report || d.session_report || null };
    };

    window.resultsImpl.formatConfidenceText = function(value) {
        if (value === null || value === undefined || value === '') return '0.00%';
        const num = typeof value === 'string' ? parseFloat(value) : value;
        if (!Number.isFinite(num)) return String(value);
        return num <= 1 ? (num * 100).toFixed(2) + '%' : num.toFixed(2) + '%';
    };

    window.resultsImpl.formatExecTime = function(value) {
        if (value === null || value === undefined || value === '') return '-';
        if (typeof value === 'number') return value >= 1000 ? (value / 1000).toFixed(2) + ' s' : value.toFixed(2) + ' ms';
        return String(value);
    };

    window.resultsImpl.formatCaptureTime = function(value) {
        if (value === null || value === undefined || value === '') return '-';
        if (typeof value === 'number') {
            const ms = value < 1e12 ? value * 1000 : value;
            const d = new Date(ms);
            return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString('zh-CN');
        }
        const d = new Date(value);
        return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString('zh-CN');
    };

    window.resultsImpl.formatSessionParseReport = function(report) {
        if (!report || typeof report !== 'object') return '';
        const parts = [];
        if (typeof report.total_packets === 'number') parts.push(`总包数 ${report.total_packets}`);
        if (typeof report.accepted_packets === 'number') parts.push(`已纳入 ${report.accepted_packets}`);
        if (typeof report.session_count === 'number') parts.push(`session 数 ${report.session_count}`);
        if (typeof report.too_short === 'number') parts.push(`过短 ${report.too_short}`);
        if (typeof report.non_ipv4 === 'number') parts.push(`非 IPv4 ${report.non_ipv4}`);
        if (typeof report.vlan_too_short === 'number') parts.push(`VLAN 过短 ${report.vlan_too_short}`);
        if (typeof report.parse_errors === 'number') parts.push(`解析异常 ${report.parse_errors}`);
        return parts.length > 0 ? parts.join('，') : '';
    };

    window.resultsImpl.buildFallbackSessionReport = function(flows) {
        const list = Array.isArray(flows) ? flows : [];
        if (!list.length) return '实时抓包未提供 session 解析统计，当前也没有可展示的流记录。';
        const mal = list.filter((f) => f && f.is_malicious).length;
        const benign = list.length - mal;
        const first = list[0] || {};
        const src = `${first.src_ip || first.src || '-'}:${first.src_port || first.sport || '-'}`;
        const dst = `${first.dst_ip || first.dst || '-'}:${first.dst_port || first.dport || '-'}`;
        return `实时抓包未返回 session 解析统计；当前展示 ${list.length} 条流记录（Malware ${mal}，Benign ${benign}）。示例流：${src} → ${dst}`;
    };

    window.resultsImpl.buildDetailVisualizationDataUri = function(name, result, conf, imageData, payloadDist, flows) {
        if (imageData && typeof imageData === 'string' && imageData.length > 0) return imageData.startsWith('data:') ? imageData : ('data:image/png;base64,' + imageData);
        const list = Array.isArray(flows) ? flows : [];
        const mal = list.filter((f) => f && f.is_malicious).length;
        const benign = list.length - mal;
        const labels = ['0x00','0x10','0x20','0x30','0x40','0x50','0x60','0x70','0x80','0x90','0xA0','0xB0','0xC0','0xD0','0xE0','0xF0'];
        const bins = Array.isArray(payloadDist) && payloadDist.length === 16 ? payloadDist : labels.map((_, i) => Math.max(4, Math.round((i % 4 + 1) * (1 + list.length / 6))));
        const maxBin = Math.max(...bins, 1);
        const barWidth = 10; const gap = 3; const chartX = 26; const chartY = 126; const chartH = 60;
        const bars = bins.map((v, i) => { const h = Math.max(2, Math.round((v / maxBin) * chartH)); const x = chartX + i * (barWidth + gap); const y = chartY + (chartH - h); return `<rect x="${x}" y="${y}" width="${barWidth}" height="${h}" rx="2" fill="url(#barGrad)" opacity="0.95"/>`; }).join('');
        const svg = `\n<svg xmlns="http://www.w3.org/2000/svg" width="560" height="260" viewBox="0 0 560 260">\n    <defs>\n        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">\n            <stop offset="0%" stop-color="#0b0f1e"/>\n            <stop offset="100%" stop-color="#101a35"/>\n        </linearGradient>\n        <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">\n            <stop offset="0%" stop-color="#22d3ee"/>\n            <stop offset="100%" stop-color="#2563eb"/>\n        </linearGradient>\n        <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">\n            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>\n            <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>\n        </filter>\n    </defs>\n    <rect width="560" height="260" fill="url(#bg)" rx="18"/>\n    <circle cx="110" cy="82" r="34" fill="rgba(34,211,238,0.15)" stroke="#22d3ee" stroke-width="2"/>\n    <circle cx="450" cy="82" r="34" fill="rgba(37,99,235,0.15)" stroke="#60a5fa" stroke-width="2"/>\n    <line x1="145" y1="82" x2="415" y2="82" stroke="#22d3ee" stroke-width="4" stroke-linecap="round" filter="url(#glow)"/>\n    <polygon points="415,82 402,74 402,90" fill="#22d3ee"/>\n    <text x="110" y="78" text-anchor="middle" fill="#e5e7eb" font-size="14" font-family="sans-serif">SRC</text>\n    <text x="110" y="98" text-anchor="middle" fill="#9ca3af" font-size="11" font-family="monospace">${(list[0] && (list[0].src_ip || list[0].src)) || 'unknown'}</text>\n    <text x="450" y="78" text-anchor="middle" fill="#e5e7eb" font-size="14" font-family="sans-serif">DST</text>\n    <text x="450" y="98" text-anchor="middle" fill="#9ca3af" font-size="11" font-family="monospace">${(list[0] && (list[0].dst_ip || list[0].dst)) || 'unknown'}</text>\n    <text x="280" y="58" text-anchor="middle" fill="#e5e7eb" font-size="18" font-weight="700" font-family="sans-serif">${name || 'Traffic Detail'}</text>\n    <text x="280" y="80" text-anchor="middle" fill="#7dd3fc" font-size="12" font-family="monospace">${result || '-'} · ${conf || '0.00%'}</text>\n    <text x="280" y="118" text-anchor="middle" fill="#cbd5e1" font-size="12" font-family="sans-serif">Flow Summary: ${list.length} records · Malware ${mal} · Benign ${benign}</text>\n    ${bars}\n    <text x="26" y="204" fill="#9ca3af" font-size="10" font-family="monospace">Payload histogram fallback when original image is unavailable</text>\n</svg>`;
        return 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(svg);
    };

    window.resultsImpl.escapeSvgText = function(text) { return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&apos;'); };

    window.resultsImpl.showDetails = function(name, result, conf, execTime, captureTime, imageData, payloadDist, flows, sessionParseReport) {
        if (window.modalHelpers && typeof window.modalHelpers.showDetails === 'function') {
            return window.modalHelpers.showDetails(name, result, conf, execTime, captureTime, imageData, payloadDist, flows, sessionParseReport);
        }
        console.warn('resultsImpl.showDetails placeholder');
        alert('详情弹窗未初始化，请刷新页面后重试。');
        return null;
    };

    window.resultsImpl.addResultRow = function(name, size, time, data, options = {}) {
        try {
            const tbody = document.getElementById('resultTableBody');
            if (!tbody) return 'row_stub_' + Date.now();
            const recordKind = options.recordKind || 'upload';
            const currentMode = window.recordTableMode || 'upload';
            if (!options.skipCache) { window.cacheRecordedRow && window.cacheRecordedRow(recordKind, { name, size, time, data: Object.assign({}, data || {}), options: Object.assign({}, options || {}, { recordKind }) }); }
            const shouldRender = !!options.forceRender || recordKind === currentMode;
            if (!shouldRender) return 'row_stub_' + Date.now();
            const row = document.createElement('tr');
            row.className = 'border-b border-cyber-border/50';
            const status = data && data.status ? String(data.status) : '-';
            const isMal = status.includes('Malicious') || status.includes('Malware');
            row.dataset.recordType = isMal ? 'malware' : 'benign';
            const displayConf = data && (data.confidence || data.conf) ? (parseFloat(data.confidence || data.conf) * 100).toFixed(2) : '-';
            function formatDuration(d) { try { if (!d || typeof d !== 'object') return '-'; if (typeof d.execution_time === 'string' && d.execution_time.trim()) return d.execution_time; if (typeof d.elapsed_time === 'string' && d.elapsed_time.trim()) return d.elapsed_time; const msKeys = ['elapsed_ms', 'processing_time_ms', 'duration_ms', 'elapsed']; const sKeys = ['duration_s', 'processing_time_s', 'elapsed_s']; for (const k of msKeys) if (k in d && typeof d[k] === 'number') return (d[k] >= 1000 ? (d[k] / 1000).toFixed(2) + ' s' : d[k].toFixed(2) + ' ms'); for (const k of sKeys) if (k in d && typeof d[k] === 'number') return d[k].toFixed(2) + ' s'; if ('time_ms' in d && typeof d.time_ms === 'number') return (d.time_ms/1000).toFixed(2) + ' s'; if (typeof d.execution_time === 'number') return d.execution_time >= 1000 ? (d.execution_time/1000).toFixed(2) + ' s' : d.execution_time.toFixed(2) + ' ms'; const possibleKeys = ['execution_time', 'executionTime', 'elapsed', 'duration', 'time']; for (const k of possibleKeys) if (k in d && typeof d[k] === 'string' && d[k].trim()) return d[k].trim(); } catch (e) {} return '-'; }
            const durationText = formatDuration(data);
            const rawModelName = data && (data.model_name || data.detected_model_name || data.model || data.detected_model || data.model_type)
                ? (data.model_name || data.detected_model_name || data.model || data.detected_model || data.model_type)
                : window.currentActiveModel;
            const modelName = window.getModelDisplayName ? window.getModelDisplayName(rawModelName) : rawModelName;
            const rowId = 'row_' + Date.now();
            window.__detailsDataMap = window.__detailsDataMap || {};
            window.__detailsDataMap[rowId] = window.resultsImpl.normalizeDetailPayload(Object.assign({ rowId, name, size, time, statusText: isMal ? 'Malware' : 'Benign', conf: data && (data.confidence || data.conf || data.malware_conf) ? String(data.confidence || data.conf || data.malware_conf) : '0', }, data || {}));
            function protoToName(p) { if (p === null || p === undefined || p === '') return '-'; const n = Number(p); const map = { 1: 'ICMP', 6: 'TCP', 17: 'UDP', 41: 'IPv6', 47: 'GRE', 50: 'ESP', 51: 'AH', 89: 'OSPF' }; if (Number.isFinite(n) && map[n]) return map[n] + ' (' + n + ')'; return String(p); }
            const displaySize = (size && size !== '-' && size !== null && size !== undefined) ? size : (data && (data.size || data.bytes || data.length) ? (data.size || data.bytes || data.length) : (data && (data.proto || data.protocol) ? protoToName(data.proto || data.protocol) : '-'));
            row.innerHTML = `<td class="px-6 py-4 font-mono text-white truncate max-w-[200px]" title="${name}">${name}</td>` + `<td class="px-6 py-4 text-gray-400">${time}</td>` + `<td class="px-6 py-4 font-mono text-cyan-400">${durationText}</td>` + `<td class="px-6 py-4 text-gray-400">${displaySize}</td>` + `<td class="px-6 py-4">${isMal ? '<span class="text-cyber-danger">Malware</span>' : '<span class="text-cyber-success">Benign</span>'}</td>` + `<td class="px-6 py-4 text-gray-300">${modelName}</td>` + `<td class="px-6 py-4 font-mono text-cyber-primary">${displayConf}%</td>` + `<td class="px-6 py-4"><button type="button" data-row-id="${rowId}" class="text-gray-400 hover:text-white transition-colors text-sm underline view-details-btn">详情</button></td>`;
            tbody.insertBefore(row, tbody.firstChild);
            return rowId;
        } catch (e) { console.warn('resultsImpl.addResultRow failed', e); return 'row_stub_' + Date.now(); }
    };

    window.resultsImpl.openDetails = function(rowId) {
        try {
            const map = window.__detailsDataMap || {};
            const d = window.resultsImpl.normalizeDetailPayload(map[rowId] || {});
            if (!d) { alert('未找到该检测结果的详情数据，请重新执行一次检测。'); return; }
            window.resultsImpl.showDetails(d.name || rowId, d.statusText || d.status || '-', window.resultsImpl.formatConfidenceText(d.confidence || d.conf), window.resultsImpl.formatExecTime(d.execTime || d.processing_time_ms || d.duration_ms || d.elapsed_ms || d.time_ms), window.resultsImpl.formatCaptureTime(d.captureTime || d.captured_at || d.timestamp || d.time), d.imageData || d.image || d.visualization || '', d.payloadDist || d.payload_dist || d.histogram || null, d.flows || d.session_flows || d.records || [], d.sessionParseReport || d.session_parse_report || d.session_report || null);
        } catch (e) { console.warn('resultsImpl.openDetails error', e); try { alert('显示详情时出错: ' + (e && e.message ? e.message : String(e))); } catch(_){} }
    };

    window.resultsImpl.bindDetailButtonEvents = function() {
        try {
            const tbody = document.getElementById('resultTableBody');
            if (!tbody || tbody._detailClickBound) return;
            tbody._detailClickBound = true;
            tbody.addEventListener('click', function(ev) {
                const btn = ev.target && ev.target.closest ? ev.target.closest('.view-details-btn') : null;
                if (!btn) return;
                const rowId = btn.getAttribute('data-row-id');
                if (rowId) window.openDetails && window.openDetails(rowId);
            });
        } catch (e) { console.warn('bindDetailButtonEvents failed', e); }
    };

    window.resultsImpl.bindClearButtons = function() {
        try {
            const uploadBtn = document.getElementById('clearUploadLogsBtn');
            if (uploadBtn && !uploadBtn._boundClick) {
                uploadBtn._boundClick = () => window.resultsImpl.clearUploadRecords();
                uploadBtn.addEventListener('click', uploadBtn._boundClick);
            }
            // keep visibility in sync with current record table mode (hide upload-clear on live view)
            function updateClearButtonsVisibility() {
                try {
                    const mode = window.recordTableMode || 'upload';
                    if (uploadBtn) uploadBtn.style.display = (mode === 'upload') ? '' : 'none';
                } catch (e) { /* ignore */ }
            }
            // initial visibility
            try { updateClearButtonsVisibility(); } catch (e) {}
            // watch tab buttons if present to update visibility on user click
            try {
                const tabLive = document.getElementById('tabLive');
                const tabUpload = document.getElementById('tabUpload');
                if (tabLive && !tabLive._clearVisBound) { tabLive._clearVisBound = true; tabLive.addEventListener('click', () => { window.recordTableMode = 'live'; updateClearButtonsVisibility(); }); }
                if (tabUpload && !tabUpload._clearVisBound) { tabUpload._clearVisBound = true; tabUpload.addEventListener('click', () => { window.recordTableMode = 'upload'; updateClearButtonsVisibility(); }); }
            } catch (e) {}
        } catch (e) { console.warn('bindClearButtons failed', e); }
    };

    window.resultsImpl.cacheRecordedRow = function(kind, entry) {
        window.recordStores = window.recordStores || { live: [], upload: [] };
        if (!window.recordStores[kind]) window.recordStores[kind] = [];
        window.recordStores[kind].push(entry);
        if (window.recordStores[kind].length > MAX_RECORDS_PER_KIND) {
            window.recordStores[kind] = window.recordStores[kind].slice(-MAX_RECORDS_PER_KIND);
        }
        // schedule persisted write (debounced) to avoid frequent localStorage writes
        window.resultsImpl.schedulePersist && window.resultsImpl.schedulePersist();
    };

    // debounce wrapper for persistRecordStores
    window.resultsImpl.schedulePersist = function() {
        try {
            window.resultsImpl = window.resultsImpl || {};
            if (window.resultsImpl._persistTimer) clearTimeout(window.resultsImpl._persistTimer);
            window.resultsImpl._persistTimer = setTimeout(() => {
                try { window.resultsImpl.persistRecordStores(); } catch (e) { console.warn('schedulePersist: persist failed', e); }
            }, PERSIST_DEBOUNCE_MS);
        } catch (e) { console.warn('schedulePersist failed', e); }
    };

    window.resultsImpl.clearUploadRecords = function() {
        try {
            window.recordStores = window.recordStores || { live: [], upload: [] };
            window.recordStores.upload = [];

            window.__detailsDataMap = window.__detailsDataMap || {};
            Object.keys(window.__detailsDataMap).forEach((key) => {
                const payload = window.__detailsDataMap[key];
                if (!payload || !payload.liveCapture) delete window.__detailsDataMap[key];
            });

            if (window.uploadTrendImpl && typeof window.uploadTrendImpl.resetTrendChart === 'function') {
                window.uploadTrendImpl.resetTrendChart();
            }

            const tbody = document.getElementById('resultTableBody');
            if (tbody) {
                tbody.innerHTML = '<tr class="border-b border-cyber-border/50"><td class="px-6 py-4 text-gray-500 text-center" colspan="8">暂无上传检测记录，请上传文件开始检测</td></tr>';
            }

            window.recordTableMode = 'upload';
            window.currentRecordFilter = 'all';
            try { if (typeof window.applyRecordFilter === 'function') window.applyRecordFilter('all'); } catch (e) {}
            window.resultsImpl.persistRecordStores();
            try { if (typeof window.persistUiState === 'function') window.persistUiState(); } catch (e) {}
        } catch (e) {
            console.warn('resultsImpl.clearUploadRecords failed', e);
        }
    };

    window.resultsImpl.renderRecordedRows = function(kind) {
        try {
            const tbody = document.getElementById('resultTableBody');
            if (!tbody) return;
            const list = (window.recordStores && Array.isArray(window.recordStores[kind])) ? window.recordStores[kind] : [];
            window.recordTableMode = kind;
            tbody.innerHTML = '';
            if (!list.length) {
                const emptyText = kind === 'live' ? '暂无实时抓包记录，启动后会只显示实时流量。' : '暂无上传检测记录，请上传文件开始检测';
                tbody.innerHTML = `<tr class="border-b border-cyber-border/50"><td class="px-6 py-4 text-gray-500 text-center" colspan="8">${emptyText}</td></tr>`;
                if (kind === 'live') window._liveSeen = new Set();
                if (window.resultsImpl && typeof window.resultsImpl.applyRecordFilter === 'function') {
                    window.resultsImpl.applyRecordFilter(window.currentRecordFilter || 'all');
                }
                return;
            }
            for (let i = list.length - 1; i >= 0; i--) {
                const item = list[i] || {};
                window.resultsImpl.addResultRow(item.name, item.size, item.time, item.data, Object.assign({}, item.options || {}, { recordKind: kind, forceRender: true, skipCache: true }));
            }
            if (window.resultsImpl && typeof window.resultsImpl.applyRecordFilter === 'function') {
                window.resultsImpl.applyRecordFilter(window.currentRecordFilter || 'all');
            }
        } catch (e) { console.warn('resultsImpl.renderRecordedRows failed', e); }
    };

    window.resultsImpl.applyRecordFilter = function(filterType) {
        try {
            const normalized = String(filterType || 'all').toLowerCase();
            const finalFilter = ['all', 'benign', 'malware'].includes(normalized) ? normalized : 'all';
            window.currentRecordFilter = finalFilter;

            const rows = document.querySelectorAll('#resultTableBody tr');
            rows.forEach((row) => {
                const type = (row.dataset && row.dataset.recordType) ? String(row.dataset.recordType).toLowerCase() : '';
                let show = finalFilter === 'all';
                if (!show) show = (finalFilter === 'benign' && type === 'benign');
                if (!show) show = (finalFilter === 'malware' && type === 'malware');
                row.style.display = show ? '' : 'none';
            });

            ['recordFilterAll', 'recordFilterBenign', 'recordFilterMalware'].forEach((id) => {
                const btn = document.getElementById(id);
                if (!btn) return;
                const isActive = (id === 'recordFilterAll' && finalFilter === 'all')
                    || (id === 'recordFilterBenign' && finalFilter === 'benign')
                    || (id === 'recordFilterMalware' && finalFilter === 'malware');
                btn.classList.toggle('bg-cyber-primary', isActive);
                btn.classList.toggle('text-black', isActive);
                btn.classList.toggle('text-gray-300', !isActive);
            });

            try { if (typeof window.updateRecordFilterIndicator === 'function') window.updateRecordFilterIndicator(finalFilter); } catch (e) {}

            try { if (typeof window.persistUiState === 'function') window.persistUiState(); } catch (e) {}
        } catch (e) {
            console.warn('resultsImpl.applyRecordFilter failed', e);
        }
    };

    // Restore persisted records early so tab switching can render cached history after refresh.
    window.resultsImpl.restoreRecordStores();
    window.resultsImpl.bindClearButtons();

})();
