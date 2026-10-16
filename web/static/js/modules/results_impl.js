/* implementations for result rendering and modal helpers */
(function(){
    window.resultsImpl = window.resultsImpl || {};
    const RECORDS_STORAGE_KEY = 'detection_records';
    const MAX_RECORDS_PER_KIND = 500;
    const MAX_PERSIST_RECORDS_PER_KIND = 300; // localStorage 中每种记录（upload/live）最多保留最近 300 条
    const MAX_PERSIST_BYTES = 3 * 1024 * 1024; // localStorage 中该 key 序列化后的字节上限：3 MB
    const PERSIST_DEBOUNCE_MS = 300; // debounce writes to localStorage
    const PERSIST_BACKOFF_MS = 60 * 1000; // backoff on quota error

    // HTML 转义：优先复用 core/utils.js 的 Utils.escapeHtml（main.js 在加载本文件之前已加载 core/utils.js），
    // 若工具模块尚未就绪则本地兜底，保证任何路径下都不会把未转义的外部数据拼进 innerHTML。
    function escapeHtml(value) {
        if (value === null || value === undefined) return '';
        if (window.Utils && typeof window.Utils.escapeHtml === 'function') return window.Utils.escapeHtml(value);
        if (typeof window.escapeHtml === 'function' && window.escapeHtml !== escapeHtml) return window.escapeHtml(value);
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    window.resultsImpl.escapeHtml = escapeHtml;

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

    // 序列化后的 UTF-8 字节数：localStorage 配额是按字节计算的，不能只看字符数（中文/emoji 占多字节）
    function jsonByteLength(str) {
        if (!str) return 0;
        try { if (typeof TextEncoder !== 'undefined') return new TextEncoder().encode(str).length; } catch (e) { /* fallthrough */ }
        return str.length * 2; // 兜底：保守估计
    }

    // 组装待写入 localStorage 的 JSON 文本，双重上限保护：
    //   1) 条数：每种记录只取最近 MAX_PERSIST_RECORDS_PER_KIND 条（upload/live 各自独立）；
    //   2) 字节：序列化后若超过 MAX_PERSIST_BYTES，就从最旧的记录开始裁剪（upload/live 交替，数组为追加序，头部即最旧），
    //      直到低于上限为止，避免 QuotaExceededError。
    function buildPersistableJson() {
        const stores = window.recordStores || {};
        const upload = (Array.isArray(stores.upload) ? stores.upload : []).slice(-MAX_PERSIST_RECORDS_PER_KIND);
        const live = (Array.isArray(stores.live) ? stores.live : []).slice(-MAX_PERSIST_RECORDS_PER_KIND);
        const serialize = () => JSON.stringify({
            upload: upload.map(window.resultsImpl.toPersistableEntry),
            live: live.map(window.resultsImpl.toPersistableEntry)
        });

        let json = serialize();
        let bytes = jsonByteLength(json);
        let turn = 0;
        let guard = 0; // 防御性上限，避免任何异常情况下出现死循环
        while (bytes > MAX_PERSIST_BYTES && (upload.length || live.length) && guard++ < 700) {
            // 按平均单条字节数估算要裁掉多少条，一次多裁一些，避免逐条重新序列化
            const total = upload.length + live.length;
            const perEntry = Math.max(1, Math.ceil(bytes / total));
            const dropCount = Math.min(total, Math.max(1, Math.ceil((bytes - MAX_PERSIST_BYTES) / perEntry)));
            for (let i = 0; i < dropCount; i++) {
                const order = (turn++ % 2 === 0) ? [upload, live] : [live, upload];
                const target = order.find((arr) => arr.length > 0);
                if (!target) break;
                target.shift(); // 从最旧的记录开始丢弃
            }
            json = serialize();
            bytes = jsonByteLength(json);
        }
        return json;
    }

    window.resultsImpl.persistRecordStores = function() {
        let acquired = false;
        try {
            window.resultsImpl = window.resultsImpl || {};
            // avoid concurrent persist runs
            if (window.resultsImpl._persistInFlight) return;
            window.resultsImpl._persistInFlight = true;
            acquired = true;
            // respect backoff after quota error
            const now = Date.now();
            if (window.resultsImpl._persistBackoffUntil && now < window.resultsImpl._persistBackoffUntil) return;

            window.recordStores = window.recordStores || { live: [], upload: [] };

            if (!(window.localStorage && window.localStorage.setItem)) return;

            // 先按条数 + 字节双重上限裁剪，再写入
            const json = buildPersistableJson();
            try {
                window.localStorage.setItem(RECORDS_STORAGE_KEY, json);
                // clear any prior backoff
                window.resultsImpl._persistBackoffUntil = 0;
                return;
            } catch (e) {
                const isQuota = e && (e.name === 'QuotaExceededError' || e.name === 'NS_ERROR_DOM_QUOTA_REACHED' || e.code === 22 || e.code === 1014 || /quota/i.test(String(e.message || '')));
                if (!isQuota) {
                    // warn but throttle repeated messages
                    const last = window.resultsImpl._lastPersistWarnTs || 0;
                    if (Date.now() - last > 5000) {
                        console.warn('resultsImpl.persistRecordStores failed', e);
                        window.resultsImpl._lastPersistWarnTs = Date.now();
                    }
                    return;
                }
                // 配额超限（裁剪后仍然写不进去，通常是浏览器总配额已被其它 key 占满）：
                // 不能再静默吞掉异常，清空该 key 并给出明确告警，同时短时退避避免刷屏。
                try { window.localStorage.removeItem(RECORDS_STORAGE_KEY); } catch (removeErr) { /* ignore */ }
                console.warn('resultsImpl.persistRecordStores: localStorage 配额超限（QuotaExceededError），已清空并重置本地检测记录（key=' + RECORDS_STORAGE_KEY + '）；本次写入放弃，' + Math.round(PERSIST_BACKOFF_MS / 1000) + ' 秒内不再重试。', e);
                window.resultsImpl._persistBackoffUntil = Date.now() + PERSIST_BACKOFF_MS;
            }
        } catch (e) {
            console.warn('resultsImpl.persistRecordStores failed', e);
        }
        finally {
            try { if (acquired) window.resultsImpl._persistInFlight = false; } catch(_){}
        }
    };

    window.resultsImpl.restoreRecordStores = function() {
        try {
            window.recordStores = window.recordStores || { live: [], upload: [] };
            if (!(window.localStorage && window.localStorage.getItem)) return;
            const raw = window.localStorage.getItem(RECORDS_STORAGE_KEY);
            if (!raw) return;
            let data;
            try {
                data = JSON.parse(raw);
            } catch (parseErr) {
                // 本地数据已损坏：清空该 key 并明确告警，避免每次启动都解析失败
                try { window.localStorage.removeItem(RECORDS_STORAGE_KEY); } catch (removeErr) { /* ignore */ }
                console.warn('resultsImpl.restoreRecordStores: 本地检测记录（key=' + RECORDS_STORAGE_KEY + '）不是合法 JSON，已清空并重置该键。', parseErr);
                return;
            }
            const restoredUpload = Array.isArray(data && data.upload) ? data.upload : [];
            const restoredLive = Array.isArray(data && data.live) ? data.live : [];
            // 恢复侧沿用原内存上限（MAX_RECORDS_PER_KIND=500），300 条只是“写盘”上限，不改动原有内存语义
            window.recordStores.upload = restoredUpload.slice(-MAX_RECORDS_PER_KIND);
            window.recordStores.live = restoredLive.slice(-MAX_RECORDS_PER_KIND);
        } catch (e) {
            console.warn('resultsImpl.restoreRecordStores failed', e);
        }
    };

    window.resultsImpl.normalizeDetailPayload = function(data) {
        const d = data && typeof data === 'object' ? data : {};
        return { name: d.name || '', statusText: d.statusText || d.status || d.confidence_label || d.result || '', conf: d.conf || d.confidence || '0', execTime: d.execution_time || d.execTime || d.duration_ms || d.time_ms || '-', captureTime: d.capture_time || d.captureTime || d.timestamp || '', imageData: d.imageData || d.image_data || d.image || d.visualization || d.visualization_data || '', payloadDist: d.payloadDist || d.payload_dist || d.payload_distribution || d.histogram || null, flows: Array.isArray(d.flows) ? d.flows : [], flowsTotal: Number.isFinite(Number(d.flows_total)) ? Number(d.flows_total) : (Array.isArray(d.flows) ? d.flows.length : 0), flowsTruncated: !!d.flows_truncated, sessionParseReport: d.sessionParseReport || d.session_parse_report || d.parse_report || d.session_report || null };
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
        const svg = `\n<svg xmlns="http://www.w3.org/2000/svg" width="560" height="260" viewBox="0 0 560 260">\n    <defs>\n        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">\n            <stop offset="0%" stop-color="#0b0f1e"/>\n            <stop offset="100%" stop-color="#101a35"/>\n        </linearGradient>\n        <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">\n            <stop offset="0%" stop-color="#22d3ee"/>\n            <stop offset="100%" stop-color="#2563eb"/>\n        </linearGradient>\n        <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">\n            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>\n            <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>\n        </filter>\n    </defs>\n    <rect width="560" height="260" fill="url(#bg)" rx="18"/>\n    <circle cx="110" cy="82" r="34" fill="rgba(34,211,238,0.15)" stroke="#22d3ee" stroke-width="2"/>\n    <circle cx="450" cy="82" r="34" fill="rgba(37,99,235,0.15)" stroke="#60a5fa" stroke-width="2"/>\n    <line x1="145" y1="82" x2="415" y2="82" stroke="#22d3ee" stroke-width="4" stroke-linecap="round" filter="url(#glow)"/>\n    <polygon points="415,82 402,74 402,90" fill="#22d3ee"/>\n    <text x="110" y="78" text-anchor="middle" fill="#e5e7eb" font-size="14" font-family="sans-serif">SRC</text>\n    <text x="110" y="98" text-anchor="middle" fill="#9ca3af" font-size="11" font-family="monospace">${window.resultsImpl.escapeSvgText((list[0] && (list[0].src_ip || list[0].src)) || 'unknown')}</text>\n    <text x="450" y="78" text-anchor="middle" fill="#e5e7eb" font-size="14" font-family="sans-serif">DST</text>\n    <text x="450" y="98" text-anchor="middle" fill="#9ca3af" font-size="11" font-family="monospace">${window.resultsImpl.escapeSvgText((list[0] && (list[0].dst_ip || list[0].dst)) || 'unknown')}</text>\n    <text x="280" y="58" text-anchor="middle" fill="#e5e7eb" font-size="18" font-weight="700" font-family="sans-serif">${window.resultsImpl.escapeSvgText(name || 'Traffic Detail')}</text>\n    <text x="280" y="80" text-anchor="middle" fill="#7dd3fc" font-size="12" font-family="monospace">${window.resultsImpl.escapeSvgText(result || '-')} · ${window.resultsImpl.escapeSvgText(conf || '0.00%')}</text>\n    <text x="280" y="118" text-anchor="middle" fill="#cbd5e1" font-size="12" font-family="sans-serif">Flow Summary: ${list.length} records · Malware ${mal} · Benign ${benign}</text>\n    ${bars}\n    <text x="26" y="204" fill="#9ca3af" font-size="10" font-family="monospace">Payload histogram fallback when original image is unavailable</text>\n</svg>`;
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
            // mode='all'（总览页 / 记录页）要同时显示两路记录；'live' / 'upload' 只显示各自那一路
            const shouldRender = !!options.forceRender || recordKind === currentMode || currentMode === 'all';
            if (!shouldRender) return 'row_stub_' + Date.now();
            const row = document.createElement('tr');
            // 新到达的行做一次单次高亮（styles.css 的 .mo-row-new，420ms 后自行褪去）：
            // 只改自身背景，不改行高、不推动其他行，也不让整张表持续动。
            row.className = 'border-b border-cyber-border/50 mo-row-new';
            setTimeout(() => row.classList.remove('mo-row-new'), 460);
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
            // 同一毫秒内可能批量插入多行（实时页每轮最多 25 行），rowId 必须唯一；
            // 否则 __detailsDataMap 后写覆盖先写，不同行会打开同一份详情数据。
            window.__rowIdSeq = (window.__rowIdSeq || 0) + 1;
            const rowId = 'row_' + Date.now() + '_' + window.__rowIdSeq;
            window.__detailsDataMap = window.__detailsDataMap || {};
            window.__detailsDataMap[rowId] = window.resultsImpl.normalizeDetailPayload(Object.assign({ rowId, name, size, time, statusText: isMal ? 'Malware' : 'Benign', conf: data && (data.confidence || data.conf || data.malware_conf) ? String(data.confidence || data.conf || data.malware_conf) : '0', }, data || {}));
            function protoToName(p) { if (p === null || p === undefined || p === '') return '-'; const n = Number(p); const map = { 1: 'ICMP', 6: 'TCP', 17: 'UDP', 41: 'IPv6', 47: 'GRE', 50: 'ESP', 51: 'AH', 89: 'OSPF' }; if (Number.isFinite(n) && map[n]) return map[n] + ' (' + n + ')'; return String(p); }
            const displaySize = (size && size !== '-' && size !== null && size !== undefined) ? size : (data && (data.size || data.bytes || data.length) ? (data.size || data.bytes || data.length) : (data && (data.proto || data.protocol) ? protoToName(data.proto || data.protocol) : '-'));
            // 外部数据（文件名 name、时间 time、耗时、大小/协议、模型名、置信度、rowId）一律先 escapeHtml 再插值，
            // 只转义插值数据，模板里的 <td>/<button>/class 等结构标签保持原样。
            // mode='all' 的表格比单路表格多一列「来源」，用于区分这一行是上传检测还是实时抓包
            const sourceTd = currentMode === 'all'
                ? `<td class="px-6 py-4">${recordKind === 'live' ? '<span class="src-badge is-live">实时</span>' : '<span class="src-badge">上传</span>'}</td>`
                : '';
            row.dataset.recordSource = recordKind;
            row.innerHTML = `<td class="px-6 py-4 font-mono text-white truncate max-w-[200px]" title="${escapeHtml(name)}">${escapeHtml(name)}</td>` + sourceTd + `<td class="px-6 py-4 text-gray-400">${escapeHtml(time)}</td>` + `<td class="px-6 py-4 font-mono text-cyan-400">${escapeHtml(durationText)}</td>` + `<td class="px-6 py-4 text-gray-400">${escapeHtml(displaySize)}</td>` + `<td class="px-6 py-4">${isMal ? '<span class="text-cyber-danger">Malware</span>' : '<span class="text-cyber-success">Benign</span>'}</td>` + `<td class="px-6 py-4 text-gray-300">${escapeHtml(modelName)}</td>` + `<td class="px-6 py-4 font-mono text-cyber-primary">${escapeHtml(displayConf)}%</td>` + `<td class="px-6 py-4"><button type="button" data-row-id="${escapeHtml(rowId)}" class="text-gray-400 hover:text-white transition-colors text-sm underline view-details-btn">详情</button></td>`;
            // 插入真实记录前先移除空态占位行（colspan 单元格）：
            // 否则“暂无…记录”的提示会一直残留在表格底部，与真实数据同时显示
            const placeholder = tbody.querySelector('tr > td[colspan]');
            if (placeholder && placeholder.parentNode) placeholder.parentNode.remove();
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

    // 清除按钮：上传页清上传、实时页清实时（含辐射图上的历史点）、总览与记录页清全部。
    window.resultsImpl.clearRecords = function() {
        const mode = pageMode();
        if (mode !== 'upload') {
            // 辐射图自己维护了一份会话/节点缓存，先让它清干净
            try { if (window.liveGraph && typeof window.liveGraph.clearLiveLogs === 'function') window.liveGraph.clearLiveLogs(); } catch (e) {}
            window.resultsImpl.clearLiveRecords();
        }
        if (mode !== 'live') {
            window.resultsImpl.clearUploadRecords();
        }
        try { if (typeof window.resultsImpl.renderRecordedRows === 'function') window.resultsImpl.renderRecordedRows(mode); } catch (e) {}
        return null;
    };

    window.resultsImpl.bindClearButtons = function() {
        try {
            const clearBtn = document.getElementById('clearUploadLogsBtn');
            if (clearBtn && !clearBtn._boundClick) {
                clearBtn._boundClick = () => window.resultsImpl.clearRecords();
                clearBtn.addEventListener('click', clearBtn._boundClick);
            }
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

    // ── 记录表的三条规则 ──────────────────────────────────────────────────
    // 1. 页面模式只由当前页面模板的 data-mode 决定（main.js 启动时写进 window.recordTableMode）；
    //    渲染函数绝不反过来修改它 —— 之前 renderRecordedRows('live') 会把总览页改成 live 模式，
    //    列数与内容跟着一起错。
    // 2. mode='all' 的表格合并两路记录（多一列「来源」），单路页面只看自己那一路。
    // 3. 清除按钮的文案由模板给出，这里只决定清哪一路。
    function pageMode() {
        const mode = String(window.recordTableMode || 'upload');
        return (mode === 'live' || mode === 'all') ? mode : 'upload';
    }
    window.resultsImpl.pageMode = pageMode;

    function emptyStateText(mode) {
        if (mode === 'live') return '暂无实时抓包记录，启动后会只显示实时流量。';
        if (mode === 'all') return '暂无检测记录，上传文件或开始抓包后这里会出现记录。';
        return '暂无上传检测记录，请上传文件开始检测';
    }

    function emptyStateRow(mode) {
        const colspan = mode === 'all' ? 9 : 8;
        return `<tr class="border-b border-cyber-border/50"><td class="px-6 py-4 text-gray-500 text-center" colspan="${colspan}">${escapeHtml(emptyStateText(mode))}</td></tr>`;
    }

    window.resultsImpl.clearUploadRecords = function() {
        try {
            window.recordStores = window.recordStores || { live: [], upload: [] };
            window.recordStores.upload = [];

            // 只丢弃「上传检测」产生的详情，实时记录对应的详情仍然有效
            window.__detailsDataMap = window.__detailsDataMap || {};
            Object.keys(window.__detailsDataMap).forEach((key) => {
                const payload = window.__detailsDataMap[key];
                if (!payload || !payload.liveCapture) delete window.__detailsDataMap[key];
            });

            if (window.uploadTrendImpl && typeof window.uploadTrendImpl.resetTrendChart === 'function') {
                window.uploadTrendImpl.resetTrendChart();
            }

            window.currentRecordFilter = 'all';
            window.resultsImpl.persistRecordStores();
            try { if (typeof window.persistUiState === 'function') window.persistUiState(); } catch (e) {}
        } catch (e) {
            console.warn('resultsImpl.clearUploadRecords failed', e);
        }
    };

    window.resultsImpl.clearLiveRecords = function() {
        try {
            window.recordStores = window.recordStores || { live: [], upload: [] };
            window.recordStores.live = [];
            window._liveSeen = new Set();
            // 清空时间点落盘：轮询会把早于它的流过滤掉，刷新页面后不会又把旧流拉回来
            window.liveLogsClearedAt = Date.now() / 1000;
            try { window.localStorage && window.localStorage.setItem && window.localStorage.setItem('liveLogsClearedAt', String(window.liveLogsClearedAt)); } catch (e) {}

            window.__detailsDataMap = window.__detailsDataMap || {};
            Object.keys(window.__detailsDataMap).forEach((key) => {
                const payload = window.__detailsDataMap[key];
                if (payload && payload.liveCapture) delete window.__detailsDataMap[key];
            });

            if (window.liveTrendImpl && typeof window.liveTrendImpl.resetTrendChart === 'function') {
                window.liveTrendImpl.resetTrendChart();
            }

            window.resultsImpl.persistRecordStores();
            try { if (typeof window.persistUiState === 'function') window.persistUiState(); } catch (e) {}
        } catch (e) {
            console.warn('resultsImpl.clearLiveRecords failed', e);
        }
    };

    window.resultsImpl.renderRecordedRows = function(kind) {
        try {
            const tbody = document.getElementById('resultTableBody');
            if (!tbody) return;
            const mode = pageMode();
            // 目标列表：总览/记录页把两路合并，其余页面只看自己那一路
            const wanted = mode === 'all' ? 'all' : ((kind === 'live' || kind === 'upload') ? kind : mode);
            const uploads = (window.recordStores && Array.isArray(window.recordStores.upload)) ? window.recordStores.upload : [];
            const lives = (window.recordStores && Array.isArray(window.recordStores.live)) ? window.recordStores.live : [];
            let list = [];
            if (wanted === 'live') {
                list = lives.map((item) => ({ item, kind: 'live' }));
            } else if (wanted === 'upload') {
                list = uploads.map((item) => ({ item, kind: 'upload' }));
            } else {
                // 合并展示：按采集时间升序（旧→新），没有时间戳的条目退化为各自数组内的原有顺序
                const keyed = uploads.map((item, i) => ({ item, kind: 'upload', at: Number(item && item.data && (item.data.captured_at || item.data.timestamp)) || 0, i }))
                    .concat(lives.map((item, i) => ({ item, kind: 'live', at: Number(item && item.data && (item.data.captured_at || item.data.timestamp)) || 0, i })));
                keyed.sort((a, b) => (a.at - b.at) || (a.i - b.i));
                list = keyed;
            }

            tbody.innerHTML = '';
            if (!list.length) {
                tbody.innerHTML = emptyStateRow(mode);
                if (mode !== 'upload') window._liveSeen = new Set();
                if (window.resultsImpl && typeof window.resultsImpl.applyRecordFilter === 'function') {
                    window.resultsImpl.applyRecordFilter(window.currentRecordFilter || 'all');
                }
                return;
            }
            // 旧→新逐条插入，而 addResultRow 是 prepend，最终最新的一条在表格顶部，
            // 与运行期新行到达的顺序一致。
            for (let i = 0; i < list.length; i++) {
                const entry = list[i];
                const item = entry.item || {};
                window.resultsImpl.addResultRow(item.name, item.size, item.time, item.data, Object.assign({}, item.options || {}, { recordKind: entry.kind, forceRender: true, skipCache: true }));
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
