/* live capture and polling implementations */
(function(){
    window.liveImpl = window.liveImpl || {};

    function ensurePanelTransition(el) {
        if (!el || el.dataset.panelTransitionReady === '1') return;
        el.dataset.panelTransitionReady = '1';
        el.style.transition = 'opacity 240ms ease, transform 240ms ease, filter 240ms ease';
        el.style.willChange = 'opacity, transform';
        el.style.transformOrigin = 'center top';
    }

    function setPanelVisible(el, visible, options) {
        if (!el) return;
        ensurePanelTransition(el);
        const config = options || {};
        const prefersReducedMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

        // Track desired state and per-transition id to avoid races from overlapping timers/listeners
        el._panelDesiredVisible = !!visible;
        el._panelTransitionId = (el._panelTransitionId || 0) + 1;
        const myTransitionId = el._panelTransitionId;

        // clear existing timers/listeners for previous transitions
        if (el._panelHideTimer) {
            clearTimeout(el._panelHideTimer);
            el._panelHideTimer = null;
        }
        if (el._panelShowTimer) {
            clearTimeout(el._panelShowTimer);
            el._panelShowTimer = null;
        }
        if (el._panelOnEnd) {
            try { el.removeEventListener('transitionend', el._panelOnEnd); } catch(e){}
            el._panelOnEnd = null;
        }

        if (visible) {
            el.classList.remove('hidden');
            el.setAttribute('aria-hidden', 'false');
            if (prefersReducedMotion) {
                el.style.opacity = '';
                el.style.transform = '';
                el.style.pointerEvents = '';
                return;
            }
            const enterTransform = config.enterTransform || 'translateX(24px) scale(0.985)';
            el.style.opacity = '0';
            el.style.transform = enterTransform;
            el.style.pointerEvents = 'none';
            void el.offsetHeight;
            requestAnimationFrame(() => {
                // If another transition started meanwhile, bail
                if (el._panelTransitionId !== myTransitionId || el._panelDesiredVisible !== true) return;
                el.style.opacity = '1';
                el.style.transform = 'translateX(0) scale(1)';
                el.style.pointerEvents = '';
            });
            el._panelShowTimer = setTimeout(() => {
                // Only finalize if this is still the active transition
                if (el._panelTransitionId === myTransitionId && el._panelDesiredVisible === true) {
                    el.style.opacity = '';
                    el.style.transform = '';
                    el.style.pointerEvents = '';
                }
                el._panelShowTimer = null;
            }, 320);
            return;
        }

        el.setAttribute('aria-hidden', 'true');
        if (prefersReducedMotion) {
            el.classList.add('hidden');
            el.style.opacity = '';
            el.style.transform = '';
            el.style.pointerEvents = '';
            return;
        }
        el.style.opacity = '0';
        el.style.transform = config.exitTransform || 'translateX(-24px) scale(0.985)';
        el.style.pointerEvents = 'none';

        const finish = () => {
            // Only hide if this transition is still current and desired state is hidden
            if (el._panelTransitionId === myTransitionId && el._panelDesiredVisible === false) {
                el.classList.add('hidden');
                el.style.opacity = '';
                el.style.transform = '';
                el.style.pointerEvents = '';
            }
        };

        const onEnd = (evt) => {
            if (evt && evt.target !== el) return;
            // ensure we remove only our handler
            if (el._panelOnEnd === onEnd) el._panelOnEnd = null;
            try { el.removeEventListener('transitionend', onEnd); } catch(e){}
            if (el._panelHideTimer) {
                clearTimeout(el._panelHideTimer);
                el._panelHideTimer = null;
            }
            finish();
        };
        el._panelOnEnd = onEnd;
        el.addEventListener('transitionend', onEnd);
        el._panelHideTimer = setTimeout(() => {
            // Only finalize if still current
            if (el._panelOnEnd === onEnd) {
                try { el.removeEventListener('transitionend', onEnd); } catch(e){}
                el._panelOnEnd = null;
            }
            el._panelHideTimer = null;
            finish();
        }, 320);
    }

    function animateMovedElement(el, opts) {
        if (!el) return;
        ensurePanelTransition(el);
        const prefersReducedMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
        if (prefersReducedMotion) return;
        const offset = (opts && typeof opts.offset === 'number') ? opts.offset : 18;
        const scale = (opts && typeof opts.scale === 'number') ? opts.scale : 0.99;
        const enterTransform = `translateX(${offset}px) scale(${scale})`;
        el.style.opacity = '0';
        el.style.transform = enterTransform;
        void el.offsetHeight;
        requestAnimationFrame(() => {
            el.style.opacity = '1';
            el.style.transform = 'translateX(0) scale(1)';
        });
        if (el._panelResetTimer) clearTimeout(el._panelResetTimer);
        el._panelResetTimer = setTimeout(() => {
            el.style.opacity = '';
            el.style.transform = '';
            el._panelResetTimer = null;
        }, 320);
    }

    function animatePageShift(direction) {
        return;
    }

    // expose helpers so other modules can reuse the same transition behavior
    try {
        if (typeof window !== 'undefined') {
            window.setPanelVisible = window.setPanelVisible || setPanelVisible;
            window.animateMovedElement = window.animateMovedElement || animateMovedElement;
        }
    } catch(e) {}

    // ensure a small keyframe animation exists for robust card entry
    function ensureCardKeyframes() {
        try {
            if (document.getElementById('dlids-card-keyframes')) return;
            const css = `@keyframes dlids-slide-in { from { opacity: 0; transform: translateX(28px) scale(0.985); } to { opacity: 1; transform: translateX(0) scale(1); } } .dlids-slide-in { animation: dlids-slide-in 320ms ease forwards; }`;
            const style = document.createElement('style');
            style.id = 'dlids-card-keyframes';
            style.appendChild(document.createTextNode(css));
            document.head.appendChild(style);
        } catch (e) { /* ignore */ }
    }

    // initialize persisted clear timestamp (seconds)
    window.liveLogsClearedAt = Number(window.localStorage && window.localStorage.getItem && window.localStorage.getItem('liveLogsClearedAt')) || window.liveLogsClearedAt || 0;

    window.liveImpl.startLive = async function() {
        const startBtn = document.getElementById('startLiveBtn');
        try {
            if (startBtn) startBtn.disabled = true;
            const iface = window.getSelectedInterface ? window.getSelectedInterface() : '';
            if (!iface) { alert('请选择网卡'); return; }
            window.liveCaptureModel = window.currentActiveModel;
            const fd = new FormData(); fd.append('iface', iface); fd.append('model_type', window.currentActiveModel);
            const r = await fetch('/api/live/start', { method: 'POST', body: fd });
            const j = await (r.ok ? r.json() : r.text().then(t=>({ error: t }))); 
            if (j && (j.status === 'started' || j.status === 'already_running')) {
                try { if (typeof window.setLiveRunningUI === 'function') window.setLiveRunningUI(true); } catch(e){}
                try { if (typeof window.pollLive === 'function') { if (window.livePoller) clearInterval(window.livePoller); window.pollLive(); window.livePoller = setInterval(window.pollLive, 1500); } } catch(e){}
                try { if (typeof window.switchToLive === 'function') window.switchToLive(); } catch(e){}
            } else {
                const msg = j && (j.message || j.error) ? (j.message || j.error) : JSON.stringify(j);
                alert('启动失败: ' + msg);
            }
        } catch (e) { alert('启动失败: ' + (e && e.message ? e.message : String(e))); } finally { if (startBtn) startBtn.disabled = false; }
    };

    window.liveImpl.switchToLive = function() {
        try {
            const rightColumn = document.getElementById('rightColumn');
            const trendChartPanel = document.getElementById('trendChartPanel');
            const liveSection = document.getElementById('liveSection');
            const liveFlowGraphWrap = document.getElementById('liveFlowGraphWrap');
            const dropZoneEl = document.getElementById('dropZone');
            const uploadContentEl = document.getElementById('uploadContent');
            const loadingStateEl = document.getElementById('loadingState');
            const tabLive = document.getElementById('tabLive');
            const tabUpload = document.getElementById('tabUpload');
            if (rightColumn && liveFlowGraphWrap && liveFlowGraphWrap.parentElement !== rightColumn) {
                try { liveFlowGraphWrap.style.opacity = '0'; liveFlowGraphWrap.style.transform = 'translateX(18px) scale(0.99)'; } catch(e){}
                rightColumn.insertBefore(liveFlowGraphWrap, rightColumn.firstChild);
            }
            if (liveSection && trendChartPanel && trendChartPanel.parentElement !== liveSection) {
                try {
                    const liveFlowList = liveSection.querySelector('#liveFlowList');
                    liveSection.insertBefore(trendChartPanel, liveFlowList || liveSection.firstChild);
                    if (typeof setPanelVisible === 'function') setPanelVisible(trendChartPanel, true, { enterTransform: 'translateX(28px) scale(0.985)' });
                } catch(e){}
            }
            animatePageShift(1);
            setPanelVisible(dropZoneEl, false, { exitTransform: 'translateX(-28px) scale(0.985)' });
            if (uploadContentEl) uploadContentEl.classList.add('hidden');
            if (loadingStateEl) loadingStateEl.classList.add('hidden');
            setPanelVisible(liveSection, true, { enterTransform: 'translateX(28px) scale(0.985)' });
            // ensure both trend chart and result panel use identical panel-visible transition
            try { if (typeof setPanelVisible === 'function') setPanelVisible(trendChartPanel, true, { enterTransform: 'translateX(28px) scale(0.985)' }); } catch(e) {}
            try { animateMovedElement(liveFlowGraphWrap); } catch(e) {}
            try { if (typeof setPanelVisible === 'function') setPanelVisible(document.getElementById('resultCardPanel'), true, { enterTransform: 'translateX(28px) scale(0.985)' }); } catch(e) {}
            try { ensureCardKeyframes(); const panelEl = document.getElementById('resultCardPanel'); if (panelEl) { panelEl.classList.add('dlids-slide-in'); setTimeout(()=>panelEl.classList.remove('dlids-slide-in'), 360); } } catch(e) {}
            try { const viewport = document.getElementById('resultTableViewport'); if (viewport) animateMovedElement(viewport, { offset: 12, scale: 0.994 }); } catch(e) {}
            if (tabLive) { tabLive.classList.add('bg-cyber-primary', 'text-black'); tabLive.classList.remove('text-gray-300'); }
            if (tabUpload) { tabUpload.classList.remove('bg-cyber-primary', 'text-black'); tabUpload.classList.add('text-gray-300'); }
            try { if (window.liveTrendImpl && typeof window.liveTrendImpl.initCharts === 'function') window.liveTrendImpl.initCharts(); } catch(e){}
            try { if (window.liveTrendImpl && typeof window.liveTrendImpl.activate === 'function') window.liveTrendImpl.activate(); } catch(e){}
            if (typeof window.activateLiveRecordCard === 'function') {
                window.activateLiveRecordCard();
            } else {
                const resultTitle = document.getElementById('resultTableTitle'); if (resultTitle) resultTitle.innerText = '实时抓包记录';
                const resultHint = document.getElementById('resultTableHint'); if (resultHint) resultHint.innerText = '仅记录当前实时抓包会话中的流量';
                window.recordTableMode = 'live';
                window.renderRecordedRows && window.renderRecordedRows('live');
            }
            try { if (typeof window.persistUiState === 'function') window.persistUiState(); } catch(e){}
            try { if (typeof window.persistLiveUiState === 'function') window.persistLiveUiState(); } catch(e){}
            try { if (window.liveTrendChart && typeof window.liveTrendChart.resize === 'function') window.liveTrendChart.resize(); } catch(e){}
        } catch (e) { console.warn('liveImpl.switchToLive failed', e); }
    };

    window.liveImpl.switchToUpload = function() {
        try {
            const rightColumn = document.getElementById('rightColumn');
            const trendChartPanel = document.getElementById('trendChartPanel');
            const liveSection = document.getElementById('liveSection');
            const liveFlowGraphWrap = document.getElementById('liveFlowGraphWrap');
            const dropZoneEl = document.getElementById('dropZone');
            const uploadContentEl = document.getElementById('uploadContent');
            const loadingStateEl = document.getElementById('loadingState');
            const tabLive = document.getElementById('tabLive');
            const tabUpload = document.getElementById('tabUpload');
            if (rightColumn && trendChartPanel && trendChartPanel.parentElement !== rightColumn) {
                try {
                    rightColumn.insertBefore(trendChartPanel, rightColumn.firstChild);
                    if (typeof setPanelVisible === 'function') setPanelVisible(trendChartPanel, true, { enterTransform: 'translateX(28px) scale(0.985)' });
                } catch(e){}
            }
            if (liveSection && liveFlowGraphWrap && liveFlowGraphWrap.parentElement !== liveSection) {
                try { liveFlowGraphWrap.style.opacity = '0'; liveFlowGraphWrap.style.transform = 'translateX(18px) scale(0.99)'; } catch(e){}
                const liveFlowList = liveSection.querySelector('#liveFlowList');
                liveSection.insertBefore(liveFlowGraphWrap, liveFlowList || liveSection.firstChild);
            }
            animatePageShift(-1);
            setPanelVisible(liveSection, false, { exitTransform: 'translateX(28px) scale(0.985)' });
            setPanelVisible(dropZoneEl, true, { enterTransform: 'translateX(-28px) scale(0.985)' });
            try { if (typeof setPanelVisible === 'function') setPanelVisible(trendChartPanel, true, { enterTransform: 'translateX(28px) scale(0.985)' }); } catch(e) {}
            try { animateMovedElement(liveFlowGraphWrap); } catch(e) {}
            try { if (typeof setPanelVisible === 'function') setPanelVisible(document.getElementById('resultCardPanel'), true, { enterTransform: 'translateX(28px) scale(0.985)' }); } catch(e) {}
            try { ensureCardKeyframes(); const panelEl = document.getElementById('resultCardPanel'); if (panelEl) { panelEl.classList.add('dlids-slide-in'); setTimeout(()=>panelEl.classList.remove('dlids-slide-in'), 360); } } catch(e) {}
            try { const viewport = document.getElementById('resultTableViewport'); if (viewport) animateMovedElement(viewport, { offset: 12, scale: 0.994 }); } catch(e) {}
            if (uploadContentEl) uploadContentEl.classList.remove('hidden');
            if (loadingStateEl) loadingStateEl.classList.add('hidden');
            if (tabUpload) { tabUpload.classList.add('bg-cyber-primary', 'text-black'); tabUpload.classList.remove('text-gray-300'); }
            if (tabLive) { tabLive.classList.remove('bg-cyber-primary', 'text-black'); tabLive.classList.add('text-gray-300'); }
            try { if (window.uploadTrendImpl && typeof window.uploadTrendImpl.activate === 'function') window.uploadTrendImpl.activate(); } catch(e){}
            if (typeof window.activateUploadRecordCard === 'function') {
                window.activateUploadRecordCard();
            } else {
                const resultTitle = document.getElementById('resultTableTitle'); if (resultTitle) resultTitle.innerText = '上传检测记录';
                const resultHint = document.getElementById('resultTableHint'); if (resultHint) resultHint.innerText = '只记录上传检测文件的结果';
                window.recordTableMode = 'upload';
                window.renderRecordedRows && window.renderRecordedRows('upload');
            }
            try { if (typeof window.persistUiState === 'function') window.persistUiState(); } catch(e){}
            try { if (typeof window.persistLiveUiState === 'function') window.persistLiveUiState(); } catch(e){}
            try { if (window.uploadTrendChart && typeof window.uploadTrendChart.resize === 'function') window.uploadTrendChart.resize(); } catch(e){}
        } catch (e) { console.warn('liveImpl.switchToUpload failed', e); }
    };

    window.liveImpl.setLiveRunningUI = function(running, options) {
        try {
            window.liveUiState = window.liveUiState || {};
            const prevRunning = !!window.liveUiState.running;
            window.liveUiState.running = !!running;
            const liveStatus = document.getElementById('liveStatus');
            const startBtn = document.getElementById('startLiveBtn');
            const stopBtn = document.getElementById('stopLiveBtn');
            if (liveStatus) liveStatus.innerText = running ? '运行中' : '停止';
            if (startBtn) { if (running) startBtn.classList.add('hidden'); else startBtn.classList.remove('hidden'); startBtn.disabled = false; }
            if (stopBtn) { if (running) stopBtn.classList.remove('hidden'); else stopBtn.classList.add('hidden'); stopBtn.disabled = false; }
            if (running) {
                // only switch to Live view when transitioning from stopped -> running
                if (!prevRunning) {
                    try { if (typeof window.switchToLive === 'function') window.switchToLive(); } catch(e){}
                }
            } else {
                // Do not automatically switch back to Upload view by default; only switch when explicitly requested
                if (options && options.autoSwitch === true) {
                    try { if (typeof window.switchToUpload === 'function') window.switchToUpload(); } catch(e){}
                }
            }
            try { if (typeof window.persistUiState === 'function') window.persistUiState(); } catch(e){}
        } catch (e) { console.warn('liveImpl.setLiveRunningUI failed', e); }
    };

    window.liveImpl.pollLive = async function() {
        try {
            const r = await fetch('/api/live/stats');
            if (!r.ok) return; const j = await r.json(); if (!j) return; const stats = j.stats || {}; const recent = Array.isArray(j.recent) ? j.recent : [];
            const isRunning = !!stats.running;
            if (!isRunning) return;
            const clearCutoff = Number(window.liveLogsClearedAt || 0) || 0;
            const liveRows = recent.filter((f) => {
                if (!clearCutoff) return true;
                const capturedAt = Number(f && f.captured_at ? f.captured_at : 0) || 0;
                return capturedAt > clearCutoff;
            });
            const uniqueSessions = new Set(); const protoCounts = {}; let inferredBytes = 0; let inferredMalware = 0;
            liveRows.forEach((f) => { const src = f && (f.src || f.src_ip || f.src_addr || '0.0.0.0'); const dst = f && (f.dst || f.dst_ip || f.dst_addr || '0.0.0.0'); const sport = f && (f.sport || f.src_port || 0); const dport = f && (f.dport || f.dst_port || 0); const proto = f && (f.proto || f.protocol || 'unknown'); uniqueSessions.add(`${src}:${sport}->${dst}:${dport}`); protoCounts[String(proto)] = (protoCounts[String(proto)] || 0) + 1; inferredBytes += Number(f && (f.bytes || f.length || 0)) || 0; if (f && (f.is_malicious || (typeof f.predicted_label === 'string' && /malware|malicious/i.test(f.predicted_label)))) inferredMalware += 1; });
            const liveSessionParseReport = { total_packets: stats.total_packets || liveRows.length, accepted_packets: stats.accepted_packets || liveRows.length, session_count: stats.session_count || uniqueSessions.size, parse_errors: stats.parse_errors || 0, non_ipv4: stats.non_ipv4 || 0, malware_count: stats.malware_count || inferredMalware, benign_count: Math.max(0, liveRows.length - (stats.malware_count || inferredMalware)), total_bytes: stats.total_bytes || inferredBytes, protocol_summary: Object.entries(protoCounts).slice(0, 4).map(([k, v]) => `${k}:${v}`).join('，') };
            const livePayloadDist = Array.from({ length: 16 }, (_, i) => { const bucket = liveRows[i % Math.max(1, liveRows.length)] || {}; const seed = `${bucket.src || bucket.src_ip || ''}|${bucket.dst || bucket.dst_ip || ''}|${bucket.sport || bucket.src_port || ''}|${bucket.dport || bucket.dst_port || ''}|${bucket.proto || bucket.protocol || ''}|${bucket.malware_conf || bucket.confidence || 0}`; let hash = 0; for (let j = 0; j < seed.length; j++) hash = ((hash << 5) - hash + seed.charCodeAt(j)) >>> 0; return Math.max(1, ((hash % 11) + 1) * (1 + Math.round(liveRows.length / 10))); });
            try { if ('running' in stats) window.setLiveRunningUI && window.setLiveRunningUI(!!stats.running); } catch(e){}
            try { const statTotal = document.getElementById('statTotal'); if (statTotal) statTotal.innerText = stats.total_packets || stats.total_sessions || statTotal.innerText || 0; const statMal = document.getElementById('statMalware'); if (statMal) statMal.innerText = stats.malware_count || 0; } catch(e){}
            window._liveSeen = window._liveSeen || new Set();
            // limit how many new rows we insert per poll to avoid blocking the UI
            const MAX_NEW_ROWS_PER_POLL = 25;
            let newRowsInserted = 0;
            for (let i = recent.length - 1; i >= 0; --i) {
                const f = recent[i]; try {
                    const capturedAt = Number(f && f.captured_at ? f.captured_at : 0) || 0;
                    if (clearCutoff && capturedAt <= clearCutoff) continue;
                    const fid = `${f.src || '0'}:${f.sport||0}->${f.dst||'0'}:${f.dport||0}@${Math.floor((f.captured_at||0)*1000)}`;
                    if (window._liveSeen.has(fid)) continue; window._liveSeen.add(fid);
                    const name = `${f.src || '-'}:${f.sport||'-'} -> ${f.dst || '-'}:${f.dport||'-'}`; const time = f.captured_at ? (new Date(f.captured_at * 1000)).toLocaleString('zh-CN') : new Date().toLocaleString('zh-CN'); const norm = Object.assign({}, f);
                    norm.model = f.model || f.model_name || f.detected_model || f.detected_model_name || f.predicted_model || f.pred_label || window.getModelDisplayName(window.liveCaptureModel || window.currentActiveModel);
                    norm.confidence = f.confidence || f.conf || f.malware_conf || f.score || null;
                    norm.is_malicious = !!f.is_malicious || (typeof f.predicted_label === 'string' && /malware|malicious/i.test(f.predicted_label)) || false;
                    norm.execution_time = f.execution_time || f.exec_time || f.executionTime || f.duration_ms || f.processing_time_ms || f.elapsed_ms || f.time_ms || null;
                    norm.size = f.size || f.bytes || f.length || f.packet_count || null;
                    norm.flows = liveRows; norm.sessionParseReport = liveSessionParseReport; norm.payloadDist = livePayloadDist; norm.liveCapture = true;
                    window.cacheRecordedRow && window.cacheRecordedRow('live', { name, size: norm.size || '-', time, data: norm, options: { recordKind: 'live' } });
                    if (newRowsInserted < MAX_NEW_ROWS_PER_POLL) {
                        if ((window.recordTableMode || 'upload') === 'live') { window.addResultRow && window.addResultRow(name, norm.size || '-', time, norm, { recordKind: 'live' }); }
                        newRowsInserted += 1;
                    }
                    try { if (window.updateTrendChart) window.updateTrendChart(!!norm.is_malicious); } catch(e){}
                } catch (e) { console.warn('pollLive row append failed', e); }
            }
            try { if (typeof window.renderLiveFlowGraph === 'function') window.renderLiveFlowGraph(liveRows); } catch(e){}
        } catch (e) { console.warn('liveImpl.pollLive failed', e); }
    };

    window.liveImpl.renderLiveFlowGraph = function(flows) { if (window.liveGraph && typeof window.liveGraph.renderLiveFlowGraph === 'function') return window.liveGraph.renderLiveFlowGraph(flows); console.warn('liveGraph module not loaded; cannot render flow graph'); };
    window.liveImpl.clearLiveLogs = function() {
        window.liveLogsClearedAt = Date.now() / 1000;
        try { window.localStorage && window.localStorage.setItem && window.localStorage.setItem('liveLogsClearedAt', String(window.liveLogsClearedAt)); } catch(e){}
        if (window.liveGraph && typeof window.liveGraph.clearLiveLogs === 'function') return window.liveGraph.clearLiveLogs();
        if (window.recordStores && Array.isArray(window.recordStores.live)) window.recordStores.live = [];
        if (window._liveSeen) window._liveSeen = new Set();
        const listDom = document.getElementById('liveFlowList');
        if (listDom) listDom.innerHTML = '<p class="text-sm text-gray-400">暂无实时流量</p>';
        try { if (typeof window.renderRecordedRows === 'function') window.renderRecordedRows('live'); } catch(e){}
    };

})();
