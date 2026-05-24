// main.js - 精简入口，只负责模块加载、事件绑定和委托

(function() {
    'use strict';

    const UI_STATE_KEY = 'dlids_ui_state';

    function loadUiState() {
        try {
            const raw = window.localStorage && window.localStorage.getItem && window.localStorage.getItem(UI_STATE_KEY);
            if (!raw) return {};
            const parsed = JSON.parse(raw);
            return parsed && typeof parsed === 'object' ? parsed : {};
        } catch (e) {
            console.warn('loadUiState failed', e);
            return {};
        }
    }

    function persistUiState() {
        try {
            const votingModeToggle = document.getElementById('votingModeToggle');
            const payload = {
                selectedInterface: window.liveUiState && window.liveUiState.selectedInterface ? window.liveUiState.selectedInterface : null,
                selectedInterfaceLabel: window.liveUiState && window.liveUiState.selectedInterfaceLabel ? window.liveUiState.selectedInterfaceLabel : null,
                tabMode: (window.recordTableMode === 'live' || window.recordTableMode === 'upload') ? window.recordTableMode : 'upload',
                votingModeEnabled: !!(votingModeToggle && votingModeToggle.checked),
                currentFilter: ['all', 'benign', 'malware'].includes(String(window.currentRecordFilter || '').toLowerCase())
                    ? String(window.currentRecordFilter).toLowerCase()
                    : 'all'
            };
            window.localStorage && window.localStorage.setItem && window.localStorage.setItem(UI_STATE_KEY, JSON.stringify(payload));
        } catch (e) {
            console.warn('persistUiState failed', e);
        }
    }
    window.persistUiState = persistUiState;

    const savedUiState = loadUiState();

    // ==================== 错误处理 ====================
    window.addEventListener('error', (ev) => {
        try {
            const msg = ev?.message || String(ev);
            let banner = document.getElementById('jsErrorBanner');
            if (!banner) {
                banner = document.createElement('div');
                banner.id = 'jsErrorBanner';
                banner.style.cssText = 'position:fixed;right:12px;bottom:12px;z-index:99999;max-width:420px;padding:10px 12px;background:rgba(255,46,99,0.95);color:#fff;border-radius:8px;font-size:13px;box-shadow:0 6px 18px rgba(0,0,0,0.4)';
                document.body.appendChild(banner);
            }
            banner.innerText = msg;
            setTimeout(() => banner.remove?.(), 8000);
        } catch (_) {}
    });

    // ==================== 全局状态初始化 ====================
    window.liveUiState = window.liveUiState || { running: false, selectedFilter: 'all', selectedInterface: null, selectedInterfaceLabel: null };
    if (savedUiState && savedUiState.selectedInterface) {
        window.liveUiState.selectedInterface = savedUiState.selectedInterface;
    }
    if (savedUiState && savedUiState.selectedInterfaceLabel) {
        window.liveUiState.selectedInterfaceLabel = savedUiState.selectedInterfaceLabel;
    }
    window.persistedVotingModeEnabled = !!(savedUiState && savedUiState.votingModeEnabled);
    window.recordStores = window.recordStores || { live: [], upload: [] };
    const savedTabMode = savedUiState && (savedUiState.tabMode === 'live' || savedUiState.tabMode === 'upload') ? savedUiState.tabMode : null;
    window.recordTableMode = savedTabMode || window.recordTableMode || 'upload';

    // Apply initial tab highlight immediately to avoid a flash where neither tab is active after reload.
    try {
        const earlyTabLive = findButtonByText('实时抓包') || document.getElementById('tabLive');
        const earlyTabUpload = findButtonByText('上传检测') || document.getElementById('tabUpload');
        if (earlyTabLive && earlyTabUpload) {
            if (window.recordTableMode === 'live') {
                earlyTabLive.classList.add('bg-cyber-primary', 'text-black');
                earlyTabLive.classList.remove('text-gray-300');
                earlyTabUpload.classList.remove('bg-cyber-primary', 'text-black');
                earlyTabUpload.classList.add('text-gray-300');
            } else {
                earlyTabUpload.classList.add('bg-cyber-primary', 'text-black');
                earlyTabUpload.classList.remove('text-gray-300');
                earlyTabLive.classList.remove('bg-cyber-primary', 'text-black');
                earlyTabLive.classList.add('text-gray-300');
            }
        }
    } catch (e) { /* non-fatal */ }
    const savedFilter = savedUiState && ['all', 'benign', 'malware'].includes(String(savedUiState.currentFilter || '').toLowerCase())
        ? String(savedUiState.currentFilter).toLowerCase()
        : 'all';
    window.currentRecordFilter = savedFilter;
    window._liveSeen = window._liveSeen || new Set();
    window.__detailsDataMap = window.__detailsDataMap || {};

    function applyInitialViewMode(mode) {
        try {
            const liveMode = mode === 'live';
            const dropZoneEl = document.getElementById('dropZone');
            const uploadContentEl = document.getElementById('uploadContent');
            const loadingStateEl = document.getElementById('loadingState');
            const liveSection = document.getElementById('liveSection');
            const tabLive = findButtonByText('实时抓包') || document.getElementById('tabLive');
            const tabUpload = findButtonByText('上传检测') || document.getElementById('tabUpload');

            if (dropZoneEl) dropZoneEl.classList.toggle('hidden', liveMode);
            if (uploadContentEl) uploadContentEl.classList.toggle('hidden', liveMode);
            if (loadingStateEl) loadingStateEl.classList.add('hidden');
            if (liveSection) liveSection.classList.toggle('hidden', !liveMode);

            if (tabLive && tabUpload) {
                if (liveMode) {
                    tabLive.classList.add('bg-cyber-primary', 'text-black');
                    tabLive.classList.remove('text-gray-300');
                    tabUpload.classList.remove('bg-cyber-primary', 'text-black');
                    tabUpload.classList.add('text-gray-300');
                } else {
                    tabUpload.classList.add('bg-cyber-primary', 'text-black');
                    tabUpload.classList.remove('text-gray-300');
                    tabLive.classList.remove('bg-cyber-primary', 'text-black');
                    tabLive.classList.add('text-gray-300');
                }
            }
        } catch (e) {
            console.warn('applyInitialViewMode failed', e);
        }
    }

    applyInitialViewMode(window.recordTableMode);

    // ==================== 委托函数 ====================
    function delegate(name, fn) {
        return function(...args) {
            try {
                const impl = window[name];
                if (impl && typeof impl[fn] === 'function') {
                    return impl[fn].apply(impl, args);
                }
            } catch(e) {
                console.warn('delegate', name, fn, e);
            }
            return null;
        };
    }

    // 导出委托API（兼容现有模块调用）
    window.initModelEvalCharts = delegate('modelEvalImpl', 'initModelEvalCharts');
    window.initCharts = delegate('uploadTrendImpl', 'initCharts');
    window.updateTrendChart = delegate('liveTrendImpl', 'updateTrendChart');
    window.updateStats = delegate('uploadTrendImpl', 'updateStats');
    window.initModalChart = delegate('uploadTrendImpl', 'initModalChart');
    window.renderLiveFlowGraph = delegate('liveTrendImpl', 'renderLiveFlowGraph');
    // ensure a robust fallback: if liveTrendImpl doesn't provide renderLiveFlowGraph,
    // delegate to liveImpl or the liveGraph module directly so the canvas gets drawn.
    (function(){
        const orig = window.renderLiveFlowGraph;
        window.renderLiveFlowGraph = function(flows) {
            try {
                // try liveTrendImpl first (orig delegate)
                if (typeof orig === 'function') {
                    const res = orig.apply(null, arguments);
                    if (res !== null) return res;
                }
                } catch (e) { console.warn('liveTrendImpl.renderLiveFlowGraph failed', e); }
            try {
                if (window.liveImpl && typeof window.liveImpl.renderLiveFlowGraph === 'function') return window.liveImpl.renderLiveFlowGraph.apply(window.liveImpl, arguments);
            } catch (e) { console.warn('liveImpl.renderLiveFlowGraph failed', e); }
            try {
                if (window.liveGraph && typeof window.liveGraph.renderLiveFlowGraph === 'function') return window.liveGraph.renderLiveFlowGraph.apply(window.liveGraph, arguments);
            } catch (e) { console.warn('liveGraph.renderLiveFlowGraph failed', e); }
            console.warn('renderLiveFlowGraph: no implementation available');
            return null;
        };
    })();

    window.handleFiles = delegate('uploadImpl', 'handleFiles');
    window.uploadFilesSequentially = delegate('uploadImpl', 'uploadFilesSequentially');
    window.uploadFile = delegate('uploadImpl', 'uploadFile');
    window.readResponsePayload = delegate('uploadImpl', 'readResponsePayload');
    window.selectModel = delegate('uploadImpl', 'selectModel');
    window.getModelDisplayName = delegate('uploadImpl', 'getModelDisplayName');

    window.startLive = delegate('liveImpl', 'startLive');
    window.switchToLive = delegate('liveImpl', 'switchToLive');
    window.switchToUpload = delegate('liveImpl', 'switchToUpload');
    window.setLiveRunningUI = delegate('liveImpl', 'setLiveRunningUI');
    window.pollLive = delegate('liveImpl', 'pollLive');
    window.clearLiveResultTable = delegate('liveImpl', 'clearLiveResultTable');

    window.addResultRow = delegate('resultsImpl', 'addResultRow');
    window.openDetails = delegate('resultsImpl', 'openDetails');
    window.cacheRecordedRow = delegate('resultsImpl', 'cacheRecordedRow');
    window.renderRecordedRows = delegate('resultsImpl', 'renderRecordedRows');
    window.applyRecordFilter = delegate('resultsImpl', 'applyRecordFilter');
    window.activateUploadRecordCard = delegate('uploadRecordCardImpl', 'activate');
    window.activateLiveRecordCard = delegate('liveRecordCardImpl', 'activate');
    // fallback in case an older cached results_impl.js is loaded without applyRecordFilter
    (function() {
        const delegated = window.applyRecordFilter;
        window.applyRecordFilter = function(filterType) {
            const delegatedResult = (typeof delegated === 'function') ? delegated.apply(null, arguments) : null;
            if (delegatedResult !== null) return delegatedResult;
            try {
                const normalized = String(filterType || 'all').toLowerCase();
                const finalFilter = ['all', 'benign', 'malware'].includes(normalized) ? normalized : 'all';
                window.currentRecordFilter = finalFilter;
                const rows = document.querySelectorAll('#resultTableBody tr');
                rows.forEach((row) => {
                    const type = row && row.dataset ? String(row.dataset.recordType || '').toLowerCase() : '';
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
                try { persistUiState(); } catch (e) {}
            } catch (e) {
                console.warn('fallback applyRecordFilter failed', e);
            }
            return null;
        };
    })();

    // ==================== 辅助函数 ====================
    function findButtonByText(text) {
        return Array.from(document.querySelectorAll('button')).find(b => 
            (b.innerText || '').trim().indexOf(text) !== -1
        );
    }

    function getSelectedInterface() {
        return window.liveUiState?.selectedInterface || null;
    }
    window.getSelectedInterface = getSelectedInterface;

    function initRecordFilterIndicator() {
        const btnAll = document.getElementById('recordFilterAll');
        const btnBenign = document.getElementById('recordFilterBenign');
        const btnMalware = document.getElementById('recordFilterMalware');
        const container = btnAll && btnAll.parentElement;
        if (!container || !btnAll || !btnBenign || !btnMalware) return;

        container.classList.add('relative', 'overflow-hidden');
        [btnAll, btnBenign, btnMalware].forEach((btn) => btn.classList.add('relative', 'z-10'));

        let indicator = container.querySelector('.record-filter-indicator');
        if (!indicator) {
            indicator = document.createElement('span');
            indicator.className = 'record-filter-indicator';
            indicator.setAttribute('aria-hidden', 'true');
            indicator.style.position = 'absolute';
            indicator.style.top = '4px';
            indicator.style.bottom = '4px';
            indicator.style.left = '0';
            indicator.style.width = '0px';
            indicator.style.borderRadius = '9999px';
            indicator.style.background = 'rgb(0 229 255)';
            indicator.style.boxShadow = '0 0 12px rgba(0,229,255,0.35)';
            indicator.style.transition = 'transform 220ms ease, width 220ms ease, opacity 180ms ease';
            indicator.style.willChange = 'transform, width';
            indicator.style.zIndex = '0';
            container.insertBefore(indicator, container.firstChild);
        }

        window.updateRecordFilterIndicator = function(filterType) {
            const f = String(filterType || window.currentRecordFilter || 'all').toLowerCase();
            const target = f === 'benign' ? btnBenign : (f === 'malware' ? btnMalware : btnAll);
            if (!target || !container || !indicator) return;
            const cRect = container.getBoundingClientRect();
            const tRect = target.getBoundingClientRect();
            const x = Math.max(0, tRect.left - cRect.left);
            indicator.style.width = `${tRect.width}px`;
            indicator.style.transform = `translateX(${x}px)`;
            indicator.style.opacity = '1';
        };

        try { window.updateRecordFilterIndicator(window.currentRecordFilter || 'all'); } catch (e) {}
        window.addEventListener('resize', () => {
            try { window.updateRecordFilterIndicator(window.currentRecordFilter || 'all'); } catch (e) {}
        });
    }

    // ==================== 网卡加载 ====================
    async function loadInterfaces() {
        const dropdownMenu = document.getElementById('dropdownMenu');
        const selectedText = document.getElementById('selectedText');
        if (!dropdownMenu) return;
        
        dropdownMenu.innerHTML = '<li class="px-3 py-2 text-sm text-gray-400">加载中...</li>';
        try {
            const r = await fetch('/api/interfaces');
            if (!r.ok) throw new Error('Failed to load');
            const body = await r.json();
            dropdownMenu.innerHTML = '';
            const list = body?.interfaces || [];
            
            if (!list.length) {
                dropdownMenu.innerHTML = '<li class="px-3 py-2 text-sm text-gray-400">无可用网卡</li>';
                return;
            }
            
            list.forEach((iface) => {
                const li = document.createElement('li');
                li.className = 'px-3 py-2 cursor-pointer hover:bg-cyber-card/30';
                const value = typeof iface === 'object' ? (iface.value || iface.name) : iface;
                li.dataset.ifaceValue = value;
                
                let display = '';
                if (typeof iface === 'object') {
                    display = (iface.name || value) + (iface.description && iface.description !== (iface.name || value) ? (' - ' + iface.description) : '');
                } else {
                    display = String(iface);
                }
                li.innerText = display;
                
                li.addEventListener('click', () => {
                    window.liveUiState = window.liveUiState || {};
                    window.liveUiState.selectedInterface = value;
                    window.liveUiState.selectedInterfaceLabel = li.innerText;
                    if (selectedText) selectedText.innerText = li.innerText;
                    dropdownMenu.classList.add('hidden');
                    try { persistUiState(); } catch(e) {}
                    try { persistLiveUiState?.(); } catch(e) {}
                });
                dropdownMenu.appendChild(li);
            });
            
            // 恢复上次选择
            const pref = window.liveUiState?.selectedInterface;
            if (pref) {
                const node = Array.from(dropdownMenu.children).find(c => c.dataset?.ifaceValue === pref);
                if (node && selectedText) {
                    selectedText.innerText = node.innerText;
                    window.liveUiState.selectedInterfaceLabel = node.innerText;
                } else if (selectedText && window.liveUiState?.selectedInterfaceLabel) {
                    selectedText.innerText = window.liveUiState.selectedInterfaceLabel;
                }
            } else if (dropdownMenu.children.length > 0) {
                const first = dropdownMenu.children[0];
                if (first && selectedText) {
                    selectedText.innerText = first.innerText;
                    window.liveUiState = window.liveUiState || {};
                    window.liveUiState.selectedInterface = first.dataset.ifaceValue;
                    window.liveUiState.selectedInterfaceLabel = first.innerText;
                    try { persistUiState(); } catch(e) {}
                }
            }
        } catch(e) {
            dropdownMenu.innerHTML = '<li class="px-3 py-2 text-sm text-red-400">加载失败</li>';
        }
    }
    window.loadInterfaces = loadInterfaces;

    // ==================== 模型指标加载 ====================
    async function loadModelMetrics() {
        try {
            const r = await fetch('/static/checkpoints/model_metrics.json');
            if (!r.ok) return;
            const body = await r.json();
            const map = {
                'CNN_BiLSTM': 'm1',
                'Classic_CNN': 'm2',
                'Lightweight_CNN_BiLSTM': 'm3',
                'Pure_BiLSTM': 'm4',
                'MLP': 'm5',
                'Transformer': 'm6'
            };
            const fieldMap = {
                pre: 'precision', rec: 'recall', f1: 'f1_score', f2: 'f2_score',
                dr: 'detection_rate', fpr: 'fpr', fnr: 'fnr', auc: 'auc',
                lat: 'latency_ms_per_sample', tps: 'throughput_samples_per_sec'
            };
            
            const formatValue = (field, value) => {
                if (value == null || value === '') return '--';
                if (field === 'auc') return String(value);
                if (field === 'lat') return value + ' ms/条';
                if (field === 'tps') return value + ' 条/s';
                const text = String(value);
                return text.endsWith('%') ? text : text + '%';
            };
            
            Object.keys(map).forEach((key) => {
                const node = body[key];
                const prefix = map[key];
                Object.entries(fieldMap).forEach(([suffix, field]) => {
                    const el = document.getElementById(prefix + '-' + suffix);
                    if (el) el.innerText = formatValue(suffix, node?.[field]);
                });
                const accEl = document.getElementById(prefix + '-acc');
                if (accEl) {
                    const val = node?.test_accuracy || node?.best_val_accuracy || node?.best_accuracy;
                    accEl.innerText = formatValue('acc', val);
                }
            });
            
            if (!window.currentActiveModel) {
                const selector = document.getElementById('modelSelector');
                window.currentActiveModel = window.persistedVotingModeEnabled ? 'ensemble_vote' : (selector?.value || 'cnn_bilstm');
            }
            if (!window.persistedVotingModeEnabled) {
                try { window.selectModel?.(window.currentActiveModel); } catch(e) {}
            }
        } catch(e) { console.warn('loadModelMetrics failed', e); }
    }
    window.loadModelMetrics = loadModelMetrics;

    // ==================== 事件绑定 ====================
    function bindEventHandlers() {
        const elements = {
            fileInput: document.getElementById('fileInput'),
            chooseFileBtn: document.getElementById('chooseFileBtn'),
            startLiveBtn: document.getElementById('startLiveBtn'),
            stopLiveBtn: document.getElementById('stopLiveBtn'),
            tabLive: findButtonByText('实时抓包') || document.getElementById('tabLive'),
            tabUpload: findButtonByText('上传检测') || document.getElementById('tabUpload'),
            selectedDisplay: document.getElementById('selectedDisplay'),
            dropdownMenu: document.getElementById('dropdownMenu'),
            refreshIfaces: document.getElementById('refreshIfaces'),
            recordFilterAll: document.getElementById('recordFilterAll'),
            recordFilterBenign: document.getElementById('recordFilterBenign'),
            recordFilterMalware: document.getElementById('recordFilterMalware')
        };

        if (elements.fileInput) {
            elements.fileInput.onchange = (e) => window.handleFiles?.(e.target.files);
        }
        if (elements.chooseFileBtn) {
            elements.chooseFileBtn.onclick = () => document.getElementById('fileInput')?.click();
        }
        if (elements.startLiveBtn) {
            elements.startLiveBtn.onclick = () => window.startLive?.();
        }
        if (elements.stopLiveBtn) {
            elements.stopLiveBtn.onclick = () => {
                fetch('/api/live/stop', { method: 'POST' }).then(() => window.setLiveRunningUI?.(false));
            };
        }
        if (elements.tabLive) elements.tabLive.onclick = () => window.switchToLive?.();
        if (elements.tabUpload) elements.tabUpload.onclick = () => window.switchToUpload?.();
        
        if (elements.selectedDisplay && elements.dropdownMenu) {
            elements.selectedDisplay.onclick = () => {
                elements.dropdownMenu.classList.toggle('hidden');
                const arrow = document.getElementById('dropdownArrow');
                if (arrow) arrow.style.transform = elements.dropdownMenu.classList.contains('hidden') ? '' : 'rotate(180deg)';
            };
        }
        if (elements.refreshIfaces) {
            elements.refreshIfaces.onclick = () => loadInterfaces();
        }
        // Model selector dropdown change -> update active model and UI
        const modelSelectorEl = document.getElementById('modelSelector');
        if (modelSelectorEl) {
            modelSelectorEl.onchange = (e) => {
                try { window.selectModel?.(e.target.value); } catch (err) { console.warn('modelSelector onchange failed', err); }
            };
        }
        if (elements.recordFilterAll) {
            elements.recordFilterAll.onclick = () => window.applyRecordFilter?.('all');
        }
        if (elements.recordFilterBenign) {
            elements.recordFilterBenign.onclick = () => window.applyRecordFilter?.('benign');
        }
        if (elements.recordFilterMalware) {
            elements.recordFilterMalware.onclick = () => window.applyRecordFilter?.('malware');
        }

        initRecordFilterIndicator();
    }

    // ==================== 模块加载器 ====================
    const assetVersion = '20260522a';
    const modulePaths = [
        '/static/js/core/state.js?v=' + assetVersion,
        '/static/js/core/utils.js?v=' + assetVersion,
        '/static/js/core/api.js?v=' + assetVersion,
        '/static/js/modules/upload_impl.js?v=' + assetVersion,
        '/static/js/modules/upload_record_card_impl.js?v=' + assetVersion,
        '/static/js/modules/live_record_card_impl.js?v=' + assetVersion,
        '/static/js/modules/upload_trend_impl.js?v=' + assetVersion,
        '/static/js/modules/live_trend_impl.js?v=' + assetVersion,
        '/static/js/modules/modelEval_impl.js?v=' + assetVersion,
        '/static/js/modules/results_impl.js?v=' + assetVersion,
        '/static/js/modules/live_impl.js?v=' + assetVersion
    ];

    function loadScript(src) {
        return new Promise((resolve, reject) => {
            if (document.querySelector(`script[src="${src}"]`)) {
                resolve();
                return;
            }
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = () => reject(new Error(`Failed: ${src}`));
            document.head.appendChild(script);
        });
    }

    async function init() {
        // 加载核心模块
        for (const path of modulePaths) {
            try {
                await loadScript(path);
                console.log(`✅ Loaded: ${path.split('/').pop()}`);
            } catch (e) {
                console.error(`❌ Failed: ${path}`, e);
            }
        }
        
        // 绑定事件
        bindEventHandlers();
        // 绑定结果表详情委托（由 results_impl 提供）
        try { window.resultsImpl && typeof window.resultsImpl.bindDetailButtonEvents === 'function' && window.resultsImpl.bindDetailButtonEvents(); } catch(e) { console.warn('bindDetailButtonEvents failed', e); }
        
        // 初始化
        setTimeout(() => {
            loadInterfaces();
            loadModelMetrics();
            window.initCharts?.();
            window.initModelEvalCharts?.();
            try {
                const uploadRecords = window.recordStores && Array.isArray(window.recordStores.upload) ? window.recordStores.upload : [];
                if (window.uploadTrendImpl && typeof window.uploadTrendImpl.restoreFromRecords === 'function') {
                    window.uploadTrendImpl.restoreFromRecords(uploadRecords);
                }
                const liveRecords = window.recordStores && Array.isArray(window.recordStores.live) ? window.recordStores.live : [];
                if (window.liveTrendImpl && typeof window.liveTrendImpl.restoreFromRecords === 'function') {
                    window.liveTrendImpl.restoreFromRecords(liveRecords);
                }
                if (window.liveGraph && typeof window.liveGraph.restoreFromRecords === 'function') {
                    window.liveGraph.restoreFromRecords(liveRecords);
                }
            } catch (e) { console.warn('restore chart state failed', e); }
            
            // 启动轮询
            if (window.pollLive) {
                window.pollLive();
                window.livePoller = setInterval(window.pollLive, 1500);
            }
            // 恢复选中标签（避免刷新后样式丢失）
            try {
                const persisted = loadUiState();
                const persistedTabMode = (persisted && (persisted.tabMode === 'live' || persisted.tabMode === 'upload')) ? persisted.tabMode : null;
                const preferLive = !!(window.liveUiState && window.liveUiState.running)
                    || (persistedTabMode === 'live')
                    || (window.recordTableMode === 'live');
                if (preferLive) {
                    try { if (typeof window.switchToLive === 'function') window.switchToLive(); } catch(e){}
                } else {
                    try { if (typeof window.switchToUpload === 'function') window.switchToUpload(); } catch(e){}
                }
            } catch(e) { console.warn('restore tab state failed', e); }
            try {
                if (typeof window.applyRecordFilter === 'function') {
                    window.applyRecordFilter(window.currentRecordFilter || 'all');
                }
            } catch (e) { console.warn('restore record filter failed', e); }
        }, 100);
        
        console.log('✅ Main initialized');
    }

    // 启动
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();