// main.js - 精简入口，只负责模块加载、事件绑定和委托

(function() {
    'use strict';

    // ---------- 图表动效规范 ----------
    // ECharts 默认入场动画 1000ms：趋势图每 1.5s 轮询刷新一次，数据一更新整张图就重新
    // "长"一遍，看起来像在抽搐。这里统一压到 220ms（与 CSS 的 --mo-base 同档），
    // 缓动固定 cubicOut，并让所有 echarts.init 显式带上这个主题。
    try {
        if (window.echarts && typeof window.echarts.registerTheme === 'function') {
            window.DLIDS_CHART_THEME = 'dlids';
            window.echarts.registerTheme('dlids', {
                animationDuration: 220,
                animationEasing: 'cubicOut',
                animationDurationUpdate: 220,
                animationEasingUpdate: 'cubicOut'
            });
        }
    } catch (e) { console.warn('registerTheme failed', e); }

    // KPI 数字变化时做一次短促高亮（styles.css 的 .mo-flash）；值没变就不动，
    // 避免轮询每秒把同一个数字闪一遍。
    window.dlidsFlash = function (el) {
        if (!el) return;
        el.classList.remove('mo-flash');
        void el.offsetWidth; // 强制重排，让同一个动画可以被连续触发
        el.classList.add('mo-flash');
        setTimeout(function () { el.classList.remove('mo-flash'); }, 460);
    };
    window.dlidsSetKpi = function (el, value) {
        if (!el) return;
        const next = String(value === null || value === undefined ? 0 : value);
        if (el.innerText === next) return;
        el.innerText = next;
        window.dlidsFlash(el);
    };

    // ---------- 全站统一统计口径 ----------
    // 为什么需要这一层：KPI 带与「威胁类型分布」原先各自记账——上传流程写 window.totalScans，
    // 实时轮询又直接把 KPI 改成"抓包累计包数"，于是同一屏里两个数字口径不同、切页面就变。
    // 现在只有一份数据、只有一个渲染函数；任何地方都不许再直接写 statTotal/statMalware/#pieChart。
    window.dlidsStats = window.dlidsStats || {
        upload: { total: 0, malware: 0 },   // 上传检测：按文件计
        live: { sessions: 0, malware: 0 }   // 实时抓包：按会话计（后端给的是累计值，取最大值保证单调）
    };

    window.dlidsStatsSnapshot = function () {
        const s = window.dlidsStats || {};
        const upload = s.upload || { total: 0, malware: 0 };
        const live = s.live || { sessions: 0, malware: 0 };
        const total = (Number(upload.total) || 0) + (Number(live.sessions) || 0);
        let malware = (Number(upload.malware) || 0) + (Number(live.malware) || 0);
        if (malware > total) malware = total;
        return { total: total, malware: malware, benign: Math.max(0, total - malware) };
    };

    // 唯一允许写 KPI 与威胁分布饼图的地方
    window.dlidsRenderStats = function () {
        const snap = window.dlidsStatsSnapshot();
        // 兼容既有读取方（详情弹窗、模型评估等处仍在读这两个全局）
        window.totalScans = snap.total;
        window.malwareCount = snap.malware;
        try { if (window.dlidsSetKpi) window.dlidsSetKpi(document.getElementById('statTotal'), snap.total); } catch (e) {}
        try { if (window.dlidsSetKpi) window.dlidsSetKpi(document.getElementById('statMalware'), snap.malware); } catch (e) {}
        // 饼图实例可能登记在 charts_impl（window.pieChart）或 upload_trend_impl（window.uploadPieChart）名下，
        // 谁先 init 成功就是谁；这里两个都认，否则会出现「KPI 有数字、饼图整块空白」。
        const pie = window.pieChart || window.uploadPieChart;
        if (pie && typeof pie.setOption === 'function') {
            try {
                const pieTotal = snap.total;
                pie.setOption({
                    title: { show: pieTotal <= 0, text: '暂无数据', left: 'center', top: 'middle', textStyle: { color: '#64748B', fontSize: 12, fontWeight: 'normal' } },
                    legend: { show: pieTotal > 0 },
                    series: [{
                        data: pieTotal > 0 ? [
                            { value: snap.benign, name: 'Normal', itemStyle: { color: '#10B981' } },
                            { value: snap.malware, name: 'Malware', itemStyle: { color: '#F43F5E' } }
                        ] : []
                    }]
                });
            } catch (e) { console.warn('dlidsRenderStats pie failed', e); }
        }
        return snap;
    };

    const UI_STATE_KEY = 'dlids_ui_state';
    // 实时抓包轮询间隔（毫秒）。pollLive 已改为按 since_id 增量拉取，这里只负责节奏。
    const LIVE_POLL_INTERVAL_MS = 1500;

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
    // 页面模式由当前页面模板上的 data-mode 决定（upload / live / all）。
    // 不再从 localStorage 的 tabMode 推断：页面之间是整页跳转，模式不该被上一次浏览的页面带过来。
    function resolvePageMode() {
        const panel = document.getElementById('resultCardPanel');
        const mode = panel && panel.dataset ? String(panel.dataset.mode || '') : '';
        return (mode === 'live' || mode === 'upload' || mode === 'all') ? mode : 'upload';
    }
    // 无刷新换页后内容区是新的 DOM，data-mode 可能变了，必须重算一次
    window.dlidsResolvePageMode = resolvePageMode;
    window.recordTableMode = resolvePageMode();

    const savedFilter = savedUiState && ['all', 'benign', 'malware'].includes(String(savedUiState.currentFilter || '').toLowerCase())
        ? String(savedUiState.currentFilter).toLowerCase()
        : 'all';
    window.currentRecordFilter = savedFilter;
    window._liveSeen = window._liveSeen || new Set();
    window.__detailsDataMap = window.__detailsDataMap || {};

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
    window.stopLive = delegate('liveImpl', 'stopLive');
    window.toggleLiveCapture = delegate('liveImpl', 'toggleLiveCapture');
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
            indicator.style.background = 'rgb(34 211 238)';
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
        // resize 监听只能挂一次：这个函数在无刷新换页后每页都会重新跑一遍，写在函数体里会让
        // 换页次数等于监听器个数。回调每次都重新查 window.updateRecordFilterIndicator，
        // 所以换页后它自动指向新页面的滑块。
        if (!window.__recordFilterResizeBound) {
            window.__recordFilterResizeBound = true;
            window.addEventListener('resize', () => {
                // 代码审查第 49 条：拖拽窗口会连续触发 resize，用 rAF 把同一帧内的多次回调合并成一次
                if (window.__recordFilterResizeRaf) cancelAnimationFrame(window.__recordFilterResizeRaf);
                window.__recordFilterResizeRaf = requestAnimationFrame(() => {
                    try { window.updateRecordFilterIndicator(window.currentRecordFilter || 'all'); } catch (e) {}
                });
            });
        }
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
            // 代码审查疑点 3：走 /api/metrics（唯一数据源），响应形如 {status, data}
            const r = await fetch('/api/metrics');
            if (!r.ok) return;
            const payload = await r.json();
            const body = (payload && payload.data) ? payload.data : payload;
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
            liveToggleBtn: document.getElementById('liveToggleBtn'),
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
        if (elements.liveToggleBtn) {
            elements.liveToggleBtn.onclick = () => window.toggleLiveCapture?.();
        }
        
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
    const assetVersion = '20260604c';
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
        '/static/js/modules/live_impl.js?v=' + assetVersion,
        '/static/js/modules/router_impl.js?v=' + assetVersion
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

    /* ==================== 每页初始化 / 拆卸 ====================
       这两个函数是无刷新换页的接口：router_impl 在替换内容区之前调 teardownPage()，
       替换完成后调 initPage()。首屏也走同一套路径，所以「整页加载」和「站内换页」
       之后的状态是一致的，不存在两套逻辑。 */
    function initPage() {
        // 页面模式由当前页面的 data-mode 决定，换页后必须重算（上传页 / 实时页 / 合并页）
        window.recordTableMode = resolvePageMode();

        bindEventHandlers();
        // 绑定结果表详情委托（由 results_impl 提供；内部按 tbody 上的标记做了幂等）
        try { window.resultsImpl && typeof window.resultsImpl.bindDetailButtonEvents === 'function' && window.resultsImpl.bindDetailButtonEvents(); } catch(e) { console.warn('bindDetailButtonEvents failed', e); }
        // 清除按钮：同样幂等，换页后要绑到新的那颗按钮上
        try { window.resultsImpl && typeof window.resultsImpl.bindClearButtons === 'function' && window.resultsImpl.bindClearButtons(); } catch(e) { console.warn('bindClearButtons failed', e); }
        // 这个模块的初始化挂在 DOMContentLoaded 上，换页不会再触发，必须手动补一次
        try { window.uploadImpl && typeof window.uploadImpl.initVotingModeControls === 'function' && window.uploadImpl.initVotingModeControls(); } catch(e) { console.warn('initVotingModeControls failed', e); }
        // 主机辐射图：只有实时页有 #liveFlowGraph，其它页面它会立刻返回
        try { window.liveGraph && typeof window.liveGraph.init === 'function' && window.liveGraph.init(); } catch(e) { console.warn('liveGraph.init failed', e); }

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

            // 顶栏状态胶囊在内容区之外，但实时页卡片上的「运行中 / 停止」在内容区里，
            // 换页后是新节点，必须强制重画一次（setLiveRunningUI 默认在状态没变时会直接返回）。
            try { if (typeof window.setLiveRunningUI === 'function') window.setLiveRunningUI(!!(window.liveUiState && window.liveUiState.running), { force: true }); } catch (e) {}

            // 按当前页面的 mode 把落盘的历史记录画进表格：上传页只画上传、实时页只画实时、
            // 总览与记录页把两路合并。
            try {
                if (typeof window.renderRecordedRows === 'function') window.renderRecordedRows(window.recordTableMode);
            } catch (e) { console.warn('restore record rows failed', e); }
            try {
                if (typeof window.applyRecordFilter === 'function') {
                    window.applyRecordFilter(window.currentRecordFilter || 'all');
                }
            } catch (e) { console.warn('restore record filter failed', e); }

            // KPI 带与饼图的新 DOM 也是刚生成的（模板里是初始的 0），而 window.dlidsStats
            // 一直活在窗口上没被换页清掉 —— 统一统计层是唯一写这两处的地方，所以最后必须
            // 按它重画一次，否则换页后 KPI 会「回到 0」，与记录表里的行数自相矛盾。
            try { if (typeof window.dlidsRenderStats === 'function') window.dlidsRenderStats(); } catch (e) { console.warn('repaint stats failed', e); }
        }, 100);
    }

    function teardownPage() {
        // 内容区即将被替换。这里不能只依赖 upload_trend_impl 的 isChartInstance()：
        // ECharts 的 dispose() 之后 setOption / getOption 仍然是函数，旧实例会被当成
        // 有效实例复用 → 新页面上 setOption 抛错被 catch 吞掉，图表永久空白。
        // 所以必须显式 dispose 并把全局引用清零。
        const dispose = (inst) => {
            try { if (inst && typeof inst.dispose === 'function' && typeof inst.setOption === 'function') inst.dispose(); } catch (e) {}
        };
        dispose(window.uploadTrendChart);
        dispose(window.uploadPieChart);
        dispose(window.liveTrendChart);
        dispose(window.lineChart);
        dispose(window.pieChart);
        window.uploadTrendChart = null;
        window.uploadPieChart = null;
        window.liveTrendChart = null;
        // 这两个名字会被 HTML 命名元素访问解析成 <div>，置 null 是让 getLineChart / getPieChart
        // 回落到 echarts.getInstanceByDom（新容器没有实例 → 重新 init）。
        window.lineChart = null;
        window.pieChart = null;

        // 模型评估页的实例登记在模块内部的 instances 表里，必须让它自己清
        try { window.modelEvalImpl && typeof window.modelEvalImpl.disposeAll === 'function' && window.modelEvalImpl.disposeAll(); } catch (e) { console.warn('modelEval disposeAll failed', e); }
        // 主机辐射图是逐帧循环画的：旧 canvas 一走就要停，否则一直空转，而且重进实时页时
        // startLoop 会被 _animLoopBound 挡住，新 canvas 永远不画。
        try { window.liveGraph && typeof window.liveGraph.stopLoop === 'function' && window.liveGraph.stopLoop(); } catch (e) { console.warn('liveGraph stopLoop failed', e); }
    }

    window.dlidsPages = { initPage: initPage, teardownPage: teardownPage };

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

        initPage();

        // 站内换页改成无刷新（见 modules/router_impl.js）。直接输网址与刷新仍是完整加载。
        try { window.routerImpl && typeof window.routerImpl.init === 'function' && window.routerImpl.init(); } catch (e) { console.warn('routerImpl init failed', e); }

        // 启动轮询（幂等：先清理可能已存在的定时器，避免 init 与 startLive 叠加出多个轮询器重复拉取）
        // 注意：这是全站唯一的轮询器，跨页面一直活着，不随换页重建 —— 否则换页会丢一帧抓包数据。
        if (window.pollLive) {
            if (window.livePoller) clearInterval(window.livePoller);
            window.pollLive();
            window.livePoller = setInterval(window.pollLive, LIVE_POLL_INTERVAL_MS);
        }

        console.log('✅ Main initialized');
    }

    // 启动
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();