// core/state.js - 全局状态管理

(function() {
    'use strict';

    const STORAGE_KEY = 'malware_detector_state';
    const MAX_PERSIST_RECORDS = 300; // localStorage 中每种记录（upload/live）最多保留最近 300 条
    const MAX_PERSIST_BYTES = 3 * 1024 * 1024; // 序列化后的字节上限：3 MB
    
    // 状态定义
    const AppState = {
        live: { running: false, selectedFilter: 'all', selectedInterface: null, captureModel: null, pollInterval: null },
        records: { live: [], upload: [] },
        tableMode: 'upload',
        currentFilter: 'all',
        liveSeen: new Set(),
        detailsData: new Map(),
        totalScans: 0,
        malwareCount: 0,
        trendLabels: [], benignTrend: [], malwareTrend: [],
        chartInstances: { lineChart: null, pieChart: null, flowGraph: null, modalBarChart: null },
        modelConfig: { currentModel: 'cnn_bilstm', modelMap: { 'cnn_bilstm': 'CNN_BiLSTM', 'classic_cnn': 'Classic_CNN', 'lightweight': 'Lightweight_CNN_BiLSTM', 'lightweight_cnn': 'Lightweight_CNN_BiLSTM', 'pure_bilstm': 'Pure_BiLSTM', 'mlp': 'MLP', 'transformer': 'Transformer' } },
        config: { maxRecordsPerType: 500, maxTrendPoints: 50, pollIntervalMs: 1500, autoSave: true }
    };

    // 序列化后的 UTF-8 字节数：localStorage 配额按字节计算，不能只看字符数
    function jsonByteLength(str) {
        if (!str) return 0;
        try { if (typeof TextEncoder !== 'undefined') return new TextEncoder().encode(str).length; } catch (e) { /* fallthrough */ }
        return str.length * 2; // 兜底：保守估计
    }

    // 组装持久化 JSON，双重上限保护：
    //   1) 条数：每种记录只取最近 MAX_PERSIST_RECORDS 条；
    //   2) 字节：序列化后超过 MAX_PERSIST_BYTES 时，从最旧的记录开始裁剪（upload/live 交替，数组为追加序，头部即最旧）。
    function buildPersistSnapshot() {
        const records = AppState.records || {};
        const upload = (Array.isArray(records.upload) ? records.upload : []).slice(-MAX_PERSIST_RECORDS);
        const live = (Array.isArray(records.live) ? records.live : []).slice(-MAX_PERSIST_RECORDS);
        const serialize = () => JSON.stringify({
            records: { upload, live },
            totalScans: AppState.totalScans,
            malwareCount: AppState.malwareCount,
            trendLabels: AppState.trendLabels.slice(-50),
            benignTrend: AppState.benignTrend.slice(-50),
            malwareTrend: AppState.malwareTrend.slice(-50),
            live: { selectedInterface: AppState.live.selectedInterface, captureModel: AppState.live.captureModel },
            modelConfig: { currentModel: AppState.modelConfig.currentModel }
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

    // 持久化
    function persist() {
        if (!AppState.config.autoSave) return;
        try {
            if (typeof localStorage === 'undefined' || !localStorage) return;
            localStorage.setItem(STORAGE_KEY, buildPersistSnapshot());
        } catch(e) {
            const isQuota = e && (e.name === 'QuotaExceededError' || e.name === 'NS_ERROR_DOM_QUOTA_REACHED' || e.code === 22 || e.code === 1014 || /quota/i.test(String(e.message || '')));
            if (isQuota) {
                // 裁剪后仍写不进去（通常是浏览器总配额已被其它 key 占满）：清空该 key 并明确告警，不再静默失败
                try { localStorage.removeItem(STORAGE_KEY); } catch (removeErr) { /* ignore */ }
                console.warn('Persist failed: localStorage 配额超限（QuotaExceededError），已清空并重置本地状态记录（key=' + STORAGE_KEY + '）；内存中的状态仍保留，但本次及其后的写入在配额恢复前都会失败。', e);
                return;
            }
            console.warn('Persist failed:', e);
        }
    }

    // 恢复
    function restore() {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) {
                let data;
                try {
                    data = JSON.parse(saved);
                } catch (parseErr) {
                    // 本地数据已损坏：清空该 key 并明确告警，避免每次启动都解析失败
                    try { localStorage.removeItem(STORAGE_KEY); } catch (removeErr) { /* ignore */ }
                    console.warn('Restore failed: 本地状态（key=' + STORAGE_KEY + '）不是合法 JSON，已清空并重置该键。', parseErr);
                    return;
                }
                if (data.records) {
                    // 恢复侧沿用原内存上限（AppState.config.maxRecordsPerType=500），300 条只是“写盘”上限
                    const memoryCap = AppState.config.maxRecordsPerType || 500;
                    if (Array.isArray(data.records.upload)) AppState.records.upload = data.records.upload.slice(-memoryCap);
                    if (Array.isArray(data.records.live)) AppState.records.live = data.records.live.slice(-memoryCap);
                }
                if (data.totalScans) AppState.totalScans = data.totalScans;
                if (data.malwareCount) AppState.malwareCount = data.malwareCount;
                if (data.trendLabels) AppState.trendLabels = data.trendLabels;
                if (data.benignTrend) AppState.benignTrend = data.benignTrend;
                if (data.malwareTrend) AppState.malwareTrend = data.malwareTrend;
                if (data.live?.selectedInterface) AppState.live.selectedInterface = data.live.selectedInterface;
                if (data.modelConfig?.currentModel) AppState.modelConfig.currentModel = data.modelConfig.currentModel;
            }
        } catch(e) { console.warn('Restore failed:', e); }
    }

    restore();

    // 导出到全局
    window.AppState = AppState;
    window.StateModule = {
        getState: () => AppState,
        persist, restore,
        addRecord: (kind, entry) => {
            if (!AppState.records[kind]) AppState.records[kind] = [];
            AppState.records[kind].push(entry);
            if (AppState.records[kind].length > 500) AppState.records[kind].shift();
            persist();
        },
        incrementScan: (isMalware) => { AppState.totalScans++; if (isMalware) AppState.malwareCount++; persist(); },
        addTrendPoint: (label, isMalware) => {
            AppState.trendLabels.push(label);
            AppState.benignTrend.push(isMalware ? 0 : 1);
            AppState.malwareTrend.push(isMalware ? 1 : 0);
            if (AppState.trendLabels.length > 50) { AppState.trendLabels.shift(); AppState.benignTrend.shift(); AppState.malwareTrend.shift(); }
            persist();
        },
        setLiveRunning: (running) => { AppState.live.running = running; persist(); },
        setTableMode: (mode) => { if (mode === 'upload' || mode === 'live') { AppState.tableMode = mode; persist(); } },
        setCurrentFilter: (filter) => { if (['all','benign','malware'].includes(filter)) { AppState.currentFilter = filter; AppState.live.selectedFilter = filter; persist(); } },
        getCurrentModel: () => AppState.modelConfig.currentModel,
        setCurrentModel: (model) => { AppState.modelConfig.currentModel = model; persist(); },
        getModelDisplayName: (modelId) => AppState.modelConfig.modelMap[modelId] || modelId,
        isSeenLiveFlow: (id) => AppState.liveSeen.has(id),
        addSeenLiveFlow: (id) => { AppState.liveSeen.add(id); if (AppState.liveSeen.size > 1000) { const it = AppState.liveSeen.values(); for (let i = 0; i < 500; i++) AppState.liveSeen.delete(it.next().value); } },
        setDetailsData: (id, data) => { AppState.detailsData.set(id, data); if (AppState.detailsData.size > 200) { const first = AppState.detailsData.keys().next().value; AppState.detailsData.delete(first); } },
        getDetailsData: (id) => AppState.detailsData.get(id)
    };

    // 兼容旧版全局函数
    window.persistLiveUiState = persist;
    window.restoreLiveUiState = restore;
})();