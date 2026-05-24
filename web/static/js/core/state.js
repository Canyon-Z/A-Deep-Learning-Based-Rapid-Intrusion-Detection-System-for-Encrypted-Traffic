// core/state.js - 全局状态管理

(function() {
    'use strict';

    const STORAGE_KEY = 'malware_detector_state';
    
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

    // 持久化
    function persist() {
        if (!AppState.config.autoSave) return;
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify({
                records: { upload: AppState.records.upload.slice(-500), live: AppState.records.live.slice(-500) },
                totalScans: AppState.totalScans,
                malwareCount: AppState.malwareCount,
                trendLabels: AppState.trendLabels.slice(-50),
                benignTrend: AppState.benignTrend.slice(-50),
                malwareTrend: AppState.malwareTrend.slice(-50),
                live: { selectedInterface: AppState.live.selectedInterface, captureModel: AppState.live.captureModel },
                modelConfig: { currentModel: AppState.modelConfig.currentModel }
            }));
        } catch(e) { console.warn('Persist failed:', e); }
    }

    // 恢复
    function restore() {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) {
                const data = JSON.parse(saved);
                if (data.records) { if (data.records.upload) AppState.records.upload = data.records.upload; if (data.records.live) AppState.records.live = data.records.live; }
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