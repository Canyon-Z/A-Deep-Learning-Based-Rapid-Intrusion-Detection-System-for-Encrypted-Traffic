// modules/results.js - 结果表格模块
window.ResultsModule = (function(Utils, AppState) {
    // Buffer for live-mode DOM inserts to avoid frequent reflows
    let liveRowBuffer = [];
    let _liveFlushScheduled = false;
    const LIVE_FLUSH_INTERVAL_MS = 1500; // flush every ~1.5s

    function scheduleLiveFlush() {
        if (_liveFlushScheduled) return;
        _liveFlushScheduled = true;
        setTimeout(() => { try { flushLiveBuffer(); } catch(e) { console.warn('flushLiveBuffer failed', e); } _liveFlushScheduled = false; }, LIVE_FLUSH_INTERVAL_MS);
    }

    function flushLiveBuffer() {
        if (!liveRowBuffer.length) return;
        const tbody = document.getElementById('resultTableBody');
        if (!tbody) { liveRowBuffer = []; return; }
        const frag = document.createDocumentFragment();
        while (liveRowBuffer.length) {
            const { rowHtml, rowId, isMal, name, time, size, data } = liveRowBuffer.shift();
            const row = document.createElement('tr');
            row.className = 'border-b border-cyber-border/50';
            row.dataset.recordType = isMal ? 'malware' : 'benign';
            row.innerHTML = rowHtml;
            // attach details handler
            row.querySelector('.view-details-btn')?.addEventListener('click', () => { window.openDetails?.(rowId); });
            frag.insertBefore(row, frag.firstChild);
        }
        tbody.insertBefore(frag, tbody.firstChild);
    }

    function addResultRow(name, size, time, data, options = {}) {
        const tbody = document.getElementById('resultTableBody');
        if (!tbody) return null;
        
        const recordKind = options.recordKind || 'upload';
        const isMal = data?.status?.includes('Malicious') || data?.status?.includes('Malware');
        
        // 缓存记录
        cacheRecordedRow(recordKind, { name, size, time, data, options });
        
        const currentMode = AppState.tableMode;
        if (!options.forceRender && recordKind !== currentMode) return null;
        
        const rowId = `row_${Date.now()}_${Math.random()}`;
        const conf = Utils.formatConfidence(data?.confidence || data?.conf);
        const duration = Utils.formatDuration(data);
        
        // 缓存详情数据
        AppState.detailsData.set(rowId, {
            name, size, time, isMal, conf,
            rawData: data
        });
        
        const rowHtml = `
            <td class="px-6 py-4 font-mono text-white truncate max-w-[200px]" title="${name}">${name}</td>
            <td class="px-6 py-4 text-gray-400">${time}</td>
            <td class="px-6 py-4 font-mono text-cyan-400">${duration}</td>
            <td class="px-6 py-4 text-gray-400">${size || '-'}</td>
            <td class="px-6 py-4">${isMal ? '<span class="text-cyber-danger">Malware</span>' : '<span class="text-cyber-success">Benign</span>'}</td>
            <td class="px-6 py-4 text-gray-300">${data?.model || '-'}</td>
            <td class="px-6 py-4 font-mono text-cyber-primary">${conf}</td>
            <td class="px-6 py-4"><button data-row-id="${rowId}" class="view-details-btn text-gray-400 hover:text-white underline">详情</button></td>
        `;

        // If this is live mode and not a forced immediate render, buffer the DOM insertion
        if (recordKind === 'live' && !options.forceRender) {
            // keep buffer bounded
            if (liveRowBuffer.length > 1000) liveRowBuffer.shift();
            liveRowBuffer.push({ rowHtml, rowId, isMal, name, time, size, data });
            scheduleLiveFlush();
            return rowId;
        }

        // Non-live or forced render: immediate insertion
        const row = document.createElement('tr');
        row.className = 'border-b border-cyber-border/50';
        row.dataset.recordType = isMal ? 'malware' : 'benign';
        row.innerHTML = rowHtml;
        tbody.insertBefore(row, tbody.firstChild);
        row.querySelector('.view-details-btn')?.addEventListener('click', () => { window.openDetails?.(rowId); });
        return rowId;
    }
    
    function cacheRecordedRow(kind, entry) {
        if (!AppState.records[kind]) AppState.records[kind] = [];
        AppState.records[kind].push(entry);
        // 限制缓存数量
        if (AppState.records[kind].length > 500) {
            AppState.records[kind].shift();
        }
    }
    
    function renderRecordedRows(kind) {
        const tbody = document.getElementById('resultTableBody');
        if (!tbody) return;
        
        const list = AppState.records[kind] || [];
        AppState.tableMode = kind;
        tbody.innerHTML = '';
        
        if (!list.length) {
            const emptyText = kind === 'live' ? '暂无实时抓包记录' : '暂无上传检测记录';
            tbody.innerHTML = `<tr><td class="px-6 py-4 text-gray-500 text-center" colspan="8">${emptyText}</td></tr>`;
            return;
        }
        
        // 倒序显示
        for (let i = list.length - 1; i >= 0; i--) {
            const item = list[i];
            addResultRow(item.name, item.size, item.time, item.data, {
                ...item.options,
                forceRender: true,
                skipCache: true
            });
        }
        
        applyRecordFilter(AppState.currentFilter);
    }
    
    function applyRecordFilter(filterType) {
        AppState.currentFilter = filterType || 'all';
        
        const rows = document.querySelectorAll('#resultTableBody tr');
        rows.forEach(row => {
            const type = row.dataset.recordType;
            let show = filterType === 'all';
            if (!show) show = (filterType === 'benign' && type === 'benign');
            if (!show) show = (filterType === 'malware' && type === 'malware');
            row.style.display = show ? '' : 'none';
        });
        
        // 更新按钮样式
        ['recordFilterAll', 'recordFilterBenign', 'recordFilterMalware'].forEach(id => {
            const btn = document.getElementById(id);
            if (!btn) return;
            const isActive = (id === 'recordFilterAll' && filterType === 'all') ||
                            (id === 'recordFilterBenign' && filterType === 'benign') ||
                            (id === 'recordFilterMalware' && filterType === 'malware');
            btn.classList.toggle('bg-cyber-primary', isActive);
            btn.classList.toggle('text-black', isActive);
            btn.classList.toggle('text-gray-300', !isActive);
        });
    }
    
    function restoreFromCache() {
        // 从localStorage恢复
        try {
            const saved = localStorage.getItem('detection_records');
            if (saved) {
                const data = JSON.parse(saved);
                if (data.upload) AppState.records.upload = data.upload;
                if (data.live) AppState.records.live = data.live;
            }
        } catch (e) {}
    }
    
    return {
        addResultRow,
        cacheRecordedRow,
        renderRecordedRows,
        applyRecordFilter,
        restoreFromCache
    };
})(window.Utils, window.AppState);