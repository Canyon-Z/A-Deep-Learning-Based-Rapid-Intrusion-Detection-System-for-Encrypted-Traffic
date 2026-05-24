/* live page trend implementation */
(function(){
    window.liveTrendImpl = window.liveTrendImpl || {};

    function getState() {
        window.liveTrendImpl.state = window.liveTrendImpl.state || {
            labels: [],
            benign: [],
            malware: [],
        };
        return window.liveTrendImpl.state;
    }

    function getLineChart() {
        return window.liveTrendChart || null;
    }

    function ensureLineChart() {
        if (getLineChart()) return getLineChart();
        if (typeof echarts === 'undefined') return null;
        const dom = document.getElementById('lineChart');
        if (!dom) return null;
        window.liveTrendChart = window.uploadTrendChart || echarts.init(dom);
        return getLineChart();
    }

    function renderCharts() {
        const state = getState();
        const lineChart = getLineChart();
        if (lineChart) {
            lineChart.setOption({ xAxis: { data: state.labels }, series: [{ data: state.benign }, { data: state.malware }] });
        }
    }

    // Throttle rendering to avoid heavy synchronous reflows when many packets arrive.
    // Coalesce updates into a single render every ~1.5s so the UI refreshes at a human-friendly rate.
    let _renderScheduled = false;
    function scheduleRender() {
        if (_renderScheduled) return;
        _renderScheduled = true;
        // coalesce updates within 1500ms window (~1.5s)
        setTimeout(() => {
            try { renderCharts(); } catch (e) { console.warn('liveTrendImpl.scheduled render failed', e); }
            _renderScheduled = false;
        }, 1500);
    }

    window.liveTrendImpl.initCharts = function() {
        try {
            ensureLineChart();
            renderCharts();
        } catch (e) { console.warn('liveTrendImpl.initCharts failed', e); }
    };

    window.liveTrendImpl.activate = function() {
        try {
            ensureLineChart();
            renderCharts();
            try { const lineChart = getLineChart(); if (lineChart && typeof lineChart.resize === 'function') lineChart.resize(); } catch (e) {}
        } catch (e) { console.warn('liveTrendImpl.activate failed', e); }
    };

    window.liveTrendImpl.resetTrendChart = function() {
        try {
            const state = getState();
            state.labels = [];
            state.benign = [];
            state.malware = [];
            renderCharts();
        } catch (e) { console.warn('liveTrendImpl.resetTrendChart failed', e); }
    };

    window.liveTrendImpl.updateTrendChart = function(isMalware) {
        try {
            ensureLineChart();
            const state = getState();
            const label = new Date().toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit' });
            if (state.labels.length > 0 && state.labels[state.labels.length - 1] === label) {
                const lastIndex = state.labels.length - 1;
                state.benign[lastIndex] = (state.benign[lastIndex] || 0) + (isMalware ? 0 : 1);
                state.malware[lastIndex] = (state.malware[lastIndex] || 0) + (isMalware ? 1 : 0);
            } else {
                state.labels.push(label);
                state.benign.push(isMalware ? 0 : 1);
                state.malware.push(isMalware ? 1 : 0);
            }
            // coalesce frequent updates
            scheduleRender();
        } catch (e) { console.warn('liveTrendImpl.updateTrendChart failed', e); }
    };

    window.liveTrendImpl.restoreFromRecords = function(records) {
        try {
            const list = Array.isArray(records) ? records : [];
            const liveRecords = list.filter((item) => item && typeof item === 'object');
            const grouped = new Map();

            liveRecords.forEach((item, index) => {
                const data = item.data && typeof item.data === 'object' ? item.data : {};
                const isMalware = !!(data.is_malicious || String(data.status || '').toLowerCase().includes('malicious') || String(data.predicted_label || '').toLowerCase().includes('malware'));
                const rawTime = item.time || data.captured_at || data.captureTime || data.capture_time || data.timestamp || `#${index + 1}`;
                const label = new Date(rawTime).toString() === 'Invalid Date'
                    ? String(rawTime)
                    : new Date(rawTime).toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit' });
                if (!grouped.has(label)) grouped.set(label, { benign: 0, malware: 0 });
                const bucket = grouped.get(label);
                if (isMalware) bucket.malware += 1; else bucket.benign += 1;
            });

            const state = getState();
            state.labels = [];
            state.benign = [];
            state.malware = [];

            for (const [label, bucket] of grouped.entries()) {
                state.labels.push(label);
                state.benign.push(bucket.benign);
                state.malware.push(bucket.malware);
            }

            renderCharts();
        } catch (e) { console.warn('liveTrendImpl.restoreFromRecords failed', e); }
    };
})();