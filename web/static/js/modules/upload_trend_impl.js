/* upload page trend/pie implementation */
(function(){
    window.uploadTrendImpl = window.uploadTrendImpl || {};

    function toSafeNumber(value, fallback) {
        const num = Number(value);
        return Number.isFinite(num) ? num : fallback;
    }

    function normalizeMinuteLabel(value) {
        if (value === null || value === undefined || value === '') return '';
        if (typeof value === 'string') {
            const text = value.trim();
            const hhmmss = text.match(/^(\d{1,2}:\d{2}):\d{2}$/);
            if (hhmmss) return hhmmss[1];
            const hhmm = text.match(/^(\d{1,2}:\d{2})$/);
            if (hhmm) return hhmm[1];
            const parsed = new Date(text);
            if (!Number.isNaN(parsed.getTime())) {
                return parsed.toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit' });
            }
            return text;
        }
        const parsed = new Date(value);
        return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit' });
    }

    function getState() {
        window.uploadTrendImpl.state = window.uploadTrendImpl.state || {
            labels: [],
            benign: [],
            malware: [],
            totalScans: 0,
            malwareCount: 0,
        };
        return window.uploadTrendImpl.state;
    }

    function getLineChart() {
        return window.uploadTrendChart || null;
    }

    function getPieChart() {
        return window.uploadPieChart || null;
    }

    function syncGlobals(state) {
        window.trendLabels = state.labels;
        window.benignTrend = state.benign;
        window.malwareTrend = state.malware;
        window.totalScans = state.totalScans;
        window.malwareCount = state.malwareCount;
    }

    function renderCharts() {
        const state = getState();
        syncGlobals(state);
        const lineChart = getLineChart();
        if (lineChart) {
            lineChart.setOption({ xAxis: { data: state.labels }, series: [{ data: state.benign }, { data: state.malware }] });
        }
        const pieChart = getPieChart();
        if (pieChart) {
            const normalCount = Math.max(0, toSafeNumber(state.totalScans, 0) - toSafeNumber(state.malwareCount, 0));
            pieChart.setOption({
                series: [{
                    data: [
                        { value: normalCount, name: 'Normal', itemStyle: { color: '#00D084' } },
                        { value: Math.max(0, toSafeNumber(state.malwareCount, 0)), name: 'Malware', itemStyle: { color: '#FF2E63' } }
                    ]
                }]
            });
        }
        const statTotal = document.getElementById('statTotal');
        const statMal = document.getElementById('statMalware');
        if (statTotal) statTotal.innerText = String(state.totalScans);
        if (statMal) statMal.innerText = String(state.malwareCount);
    }

    window.uploadTrendImpl.initCharts = function() {
        try {
            if (!getLineChart() && typeof echarts !== 'undefined') {
                window.uploadTrendChart = echarts.init(document.getElementById('lineChart'));
            }
            if (getLineChart()) {
                getLineChart().setOption({
                    backgroundColor: 'transparent',
                    tooltip: { trigger: 'axis' },
                    grid: { top: '15%', left: '5%', right: '5%', bottom: '10%', containLabel: true },
                    xAxis: { type: 'category', boundaryGap: false, data: [], axisLine: { lineStyle: { color: '#2A3447' } }, axisLabel: { color: '#64748B' } },
                    yAxis: { type: 'value', splitLine: { lineStyle: { color: '#2A3447', type: 'dashed' } }, axisLabel: { color: '#64748B' } },
                    series: [
                        { name: '正常流量', type: 'line', smooth: true, itemStyle: { color: '#00D084' }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(0, 208, 132, 0.3)' },{ offset: 1, color: 'rgba(0, 208, 132, 0)' }]) }, data: [] },
                        { name: '恶意流量', type: 'line', smooth: true, itemStyle: { color: '#FF2E63' }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(255, 46, 99, 0.3)' },{ offset: 1, color: 'rgba(255, 46, 99, 0)' }]) }, data: [] }
                    ]
                }, true);
            }

            if (!getPieChart() && typeof echarts !== 'undefined') {
                window.uploadPieChart = echarts.init(document.getElementById('pieChart'));
            }
            if (getPieChart()) {
                getPieChart().setOption({
                    backgroundColor: 'transparent',
                    tooltip: { trigger: 'item' },
                    legend: { bottom: '0%', textStyle: { color: '#94A3B8' }, icon: 'circle' },
                    series: [{ name: '威胁类型', type: 'pie', radius: ['40%', '70%'], avoidLabelOverlap: false, itemStyle: { borderRadius: 10, borderColor: '#151B2D', borderWidth: 2 }, label: { show: false, position: 'center' }, emphasis: { label: { show: true, fontSize: '18', fontWeight: 'bold', color: '#fff' } }, labelLine: { show: false }, data: [ { value: 0, name: 'Normal', itemStyle: { color: '#00D084' } }, { value: 0, name: 'Malware', itemStyle: { color: '#FF2E63' } } ] }]
                }, true);
            }
            renderCharts();
        } catch (e) { console.warn('uploadTrendImpl.initCharts failed', e); }
    };

    window.uploadTrendImpl.activate = function() {
        try {
            renderCharts();
            try { const lineChart = getLineChart(); if (lineChart && typeof lineChart.resize === 'function') lineChart.resize(); } catch (e) {}
            try { const pieChart = getPieChart(); if (pieChart && typeof pieChart.resize === 'function') pieChart.resize(); } catch (e) {}
        } catch (e) { console.warn('uploadTrendImpl.activate failed', e); }
    };

    window.uploadTrendImpl.resetTrendChart = function() {
        try {
            const state = getState();
            state.labels = [];
            state.benign = [];
            state.malware = [];
            state.totalScans = 0;
            state.malwareCount = 0;
            renderCharts();
        } catch (e) { console.warn('uploadTrendImpl.resetTrendChart failed', e); }
    };

    window.uploadTrendImpl.restoreFromRecords = function(records) {
        try {
            const list = Array.isArray(records) ? records : [];
            const uploadRecords = list.filter((item) => item && typeof item === 'object');
            const grouped = new Map();

            uploadRecords.forEach((item, index) => {
                const data = item.data && typeof item.data === 'object' ? item.data : {};
                const status = String(data.status || '').toLowerCase();
                const isMalware = !!(data.is_malicious || status.includes('malicious') || status.includes('malware'));
                const label = normalizeMinuteLabel(item.time || data.captureTime || data.capture_time || `#${index + 1}`);
                if (!grouped.has(label)) grouped.set(label, { benign: 0, malware: 0 });
                const bucket = grouped.get(label);
                if (isMalware) bucket.malware += 1; else bucket.benign += 1;
            });

            const state = getState();
            state.labels = [];
            state.benign = [];
            state.malware = [];
            state.totalScans = uploadRecords.length;
            state.malwareCount = 0;

            for (const [label, bucket] of grouped.entries()) {
                state.labels.push(label);
                state.benign.push(bucket.benign);
                state.malware.push(bucket.malware);
                state.malwareCount += bucket.malware;
            }

            renderCharts();
        } catch (e) { console.warn('uploadTrendImpl.restoreFromRecords failed', e); }
    };

    window.uploadTrendImpl.updateTrendChart = function(isMalware) {
        try {
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
            renderCharts();
        } catch (e) { console.warn('uploadTrendImpl.updateTrendChart failed', e); }
    };

    window.uploadTrendImpl.updateStats = function(data) {
        try {
            const state = getState();
            state.totalScans = toSafeNumber(state.totalScans, 0) + 1;
            const isMal = !!(data && (String(data.status || '').includes('Malicious') || String(data.status || '').includes('Malware')));
            if (isMal) state.malwareCount = toSafeNumber(state.malwareCount, 0) + 1;
            renderCharts();
            window.uploadTrendImpl.updateTrendChart(isMal);
        } catch (e) { console.warn('uploadTrendImpl.updateStats failed', e); }
    };

    window.uploadTrendImpl.initModalChart = function() {
        try {
            if (window.modalHelpers && typeof window.modalHelpers.initModalChart === 'function') {
                return window.modalHelpers.initModalChart.apply(window.modalHelpers, arguments);
            }
            const chartDom = document.getElementById('modalBarChart');
            if (!chartDom || typeof echarts === 'undefined') return;
            if (!window.modalBarChart) window.modalBarChart = echarts.init(chartDom);
            const labels = ['0x00','0x10','0x20','0x30','0x40','0x50','0x60','0x70','0x80','0x90','0xA0','0xB0','0xC0','0xD0','0xE0','0xF0'];
            let data = window.currentPayloadDist;
            if (!Array.isArray(data) || data.length !== 16) data = Array.from({ length: 16 }, () => 0);
            const option = {
                backgroundColor: 'transparent',
                tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
                grid: { top: '10%', left: '0%', right: '0%', bottom: '5%', containLabel: true },
                xAxis: { type: 'category', data: labels, axisLabel: { color: '#9ca3af', fontSize: 10 }, axisLine: { lineStyle: { color: '#374151' } } },
                yAxis: { type: 'value', axisLabel: { color: '#9ca3af', fontSize: 10 }, splitLine: { lineStyle: { color: '#1f2937' } } },
                series: [{ name: 'Bytes', type: 'bar', data: data, itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#22d3ee' }, { offset: 1, color: '#2563eb' }]) } }]
            };
            window.modalBarChart.setOption(option, true);
            try { window.modalBarChart.resize(); } catch(e){}
        } catch (e) { console.warn('uploadTrendImpl.initModalChart failed', e); }
    };
})();