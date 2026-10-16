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

    // 注意：页面上有 <div id="pieChart"> 与 <div id="lineChart">，浏览器会把 window.pieChart /
    // window.lineChart 自动解析成 DOM 元素（HTML 命名元素访问）。所以取实例前必须验证「它是不是
    // ECharts 实例」，否则 DOM 会被当成图表用：init 被跳过、setOption 不存在 → 整块卡片空白。
    function isChartInstance(c) {
        return !!c && typeof c.setOption === 'function' && typeof c.getOption === 'function';
    }

    function instanceFromDom(id) {
        try {
            const el = document.getElementById(id);
            if (!el || !window.echarts || typeof window.echarts.getInstanceByDom !== 'function') return null;
            const inst = window.echarts.getInstanceByDom(el);
            return isChartInstance(inst) ? inst : null;
        } catch (e) { return null; }
    }

    function getLineChart() {
        if (isChartInstance(window.uploadTrendChart)) return window.uploadTrendChart;
        if (isChartInstance(window.lineChart)) return window.lineChart;
        return instanceFromDom('lineChart');
    }

    function getPieChart() {
        if (isChartInstance(window.uploadPieChart)) return window.uploadPieChart;
        if (isChartInstance(window.pieChart)) return window.pieChart;
        return instanceFromDom('pieChart');
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
        // upload_trend_impl 是本项目里唯一被加载、被调用的上传统计维护者（charts_impl.js 没有进 main.js
        // 的模块清单），所以上传这一路的口径在这里写入统一统计层。
        if (window.dlidsStats) {
            window.dlidsStats.upload = {
                total: Math.max(0, toSafeNumber(state.totalScans, 0)),
                malware: Math.max(0, toSafeNumber(state.malwareCount, 0))
            };
        }
        syncGlobals(state);
        const lineChart = getLineChart();
        if (lineChart) {
            lineChart.setOption({
                // 没有数据时给个明确空态，免得只看到一片空的坐标区（饼图那边是「暂无数据」）
                title: { show: state.labels.length === 0, text: '暂无数据', left: 'center', top: 'middle', textStyle: { color: '#64748B', fontSize: 12, fontWeight: 'normal' } },
                xAxis: { data: state.labels },
                series: [{ data: state.benign }, { data: state.malware }]
            });
        }
        // KPI 与威胁分布饼图统一由 dlidsRenderStats 输出，保证与实时页同一口径、切页面不变
        if (window.dlidsRenderStats) window.dlidsRenderStats();
    }

    window.uploadTrendImpl.initCharts = function() {
        try {
            // 拆成 5 个页面后，趋势图和威胁分布饼图不再总是同时存在（例如「实时抓包」页只有饼图）。
            // 这里必须逐个判断容器是否存在：echarts.init(null) 会抛错，一旦抛在折线图那一步，
            // 后面的饼图初始化就被跳过 —— 表现就是实时页的威胁分布永远空白。
            const lineEl = document.getElementById('lineChart');
            const pieEl = document.getElementById('pieChart');

            if (!getLineChart() && lineEl && typeof echarts !== 'undefined') {
                window.uploadTrendChart = echarts.init(lineEl, window.DLIDS_CHART_THEME || null);
                // 显式写回：<div id="lineChart"> 会让 window.lineChart 被浏览器解析成 DOM 元素，
                // 这里用真正的实例覆盖它，免得别处再把 DOM 当图表用。
                window.lineChart = window.uploadTrendChart;
            }
            if (getLineChart()) {
                getLineChart().setOption({
                    backgroundColor: 'transparent',
                    tooltip: { trigger: 'axis' },
                    grid: { top: '15%', left: '5%', right: '5%', bottom: '10%', containLabel: true },
                    xAxis: { type: 'category', boundaryGap: false, data: [], axisLine: { lineStyle: { color: '#222C3A' } }, axisLabel: { color: '#64748B' } },
                    yAxis: { type: 'value', splitLine: { lineStyle: { color: '#222C3A', type: 'dashed' } }, axisLabel: { color: '#64748B' } },
                    series: [
                        { name: '正常流量', type: 'line', smooth: true, itemStyle: { color: '#10B981' }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(16, 185, 129, 0.3)' },{ offset: 1, color: 'rgba(16, 185, 129, 0)' }]) }, data: [] },
                        { name: '恶意流量', type: 'line', smooth: true, itemStyle: { color: '#F43F5E' }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(244, 63, 94, 0.3)' },{ offset: 1, color: 'rgba(244, 63, 94, 0)' }]) }, data: [] }
                    ]
                }, true);
            }

            if (!getPieChart() && pieEl && typeof echarts !== 'undefined') {
                window.uploadPieChart = echarts.init(pieEl, window.DLIDS_CHART_THEME || null);
                window.pieChart = window.uploadPieChart;
            }
            // 无论实例是谁 init 的，这里都先给它一次完整的 pie 定义（含 type:'pie'）：
            // dlidsRenderStats 之后用 merge 只更新数据，若初始 series 缺失，merge 会建出默认的 line series。
            if (getPieChart()) {
                getPieChart().setOption({
                    backgroundColor: 'transparent',
                    tooltip: { trigger: 'item' },
                    title: { show: true, text: '暂无数据', left: 'center', top: 'middle', textStyle: { color: '#64748B', fontSize: 12, fontWeight: 'normal' } },
                    legend: { show: false, bottom: '0%', textStyle: { color: '#94A3B8' }, icon: 'circle' },
                    series: [{ name: '威胁类型', type: 'pie', radius: ['40%', '70%'], avoidLabelOverlap: false, stillShowZeroSum: false, emptyCircleStyle: { color: 'transparent', borderColor: 'transparent' }, itemStyle: { borderRadius: 10, borderColor: '#151C28', borderWidth: 2 }, label: { show: false }, emphasis: { label: { show: true, fontSize: '18', fontWeight: 'bold', color: '#fff' } }, labelLine: { show: false }, data: [] }]
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
            if (!window.modalBarChart) window.modalBarChart = echarts.init(chartDom, window.DLIDS_CHART_THEME || null);
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