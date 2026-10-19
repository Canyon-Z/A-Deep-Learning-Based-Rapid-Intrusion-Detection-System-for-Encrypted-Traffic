/* upload page trend/pie implementation */
(function(){
    window.uploadTrendImpl = window.uploadTrendImpl || {};

    // 折线两色只在这里定义一次，取值与 styles.css 的 --ds-chart-normal / --ds-chart-malware
    // 必须一致：图例是 HTML 画的、线是 canvas 画的，两处各写一份颜色就会对不上
    // （过去图例是青色方块、线是绿色）。ECharts 不认 CSS 变量，只能写字面值，
    // 所以这里保留一份镜像常量，改颜色时两个文件一起改。
    const CHART_NORMAL = '#10B981';
    const CHART_MALWARE = '#F43F5E';
    const AREA_NORMAL = ['rgba(16, 185, 129, 0.3)', 'rgba(16, 185, 129, 0)'];
    const AREA_MALWARE = ['rgba(244, 63, 94, 0.3)', 'rgba(244, 63, 94, 0)'];

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
            // true 表示当前曲线是「上传 + 实时」合并数据（总览页），不写 dlidsStats.upload
            suppressStats: false,
        };
        return window.uploadTrendImpl.state;
    }

    // 注意：页面上有 <div id="lineChart">，浏览器会把 window.lineChart 自动解析成 DOM 元素
    // （HTML 命名元素访问）。所以取实例前必须验证「它是不是 ECharts 实例」，否则 DOM 会被当成
    // 图表用：init 被跳过、setOption 不存在 → 整块卡片空白。
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
        // 但总览页画的是「上传 + 实时」合并曲线（见 restoreFromRecords 的 mergeStats 选项），
        // 那份合并条数不能写进 upload 计数：dlidsStatsSnapshot 已经把 upload.total 与
        // live.sessions 相加，再混入实时条数会重复计算，KPI 直接翻倍。
        if (window.dlidsStats && !state.suppressStats) {
            window.dlidsStats.upload = {
                total: Math.max(0, toSafeNumber(state.statsTotal, toSafeNumber(state.totalScans, 0))),
                malware: Math.max(0, toSafeNumber(state.statsMalware, toSafeNumber(state.malwareCount, 0)))
            };
        } else if (window.dlidsStats && state.suppressStats) {
            // 合并模式：曲线是两路之和，但 KPI 只认真实上传条数（实时那一路由
            // dlidsStats.live.sessions 提供，混进来会重复计数）。
            window.dlidsStats.upload = {
                total: Math.max(0, toSafeNumber(state.statsTotal, 0)),
                malware: Math.max(0, toSafeNumber(state.statsMalware, 0))
            };
        }
        syncGlobals(state);
        const lineChart = getLineChart();
        if (lineChart) {
            lineChart.setOption({
                // 没有数据时给个明确空态，免得只看到一片空的坐标区
                title: { show: state.labels.length === 0, text: '暂无数据', left: 'center', top: 'middle', textStyle: { color: '#64748B', fontSize: 12, fontWeight: 'normal' } },
                xAxis: { data: state.labels },
                series: [{ data: state.benign }, { data: state.malware }]
            });
        }
        // KPI 与威胁类型分布统一由 dlidsRenderStats 输出，保证与实时页同一口径、切页面不变
        if (window.dlidsRenderStats) window.dlidsRenderStats();
    }

    window.uploadTrendImpl.initCharts = function() {
        try {
            // 五个页面里趋势图不总是存在（「实时抓包」页只有威胁分布卡片）。
            // echarts.init(null) 会抛错，所以必须先判断容器在不在。
            const lineEl = document.getElementById('lineChart');

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
                        { name: '正常流量', type: 'line', smooth: true, itemStyle: { color: CHART_NORMAL }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: AREA_NORMAL[0] },{ offset: 1, color: AREA_NORMAL[1] }]) }, data: [] },
                        { name: '恶意流量', type: 'line', smooth: true, itemStyle: { color: CHART_MALWARE }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: AREA_MALWARE[0] },{ offset: 1, color: AREA_MALWARE[1] }]) }, data: [] }
                    ]
                }, true);
            }

            // 威胁类型分布改成六分类横向条形后不再走 ECharts（DOM 渲染，见 main.js 的
            // dlidsRenderThreatBars）：它要按「六类固定行位置」展示，饼图做不到这一点，
            // 也不需要 canvas 实例。所以这里只剩折线图需要 init。
            renderCharts();
        } catch (e) { console.warn('uploadTrendImpl.initCharts failed', e); }
    };

    window.uploadTrendImpl.activate = function() {
        try {
            renderCharts();
            try { const lineChart = getLineChart(); if (lineChart && typeof lineChart.resize === 'function') lineChart.resize(); } catch (e) {}
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
            // 清空后回到「只统计上传」的正常模式，否则清日志后 KPI 第 1 格再也不更新
            state.suppressStats = false;
            renderCharts();
        } catch (e) { console.warn('uploadTrendImpl.resetTrendChart failed', e); }
    };

    // options.mergeStats=true 表示传进来的是「上传 + 实时」合并记录（总览页用）：
    // 折线照画，但不把合并条数写回 dlidsStats.upload（理由见 renderCharts）。
    window.uploadTrendImpl.restoreFromRecords = function(records, options) {
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
            state.suppressStats = !!(options && options.mergeStats);
            // 合并模式下另外传一份「真实上传记录」用来算 KPI 计数，不传就退回当前这批
            const statsList = (options && Array.isArray(options.statsRecords)) ? options.statsRecords : uploadRecords;
            state.statsTotal = statsList.filter((item) => item && typeof item === 'object').length;
            state.statsMalware = statsList.filter((item) => {
                const data = item && item.data && typeof item.data === 'object' ? item.data : {};
                const status = String(data.status || '').toLowerCase();
                return !!(data.is_malicious || status.includes('malicious') || status.includes('malware'));
            }).length;
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
            // 威胁类型分布走六分类口径。/analyze 现在也返回 threat_summary
            // （pcap → 78 维流特征 → flow_multiclass），结构与 /analyze-flow 的
            // class_summary 一致，所以两条链路共用同一个吸收函数。
            // 模型未加载、或这份 pcap 一条流都没解析出来时该字段是 null：直接跳过，
            // 让卡片保留上一次的累计值，而不是把已有数据清空。
            if (data && data.threat_summary && typeof window.dlidsAbsorbThreatSummary === 'function') {
                window.dlidsAbsorbThreatSummary(data.threat_summary);
            }
            // 逐流明细：卡片上点开某个分类看的就是这批流，也是卡片计数的唯一来源。
            // 按双向五元组合并去重（不是按事件追加），和记录表同一口径；
            // 上限 1500 条是防内存——单次上传最多回 200 条（后端 THREAT_FLOW_ROWS_LIMIT）。
            // 必须在 threat_summary 判断之外执行：summary 缺失但明细还在时也要能显示。
            const rows = Array.isArray(data && data.threat_flows) ? data.threat_flows : [];
            if (rows.length && typeof window.dlidsMergeThreatFlows === 'function') {
                window.dlidsMergeThreatFlows(rows, { key: 'upload', max: 1500 });
            }
            if (data && data.threat_meta && typeof window.dlidsSetThreatMeta === 'function') {
                window.dlidsSetThreatMeta(data.threat_meta, { key: 'upload' });
            }
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