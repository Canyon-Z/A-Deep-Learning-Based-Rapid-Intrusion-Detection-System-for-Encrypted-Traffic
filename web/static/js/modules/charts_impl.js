/* implementations for charts and trend/pie state
 *
 * ⚠️ 这个文件目前【没有】被加载：main.js 的模块清单里没有它，全仓也没有任何地方调用 window.chartsImpl.*。
 * 真正在跑的上传统计 / 趋势曲线 / 威胁分布饼图实现是 web/static/js/modules/upload_trend_impl.js。
 * 保留它是为了不动历史产物，但改统计口径时请改 upload_trend_impl.js。
 *
 * 顺带记一个坑：本文件开头把 window.lineChart / window.pieChart 显式设为 null。未加载时这两个名字会被
 * 浏览器按 HTML 命名元素访问解析成 <div id="lineChart"> / <div id="pieChart"> 元素本身，
 * 于是 `window.pieChart.setOption(...)` 会以「不是函数」失败——取实例前必须校验 setOption/getOption。
 */
(function(){
    window.chartsImpl = window.chartsImpl || {};

    function toSafeNumber(value, fallback) {
        const num = Number(value);
        return Number.isFinite(num) ? num : fallback;
    }

    window.lineChart = null;
    window.pieChart = null;
    window.trendLabels = window.trendLabels || [];
    window.benignTrend = window.benignTrend || [];
    window.malwareTrend = window.malwareTrend || [];
    window.MAX_TREND_POINTS = window.MAX_TREND_POINTS || 20;

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

    window.chartsImpl.initCharts = function() {
        try {
            window.lineChart = echarts.init(document.getElementById('lineChart'), window.DLIDS_CHART_THEME || null);
            const lineOption = {
                backgroundColor: 'transparent',
                tooltip: { trigger: 'axis' },
                grid: { top: '15%', left: '5%', right: '5%', bottom: '10%', containLabel: true },
                xAxis: { type: 'category', boundaryGap: false, data: window.trendLabels, axisLine: { lineStyle: { color: '#222C3A' } }, axisLabel: { color: '#64748B' } },
                yAxis: { type: 'value', splitLine: { lineStyle: { color: '#222C3A', type: 'dashed' } }, axisLabel: { color: '#64748B' } },
                series: [
                    { name: '正常流量', type: 'line', smooth: true, itemStyle: { color: '#10B981' }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(16, 185, 129, 0.3)' },{ offset: 1, color: 'rgba(16, 185, 129, 0)' }]) }, data: window.benignTrend },
                    { name: '恶意流量', type: 'line', smooth: true, itemStyle: { color: '#F43F5E' }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(244, 63, 94, 0.3)' },{ offset: 1, color: 'rgba(244, 63, 94, 0)' }]) }, data: window.malwareTrend }
                ]
            };
            window.lineChart.setOption(lineOption);

            window.pieChart = echarts.init(document.getElementById('pieChart'), window.DLIDS_CHART_THEME || null);
            const pieOption = {
                backgroundColor: 'transparent',
                tooltip: { trigger: 'item' },
                title: { show: true, text: '暂无数据', left: 'center', top: 'middle', textStyle: { color: '#64748B', fontSize: 12, fontWeight: 'normal' } },
                legend: { show: false, bottom: '0%', textStyle: { color: '#94A3B8' }, icon: 'circle' },
                series: [{ name: '威胁类型', type: 'pie', radius: ['40%', '70%'], avoidLabelOverlap: false, stillShowZeroSum: false, emptyCircleStyle: { color: 'transparent', borderColor: 'transparent' }, itemStyle: { borderRadius: 10, borderColor: '#151C28', borderWidth: 2 }, label: { show: false }, emphasis: { label: { show: true, fontSize: '18', fontWeight: 'bold', color: '#fff' } }, labelLine: { show: false }, data: [] }]
            };
            window.pieChart.setOption(pieOption);
            // 图表实例就绪后按统一口径渲染一次（记录恢复可能早于 initCharts）
            if (window.dlidsRenderStats) window.dlidsRenderStats();
        } catch (e) { console.warn('chartsImpl.initCharts failed', e); }
    };

    window.chartsImpl.resetTrendChart = function() {
        try {
            window.trendLabels = [];
            window.benignTrend = [];
            window.malwareTrend = [];
            if (window.lineChart) {
                window.lineChart.setOption({
                    xAxis: { data: [] },
                    series: [{ data: [] }, { data: [] }]
                });
            }
        } catch (e) { console.warn('chartsImpl.resetTrendChart', e); }
    };

    window.chartsImpl.restoreChartsFromRecords = function(records) {
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

            const reconstructedLabels = [];
            const reconstructedBenign = [];
            const reconstructedMalware = [];
            let malwareCount = 0;
            for (const [label, bucket] of grouped.entries()) {
                reconstructedLabels.push(label);
                reconstructedBenign.push(bucket.benign);
                reconstructedMalware.push(bucket.malware);
                malwareCount += bucket.malware;
            }

            window.trendLabels = reconstructedLabels.slice(-window.MAX_TREND_POINTS);
            window.benignTrend = reconstructedBenign.slice(-window.MAX_TREND_POINTS);
            window.malwareTrend = reconstructedMalware.slice(-window.MAX_TREND_POINTS);
            window.totalScans = uploadRecords.length;
            window.malwareCount = malwareCount;

            // 恢复时重建上传这一路的口径；KPI 与饼图统一由 dlidsRenderStats 输出
            if (window.dlidsStats) window.dlidsStats.upload = { total: uploadRecords.length, malware: malwareCount };

            if (window.lineChart) {
                window.lineChart.setOption({ xAxis: { data: window.trendLabels }, series: [{ data: window.benignTrend }, { data: window.malwareTrend }] });
            }
            if (window.dlidsRenderStats) window.dlidsRenderStats();
        } catch (e) { console.warn('chartsImpl.restoreChartsFromRecords failed', e); }
    };

    window.chartsImpl.updateTrendChart = function(isMalware) {
        try {
            const now = new Date();
            const label = now.toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit' });
            if (window.trendLabels.length > 0 && window.trendLabels[window.trendLabels.length - 1] === label) {
                const lastIndex = window.trendLabels.length - 1;
                window.benignTrend[lastIndex] = (window.benignTrend[lastIndex] || 0) + (isMalware ? 0 : 1);
                window.malwareTrend[lastIndex] = (window.malwareTrend[lastIndex] || 0) + (isMalware ? 1 : 0);
            } else {
                window.trendLabels.push(label);
                window.benignTrend.push(isMalware ? 0 : 1);
                window.malwareTrend.push(isMalware ? 1 : 0);
            }
            if (window.lineChart) {
                window.lineChart.setOption({ xAxis: { data: window.trendLabels }, series: [{ data: window.benignTrend }, { data: window.malwareTrend }] });
            }
        } catch (e) { console.warn('chartsImpl.updateTrendChart', e); }
    };

    window.chartsImpl.updateStats = function(data) {
        try {
            const isMal = (data && (String(data.status || '').includes('Malicious') || String(data.status || '').includes('Malware')));
            // 只累加「上传这一路」的口径，展示统一交给 dlidsRenderStats
            if (window.dlidsStats) {
                const up = window.dlidsStats.upload = window.dlidsStats.upload || { total: 0, malware: 0 };
                up.total = (Number(up.total) || 0) + 1;
                if (isMal) up.malware = (Number(up.malware) || 0) + 1;
            }
            if (window.dlidsRenderStats) window.dlidsRenderStats();
            window.chartsImpl.updateTrendChart(isMal);
        } catch (e) { console.warn('chartsImpl.updateStats', e); }
    };

    window.chartsImpl.initModalChart = function() {
        try {
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
        } catch (e) { console.warn('chartsImpl.initModalChart failed', e); }
    };

})();
