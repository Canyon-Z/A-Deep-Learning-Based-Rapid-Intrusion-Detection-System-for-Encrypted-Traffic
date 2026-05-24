/* implementations for charts and trend/pie state */
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
            window.lineChart = echarts.init(document.getElementById('lineChart'));
            const lineOption = {
                backgroundColor: 'transparent',
                tooltip: { trigger: 'axis' },
                grid: { top: '15%', left: '5%', right: '5%', bottom: '10%', containLabel: true },
                xAxis: { type: 'category', boundaryGap: false, data: window.trendLabels, axisLine: { lineStyle: { color: '#2A3447' } }, axisLabel: { color: '#64748B' } },
                yAxis: { type: 'value', splitLine: { lineStyle: { color: '#2A3447', type: 'dashed' } }, axisLabel: { color: '#64748B' } },
                series: [
                    { name: '正常流量', type: 'line', smooth: true, itemStyle: { color: '#00D084' }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(0, 208, 132, 0.3)' },{ offset: 1, color: 'rgba(0, 208, 132, 0)' }]) }, data: window.benignTrend },
                    { name: '恶意流量', type: 'line', smooth: true, itemStyle: { color: '#FF2E63' }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(255, 46, 99, 0.3)' },{ offset: 1, color: 'rgba(255, 46, 99, 0)' }]) }, data: window.malwareTrend }
                ]
            };
            window.lineChart.setOption(lineOption);

            window.pieChart = echarts.init(document.getElementById('pieChart'));
            const pieOption = {
                backgroundColor: 'transparent',
                tooltip: { trigger: 'item' },
                legend: { bottom: '0%', textStyle: { color: '#94A3B8' }, icon: 'circle' },
                series: [{ name: '威胁类型', type: 'pie', radius: ['40%', '70%'], avoidLabelOverlap: false, itemStyle: { borderRadius: 10, borderColor: '#151B2D', borderWidth: 2 }, label: { show: false, position: 'center' }, emphasis: { label: { show: true, fontSize: '18', fontWeight: 'bold', color: '#fff' } }, labelLine: { show: false }, data: [ { value: 0, name: 'Normal', itemStyle: { color: '#00D084' } }, { value: 0, name: 'Malware', itemStyle: { color: '#FF2E63' } } ] }]
            };
            window.pieChart.setOption(pieOption);
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

            const normalCount = Math.max(0, window.totalScans - window.malwareCount);
            const statTotal = document.getElementById('statTotal');
            const statMal = document.getElementById('statMalware');
            if (statTotal) statTotal.innerText = String(window.totalScans);
            if (statMal) statMal.innerText = String(window.malwareCount);

            if (window.lineChart) {
                window.lineChart.setOption({ xAxis: { data: window.trendLabels }, series: [{ data: window.benignTrend }, { data: window.malwareTrend }] });
            }
            if (window.pieChart) {
                window.pieChart.setOption({
                    series: [{
                        data: [
                            { value: normalCount, name: 'Normal', itemStyle: { color: '#00D084' } },
                            { value: window.malwareCount, name: 'Malware', itemStyle: { color: '#FF2E63' } }
                        ]
                    }]
                });
            }
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
            window.totalScans = toSafeNumber(window.totalScans, 0) + 1;
            window.malwareCount = toSafeNumber(window.malwareCount, 0);
            const isMal = (data && (String(data.status || '').includes('Malicious') || String(data.status || '').includes('Malware')));
            if (isMal) window.malwareCount = (typeof window.malwareCount === 'number' ? window.malwareCount : 0) + 1;
            const normalCount = Math.max(0, toSafeNumber(window.totalScans, 0) - toSafeNumber(window.malwareCount, 0));
            const malwareCount = Math.max(0, toSafeNumber(window.malwareCount, 0));
            const statTotal = document.getElementById('statTotal'); if (statTotal) statTotal.innerText = window.totalScans;
            const statMal = document.getElementById('statMalware'); if (statMal) statMal.innerText = window.malwareCount;
            if (window.pieChart) {
                window.pieChart.setOption({
                    series: [{
                        data: [
                            { value: normalCount, name: 'Normal', itemStyle: { color: '#00D084' } },
                            { value: malwareCount, name: 'Malware', itemStyle: { color: '#FF2E63' } }
                        ]
                    }]
                });
            }
            window.chartsImpl.updateTrendChart(isMal);
        } catch (e) { console.warn('chartsImpl.updateStats', e); }
    };

    window.chartsImpl.initModalChart = function() {
        try {
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
        } catch (e) { console.warn('chartsImpl.initModalChart failed', e); }
    };

})();
