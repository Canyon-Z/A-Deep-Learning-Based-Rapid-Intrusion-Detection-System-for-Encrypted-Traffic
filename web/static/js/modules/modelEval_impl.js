/* model evaluation charts (confusion matrix, ROC) implementation */
(function(){
    window.modelEvalImpl = window.modelEvalImpl || {};

    window.modelEvalImpl.initModelEvalCharts = function() {
        try {
            function initCM(domId, data) {
                const chart = echarts.init(document.getElementById(domId));
                const option = {
                    grid: { top: 10, bottom: 25, left: 30, right: 10 },
                    tooltip: { show: true, formatter: 'Values: {c}' },
                    xAxis: { type: 'category', data: ['Neg', 'Pos'], axisLabel: { fontSize: 9, color: '#94A3B8' }, axisLine: { show: false }, axisTick: { show: false } },
                    yAxis: { type: 'category', data: ['Neg', 'Pos'], axisLabel: { fontSize: 9, color: '#94A3B8' }, axisLine: { show: false }, axisTick: { show: false } },
                    visualMap: { show: false, min: 0, max: 100, inRange: { color: ['#151B2D', '#00E5FF'] } },
                    series: [{ type: 'heatmap', data: data, label: { show: true, color: '#fff', fontSize: 10 }, itemStyle: { borderWidth: 1, borderColor: '#2A3447' } }]
                };
                chart.setOption(option);
                return chart;
            }

            function initROC(domId, rocData) {
                const chart = echarts.init(document.getElementById(domId));
                const option = {
                    grid: { top: 10, bottom: 20, left: 25, right: 10 },
                    tooltip: { show: true, trigger: 'axis', formatter: 'FPR: {b}<br/>TPR: {c}' },
                    xAxis: { type: 'category', boundaryGap: false, data: ['0', '0.2', '0.4', '0.6', '0.8', '1'], axisLabel: { fontSize: 9, color: '#94A3B8' }, splitLine: { show: false } },
                    yAxis: { type: 'value', min: 0, max: 1, splitNumber: 3, axisLabel: { fontSize: 9, color: '#94A3B8' }, splitLine: { show: true, lineStyle: { color: '#2A3447', type: 'dashed' } } },
                    series: [ { name: 'ROC', type: 'line', data: rocData, smooth: true, symbolSize: 0, lineStyle: { color: '#00E5FF', width: 2 }, areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,[{offset: 0, color: 'rgba(0, 229, 255, 0.3)'},{offset: 1, color: 'rgba(0, 229, 255, 0)'}]) } }, { name: 'Random', type: 'line', data: [0, 0.2, 0.4, 0.6, 0.8, 1], symbolSize: 0, lineStyle: { color: '#64748B', type: 'dashed', width: 1 } } ]
                };
                chart.setOption(option);
                return chart;
            }

            const cm1 = initCM('chart-cm-m1', [[0,0,98], [0,1,2], [1,0,3], [1,1,97]]);
            const roc1 = initROC('chart-roc-m1', [0, 0.88, 0.95, 0.98, 0.99, 1.0]);

            const cm2 = initCM('chart-cm-m2', [[0,0,94], [0,1,6], [1,0,8], [1,1,92]]);
            const roc2 = initROC('chart-roc-m2', [0, 0.70, 0.85, 0.92, 0.96, 1.0]);

            const cm3 = initCM('chart-cm-m3', [[0,0,96], [0,1,4], [1,0,5], [1,1,95]]);
            const roc3 = initROC('chart-roc-m3', [0, 0.80, 0.90, 0.95, 0.97, 1.0]);

            const cm4 = initCM('chart-cm-m4', [[0,0,93], [0,1,7], [1,0,8], [1,1,92]]);
            const roc4 = initROC('chart-roc-m4', [0, 0.75, 0.88, 0.93, 0.96, 1.0]);

            const cm5 = initCM('chart-cm-m5', [[0,0,97], [0,1,3], [1,0,4], [1,1,96]]);
            const roc5 = initROC('chart-roc-m5', [0, 0.85, 0.93, 0.97, 0.98, 1.0]);

            const cm6 = initCM('chart-cm-m6', [[0,0,91], [0,1,9], [1,0,10], [1,1,90]]);
            const roc6 = initROC('chart-roc-m6', [0, 0.72, 0.84, 0.90, 0.95, 1.0]);

            window.addEventListener('resize', () => {
                cm1.resize(); roc1.resize();
                cm2.resize(); roc2.resize();
                cm3.resize(); roc3.resize();
                cm4.resize(); roc4.resize();
                cm5.resize(); roc5.resize();
                cm6.resize(); roc6.resize();
            });
        } catch (e) { console.warn('modelEvalImpl.initModelEvalCharts failed', e); }
    };

})();
