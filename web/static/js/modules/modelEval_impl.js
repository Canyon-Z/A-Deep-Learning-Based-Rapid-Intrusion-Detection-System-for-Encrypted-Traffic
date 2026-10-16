/* 模型效能评估图表：全部数据来自 /api/metrics（后端读 checkpoints/model_metrics.json）
   本文件不包含任何硬编码指标。 */
(function () {
    window.modelEvalImpl = window.modelEvalImpl || {};

    const MODEL_LIST = [
        { key: 'cnn_bilstm',      prefix: 'm1', json: 'CNN_BiLSTM',             display: 'CNN-BiLSTM' },
        { key: 'classic_cnn',     prefix: 'm2', json: 'Classic_CNN',            display: 'Classic-CNN' },
        { key: 'lightweight_cnn', prefix: 'm3', json: 'Lightweight_CNN_BiLSTM', display: 'Lightweight CNN-BiLSTM' },
        { key: 'pure_bilstm',     prefix: 'm4', json: 'Pure_BiLSTM',            display: 'Pure BiLSTM' },
        { key: 'mlp',             prefix: 'm5', json: 'MLP',                    display: 'MLP' },
        { key: 'transformer',     prefix: 'm6', json: 'Transformer',            display: 'Transformer' }
    ];

    const ACCENT = 'rgba(34, 211, 238, 0.75)';
    const ACCENT_HI = 'rgba(34, 211, 238, 0.95)';
    const MUTED = 'rgba(148, 163, 184, 0.35)';
    const TEXT_DIM = '#94A3B8';
    const GRID_LINE = '#222C3A';

    const instances = {};

    const num = (v) => {
        if (v === null || v === undefined || v === '') return null;
        const n = parseFloat(String(v).replace('%', '').trim());
        return Number.isFinite(n) ? n : null;
    };

    function axisBase() {
        return {
            axisLabel: { fontSize: 10, color: TEXT_DIM },
            axisLine: { show: false },
            axisTick: { show: false },
            splitLine: { lineStyle: { color: GRID_LINE } }
        };
    }

    function getChart(id) {
        const el = document.getElementById(id);
        if (!el) return null;
        if (!instances[id]) instances[id] = echarts.init(el, window.DLIDS_CHART_THEME || null, { renderer: 'canvas' });
        return instances[id];
    }

    function buildMetricsOption(node) {
        const rows = [
            ['准确率', num(node.test_accuracy != null ? node.test_accuracy : node.best_accuracy)],
            ['精确率', num(node.precision)],
            ['召回率', num(node.recall)],
            ['F1', num(node.f1_score)],
            ['F2', num(node.f2_score)],
            ['AUC×100', num(node.auc) != null ? num(node.auc) * 100 : null]
        ].filter((row) => row[1] !== null);

        const labels = rows.map((r) => r[0]).reverse();
        const values = rows.map((r) => r[1]).reverse();

        return {
            grid: { top: 8, bottom: 18, left: 64, right: 48 },
            tooltip: { trigger: 'axis', axisPointer: { type: 'none' } },
            xAxis: Object.assign({ type: 'value', min: 50, max: 100, splitNumber: 4 }, axisBase()),
            yAxis: Object.assign({ type: 'category', data: labels }, axisBase()),
            series: [{
                type: 'bar',
                barWidth: 10,
                data: values,
                itemStyle: { color: ACCENT, borderRadius: [0, 3, 3, 0] },
                label: {
                    show: true, position: 'right', color: TEXT_DIM, fontSize: 10,
                    formatter: (p) => Number(p.value).toFixed(2)
                }
            }]
        };
    }

    function buildCompareOption(model, data) {
        const rows = MODEL_LIST
            .map((m) => ({ key: m.key, name: m.display, v: num((data[m.json] || {}).auc) }))
            .filter((r) => r.v !== null);

        const labels = rows.map((r) => r.name).reverse();
        const values = rows.map((r) => ({
            value: r.v,
            itemStyle: { color: r.key === model.key ? ACCENT_HI : MUTED, borderRadius: [0, 3, 3, 0] }
        })).reverse();

        return {
            grid: { top: 8, bottom: 18, left: 132, right: 52 },
            tooltip: { trigger: 'axis', axisPointer: { type: 'none' } },
            xAxis: Object.assign({ type: 'value', min: 0.5, max: 1, splitNumber: 5 }, axisBase()),
            yAxis: Object.assign({ type: 'category', data: labels }, axisBase()),
            series: [{
                type: 'bar',
                barWidth: 10,
                data: values,
                label: {
                    show: true, position: 'right', color: TEXT_DIM, fontSize: 10,
                    formatter: (p) => Number(p.value).toFixed(4)
                }
            }]
        };
    }

    function fillDetailText(model, node) {
        const set = (suffix, text) => {
            const el = document.getElementById(model.prefix + '-' + suffix);
            if (el) el.innerText = text;
        };
        const accVal = node.test_accuracy != null ? node.test_accuracy : node.best_accuracy;
        set('acc-detail', accVal != null ? String(accVal) : '--');
        ['pre', 'rec', 'f1', 'f2', 'dr', 'fpr', 'fnr', 'auc'].forEach((suffix) => {
            const field = {
                pre: 'precision', rec: 'recall', f1: 'f1_score', f2: 'f2_score',
                dr: 'detection_rate', fpr: 'fpr', fnr: 'fnr', auc: 'auc'
            }[suffix];
            const v = node[field];
            set(suffix, v === null || v === undefined || v === '' ? '--' : String(v));
        });
        const lat = num(node.latency_ms_per_sample);
        set('lat', lat === null ? '--' : lat.toFixed(4) + ' ms/条');
        const tps = num(node.throughput_samples_per_sec);
        set('tps', tps === null ? '--' : Math.round(tps).toLocaleString('en-US') + ' 条/s');
    }

    function renderModel(model, data) {
        const node = data && data[model.json];
        if (!node) return;
        fillDetailText(model, node);
        const metricsChart = getChart('chart-metrics-' + model.prefix);
        if (metricsChart) metricsChart.setOption(buildMetricsOption(node), true);
        const compareChart = getChart('chart-compare-' + model.prefix);
        if (compareChart) compareChart.setOption(buildCompareOption(model, data), true);
    }

    function resizeAll() {
        Object.keys(instances).forEach((id) => { try { instances[id].resize(); } catch (e) {} });
    }
    window.modelEvalImpl.resizeAll = resizeAll;

    // 无刷新换页前调用：内容区被替换后旧实例还登记在 instances 表里，getChart 会把它们
    // 当成新容器的图表返回（setOption 抛错被 catch 吞掉）→ 模型评估页的图永久空白。
    function disposeAll() {
        Object.keys(instances).forEach((id) => {
            try { instances[id].dispose(); } catch (e) {}
            delete instances[id];
        });
    }
    window.modelEvalImpl.disposeAll = disposeAll;

    window.modelEvalImpl.showDetail = function (modelId) {
        document.querySelectorAll('[data-model-detail]').forEach((el) => {
            el.classList.toggle('hidden', el.getAttribute('data-model-detail') !== modelId);
        });
        resizeAll();
    };

    window.modelEvalImpl.initModelEvalCharts = async function () {
        try {
            let data = null;
            // 代码审查疑点 3：与 /api/metrics 同源，避免两份 json 指标打架
            const res = await fetch('/api/metrics');
            if (res.ok) {
                const payload = await res.json();
                data = (payload && payload.data) ? payload.data : payload;
            }
            if (!data) { console.warn('modelEvalImpl: /api/metrics 不可用，评估区保持空态'); return; }

            MODEL_LIST.forEach((model) => renderModel(model, data));

            if (!window._modelEvalResizeBound) {
                window._modelEvalResizeBound = true;
                window.addEventListener('resize', () => {
                    // 代码审查第 49 条：rAF 合并，避免拖拽窗口时每帧重算 12 张图
                    if (window._modelEvalResizeRaf) cancelAnimationFrame(window._modelEvalResizeRaf);
                    window._modelEvalResizeRaf = requestAnimationFrame(resizeAll);
                });
            }

            const anyVisible = document.querySelector('[data-model-detail]:not(.hidden)');
            if (!anyVisible) {
                const current = window.currentActiveModel;
                const target = MODEL_LIST.some((m) => m.key === current) ? current : MODEL_LIST[0].key;
                // 这里只能「展开某张卡的明细」，不能走 selectModel：投票模式下
                // currentActiveModel 是 ensemble_vote，不在 MODEL_LIST 里，回退到 MODEL_LIST[0]
                // 再调 selectModel 会把用户勾的「6 模型投票」清掉（用户报的刷新丢勾选）。
                if (typeof window.selectModel === 'function'
                    && !(typeof window.dlidsIsEnsembleMode === 'function' && window.dlidsIsEnsembleMode())) {
                    window.selectModel(target);
                } else {
                    window.modelEvalImpl.showDetail(target);
                }
            } else {
                resizeAll();
            }
        } catch (e) {
            console.warn('modelEvalImpl.initModelEvalCharts failed', e);
        }
    };
})();
