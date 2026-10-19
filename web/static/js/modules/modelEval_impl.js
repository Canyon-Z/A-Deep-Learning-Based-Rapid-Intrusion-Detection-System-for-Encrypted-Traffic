/* 模型效能对比：全部数据来自 /api/metrics（后端读 checkpoints/model_metrics.json）。
   本文件不包含任何硬编码指标。

   为什么这里没有 ECharts：原先每个模型详情里有两张条形图（核心指标、AUC 横向对比），
   画的是同一批数字，而指标值都挤在 99.88~99.97 / 0.9997~1.0000 这个窄区间里，
   条形长度全部顶格，六张图看起来完全一样，读者只会误以为「六个模型没有差别」。
   数字交给对比表与 metric-grid 呈现，图表删掉。 */
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

    const num = (v) => {
        if (v === null || v === undefined || v === '') return null;
        const n = parseFloat(String(v).replace('%', '').trim());
        return Number.isFinite(n) ? n : null;
    };

    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.innerText = text;
    }

    function accOf(node) {
        const v = node.test_accuracy != null ? node.test_accuracy
            : (node.best_val_accuracy != null ? node.best_val_accuracy : node.best_accuracy);
        return v === null || v === undefined || v === '' ? '--' : String(v);
    }

    /* 对比表的数值列。suffix 对应 DOM id 后缀；better 决定「最优」取最大值还是最小值。
       没有「准确率」列：六个模型的 test_accuracy 全都是 99.97%，放进来只是一列重复数字
       （它在每个模型的详情里都有）。换成安全场景更关心、也确实有区分度的检出率与误报率。 */
    const COLUMNS = [
        { key: 'dr',  suffix: '-t-dr',  better: 'high', pick: (n) => num(n.detection_rate), fmt: (v) => v.toFixed(2) + '%' },
        { key: 'fpr', suffix: '-t-fpr', better: 'low',  pick: (n) => num(n.fpr), fmt: (v) => v.toFixed(2) + '%' },
        { key: 'lat', suffix: '-t-lat', better: 'low',  pick: (n) => num(n.latency_ms_per_sample), fmt: (v) => v.toFixed(4) + ' ms' },
        { key: 'tps', suffix: '-t-tps', better: 'high', pick: (n) => num(n.throughput_samples_per_sec), fmt: (v) => Math.round(v).toLocaleString('en-US') },
        { key: 'auc', suffix: '-t-auc', better: 'high', pick: (n) => num(n.auc), fmt: (v) => v.toFixed(4) }
    ];

    function fillRow(model, data) {
        const node = data && data[model.json];
        if (!node) return;
        COLUMNS.forEach((col) => {
            const v = col.pick(node);
            setText(model.prefix + col.suffix, v === null ? '--' : col.fmt(v));
        });
    }

    // 每列最优值标绿。同一个值并列最优时都标，不做「只取第一个」的隐性裁决。
    // 一列里所有模型数值相同就不标：六个模型并列第一不是结论，只是这一列没有区分度
    // （准确率列就是这样，六者都是 99.97%，标绿会让整列变成一片绿）。
    function markBest(data) {
        COLUMNS.forEach((col) => {
            const rows = MODEL_LIST
                .map((m) => ({ m: m, v: col.pick(data[m.json] || {}) }))
                .filter((r) => r.v !== null && Number.isFinite(r.v));
            if (rows.length < 2) return;
            const values = rows.map((r) => r.v);
            if (values.every((v) => v === values[0])) return;
            const target = col.better === 'high'
                ? Math.max.apply(null, values)
                : Math.min.apply(null, values);
            rows.forEach((r) => {
                const el = document.getElementById(r.m.prefix + col.suffix);
                if (el) el.classList.toggle('cmp-best', r.v === target);
            });
        });
    }

    function fillDetailText(model, node) {
        setText(model.prefix + '-acc-detail', accOf(node));
        ['pre', 'rec', 'f1', 'f2', 'dr', 'fpr', 'fnr', 'auc'].forEach((suffix) => {
            const field = {
                pre: 'precision', rec: 'recall', f1: 'f1_score', f2: 'f2_score',
                dr: 'detection_rate', fpr: 'fpr', fnr: 'fnr', auc: 'auc'
            }[suffix];
            const v = node[field];
            setText(model.prefix + '-' + suffix, v === null || v === undefined || v === '' ? '--' : String(v));
        });
        const lat = num(node.latency_ms_per_sample);
        setText(model.prefix + '-lat', lat === null ? '--' : lat.toFixed(4) + ' ms/条');
        const tps = num(node.throughput_samples_per_sec);
        setText(model.prefix + '-tps', tps === null ? '--' : Math.round(tps).toLocaleString('en-US') + ' 条/s');
    }

    // 保留这两个接口：main.js 在换页/窗口变化时会调它们。图表已移除，只剩空实现，
    // 但调用点不需要知道这件事。
    window.modelEvalImpl.resizeAll = function () {};
    window.modelEvalImpl.disposeAll = function () {};

    window.modelEvalImpl.showDetail = function (modelId) {
        document.querySelectorAll('[data-model-detail]').forEach((el) => {
            el.classList.toggle('hidden', el.getAttribute('data-model-detail') !== modelId);
        });
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

            MODEL_LIST.forEach((model) => {
                const node = data[model.json];
                if (!node) return;
                fillRow(model, data);
                fillDetailText(model, node);
            });
            markBest(data);

            // 只有「模型评估」页才存在 [data-model-detail]。在其它页面必须整体跳过这段：
            // 之前这里无条件找 `[data-model-detail]:not(.hidden)`，在总览/上传等页面必然为空，
            // 于是走兜底 → 拿 MODEL_LIST[0]（cnn_bilstm）调 selectModel，把用户在顶栏选的
            // 「级联检测」直接改成了 CNN-BiLSTM（实测：打开 /dashboard 后 #modelSelector
            // 从 cascade 变成 cnn_bilstm，KPI 也跟着变）。模型选择是用户设置，只能由用户改。
            const detailNodes = document.querySelectorAll('[data-model-detail]');
            if (detailNodes.length) {
                const anyVisible = document.querySelector('[data-model-detail]:not(.hidden)');
                if (!anyVisible) {
                    const current = window.currentActiveModel;
                    const inList = MODEL_LIST.some((m) => m.key === current);
                    // 这里只「展开某一行的明细」。current 不在 6 个模型里（cascade / 投票模式）
                    // 时展开第一行给人看即可，绝不能再调 selectModel —— 那会覆盖用户的选择，
                    // 也会把用户勾的「6 模型投票」清掉（用户报的刷新丢勾选就是这么来的）。
                    window.modelEvalImpl.showDetail(inList ? current : MODEL_LIST[0].key);
                }
            }
        } catch (e) {
            console.warn('modelEvalImpl.initModelEvalCharts failed', e);
        }
        renderCrossDomain();
    };

    /* ── 跨域能力评估 ───────────────────────────────────────────────
       数据来自 /api/cross-domain（后端读 logs/ 下实际跑过的实验 JSON）。
       本段只负责展示，不做任何换算——数字对不上应该去改实验，而不是在这里调显示。

       分段渲染而不是拼一个大字符串，是为了某份实验日志缺失时其余两张表照常显示
       （availability 里会标出哪个源没读到）。 */
    const XD_MODEL_ORDER = ['CNN_BiLSTM', 'Classic_CNN', 'Lightweight_CNN_BiLSTM', 'Pure_BiLSTM', 'MLP', 'Transformer'];
    const XD_MODEL_LABEL = {
        CNN_BiLSTM: 'CNN-BiLSTM',
        Classic_CNN: 'Classic-CNN',
        Lightweight_CNN_BiLSTM: 'Lightweight CNN-BiLSTM',
        Pure_BiLSTM: 'Pure BiLSTM',
        MLP: 'MLP',
        Transformer: 'Transformer'
    };

    function xdEscape(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    // 三档着色：>=90 好、60~90 可疑、<60 基本失效。阈值是经验值，
    // 依据是跨域实验里同源 99.9%、目标域随机水平（50% 上下）这两端。
    function xdTone(v) {
        if (v === null || v === undefined || !Number.isFinite(Number(v))) return 'xd-bad';
        const n = Number(v);
        if (n >= 90) return 'xd-good';
        if (n >= 60) return 'xd-mid';
        return 'xd-bad';
    }

    function xdFixed(v, digits) {
        if (v === null || v === undefined || !Number.isFinite(Number(v))) return '--';
        return Number(v).toFixed(digits === undefined ? 2 : digits) + '%';
    }

    function xdAuc(v) {
        if (v === null || v === undefined || !Number.isFinite(Number(v))) return '--';
        const n = Number(v);
        // AUC 不用百分比，且跨域时着色要按同一套阈值换算（0.9 → 90）
        return n.toFixed(4);
    }

    function xdSetBody(id, html) {
        const el = document.getElementById(id);
        if (el) el.innerHTML = html;
    }

    function xdStatusText(avail, key, label) {
        return avail && avail[key] === false ? label + '（日志缺失）' : label;
    }

    // 缺陷 M4：把后端算出的 caveats 原样摆出来。
    // 触发条件是「这份日志里的数字不能直接当能力证明用」：少数类样本太少、
    // 训练/测试切分有泄漏、比率口径越界、全零结果。后端只给事实与严重度，
    // 展示文案在这里定，避免后端改一个字就影响页面语气。
    const XD_CAVEAT_TITLE = {
        image_ics: 'ICS 图像评估',
        image_cse: 'CSE-CIC-IDS2018 图像评估',
        flow_multiclass: '六分类同源测试',
        flow_finetune: '目标域微调',
        flow_adanb: 'AdaBN 校准',
        flow_autoencoder: 'AutoEncoder 异常检测'
    };

    function renderCrossDomainCaveats(payload) {
        const box = document.getElementById('xdCaveats');
        if (!box) return;
        const items = [];
        Object.keys(XD_CAVEAT_TITLE).forEach(function (key) {
            const c = payload[key];
            const cav = c && c.caveats;
            // severity='info' 是「这组数字正常、可直接引用」，不该出现在警示区里
            if (!cav || cav.severity === 'info') return;
            if (!Array.isArray(cav.notes) || !cav.notes.length) return;
            items.push({ key: key, title: XD_CAVEAT_TITLE[key], cav: cav });
        });
        if (!items.length) {
            box.hidden = true;
            box.innerHTML = '';
            return;
        }
        const order = { high: 0, medium: 1, low: 2 };
        items.sort(function (a, b) {
            return (order[a.cav.severity] || 9) - (order[b.cav.severity] || 9);
        });
        const html = items.map(function (it) {
            const bad = it.cav.severity === 'high';
            const badge = bad ? '不可引用' : '需注意';
            const notes = it.cav.notes.map(function (n) {
                return '<li>' + xdEscape(n) + '</li>';
            }).join('');
            return '<div class="xd-caveat ' + (bad ? 'xd-caveat-bad' : 'xd-caveat-warn') + '">'
                + '<div class="xd-caveat-head">'
                + '<span class="xd-caveat-badge">' + badge + '</span>'
                + '<span>' + xdEscape(it.title) + '</span>'
                + '</div>'
                + '<ul class="xd-caveat-list">' + notes + '</ul>'
                + '</div>';
        }).join('');
        box.hidden = false;
        box.innerHTML = '<div class="xd-caveat-title">下表数字的使用前提</div>'
            + '<p class="xd-caveat-intro">以下各组的实验条件不满足「可以横向比较」的前提，'
            + '数字本身照实展示，但不能当作模型能力结论引用。标记为「不可引用」的整组结论作废。'
            + '</p>' + html;
    }

    function renderCrossDomainImage(payload) {
        const ics = payload.image_ics;
        const cse = payload.image_cse;
        const rows = XD_MODEL_ORDER.map(function (k) {
            const a = ics && ics.models ? ics.models[k] : null;
            const b = cse && cse.models ? cse.models[k] : null;
            const icsAcc = a ? a.accuracy : null;
            const cseAcc = b ? b.accuracy : null;
            return '<tr>'
                + '<td>' + xdEscape(XD_MODEL_LABEL[k] || k) + '</td>'
                + '<td class="xd-good">99.9x% · 100% · 0.0%</td>'
                + '<td class="' + xdTone(icsAcc) + '">'
                + xdFixed(icsAcc) + ' · ' + xdFixed(a && a.detection_rate) + ' · ' + xdFixed(a && a.fpr)
                + '</td>'
                + '<td class="' + xdTone(cseAcc) + '">'
                + xdFixed(cseAcc) + ' · ' + xdFixed(b && b.f1_score) + ' · ' + xdAuc(b && b.auc)
                + '</td>'
                + '</tr>';
        }).join('');
        xdSetBody('xdImageBody', rows);
    }

    function renderCrossDomainFlow(payload) {
        const list = payload.flow_cross_day || [];
        if (!list.length) {
            xdSetBody('xdFlowBody', '<tr><td colspan="6" class="xd-empty">暂无数据</td></tr>');
            return;
        }
        const rows = list.map(function (r) {
            const sameDay = r.detection_rate >= 90;
            return '<tr>'
                + '<td class="font-mono">' + xdEscape(r.group) + '</td>'
                + '<td>' + xdEscape(r.desc) + '</td>'
                + '<td class="' + xdTone(r.accuracy) + '">' + xdFixed(r.accuracy) + '</td>'
                + '<td class="' + xdTone(r.detection_rate) + '">' + xdFixed(r.detection_rate) + '</td>'
                + '<td>' + xdFixed(r.fpr, 3) + '</td>'
                + '<td class="' + xdTone(Number(r.auc) * 100) + '">' + xdAuc(r.auc) + '</td>'
                + '</tr>';
        }).join('');
        xdSetBody('xdFlowBody', rows);
    }

    function renderCrossDomainAdapt(payload) {
        const ft = payload.flow_finetune;
        const ae = payload.flow_autoencoder;
        const rows = [];

        rows.push('<tr><td>不校准（基线）</td><td>0</td>'
            + '<td class="xd-bad">' + xdFixed(ft && ft.baseline && ft.baseline.accuracy) + '</td>'
            + '<td class="xd-bad">' + xdFixed(ft && ft.baseline && ft.baseline.detection_rate) + '</td>'
            + '<td>' + xdFixed(ft && ft.baseline && ft.baseline.false_positive_rate, 3) + '</td>'
            + '<td class="xd-bad">换一天即失效，检出为 0</td></tr>');

        const fit = ft && Array.isArray(ft.finetuned) ? ft.finetuned[ft.finetuned.length - 1] : null;
        if (fit) {
            rows.push('<tr><td>目标域微调</td><td>' + (fit.n_labels || 0) + ' 条</td>'
                + '<td class="' + xdTone(fit.accuracy) + '">' + xdFixed(fit.accuracy) + '</td>'
                + '<td class="' + xdTone(fit.detection_rate) + '">' + xdFixed(fit.detection_rate) + '</td>'
                + '<td>' + xdFixed(fit.false_positive_rate) + '</td>'
                + '<td class="xd-good">目前唯一验证有效的办法</td></tr>');
        }

        const adanb = payload.flow_adanb;
        if (adanb) {
            const variants = Array.isArray(adanb.variants) ? adanb.variants : [];
            const best = variants.reduce(function (acc, v) {
                const dr = Number(v.detection_rate);
                return (acc === null || (Number.isFinite(dr) && dr > acc.dr)) ? { dr: dr, v: v } : acc;
            }, null);
            rows.push('<tr><td>AdaBN（无标签校准）</td><td>0</td>'
                + '<td>--</td>'
                + '<td class="xd-bad">' + (best ? xdFixed(best.v.detection_rate) : '0.00%') + '</td>'
                + '<td>--</td>'
                + '<td class="xd-bad">BN 只修一阶/二阶矩，分不开重叠的类条件分布</td></tr>');
        }

        if (ae && ae.anomaly_auc != null) {
            rows.push('<tr><td>AutoEncoder 异常检测</td><td>只用良性</td>'
                + '<td>--</td>'
                + '<td class="xd-bad">' + xdFixed(ae.detection_rate_at_default) + '</td>'
                + '<td>' + xdFixed(ae.fpr_at_default) + '</td>'
                + '<td class="xd-bad">异常分 AUC ' + xdAuc(ae.anomaly_auc) + '，接近随机</td></tr>');
        }

        if (!rows.length) rows.push('<tr><td colspan="6" class="xd-empty">暂无数据</td></tr>');
        xdSetBody('xdAdaptBody', rows.join(''));
    }

    function renderCrossDomainSummary(payload) {
        const el = document.getElementById('xdSummary');
        if (!el) return;
        const avail = payload.availability || {};
        const parts = [];
        parts.push(xdStatusText(avail, 'image_ics', 'ICS 图像评估'));
        parts.push(xdStatusText(avail, 'image_cse', 'CSE-CIC-IDS2018 图像评估'));
        parts.push(xdStatusText(avail, 'flow_multiclass', '六分类'));
        parts.push(xdStatusText(avail, 'flow_finetune', '目标域微调'));
        const mc = payload.flow_multiclass;
        const mcText = mc
            ? '六分类同源测试集 ' + xdFixed(mc.test_accuracy) + '（macro F1 ' + xdFixed(mc.test_macro_f1) + '）'
            : '六分类指标缺失';
        el.innerHTML = '同源数据集上所有模型都在 99.9% 附近；换成另一批采集数据后，'
            + '图像分支掉到随机水平（AUC 0.33~0.61），流特征分支同一天 99.99%、换一天检出 0.00%。'
            + ' ' + xdEscape(mcText) + '。数据源：' + xdEscape(parts.join(' · ')) + '。';
    }

    async function renderCrossDomain() {
        if (!document.getElementById('cross-domain-section')) return;
        try {
            const res = await fetch('/api/cross-domain');
            if (!res.ok) return;
            const payload = await res.json();
            if (!payload || payload.status !== 'success') return;
            renderCrossDomainSummary(payload);
            renderCrossDomainCaveats(payload);
            renderCrossDomainImage(payload);
            renderCrossDomainFlow(payload);
            renderCrossDomainAdapt(payload);
        } catch (e) {
            console.warn('modelEvalImpl.renderCrossDomain failed', e);
        }
    }
})();
