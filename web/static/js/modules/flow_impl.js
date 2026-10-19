/**
 * 流量分类页（/flow）：CSV 流特征上传、六分类结果渲染、三档风险分流。
 *
 * 与 upload_impl.js 的分工：那条走 pcap（28x28 字节图 + 二分类 6 模型），
 * 这条走 CSV（78 维流特征 + 单个六分类模型）。两条链路的后端接口也不同
 * （/analyze vs /analyze-flow），所以前端不复用。
 */
(function () {
    'use strict';

    var TIERS = {
        auto: { label: '自动处置', color: 'var(--ds-success)' },
        review: { label: '告警待确认', color: 'var(--ds-warning)' },
        manual: { label: '人工复核', color: 'var(--ds-danger)' }
    };

    // 六类的固定配色：Benign 用成功色，攻击类型按危险度从强调色到危险色铺开。
    // 固定映射而不是按出现顺序轮转，避免同一类在不同上传里换颜色。
    // 色相按相邻行两两拉开（理由与 main.js 的 THREAT_CLASSES 一致，两处必须同步改）。
    var CLASS_COLORS = {
        'Benign': 'var(--ds-success)',
        'Bot': 'var(--ds-accent)',
        'DoS-GoldenEye': 'var(--ds-warning)',
        'DoS-Slowloris': '#A78BFA',
        'DDoS-HOIC': '#60A5FA',
        'DDoS-LOIC-UDP': 'var(--ds-danger)',
        // 未知攻击是六类之外的统计项：这些行的预测值仍是六类之一，
        // 只是离该类中心太远，单独用中性灰标出来避免被当成正常结果。
        '未知攻击': '#94A3B8'
    };

    var state = { result: null, filter: 'all' };

    function el(id) { return document.getElementById(id); }

    function escapeHtml(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    function colorOf(name) { return CLASS_COLORS[name] || 'var(--ds-accent)'; }

    /* --------------------------- 渲染：类别分布 --------------------------- */
    function renderClassBars(summary, total) {
        var host = el('flowClassBars');
        var empty = el('flowClassEmpty');
        if (!host || !empty) return;

        var names = Object.keys(summary || {});
        if (!names.length || !total) {
            host.innerHTML = '';
            empty.classList.remove('hidden');
            return;
        }
        empty.classList.add('hidden');

        var maxCount = names.reduce(function (m, n) {
            return Math.max(m, summary[n].count || 0);
        }, 1);

        host.innerHTML = names.map(function (name) {
            var item = summary[name] || {};
            var count = item.count || 0;
            var ratio = total ? (count / total) * 100 : 0;
            // 条形宽度按「相对最大值」而不是「相对总数」：长尾类别（比如 346 条的 LOIC-UDP
            // 对 42 万条的 Benign）按总数算会短到看不见，相对最大值至少能看出存在。
            var width = maxCount ? (count / maxCount) * 100 : 0;
            var conf = typeof item.avg_confidence === 'number'
                ? item.avg_confidence.toFixed(4) : '-';
            return ''
                + '<div>'
                + '  <div class="flex items-baseline justify-between mb-1">'
                + '    <span class="text-[13px] text-gray-300">' + escapeHtml(name) + '</span>'
                + '    <span class="text-xs font-mono text-gray-500">'
                +        count.toLocaleString() + ' · ' + ratio.toFixed(2) + '% · ' + conf
                + '    </span>'
                + '  </div>'
                + '  <div class="h-1.5 rounded-full overflow-hidden" style="background: var(--ds-bg);">'
                + '    <div class="h-full rounded-full" style="width:' + width.toFixed(2) + '%;'
                +      ' background:' + colorOf(name) + '; transition: width .35s ease;"></div>'
                + '  </div>'
                + '</div>';
        }).join('');
    }

    /* --------------------------- 渲染：三档分流 --------------------------- */
    // 三档全为 0（还没跑过分类）时整个面板切成空态：三组数字收起，换成一行说明。
    // 否则空态是三个 24px 的 "--" 加三条横贯卡片的进度条槽，读起来像排版错误。
    function renderTiers(tierSummary) {
        var total = 0;
        var panel = el('flowTierPanel');
        Object.keys(TIERS).forEach(function (key) {
            var item = (tierSummary && tierSummary[key]) || {};
            total += (item.count || 0);
        });
        if (panel) panel.classList.toggle('is-empty', total === 0);

        Object.keys(TIERS).forEach(function (key) {
            var item = (tierSummary && tierSummary[key]) || {};
            var count = item.count || 0;
            var ratio = typeof item.ratio === 'number' ? item.ratio : 0;
            var suffix = key === 'auto' ? 'Auto' : (key === 'review' ? 'Review' : 'Manual');

            var countEl = el('flowTier' + suffix + 'Count');
            var ratioEl = el('flowTier' + suffix + 'Ratio');
            var barEl = el('flowTier' + suffix + 'Bar');

            if (countEl) countEl.innerText = count.toLocaleString();
            if (ratioEl) ratioEl.innerText = ratio.toFixed(2) + '%';
            // 进度条按「本档占比」画，三档加起来 100%，读出来就是分流结构
            if (barEl) barEl.style.width = Math.min(100, ratio).toFixed(2) + '%';
        });
    }

    /* --------------------------- 复核辅助（本地 LLM） --------------------------- */
    // 只在高风险档位给出入口：已经自动定案的干净行既不需要解释，也不该占本地算力。
    function canExplain(d) {
        return !!d && (d.tier === 'review' || d.tier === 'manual');
    }

    function setAiPanel(visible, meta, content) {
        var panel = el('flowAiPanel');
        if (!panel) return;
        if (!visible) {
            panel.classList.add('hidden');
            return;
        }
        panel.classList.remove('hidden');
        if (meta !== undefined) {
            var m = el('flowAiMeta');
            if (m) m.innerText = meta || '';
        }
        if (content !== undefined) {
            var c = el('flowAiContent');
            if (c) c.innerText = content || '';
        }
    }

    // 小模型（3B）几乎必然无视「不要 markdown」的指令，仍然输出 **加粗**、`代码`
    // 和行首列表符号。面板是纯文本渲染，这些符号会原样显示成杂讯，所以在展示前剥掉。
    function stripMarkdown(text) {
        return String(text == null ? '' : text)
            .replace(/\*\*|__/g, '')
            .replace(/`+/g, '')
            .replace(/^\s{0,3}#{1,6}\s*/gm, '')
            .replace(/^\s{0,3}[-*+]\s+/gm, '· ')
            .replace(/^\s{0,3}\d+[.)]\s+/gm, '')
            .replace(/\n{3,}/g, '\n\n')
            .trim();
    }

    // 注意：这里只请求「解释」，拿回来的文字不回流到任何判定字段。
    // 判定（预测类别、置信度、风险档位）始终以后端模型输出为准。
    async function explainRow(rowIndex, detail) {
        setAiPanel(true, '分析中…', '本地模型正在解读这条流量，通常需要 1~3 秒…');
        try {
            var resp = await fetch('/api/flow/explain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'same-origin',
                body: JSON.stringify({ row_index: Number(rowIndex), detail: detail || {} })
            });
            var data = await resp.json();
            if (!resp.ok) {
                setAiPanel(true, '未完成',
                    (data.error || ('HTTP ' + resp.status)) + (data.hint ? '\n' + data.hint : ''));
                return;
            }
            setAiPanel(true, (data.model || '本地模型') + ' · ' + (data.elapsed || ''), stripMarkdown(data.content));
        } catch (err) {
            setAiPanel(true, '请求出错', String(err));
        }
    }

    /* --------------------------- 渲染：明细表 --------------------------- */
    function renderRows(details, filter) {
        var body = el('flowResultsBody');
        var foot = el('flowResultsFoot');
        if (!body) return;

        var rows = (details || []).filter(function (d) {
            return filter === 'all' || d.tier === filter;
        });

        if (!rows.length) {
            body.innerHTML = '<tr><td colspan="6" class="text-center text-gray-500 py-10">'
                + (state.result ? '当前筛选条件下没有记录' : '暂无数据，请上传 CSV 开始分类')
                + '</td></tr>';
            if (foot) foot.innerText = '';
            return;
        }

        body.innerHTML = rows.map(function (d) {
            var tier = TIERS[d.tier] || { label: d.tier, color: 'var(--ds-text-muted)' };
            var probStr = Object.keys(d.probabilities || {}).map(function (k) {
                var v = d.probabilities[k];
                // 只显示非零概率，六类全列出会挤成一行读不了
                if (v < 0.0005) return null;
                return escapeHtml(k) + ' ' + (v * 100).toFixed(1) + '%';
            }).filter(Boolean).join('  ·  ');
            return ''
                + '<tr class="border-b" style="border-color: var(--ds-border);">'
                + '  <td class="px-4 py-2 font-mono text-xs text-gray-500">' + d.row + '</td>'
                + '  <td class="px-4 py-2 text-gray-200">' + escapeHtml(d.predicted)
                + (d.is_unknown
                    ? ' <span class="text-[10px] px-1 rounded" style="background:rgba(148,163,184,.15);'
                      + 'color:#94A3B8;border:1px solid rgba(148,163,184,.35)">未知</span>'
                    : '')
                + '  </td>'
                + '  <td class="px-4 py-2 font-mono text-xs text-gray-300">'
                +      (d.confidence * 100).toFixed(2) + '%</td>'
                + '  <td class="px-4 py-2">'
                + '    <span class="text-xs font-mono" style="color:' + tier.color + '">'
                +      escapeHtml(tier.label) + '</span>'
                + '  </td>'
                + '  <td class="px-4 py-2 text-xs text-gray-500 font-mono">' + (probStr || '-') + '</td>'
                + '  <td class="px-4 py-2">'
                +      (canExplain(d)
                    ? '<button type="button" class="text-[11px] text-cyber-primary hover:underline" '
                      + 'data-flow-explain="' + d.row + '">'
                      + '<i class="fa-solid fa-robot mr-1"></i>AI 分析</button>'
                    : '<span class="text-[10px] text-gray-600">&mdash;</span>')
                + '  </td>'
                + '</tr>';
        }).join('');

        if (foot) {
            var total = (details || []).length;
            foot.innerText = '显示 ' + rows.length + ' 条 / 共 ' + total + ' 条'
                + (state.result && state.result.details_truncated ? '（后端只返回最近 200 条）' : '');
        }
    }

    /* --------------------------- 上传流程 --------------------------- */
    function setLoading(on, text) {
        var content = el('flowUploadContent');
        var loading = el('flowLoadingState');
        var status = el('flowProcessStatus');
        if (content) content.classList.toggle('hidden', on);
        if (loading) loading.classList.toggle('hidden', !on);
        if (status && text) status.innerText = text;
    }

    function readThreshold(id, fallback) {
        var node = el(id);
        if (!node) return fallback;
        var v = parseFloat(node.value);
        if (!isFinite(v) || v < 0 || v > 1) return fallback;
        return v;
    }

    async function upload(file) {
        if (!file) return;
        if (!/\.csv$/i.test(file.name)) {
            window.alert('这里只接受 .csv 文件；pcap 请到「检测工作台」页。');
            return;
        }

        var form = new FormData();
        form.append('file', file);
        form.append('confidence_high', String(readThreshold('flowHighThreshold', 0.90)));
        form.append('confidence_low', String(readThreshold('flowLowThreshold', 0.50)));

        setLoading(true, file.name + ' · ' + (file.size / 1024 / 1024).toFixed(2) + ' MB');

        try {
            var resp = await fetch('/analyze-flow', { method: 'POST', body: form });
            var data = await resp.json();
            if (!resp.ok) {
                setLoading(false);
                window.alert('分类失败：' + (data.error || ('HTTP ' + resp.status)));
                return;
            }
            state.result = data;
            state.filter = 'all';
            document.querySelectorAll('[data-flow-filter]').forEach(function (btn) {
                btn.classList.toggle('is-active', btn.dataset.flowFilter === 'all');
            });
            renderClassBars(data.class_summary, data.rows);
            renderTiers(data.tier_summary);
            renderRows(data.details, 'all');
            setLoading(false);
        } catch (err) {
            setLoading(false);
            window.alert('请求出错：' + err);
        }
    }

    /* --------------------------- 环境校准 --------------------------- */
    // 校准与上面的「分类」不是同一件事：分类是拿现成模型去判，校准是让模型适应
    // 这个环境。跨环境实测过，源域模型的检出率会掉到 0%，而几百条本环境标注微调
    // 之后能回到 97%，所以两个入口完全独立，谁都不该顺手触发对方。
    var calibState = { file: null, running: false };

    function setCalibMessage(text, tone) {
        var box = el('flowCalibMessage');
        if (!box) return;
        box.textContent = text || '';
        box.style.color = tone === 'error' ? 'var(--ds-danger)'
            : tone === 'success' ? 'var(--ds-success)' : '';
    }

    function setCalibLoading(on) {
        calibState.running = !!on;
        var content = el('flowCalibContent');
        var loading = el('flowCalibLoading');
        if (content) content.classList.toggle('hidden', !!on);
        if (loading) loading.classList.toggle('hidden', !on);
        var btn = el('flowCalibRunBtn');
        if (btn) btn.disabled = !!on || !calibState.file;
    }

    function pickCalibFile(file) {
        if (!file) return;
        if (!/\.csv$/i.test(file.name)) {
            setCalibMessage('请选择 .csv 文件。', 'error');
            return;
        }
        calibState.file = file;
        var nameEl = el('flowCalibFileName');
        if (nameEl) {
            nameEl.textContent = file.name + '  ·  ' + (file.size / 1024).toFixed(0) + ' KB';
        }
        setCalibMessage('已选择文件，点击「开始校准」。');
        var btn = el('flowCalibRunBtn');
        if (btn) btn.disabled = false;
    }

    function pctText(v) {
        return (v == null || isNaN(v)) ? '--' : Number(v).toFixed(2) + '%';
    }

    function renderCalibResult(data) {
        var b = data.before || {};
        var a = data.after || {};
        var delta = (a.macro_f1 || 0) - (b.macro_f1 || 0);
        var parts = [
            '样本 ' + data.usable_rows + ' 行（训练 ' + data.fit_samples + ' / 评估 ' + data.eval_samples + '）',
            'macro F1 ' + pctText(b.macro_f1) + ' → ' + pctText(a.macro_f1)
                + '（' + (delta >= 0 ? '+' : '') + delta.toFixed(2) + '）',
            '准确率 ' + pctText(b.accuracy) + ' → ' + pctText(a.accuracy),
            '权重 ' + data.checkpoint + (data.activated ? '（已启用）' : '（已保存未启用）')
        ];
        if (!data.improved) {
            parts.push('校准后指标没有提升，先确认标注 CSV 的 Label 是否用了六类口径的名字。');
        }
        setCalibMessage(parts.join('　·　'), data.improved ? 'success' : 'error');
    }

    async function runCalib() {
        if (!calibState.file || calibState.running) return;
        var fd = new FormData();
        fd.append('file', calibState.file);
        fd.append('epochs', (el('flowCalibEpochs') || {}).value || '8');
        fd.append('holdout', (el('flowCalibHoldout') || {}).value || '0.2');
        fd.append('activate', (el('flowCalibActivate') || {}).checked ? 'true' : 'false');

        setCalibLoading(true);
        setCalibMessage('正在微调，数据量大时需要几十秒…');
        try {
            var resp = await fetch('/api/flow/calibrate', {
                method: 'POST', body: fd, credentials: 'same-origin'
            });
            var data = await resp.json();
            setCalibLoading(false);
            if (!resp.ok) {
                setCalibMessage('校准失败：' + (data.error || ('HTTP ' + resp.status)), 'error');
                return;
            }
            renderCalibResult(data);
        } catch (err) {
            setCalibLoading(false);
            setCalibMessage('请求出错：' + err, 'error');
        }
    }

    /* --------------------------- 事件绑定 --------------------------- */
    // 无刷新换页（router_impl）会把 .app-content 整段替换掉，DOM 上的监听随之丢失，
    // 所以入口必须可重复调用：initPage() 每次进入 /flow 都会调一次 init()。
    // 同一份 DOM 上重复 addEventListener 会让「选择文件」连弹两次文件框，
    // 因此按 dropzone 上的标记只绑一次。
    function init() {
        var zone = el('flowDropZone');
        var input = el('flowFileInput');
        var choose = el('flowChooseBtn');
        if (!zone || !input) return;
        if (zone.dataset.flowBound === '1') { renderAll(); return; }
        zone.dataset.flowBound = '1';

        if (choose) {
            choose.addEventListener('click', function (e) {
                e.stopPropagation();
                input.click();
            });
        }
        zone.addEventListener('click', function () { input.click(); });
        input.addEventListener('change', function () {
            if (input.files && input.files[0]) upload(input.files[0]);
            input.value = '';
        });

        ['dragenter', 'dragover'].forEach(function (evt) {
            zone.addEventListener(evt, function (e) {
                e.preventDefault();
                zone.style.borderColor = 'var(--ds-accent)';
            });
        });
        ['dragleave', 'drop'].forEach(function (evt) {
            zone.addEventListener(evt, function (e) {
                e.preventDefault();
                zone.style.borderColor = '';
            });
        });
        zone.addEventListener('drop', function (e) {
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
                upload(e.dataTransfer.files[0]);
            }
        });

        var group = el('flowFilterGroup');
        if (group) {
            group.addEventListener('click', function (e) {
                var btn = e.target.closest('[data-flow-filter]');
                if (!btn) return;
                state.filter = btn.dataset.flowFilter;
                group.querySelectorAll('[data-flow-filter]').forEach(function (b) {
                    b.classList.toggle('is-active', b === btn);
                });
                if (state.result) renderRows(state.result.details, state.filter);
            });
        }

        // AI 分析按钮用事件委托：表格每次筛选都会整体重绘，逐行绑定会漏。
        var resultsBody = el('flowResultsBody');
        if (resultsBody) {
            resultsBody.addEventListener('click', function (e) {
                var btn = e.target.closest('[data-flow-explain]');
                if (!btn || !state.result) return;
                var idx = Number(btn.dataset.flowExplain);
                var detail = (state.result.details || []).filter(function (d) {
                    return d.row === idx;
                })[0];
                explainRow(idx, detail);
            });
        }
        var aiClose = el('flowAiClose');
        if (aiClose) {
            aiClose.addEventListener('click', function () { setAiPanel(false); });
        }

        // 环境校准面板：独立的上传区与按钮，和上面的分类入口互不干扰
        var czone = el('flowCalibDropZone');
        var cinput = el('flowCalibFileInput');
        var crun = el('flowCalibRunBtn');
        if (czone && cinput) {
            czone.addEventListener('click', function () { cinput.click(); });
            cinput.addEventListener('change', function () {
                if (cinput.files && cinput.files[0]) pickCalibFile(cinput.files[0]);
                cinput.value = '';
            });
            ['dragenter', 'dragover'].forEach(function (evt) {
                czone.addEventListener(evt, function (e) {
                    e.preventDefault();
                    czone.style.borderColor = 'var(--ds-accent)';
                });
            });
            ['dragleave', 'drop'].forEach(function (evt) {
                czone.addEventListener(evt, function (e) {
                    e.preventDefault();
                    czone.style.borderColor = '';
                });
            });
            czone.addEventListener('drop', function (e) {
                if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
                    pickCalibFile(e.dataTransfer.files[0]);
                }
            });
        }
        if (crun) {
            crun.addEventListener('click', function (e) {
                e.stopPropagation();
                runCalib();
            });
        }
    }

    // 换页回来后把上一次的批次结果重新铺到新 DOM 上（结果留在内存里，不重复请求后端）
    function renderAll() {
        if (!state.result) return;
        renderClassBars(state.result.class_summary || {}, state.result.rows || 0);
        renderTiers(state.result.tier_summary || {});
        renderRows(state.result.details || [], state.filter);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    window.flowImpl = { upload: upload, state: state, init: init, renderAll: renderAll };
})();
