/**
 * 分域投票权重面板。
 *
 * 只做两件事：把 /api/ensemble/weights 的三域权重与实测指标画成一张表，以及把开关写回去。
 * 权重本身由后端按评估产物现算，前端不做任何加权运算 —— 界面上的数字必须和真正投票用的
 * 是同一份数据，否则这张表就成了「看起来在解释」的装饰。
 */
(function () {
    'use strict';

    var MODE_LABEL = {
        off: '等权投票（当前）',
        domain: '按域加权（当前）',
        auto: '自适应加权（当前）',
    };
    var MODE_DESC = {
        off: '六个模型各投一票，票票等重。',
        domain: '先估这批数据有多像「训练分布之外」（工业协议占比 + 马氏距离），再在同源权重与'
            + '工业权重之间按分数插值；模型票数僵持时还会向跨域稳健模型再掺一次。实时抓包用同源权重。',
        auto: '保留给后续自适应策略，目前与「按域加权」同义。',
    };

    var state = { payload: null, busy: false, bound: false };
    var pendingTimer = null;

    function el(id) { return document.getElementById(id); }

    function esc(text) {
        return String(text == null ? '' : text)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function fmtWeight(v) {
        if (v == null || isNaN(v)) { return '—'; }
        return Number(v).toFixed(3);
    }

    function fmtPct(v, digits) {
        if (v == null || isNaN(v)) { return '—'; }
        return Number(v).toFixed(digits == null ? 2 : digits) + '%';
    }

    function weightClass(weight) {
        if (weight == null || isNaN(weight)) { return ''; }
        if (weight >= 1.2) { return 'is-hi'; }
        if (weight <= 0.5) { return 'is-lo'; }
        return '';
    }

    function renderMode(payload) {
        var hint = el('ewModeHint');
        var note = el('ewModeNote');
        var group = el('ewModeGroup');
        var mode = payload.mode || 'off';
        if (hint) {
            hint.textContent = MODE_LABEL[mode] || mode;
            hint.title = payload.note || '';
        }
        if (note) {
            var desc = MODE_DESC[mode] || '';
            if (mode !== 'off') {
                // 开关是进程内状态：说清楚这点，免得重启后用户以为是坏了。
                desc += '（本次运行有效，重启后端会回到等权投票）';
            }
            note.textContent = desc;
        }
        if (group) {
            Array.prototype.forEach.call(group.querySelectorAll('[data-ew-mode]'), function (btn) {
                btn.classList.toggle('is-active', btn.getAttribute('data-ew-mode') === mode);
                btn.disabled = !!state.busy;
            });
        }
    }

    function renderTable(payload) {
        var body = el('ewTableBody');
        if (!body) { return; }
        var domains = payload.domains || [];
        var rows = payload.rows || [];
        var headCells = document.querySelectorAll('#ewTable thead th');
        // 表头带「该域样本量」，直接把「这份权重是在多少数据上量出来的」写进表里，
        // 工业域只有 4 条恶意样本，读者需要立刻看到这个前提。
        if (headCells.length === 4) {
            domains.forEach(function (d, i) {
                var cell = headCells[i + 1];
                if (!cell) { return; }
                var sub = cell.querySelector('.ew-col-sub');
                if (!sub) { return; }
                var parts = [];
                if (d.total) { parts.push(d.total.toLocaleString() + ' 条'); }
                if (d.class_counts && d.class_counts.Malware != null) {
                    parts.push('恶意 ' + d.class_counts.Malware);
                }
                if (!d.available) { parts.push('评估缺失'); }
                // 拿不到样本量时不要用 label 覆盖表头 —— defualt 域的 label 本身就是
                // 「同源（USTC-TFC2016 测试集）」，覆盖上去会和列标题「同源」重复一遍。
                if (parts.length) { sub.textContent = parts.join(' · '); }
                sub.title = (d.label || '') + (d.source ? '\n' + d.source : '');
            });
        }

        if (!rows.length) {
            body.innerHTML = '<tr><td colspan="4" class="ew-loading">没有读到评估产物</td></tr>';
            return;
        }

        var html = rows.map(function (row) {
            var cells = domains.map(function (d) {
                var w = (row.weights || {})[d.domain];
                var m = (row.metrics || {})[d.domain] || {};
                if (w == null) {
                    return '<td class="ew-cell"><span class="ew-weight">—</span>'
                        + '<span class="ew-metric">无评估</span></td>';
                }
                return '<td class="ew-cell">'
                    + '<span class="ew-weight ' + weightClass(w) + '">' + fmtWeight(w) + '</span>'
                    + '<span class="ew-metric">F1 ' + fmtPct(m.f1, 1)
                    + ' · ACC ' + fmtPct(m.accuracy, 1)
                    + ' · FPR ' + fmtPct(m.fpr, 1) + '</span>'
                    + '</td>';
            }).join('');
            return '<tr><td class="ew-name">' + esc(row.display || row.model) + '</td>' + cells + '</tr>';
        }).join('');
        body.innerHTML = html;
    }

    function renderFootnote(payload) {
        var box = el('ewFootnote');
        if (!box || !payload) { return; }
        var sources = (payload.domains || [])
            .filter(function (d) { return d.available; })
            .map(function (d) { return esc(d.source || ''); });
        var base = '权重 = max(该域实测 F1, 1.0) 后按均值归一化到「均值 = 1」，'
            + '只改模型之间的相对分量，不改整体票量尺度。用 F1 而不是准确率：'
            + '工业域是 45497:4 的极端不平衡集，全部判成良性也有 99.91% 的准确率，'
            + '只有 F1 反映真实检出能力。';
        if (sources.length) {
            base += ' 数据来源：' + sources.join('；') + '。';
        }

        // 光有静态表格还不够：要让读者看到「推理时这套权重是怎么被用的」——
        // 域是靠连续分数插出来的，模型吵起来时权重还会再向跨域稳健模型掺一次。
        var extra = '';
        var score = payload.domain_score || {};
        var contest = payload.contest || {};
        var robust = payload.robust || {};
        if (score.signals) {
            extra += '<br><br><b>怎么定域：</b>不再二选一，用连续域外分数 '
                + '0.6×端口命中率 + ' + (score.signals.dist || 0) + '×马氏距离比，'
                + '在两个阈值之间线性插值同源权重与工业权重。'
                + '距离比 = 批次平均马氏距离 / p99，低于 ' + score.dist_ratio_low
                + ' 记 0 分、高于 ' + score.dist_ratio_high + ' 饱和。'
                + '分数为 0 时结果就是等权，所以域判错最坏也不会比等权更差。';
        }
        if (contest.strength) {
            extra += '<br><br><b>怎么调分歧：</b>6 个模型票数越接近越不可信，'
                + '分歧度超过 ' + contest.start + ' 起逐级介入，最高把 '
                + Math.round(contest.strength * 100) + '% 的分量换成「跨域稳健权重」'
                + '（依据是' + esc(robust.basis || '两域最大误报率') + '）。';
            var rw = robust.weights || {};
            var items = Object.keys(rw).map(function (k) {
                return esc(k) + ' ' + Number(rw[k]).toFixed(3);
            });
            if (items.length) {
                extra += ' 稳健权重：' + items.join(' · ') + '。';
            }
        }
        if (payload.plan_doc) {
            extra += '<br><br>长期路线（需要多少标注数据、域分类器怎么做、其他路线与上线门槛）见 '
                + '<code>' + esc(payload.plan_doc) + '</code>。';
        }
        box.innerHTML = base + extra;
    }

    function renderAll(payload) {
        state.payload = payload;
        renderMode(payload);
        renderTable(payload);
        renderFootnote(payload);
    }

    function getJson(url) {
        return fetch(url, { credentials: 'same-origin', headers: { 'X-Requested-With': 'fetch' } })
            .then(function (r) {
                if (!r.ok) { throw new Error('HTTP ' + r.status); }
                return r.json();
            });
    }

    function postForm(url, data) {
        var body = new URLSearchParams();
        Object.keys(data).forEach(function (k) { body.append(k, data[k]); });
        return fetch(url, {
            method: 'POST',
            credentials: 'same-origin',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-Requested-With': 'fetch',
            },
            body: body.toString(),
        }).then(function (r) {
            return r.json().then(function (payload) {
                return { ok: r.ok, status: r.status, payload: payload };
            }, function () {
                return { ok: r.ok, status: r.status, payload: null };
            });
        });
    }

    function setNote(text, tone) {
        var note = el('ewModeNote');
        if (!note) { return; }
        note.textContent = text;
        note.className = 'ew-toolbar-note' + (tone ? ' ' + tone : '');
    }

    function switchMode(mode) {
        if (!mode || state.busy) { return; }
        if (state.payload && state.payload.mode === mode) { return; }
        state.busy = true;
        renderMode(state.payload || { mode: mode });
        setNote('正在切换…', '');
        postForm('/api/ensemble/weights', { mode: mode }).then(function (res) {
            state.busy = false;
            if (res.ok && res.payload) {
                renderAll(res.payload);
                setNote(res.payload.enabled
                    ? '已改为按域加权：下一次推理起使用该域权重。'
                    : '已改为等权投票：六个模型票票等重。', 'is-ok');
            } else {
                var msg = res.payload && res.payload.message ? res.payload.message : ('HTTP ' + res.status);
                setNote('切换失败：' + msg, 'is-err');
                if (state.payload) { renderMode(state.payload); }
            }
        }).catch(function (err) {
            state.busy = false;
            setNote('切换失败：' + err.message, 'is-err');
            if (state.payload) { renderMode(state.payload); }
        });
    }

    function bindEvents() {
        // 幂等标记挂在区块元素上：router 换页会替换 .app-content，
        // 模块级 boolean 会让新 DOM 永远绑不上（点击无反应，刷新才恢复）。
        var section = el('ensembleWeightSection');
        if (!section || section.dataset.ewBound === '1') { return; }
        section.dataset.ewBound = '1';
        var group = el('ewModeGroup');
        if (group) {
            group.addEventListener('click', function (e) {
                var btn = e.target.closest('[data-ew-mode]');
                if (!btn) { return; }
                e.preventDefault();
                switchMode(btn.getAttribute('data-ew-mode'));
            });
        }
    }

    function load() {
        if (!el('ensembleWeightSection')) { return; }
        getJson('/api/ensemble/weights').then(renderAll).catch(function (err) {
            var body = el('ewTableBody');
            if (body) {
                body.innerHTML = '<tr><td colspan="4" class="ew-loading">读取失败：'
                    + esc(err.message) + '</td></tr>';
            }
            setNote('读取失败：' + err.message, 'is-err');
        });
    }

    function init() {
        if (!el('ensembleWeightSection')) { return; }
        bindEvents();
        load();
        // 生产环境里评估产物会被重训覆盖，用低频轮询让表跟着刷新（切走页面时会清掉）。
        if (pendingTimer) { clearInterval(pendingTimer); }
        pendingTimer = setInterval(function () {
            if (document.hidden || state.busy) { return; }
            if (!document.body.contains(el('ensembleWeightSection'))) { return; }
            load();
        }, 60000);
    }

    function teardown() {
        if (pendingTimer) { clearInterval(pendingTimer); pendingTimer = null; }
    }

    window.ensembleImpl = {
        init: init,
        teardown: teardown,
        load: load,
        switchMode: switchMode,
    };
})();
