/* ============================================================================
   复核中心（/review）
   ----------------------------------------------------------------------------
   上传检测（pcap 与流特征 CSV）与实时抓包产生的「待确认 / 人工复核」流量都汇总在
   这一页。两种处置方式并存，但落库口径一致：
     · 人工：每条三个按钮（确认攻击 / 确认误报 / 忽略）
     · LLM：给建议（不落库）或自主处置（结论直接落库）
   后端 `/api/review/auto` 负责调用本地 LLM，前端只负责展示与触发。
   ========================================================================== */
(function () {
    'use strict';

    var REFRESH_MS = 5000;
    var SOURCE_LABEL = { live: '实时抓包', upload: '上传 pcap', csv: '流特征 CSV' };
    var TIER_LABEL = { auto: '自动处置', review: '告警待确认', manual: '人工复核' };
    var VERDICT_LABEL = { malicious: '确认攻击', benign: '确认误报', ignore: '忽略' };

    var state = {
        items: [],
        stats: {},
        sources: {},
        filter: 'all',
        timer: null,
        busy: false,
        // undefined = 还没探过；true/false = 上次探测结果
        llmAvailable: undefined,
        selected: {}
    };
    var bound = false;

    function el(id) { return document.getElementById(id); }

    function esc(value) {
        return String(value == null ? '' : value)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    // 结论是谁给的 —— 自动策略落库的结论必须和人工结论在界面上分得开，
    // 否则「真实误报率」这类统计会被机器判的条目悄悄污染。
    var REVIEWER_LABEL = {
        human: '人工判定',
        'auto:vote': '机器 · 6 模型投票',
        'auto:llm': '机器 · LLM'
    };

    function reviewerLabel(item) {
        var by = String((item && item.reviewed_by) || 'human');
        return REVIEWER_LABEL[by] || by;
    }

    function postForm(url, data) {
        var body = new URLSearchParams();
        Object.keys(data || {}).forEach(function (k) {
            body.append(k, String(data[k]));
        });
        return fetch(url, {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8' },
            body: body.toString()
        }).then(function (resp) {
            return resp.json().catch(function () { return {}; }).then(function (json) {
                if (!resp.ok) throw new Error(json.error || ('HTTP ' + resp.status));
                return json;
            });
        });
    }

    function getJson(url) {
        return fetch(url, { credentials: 'same-origin' })
            .then(function (resp) {
                return resp.json().catch(function () { return {}; }).then(function (json) {
                    if (!resp.ok) throw new Error(json.error || ('HTTP ' + resp.status));
                    return json;
                });
            });
    }

    // ── 渲染 ────────────────────────────────────────────────────────────
    function flowText(item) {
        if (item.src_ip || item.dst_ip) {
            return esc(item.src_ip || '?') + ':' + esc(item.src_port || '-')
                + ' → ' + esc(item.dst_ip || '?') + ':' + esc(item.dst_port || '-')
                + (item.proto ? ' · proto ' + esc(item.proto) : '');
        }
        var file = item.file ? esc(item.file) : '未知文件';
        return file + (item.row == null ? '' : ' · 第 ' + esc(item.row) + ' 行');
    }

    function probText(item) {
        var probs = item.probabilities;
        var pairs = [];
        if (Array.isArray(probs)) {
            pairs = probs.map(function (p) { return [p.label, p.value]; });
        } else if (probs && typeof probs === 'object') {
            pairs = Object.keys(probs).map(function (k) { return [k, probs[k]]; });
        }
        if (!pairs.length) return '';
        pairs.sort(function (a, b) { return Number(b[1] || 0) - Number(a[1] || 0); });
        return pairs.slice(0, 3).map(function (p) {
            return esc(p[0]) + ' ' + (Number(p[1] || 0) * 100).toFixed(1) + '%';
        }).join(' · ');
    }

    function filteredItems() {
        if (state.filter === 'all') return state.items;
        return state.items.filter(function (it) {
            return String(it.source || 'live') === state.filter;
        });
    }

    function renderStats() {
        var stats = state.stats || {};
        var by = stats.by_verdict || {};
        if (el('reviewStatPending')) el('reviewStatPending').textContent = String(stats.pending || 0);
        if (el('reviewStatMalicious')) el('reviewStatMalicious').textContent = String(by.malicious || 0);
        if (el('reviewStatBenign')) el('reviewStatBenign').textContent = String(by.benign || 0);
        if (el('reviewStatIgnore')) el('reviewStatIgnore').textContent = String(by.ignore || 0);
        var hint = el('reviewQueueHint');
        if (hint) {
            hint.textContent = (stats.pending || 0) + ' 待复核 · ' + (stats.reviewed || 0) + ' 已处理';
        }
    }

    function renderQueue() {
        var list = el('reviewQueueList');
        if (!list) return;
        var items = filteredItems();
        var listHint = el('reviewListHint');
        if (listHint) {
            listHint.textContent = state.filter === 'all'
                ? '按可疑度降序 · ' + items.length + ' 条'
                : (SOURCE_LABEL[state.filter] || state.filter) + ' · ' + items.length + ' 条';
        }
        if (!items.length) {
            list.innerHTML = '<div class="text-xs text-gray-500 text-center py-8">'
                + (state.items.length ? '当前筛选下没有待复核流量' : '暂无待复核流量')
                + '</div>';
            return;
        }
        var html = '';
        for (var i = 0; i < items.length; ++i) {
            var it = items[i] || {};
            var source = String(it.source || 'live');
            var tier = String(it.tier || 'review');
            var tierText = it.is_unknown ? '未知攻击' : (TIER_LABEL[tier] || tier);
            var tierClass = it.is_unknown ? 'is-unknown' : (tier === 'manual' ? 'is-manual' : 'is-review');
            var conf = Number(it.confidence);
            var ood = Number(it.ood_distance);
            var meta = esc(it.predicted || '-') + ' · 置信度 '
                + (isFinite(conf) ? (conf * 100).toFixed(1) + '%' : '--')
                + (isFinite(ood) ? ' · 离群 ' + ood.toFixed(1) : '');
            var probs = probText(it);
            var key = esc(it.key || '');
            var checked = state.selected[it.key] ? ' checked' : '';
            html += '<div class="review-item">'
                + '<div class="review-item-head">'
                + '<label class="review-item-check" title="选中后可批量处置">'
                + '<input type="checkbox" data-review-check="' + key + '"' + checked + '>'
                + '</label>'
                + '<div class="review-item-main">'
                + '<div class="review-item-flow">'
                + '<span class="review-item-source">' + esc(SOURCE_LABEL[source] || source) + '</span>'
                + flowText(it) + '</div>'
                + '<div class="review-item-meta">' + meta + '</div>'
                + (probs ? '<div class="review-item-meta">' + probs + '</div>' : '')
                + '</div>'
                + '<span class="review-item-tier ' + tierClass + '">' + esc(tierText) + '</span>'
                + '</div>'
                + '<div class="review-item-actions" data-key="' + key + '">'
                + '<button type="button" class="review-action is-attack" data-verdict="malicious">确认攻击</button>'
                + '<button type="button" class="review-action is-benign" data-verdict="benign">确认误报</button>'
                + '<button type="button" class="review-action is-ignore" data-verdict="ignore">忽略</button>'
                + '<button type="button" class="review-action" data-review-ai="1">AI 判断</button>'
                + '</div>'
                + '</div>';
        }
        list.innerHTML = html;
        syncSelectionUi();
    }

    // 选中集合 → 顶部工具条：计数、全选态、批量按钮可用性。
    function syncSelectionUi() {
        var visible = filteredItems();
        var visibleKeys = visible.map(function (it) { return it.key; });
        // 只保留仍在队列里的 key，避免清空/筛选后残留幽灵选中
        Object.keys(state.selected).forEach(function (k) {
            if (!state.items.some(function (it) { return it.key === k; })) delete state.selected[k];
        });
        var count = Object.keys(state.selected).length;
        var countNode = el('reviewSelectionHint');
        if (countNode) countNode.textContent = count ? ('已选 ' + count + ' 条') : '未选中';
        var selectAll = el('reviewSelectAll');
        if (selectAll) {
            var selectedVisible = visibleKeys.filter(function (k) { return state.selected[k]; }).length;
            selectAll.checked = visibleKeys.length > 0 && selectedVisible === visibleKeys.length;
            selectAll.indeterminate = selectedVisible > 0 && selectedVisible < visibleKeys.length;
            selectAll.disabled = visibleKeys.length === 0;
        }
        Array.prototype.forEach.call(document.querySelectorAll('[data-review-batch]'), function (btn) {
            btn.disabled = count === 0;
        });
    }

    function setSelectionForVisible(checked) {
        filteredItems().forEach(function (it) {
            if (checked) state.selected[it.key] = true;
            else delete state.selected[it.key];
        });
        renderQueue();
    }

    function runBatch(verdict) {
        var keys = Object.keys(state.selected);
        if (!keys.length) return;
        if (state.busy) return;
        state.busy = true;
        setLlmStatus('批量处置 ' + keys.length + ' 条…');
        postForm('/api/review/verdict-batch', { keys: keys.join(','), verdict: verdict })
            .then(function (data) {
                var done = (data && data.reviewed) || 0;
                setLlmStatus('已处置 ' + done + ' 条', 'ok');
                state.selected = {};
                state.items = Array.isArray(data.items) ? data.items : state.items;
                state.stats = data.stats || state.stats;
                renderStats();
                renderQueue();
                loadHistory();
            })
            .catch(function (e) {
                setLlmStatus('批量处置失败：' + (e.message || e), 'error');
            })
            .then(function () { state.busy = false; });
    }

    function runBatchLlm() {
        var keys = Object.keys(state.selected);
        if (!keys.length) return;
        if (state.busy) return;
        if (state.llmAvailable === false) {
            probeLlm().then(function (ok) {
                if (!ok) { setLlmStatus('本地 LLM 未在线，无法批量判断', 'error'); return; }
                startBatchLlm(keys);
            });
            return;
        }
        startBatchLlm(keys);
    }

    function startBatchLlm(keys) {
        state.busy = true;
        setLlmStatus('让 LLM 判断 ' + keys.length + ' 条…');
        postForm('/api/review/auto', { keys: keys.join(','), limit: keys.length, apply: 'false' })
            .then(function (data) {
                showAiResults(data);
                setLlmStatus('完成 · ' + ((data && data.results) || []).length + ' 条', 'ok');
            })
            .catch(function (e) { setLlmStatus('LLM 调用失败：' + (e.message || e), 'error'); })
            .then(function () { state.busy = false; });
    }

    function renderHistory(items) {
        var list = el('reviewHistoryList');
        if (!list) return;
        var arr = Array.isArray(items) ? items : [];
        var hint = el('reviewHistoryHint');
        if (hint) hint.textContent = arr.length + ' 条';
        if (!arr.length) {
            list.innerHTML = '<div class="text-xs text-gray-500 text-center py-8">还没有复核记录</div>';
            return;
        }
        var html = '';
        for (var i = 0; i < arr.length; ++i) {
            var it = arr[i] || {};
            var verdict = String(it.verdict || '');
            var when = it.reviewed_at ? new Date(Number(it.reviewed_at) * 1000).toLocaleString() : '';
            var cls = verdict === 'malicious' ? 'is-unknown'
                : (verdict === 'benign' ? 'is-review' : 'is-manual');
            html += '<div class="review-item">'
                + '<div class="review-item-head">'
                + '<div class="review-item-main">'
                + '<div class="review-item-flow">'
                + '<span class="review-item-source">'
                + esc(SOURCE_LABEL[String(it.source || 'live')] || String(it.source || 'live'))
                + '</span>' + flowText(it) + '</div>'
                + '<div class="review-item-meta">' + esc(it.predicted || '-')
                + ' · 置信度 ' + (isFinite(Number(it.confidence))
                    ? (Number(it.confidence) * 100).toFixed(1) + '%' : '--')
                + ' · ' + esc(reviewerLabel(it))
                + (when ? ' · ' + esc(when) : '') + '</div>'
                + '</div>'
                + '<span class="review-item-tier ' + cls + '"'
                + (it.note ? ' title="' + esc(it.note) + '"' : '') + '>'
                + esc(VERDICT_LABEL[verdict] || verdict) + '</span>'
                + '</div></div>';
        }
        list.innerHTML = html;
    }

    // ── 数据 ────────────────────────────────────────────────────────────
    function loadQueue() {
        return getJson('/api/review/queue').then(function (data) {
            state.items = Array.isArray(data.items) ? data.items : [];
            state.stats = data.stats || {};
            state.sources = data.sources || {};
            renderStats();
            renderQueue();
            return data;
        }).catch(function (e) {
            var list = el('reviewQueueList');
            if (list) {
                list.innerHTML = '<div class="text-xs text-gray-500 text-center py-8">队列读取失败：'
                    + esc(e.message || e) + '</div>';
            }
        });
    }

    function loadHistory() {
        return getJson('/api/review/history').then(function (data) {
            renderHistory(data.items);
        }).catch(function () {});
    }

    // ── 默认处置策略 ────────────────────────────────────────────────────
    // manual = 全部人工；vote = 入队即按 6 模型投票定案；llm = 入队即交给本地 LLM。
    // 切换后顺便把当前队列里能自动判的一次性消化掉 —— 用户动这个下拉的意图就是
    // 「别再让我人肉点这几百条」，只改策略不动存量等于没生效。
    function applyPolicy(payload) {
        var sel = el('reviewPolicySelect');
        var hint = el('reviewPolicyHint');
        var policy = (payload && payload.policy) || {};
        var modes = Array.isArray(policy.modes) ? policy.modes : [];
        if (!sel) return;
        sel.value = policy.mode || 'manual';
        if (!hint) return;
        var desc = '';
        for (var i = 0; i < modes.length; ++i) {
            if (modes[i].value === sel.value) desc = modes[i].desc || '';
        }
        hint.textContent = desc || '所有待复核流量都留在队列里，等人逐条点结论。';
    }

    function loadPolicy() {
        return getJson('/api/review/policy').then(applyPolicy).catch(function () {});
    }

    function setPolicy(mode) {
        var sel = el('reviewPolicySelect');
        if (sel) sel.disabled = true;
        return postForm('/api/review/policy', { mode: mode, apply_existing: 'true' })
            .then(function (data) {
                applyPolicy(data);
                state.items = Array.isArray(data.items) ? data.items : [];
                state.stats = data.stats || state.stats;
                renderStats();
                renderQueue();
                loadHistory();
                if (data.handled) {
                    setLlmStatus('已自动处置 ' + data.handled + ' 条', 'ok');
                } else {
                    setLlmStatus('策略已切换', 'ok');
                }
            })
            .catch(function (e) {
                alert('切换默认处置失败：' + (e.message || e));
                loadPolicy();
            })
            .then(function () { if (sel) sel.disabled = false; });
    }

    // ── 人工定案 ────────────────────────────────────────────────────────
    function execVerdict(key, verdict, btn) {
        if (!key || !verdict) return;
        if (btn) btn.disabled = true;
        postForm('/api/review/verdict', { key: key, verdict: verdict })
            .then(function (data) {
                state.items = Array.isArray(data.items) ? data.items : [];
                state.stats = data.stats || {};
                renderStats();
                renderQueue();
                loadHistory();
            })
            .catch(function (e) {
                alert('复核失败：' + (e.message || e));
                if (btn) btn.disabled = false;
            });
    }

    // ── LLM ─────────────────────────────────────────────────────────────
    function setLlmStatus(text, tone) {
        var node = el('reviewLlmStatus');
        if (!node) return;
        node.textContent = text;
        node.classList.toggle('text-cyber-danger', tone === 'error');
        node.classList.toggle('text-cyber-primary', tone === 'ok');
    }

    // 探测本地 LLM：进页面就探一次。离线时弹横幅并说明「怎么起来」，
    // 而不是等用户点「LLM 自主处置」拿到 503 才第一次知道它没开。
    function probeLlm() {
        return getJson('/api/flow/llm-status').then(function (data) {
            var ok = !!(data && data.available);
            var banner = el('reviewLlmBanner');
            var hint = el('reviewLlmBannerHint');
            if (ok) {
                if (banner) banner.classList.add('hidden');
                setLlmStatus('LLM 就绪 · ' + (data.model || '未知模型'), 'ok');
            } else {
                if (banner) banner.classList.remove('hidden');
                if (hint) {
                    hint.textContent = (data && data.hint)
                        ? data.hint
                        : '在 LM Studio 里打开 Server（默认 http://127.0.0.1:1234/v1），或用 DLIDS_LLM_URL 指向其它 OpenAI 兼容服务。';
                }
                setLlmStatus('本地 LLM 未在线' + (data && data.base_url ? ' · ' + data.base_url : ''), 'error');
            }
            state.llmAvailable = ok;
            return ok;
        }).catch(function (e) {
            var banner = el('reviewLlmBanner');
            if (banner) banner.classList.remove('hidden');
            setLlmStatus('探测失败：' + (e && e.message ? e.message : e), 'error');
            state.llmAvailable = false;
            return false;
        });
    }

    function showAiResults(payload) {        var box = el('reviewAiContent');
        var meta = el('reviewAiMeta');
        if (!box) return;
        var results = (payload && payload.results) || [];
        if (meta) {
            meta.textContent = (payload && payload.model ? payload.model : '模型未知')
                + (payload && payload.applied ? ' · 已落库' : ' · 仅建议');
        }
        if (!results.length) {
            box.innerHTML = '<span class="text-gray-500">队列里没有可判断的条目。</span>';
            return;
        }
        var html = '';
        for (var i = 0; i < results.length; ++i) {
            var r = results[i] || {};
            if (r.error) {
                html += '<div class="review-ai-row"><span class="text-cyber-danger">判断失败：</span>'
                    + esc(r.error) + '</div>';
                continue;
            }
            var verdict = String(r.verdict || '');
            var conf = Number(r.confidence);
            html += '<div class="review-ai-row">'
                + '<div><b>' + esc(VERDICT_LABEL[verdict] || (verdict || '无法判断')) + '</b>'
                + (isFinite(conf) ? ' · 自评 ' + (conf * 100).toFixed(0) + '%' : '')
                + (r.elapsed ? ' · ' + esc(r.elapsed) + 's' : '') + '</div>'
                + '<div class="mt-1 text-gray-400">' + esc(r.reason || '（模型未给理由）') + '</div>'
                + '</div>';
        }
        box.innerHTML = html;
    }

    function runLlm(apply) {
        if (state.busy) return;
        // 已知离线时先探一次再决定要不要发请求：直接发会拿到 503，
        // 用户看到的是一个信息量很低的失败提示。
        if (state.llmAvailable === false) {
            probeLlm().then(function (ok) {
                if (!ok) {
                    setLlmStatus('本地 LLM 未在线，无法' + (apply ? '自主处置' : '给建议'), 'error');
                    return;
                }
                startLlmRequest(apply);
            });
            return;
        }
        startLlmRequest(apply);
    }

    function startLlmRequest(apply) {
        state.busy = true;
        var limitNode = el('reviewLlmLimit');
        var limit = limitNode ? Number(limitNode.value) || 5 : 5;
        setLlmStatus(apply ? '让 LLM 自主处置中…' : '让 LLM 给建议中…');
        var targetKeys = state.filter === 'all' ? ''
            : filteredItems().slice(0, limit).map(function (it) { return it.key; }).join(',');
        postForm('/api/review/auto', {
            keys: targetKeys,
            limit: limit,
            apply: apply ? 'true' : 'false'
        }).then(function (data) {
            showAiResults(data);
            setLlmStatus('完成 · ' + ((data.results || []).length) + ' 条', 'ok');
            state.items = Array.isArray(data.items) ? data.items : state.items;
            state.stats = data.stats || state.stats;
            renderStats();
            renderQueue();
            if (apply) loadHistory();
        }).catch(function (e) {
            setLlmStatus('LLM 调用失败：' + (e.message || e), 'error');
            var box = el('reviewAiContent');
            if (box) {
                box.innerHTML = '<span class="text-cyber-danger">' + esc(e.message || e) + '</span>';
            }
        }).then(function () {
            state.busy = false;
        });
    }

    // ── 绑定 ────────────────────────────────────────────────────────────
    function bindEvents() {
        // 幂等标记必须挂在「这一页的 DOM」上，不能用模块级变量：
        // 站内换页时 router 会把 .app-content 整块换掉，旧节点的监听器全没了，
        // 而模块变量还是 true —— 新 DOM 上就再也不会绑定，按钮点了没反应。
        var scope = el('reviewQueueList') || el('reviewPolicySelect');
        if (!scope) return;
        var pageRoot = (scope.closest && scope.closest('.app-content')) || scope;
        if (pageRoot.dataset.reviewBound === '1') return;
        pageRoot.dataset.reviewBound = '1';

        var list = el('reviewQueueList');
        if (list) {
            list.addEventListener('click', function (ev) {
                // 多选复选框：只切换选中态，不触发任何判定
                var check = ev.target.closest('[data-review-check]');
                if (check) {
                    var checkKey = check.getAttribute('data-review-check') || '';
                    if (checkKey) {
                        if (check.checked) state.selected[checkKey] = true;
                        else delete state.selected[checkKey];
                        syncSelectionUi();
                    }
                    return;
                }
                var aiBtn = ev.target.closest('[data-review-ai]');
                if (aiBtn) {
                    var wrap = aiBtn.closest('.review-item-actions');
                    var aiKey = wrap ? wrap.getAttribute('data-key') : '';
                    if (!aiKey) return;
                    setLlmStatus('让 LLM 判断这一条…');
                    postForm('/api/review/auto', { keys: aiKey, limit: 1, apply: 'false' })
                        .then(function (data) {
                            showAiResults(data);
                            setLlmStatus('完成 · 1 条', 'ok');
                        })
                        .catch(function (e) { setLlmStatus('LLM 调用失败：' + (e.message || e), 'error'); });
                    return;
                }
                var btn = ev.target.closest('.review-action');
                if (!btn) return;
                var row = btn.closest('.review-item-actions');
                var key = row ? row.getAttribute('data-key') : '';
                var verdict = btn.getAttribute('data-verdict');
                execVerdict(key, verdict, btn);
            });
        }

        var filter = el('reviewSourceFilter');
        if (filter) {
            filter.addEventListener('click', function (ev) {
                var chip = ev.target.closest('[data-review-source]');
                if (!chip) return;
                state.filter = chip.getAttribute('data-review-source') || 'all';
                Array.prototype.forEach.call(
                    filter.querySelectorAll('[data-review-source]'),
                    function (node) {
                        node.classList.toggle('is-active',
                            node.getAttribute('data-review-source') === state.filter);
                    });
                renderQueue();
            });
        }

        var suggestBtn = el('reviewLlmSuggest');
        if (suggestBtn) suggestBtn.addEventListener('click', function () { runLlm(false); });
        var applyBtn = el('reviewLlmApply');
        if (applyBtn) applyBtn.addEventListener('click', function () { runLlm(true); });
        var retryBtn = el('reviewLlmRetry');
        if (retryBtn) {
            retryBtn.addEventListener('click', function () {
                setLlmStatus('正在重新探测…');
                probeLlm().then(function (ok) {
                    if (ok) setLlmStatus(el('reviewLlmStatus') ? el('reviewLlmStatus').textContent : 'LLM 就绪', 'ok');
                });
            });
        }

        var selectAll = el('reviewSelectAll');
        if (selectAll) {
            selectAll.addEventListener('change', function () { setSelectionForVisible(selectAll.checked); });
        }
        Array.prototype.forEach.call(document.querySelectorAll('[data-review-batch]'), function (btn) {
            btn.addEventListener('click', function () {
                var mode = btn.getAttribute('data-review-batch');
                if (mode === 'llm') runBatchLlm();
                else runBatch(mode);
            });
        });

        var policySel = el('reviewPolicySelect');
        if (policySel) {
            policySel.addEventListener('change', function () { setPolicy(policySel.value); });
        }

        var clearQueueBtn = el('reviewQueueClear');
        if (clearQueueBtn) {
            clearQueueBtn.addEventListener('click', function () {
                if (!confirm('清空待复核队列？已复核的结论会保留。')) return;
                fetch('/api/review/queue', { method: 'DELETE', credentials: 'same-origin' })
                    .then(function (r) { return r.json(); })
                    .then(function (data) {
                        state.items = [];
                        state.stats = data.stats || state.stats;
                        renderStats();
                        renderQueue();
                    })
                    .catch(function (e) { alert('清空失败：' + (e.message || e)); });
            });
        }

        var clearHistoryBtn = el('reviewClearHistory');
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener('click', function () {
                if (!confirm('清空已复核记录？统计里的已处理量会归零。')) return;
                fetch('/api/live/review', { method: 'DELETE', credentials: 'same-origin' })
                    .then(function (r) { return r.json(); })
                    .then(function (data) {
                        state.stats = data.stats || state.stats;
                        renderStats();
                        renderHistory([]);
                    })
                    .catch(function (e) { alert('清空失败：' + (e.message || e)); });
            });
        }

        var exportBtn = el('reviewExportCalib');
        if (exportBtn) {
            exportBtn.addEventListener('click', function () {
                // 该接口返回的是 CSV 附件（Content-Disposition: attachment），不是 JSON。
                // 之前这里先 getJson() 再 location.href：前者把 CSV 当 JSON 解析必然失败，
                // 于是用户每次点导出都先弹一个「导出失败」，而下载其实已经开始。
                // 用隐藏 <a download> 触发即可，既不解析响应，也不会把当前页导航走。
                var a = document.createElement('a');
                a.href = '/api/live/export-calib';
                a.download = '';
                a.style.display = 'none';
                document.body.appendChild(a);
                a.click();
                window.setTimeout(function () { a.remove(); }, 0);
            });
        }
    }

    function startAutoRefresh() {
        stopAutoRefresh();
        state.timer = window.setInterval(function () {
            if (document.hidden) return;
            loadQueue();
        }, REFRESH_MS);
    }

    function stopAutoRefresh() {
        if (state.timer) {
            window.clearInterval(state.timer);
            state.timer = null;
        }
    }

    function init() {
        if (!el('reviewQueueList')) return;      // 不是复核中心
        bindEvents();
        // 后端探测本地 LLM 时要逐个候选地址试连，首次可能耗时数秒。
        // 这段时间里如果什么都不写，界面会一直停在模板初值「未探测」，
        // 用户会以为系统根本没提示。先给一个明确的过程态。
        setLlmStatus('正在探测本地 LLM…');
        probeLlm();
        loadPolicy();
        loadQueue().then(function () { loadHistory(); });
        startAutoRefresh();
    }

    function teardown() {
        stopAutoRefresh();
    }

    window.reviewImpl = {
        init: init, teardown: teardown, loadQueue: loadQueue, loadPolicy: loadPolicy,
        probeLlm: probeLlm
    };
})();
