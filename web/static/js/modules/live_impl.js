/* live capture and polling implementations */
(function(){
    window.liveImpl = window.liveImpl || {};

    // 与 main.js 的 LIVE_POLL_INTERVAL_MS 保持一致：实时抓包轮询节奏（毫秒）
    const LIVE_POLL_INTERVAL_MS = 1500;

    function ensurePanelTransition(el) {
        if (!el || el.dataset.panelTransitionReady === '1') return;
        el.dataset.panelTransitionReady = '1';
        // 显隐只做透明度：位移入场在密集控制台里是噪声，不承载任何信息。
        // 时长与缓动一律取自 CSS 的动效规范变量，不在 JS 里另造一套。
        el.style.transition = 'opacity var(--mo-base) var(--mo-ease)';
        el.style.willChange = 'opacity';
    }

    function setPanelVisible(el, visible, options) {
        if (!el) return;
        ensurePanelTransition(el);
        const prefersReducedMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

        // Track desired state and per-transition id to avoid races from overlapping timers/listeners
        el._panelDesiredVisible = !!visible;
        el._panelTransitionId = (el._panelTransitionId || 0) + 1;
        const myTransitionId = el._panelTransitionId;

        // clear existing timers/listeners for previous transitions
        if (el._panelHideTimer) {
            clearTimeout(el._panelHideTimer);
            el._panelHideTimer = null;
        }
        if (el._panelShowTimer) {
            clearTimeout(el._panelShowTimer);
            el._panelShowTimer = null;
        }
        if (el._panelOnEnd) {
            try { el.removeEventListener('transitionend', el._panelOnEnd); } catch(e){}
            el._panelOnEnd = null;
        }

        if (visible) {
            el.classList.remove('hidden');
            el.setAttribute('aria-hidden', 'false');
            if (prefersReducedMotion) {
                el.style.opacity = '';
                el.style.transform = '';
                el.style.pointerEvents = '';
                return;
            }
            el.style.opacity = '0';
            el.style.pointerEvents = 'none';
            void el.offsetHeight;
            requestAnimationFrame(() => {
                // If another transition started meanwhile, bail
                if (el._panelTransitionId !== myTransitionId || el._panelDesiredVisible !== true) return;
                el.style.opacity = '1';
                el.style.pointerEvents = '';
            });
            el._panelShowTimer = setTimeout(() => {
                // Only finalize if this is still the active transition
                if (el._panelTransitionId === myTransitionId && el._panelDesiredVisible === true) {
                    el.style.opacity = '';
                    el.style.transform = '';
                    el.style.pointerEvents = '';
                }
                el._panelShowTimer = null;
            }, 260);
            return;
        }

        el.setAttribute('aria-hidden', 'true');
        if (prefersReducedMotion) {
            el.classList.add('hidden');
            el.style.opacity = '';
            el.style.transform = '';
            el.style.pointerEvents = '';
            return;
        }
        el.style.opacity = '0';
        el.style.pointerEvents = 'none';

        const finish = () => {
            // Only hide if this transition is still current and desired state is hidden
            if (el._panelTransitionId === myTransitionId && el._panelDesiredVisible === false) {
                el.classList.add('hidden');
                el.style.opacity = '';
                el.style.transform = '';
                el.style.pointerEvents = '';
            }
        };

        const onEnd = (evt) => {
            if (evt && evt.target !== el) return;
            // ensure we remove only our handler
            if (el._panelOnEnd === onEnd) el._panelOnEnd = null;
            try { el.removeEventListener('transitionend', onEnd); } catch(e){}
            if (el._panelHideTimer) {
                clearTimeout(el._panelHideTimer);
                el._panelHideTimer = null;
            }
            finish();
        };
        el._panelOnEnd = onEnd;
        el.addEventListener('transitionend', onEnd);
        el._panelHideTimer = setTimeout(() => {
            // Only finalize if still current
            if (el._panelOnEnd === onEnd) {
                try { el.removeEventListener('transitionend', onEnd); } catch(e){}
                el._panelOnEnd = null;
            }
            el._panelHideTimer = null;
            finish();
        }, 260);
    }

    // DOM 搬移（把拓扑图/趋势图在左右列之间挪位置）本身不需要入场动画：
    // 位置变化已经说明了"视图换了"，再叠一层位移只会让人看不清谁动过。
    // 这里只做一次很轻的淡入，时长同样取自 CSS 的动效规范。
    function animateMovedElement(el) {
        if (!el) return;
        ensurePanelTransition(el);
        const prefersReducedMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
        if (prefersReducedMotion) return;
        el.style.opacity = '0';
        void el.offsetHeight;
        requestAnimationFrame(() => {
            el.style.opacity = '1';
        });
        if (el._panelResetTimer) clearTimeout(el._panelResetTimer);
        el._panelResetTimer = setTimeout(() => {
            el.style.opacity = '';
            el._panelResetTimer = null;
        }, 260);
    }

    function animatePageShift(direction) {
        return;
    }

    // ---------- 面板位置快照 ----------
    // 实时抓包页需要把 #liveFlowGraphWrap / #trendChartPanel 在左右列之间来回搬移。
    // 这里不再假设“节点必须是某个容器的直接子元素”（否则模板外面包一层 div 就会让切换失效），
    // 而是在第一次搬移之前记录被搬移节点原始的 parentNode 与 nextSibling，之后一律按记录还原。
    // nextSibling 可能是文本节点（合法且常用的 insertBefore 参照点），也可能为 null，
    // 表示搬移前它本来就是父容器的最后一个子节点。
    let _placement = null;

    // 幂等记录：同一节点只记录第一次搬移之前的位置
    function rememberPlacement(key, node) {
        if (!node || !node.parentNode) return;
        if (!_placement) _placement = {};
        if (_placement[key] && _placement[key].node === node) return;
        _placement[key] = { node: node, parent: node.parentNode, ref: node.nextSibling || null };
    }

    // 当前是否已经偏离记录位置（原始父容器已不在文档中时视为无需还原）
    function placementNeedsRestore(key) {
        const entry = _placement && _placement[key];
        if (!entry || !entry.node) return false;
        const parent = entry.parent;
        if (!parent || (document.contains && !document.contains(parent))) return false;
        return !(entry.node.parentNode === parent && entry.node.nextSibling === entry.ref);
    }

    // 按记录还原。参照点若已不再是该父容器的子节点，退化为 appendChild，绝不抛 NotFoundError。
    function restorePlacement(key) {
        const entry = _placement && _placement[key];
        if (!entry || !entry.node) return false;
        const node = entry.node;
        const parent = entry.parent;
        if (!parent || (document.contains && !document.contains(parent))) return false;
        let ref = entry.ref;
        if (ref && ref.parentNode !== parent) ref = null;
        try {
            parent.insertBefore(node, ref);
        } catch (e) {
            try { parent.appendChild(node); } catch (e2) { console.warn('restorePlacement(' + key + ') failed', e2); }
        }
        return true;
    }

    // expose helpers so other modules can reuse the same transition behavior
    try {
        if (typeof window !== 'undefined') {
            window.setPanelVisible = window.setPanelVisible || setPanelVisible;
            window.animateMovedElement = window.animateMovedElement || animateMovedElement;
        }
    } catch(e) {}

    // 卡片入场不再由 JS 注入 @keyframes：统一走 styles.css 的 .mo-in（只淡入、不位移）。

    // 页面默认呈现上传视图：作为 tab 切换的视图守卫初值，避免重复点击重放进入动画
    window.dlidsView = window.dlidsView || 'upload';

    // initialize persisted clear timestamp (seconds)
    window.liveLogsClearedAt = Number(window.localStorage && window.localStorage.getItem && window.localStorage.getItem('liveLogsClearedAt')) || window.liveLogsClearedAt || 0;

    window.liveImpl.startLive = async function() {
        // 开始/停止按钮在顶栏，全站只有这一颗
        const startBtn = document.getElementById('liveToggleBtn');
        try {
            if (startBtn) startBtn.disabled = true;
            const iface = window.getSelectedInterface ? window.getSelectedInterface() : '';
            if (!iface) { alert('请先在「实时抓包」页选择网卡'); return; }
            window.liveCaptureModel = window.currentActiveModel;
            const fd = new FormData(); fd.append('iface', iface); fd.append('model_type', window.currentActiveModel);
            const r = await fetch('/api/live/start', { method: 'POST', body: fd });
            const j = await (r.ok ? r.json() : r.text().then(t=>({ error: t }))); 
            if (j && (j.status === 'started' || j.status === 'already_running')) {
                try { if (typeof window.setLiveRunningUI === 'function') window.setLiveRunningUI(true); } catch(e){}
                try { if (typeof window.pollLive === 'function') { if (window.livePoller) clearInterval(window.livePoller); window.pollLive(); window.livePoller = setInterval(window.pollLive, LIVE_POLL_INTERVAL_MS); } } catch(e){}
            } else {
                const msg = j && (j.message || j.error) ? (j.message || j.error) : JSON.stringify(j);
                alert('启动失败: ' + msg);
            }
        } catch (e) { alert('启动失败: ' + (e && e.message ? e.message : String(e))); } finally { if (startBtn) startBtn.disabled = false; }
    };

    // 信息架构重构：上传检测与实时抓包合成了「检测工作台」一页里的两个子视图
    // （2026-11-01 起侧栏只留一个入口），「切换」由顶部的 tab 完成，
    // 因此原先那套「把 trendChartPanel / liveFlowGraphWrap 在两个列之间搬来搬去」的 DOM 搬移
    // 逻辑整体作废。历史上的动画重播、上传卡收不干净、详情数据串包都源于这段搬移，
    // 删除后这些症状从根上不再可能出现。

    // 顶栏的抓包状态胶囊 + 唯一那颗开始/停止按钮，全部由这里更新。
    // 状态没变化就直接返回：pollLive 每 1.5 秒会调用一次，重复写 DOM 会让按钮一直闪。
    // 例外：无刷新换页后实时页卡片上的 #liveStatus 是新的 DOM 节点，上面那条记录的那个
    // 已经被丢掉了。只比「状态值」会把这次重画一起挡掉，于是重进实时页标题还写着「停止」——
    // 所以这里额外比一次「上次画的是哪个节点」。
    window.liveImpl.setLiveRunningUI = function(running, options) {
        try {
            window.liveUiState = window.liveUiState || {};
            const nextRunning = !!running;
            const prevRunning = !!window.liveUiState.running;
            const liveStatus = document.getElementById('liveStatus');
            const domChanged = liveStatus !== window.__liveStatusPaintedEl;
            if (!(options && options.force) && !domChanged && prevRunning === nextRunning) return;
            window.liveUiState.running = nextRunning;
            window.__liveStatusPaintedEl = liveStatus;

            if (liveStatus) {
                // liveStatus 现在是「状态点 + 文字」的组合节点，直接写 innerText 会把状态点一起抹掉，
                // 所以只写内层文字、用 is-running 类切换绿点与呼吸。liveStatusText 兜底：老结构里没有它就退回整段写。
                liveStatus.classList.toggle('is-running', nextRunning);
                const liveStatusText = document.getElementById('liveStatusText');
                if (liveStatusText) liveStatusText.innerText = nextRunning ? '运行中' : '停止';
                else liveStatus.innerText = nextRunning ? '运行中' : '停止';
            }

            // 顶栏常驻状态：在任意页面都能看到抓包是否在跑
            const captureState = document.getElementById('captureState');
            if (captureState) captureState.classList.toggle('is-running', nextRunning);
            const captureText = document.getElementById('captureText');
            if (captureText) captureText.innerText = nextRunning ? '抓包中' : '未抓包';

            // 开始/停止合并为一个按钮：切换文案、图标与配色，不再隐藏/显示两颗按钮
            const toggleBtn = document.getElementById('liveToggleBtn');
            if (toggleBtn) {
                toggleBtn.dataset.running = nextRunning ? 'true' : 'false';
                toggleBtn.classList.toggle('btn-primary', !nextRunning);
                toggleBtn.classList.toggle('btn-danger', nextRunning);
                toggleBtn.disabled = false;
                const labelEl = document.getElementById('liveToggleLabel');
                if (labelEl) labelEl.innerText = nextRunning ? '停止抓包' : '开始抓包';
                const iconEl = document.getElementById('liveToggleIcon');
                if (iconEl) iconEl.className = nextRunning ? 'fa-solid fa-stop mr-1.5' : 'fa-solid fa-play mr-1.5';
            }
            try { if (typeof window.persistUiState === 'function') window.persistUiState(); } catch (e) {}
        } catch (e) { console.warn('liveImpl.setLiveRunningUI failed', e); }
    };

    window.liveImpl.stopLive = async function() {
        const toggleBtn = document.getElementById('liveToggleBtn');
        try {
            if (toggleBtn) toggleBtn.disabled = true;
            await fetch('/api/live/stop', { method: 'POST' });
        } catch (e) {
            console.warn('liveImpl.stopLive failed', e);
        } finally {
            try { if (typeof window.setLiveRunningUI === 'function') window.setLiveRunningUI(false); } catch (e) {}
            if (toggleBtn) toggleBtn.disabled = false;
        }
    };

    // 一颗按钮承担开始与停止：按当前状态决定动作，用户不用在两颗按钮之间找
    window.liveImpl.toggleLiveCapture = function() {
        const running = !!(window.liveUiState && window.liveUiState.running);
        if (running) return window.liveImpl.stopLive();
        return window.liveImpl.startLive();
    };

    // ---------- 增量轮询游标 ----------
    // lastFlowId 是本轮之前已取到的最大流 id：首轮不带游标（等价旧的全量请求），
    // 之后每轮带 since_id，只取新流。seenFlowIds / knownFlows 用于按 id 去重，
    // 保证游标边界上的流不会被重复渲染，也让某轮返回空数组时已有内容保持不变。
    let lastFlowId = 0;
    const seenFlowIds = new Set();
    const knownFlows = [];
    const MAX_KNOWN_FLOWS = 500;   // 本地缓存上限，避免长会话内存无限增长
    const MAX_LIVE_ROWS = 200;     // 交给图表/统计的行数上限
    let stoppedGraphRestored = false;   // 停止状态下是否已用本地记录补画过辐射图

    // 双向五元组归一：与后端 src/preprocessing/flow_features.py 的 _flow_key 同一条规则
    // （IP 字符串字典序小的一侧固定为前向），让同一条连接的正反向算作同一条流。
    // 记录表和「威胁类型分布」卡片必须共用这个键，否则卡片说 41 条、记录表却有八十多行。
    function flowConnKey(f) {
        if (!f) return '';
        const a = String(f.src || f.src_ip || f.src_addr || '0');
        const b = String(f.dst || f.dst_ip || f.dst_addr || '0');
        const ap = Number(f.sport || f.src_port) || 0;
        const bp = Number(f.dport || f.dst_port) || 0;
        const proto = String(f.proto || f.protocol || '');
        return a <= b
            ? a + ':' + ap + '>' + b + ':' + bp + '|' + proto
            : b + ':' + bp + '>' + a + ':' + ap + '|' + proto;
    }

    // 把记录表收敛成「每个连接一条最新」。
    // 实时链路里的一条「流」是连接状态，不是事件流水：后端 LiveFlowTracker 用双向五元组
    // 做键，同一连接空闲超时后重开会覆盖掉上一段，所以卡片（按 tracker 统计）永远只算
    // 每连接一条；而记录表是每出现一次就 push 一行，抓久了会滚成几十上百行，
    // 于是「卡片 41 条 / 表格 128 行」这种对不上的数字就出现了。这里按同一个键收敛。
    // 返回是否真的删掉了行（调用方据此决定要不要重渲染）。
    function compactLiveRecordStore() {
        const store = window.recordStores && window.recordStores.live;
        if (!Array.isArray(store) || store.length < 2) return false;
        const seen = Object.create(null);
        const kept = [];
        for (let i = store.length - 1; i >= 0; i--) {
            const entry = store[i];
            const d = entry && entry.data;
            if (!d) { kept.push(entry); continue; }
            const k = flowConnKey(d);
            if (seen[k]) continue;
            seen[k] = 1;
            kept.push(entry);
        }
        if (kept.length === store.length) return false;
        kept.reverse();   // 恢复「旧→新」，与表里最新的在最上面保持一致
        window.recordStores.live = kept;
        return true;
    }

    function isFlowId(value) {
        return value !== undefined && value !== null && value !== '' && Number.isFinite(Number(value));
    }

    // 把新流按 id 去重后并入本地缓存；不返回任何值，空数组即空操作
    function rememberIncomingFlows(flows) {
        if (!Array.isArray(flows)) return;
        flows.forEach((f) => {
            if (!f) return;
            if (isFlowId(f.id)) {
                const key = String(f.id);
                if (seenFlowIds.has(key)) return;
                seenFlowIds.add(key);
            }
            knownFlows.push(f);
        });
        // 同一连接只留最新一段：后端 LiveFlowTracker 用双向五元组做键，
        // 一条连接空闲超时后重开会覆盖掉上一段，所以它那边每条连接永远只有一条。
        // 这里如果按事件累积，liveRows 的条数就会跟卡片越差越远（实测卡片 41、记录表 128）。
        if (knownFlows.length > 1) {
            const byConn = Object.create(null);
            const merged = [];
            knownFlows.forEach((f) => {
                const k = flowConnKey(f);
                if (!k) { merged.push(f); return; }
                byConn[k] = f;          // 后到的覆盖先到的，保留最新状态
            });
            Object.keys(byConn).forEach((k) => merged.push(byConn[k]));
            if (merged.length !== knownFlows.length) {
                knownFlows.length = 0;
                merged.forEach((f) => knownFlows.push(f));
            }
        }
        if (knownFlows.length > MAX_KNOWN_FLOWS) {
            knownFlows.splice(0, knownFlows.length - MAX_KNOWN_FLOWS).forEach((f) => {
                if (f && isFlowId(f.id)) seenFlowIds.delete(String(f.id));
            });
        }
    }

    // 从本地记录把实时流原始数据取出来重建辐射图。
    // 记录存储里每条形如 { name, size, time, data, options }，流数据在 data.flows 上。
    function restoreGraphFromRecords() {
        const records = (window.recordStores && Array.isArray(window.recordStores.live)) ? window.recordStores.live : [];
        if (!records.length) return 0;
        if (window.liveGraph && typeof window.liveGraph.restoreFromRecords === 'function') {
            return window.liveGraph.restoreFromRecords(records) || 0;
        }
        const flows = [];
        records.forEach((r) => {
            const d = (r && typeof r.data === 'object') ? r.data : null;
            if (d && Array.isArray(d.flows)) d.flows.forEach((f) => { if (f) flows.push(f); });
        });
        if (!flows.length) return 0;
        if (typeof window.renderLiveFlowGraph === 'function') window.renderLiveFlowGraph(flows);
        return flows.length;
    }

    function reviewEscape(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    // 待复核队列渲染。入队判据在后端（档位 + 未知攻击），这里只负责显示：
    // 每条给出五元组、预测类型、置信度和档位标签，点「AI 分析」仍走记录表那套逻辑。
    // 导出本机流量当校准数据。导出的 CSV = 当前抓到的流（一律标 Benign）+ 六类均衡锚点，
    // 后端拼好直接下载，省得用户手动拼表。
    function bindExportCalib() {
        const btn = document.getElementById('liveExportCalib');
        if (!btn || btn.dataset.bound === '1') return;
        btn.dataset.bound = '1';
        btn.addEventListener('click', function () {
            btn.disabled = true;
            fetch('/api/live/export-calib', { credentials: 'same-origin' })
                .then(function (r) {
                    if (!r.ok) {
                        return r.json().then(function (j) { throw new Error(j.error || ('HTTP ' + r.status)); });
                    }
                    const liveRows = r.headers.get('X-Live-Rows') || '?';
                    const anchorRows = r.headers.get('X-Anchor-Rows') || '0';
                    return r.blob().then(function (b) {
                        const url = URL.createObjectURL(b);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'live_calib.csv';
                        document.body.appendChild(a);
                        a.click();
                        a.remove();
                        URL.revokeObjectURL(url);
                        alert('已导出：本机流量 ' + liveRows + ' 行 + 锚点样本 ' + anchorRows + ' 行。\n'
                            + '下一步到「流量分类」页上传这份 CSV 做环境校准。');
                    });
                })
                .catch(function (e) { alert('导出失败：' + (e.message || e)); })
                .then(function () { btn.disabled = false; });
        });
    }

    function renderReviewQueue(items, stats) {
        const list = document.getElementById('liveReviewList');
        const hint = document.getElementById('liveReviewHint');
        if (!list) return;
        const arr = Array.isArray(items) ? items : [];
        // 页头同时给待办量与已处理量：只显示「N 条」的话，复核完的流从列表消失，
        // 计数又变了但看不出是「处理掉了」还是「被新一轮抓包冲掉了」。
        if (hint) {
            const done = stats && isFinite(Number(stats.reviewed)) ? Number(stats.reviewed) : 0;
            hint.textContent = arr.length + ' 待复核 · ' + done + ' 已处理';
        }
        if (!arr.length) {
            list.innerHTML = '<div class="text-xs text-gray-500 text-center py-8">暂无待复核流量</div>';
            return;
        }
        let html = '';
        for (let i = 0; i < arr.length; ++i) {
            const it = arr[i] || {};
            const tierText = it.is_unknown ? '未知攻击' : (it.tier === 'manual' ? '人工复核' : '待确认');
            const tierClass = it.is_unknown ? 'is-unknown' : (it.tier === 'manual' ? 'is-manual' : 'is-review');
            const conf = Number(it.confidence);
            const confText = isFinite(conf) ? (conf * 100).toFixed(1) + '%' : '--';
            const five = reviewEscape(it.src_ip) + ':' + (it.src_port || '-') + ' → ' + reviewEscape(it.dst_ip) + ':' + (it.dst_port || '-');
            const k = reviewEscape(it.key || '');
            const ood = Number(it.ood_distance);
            html += '<div class="review-item">'
                + '<div class="review-item-head">'
                + '<div class="review-item-main">'
                + '<div class="review-item-flow" title="' + five + '">' + five + '</div>'
                + '<div class="review-item-meta">' + reviewEscape(it.predicted || '-')
                + ' · 置信度 ' + confText
                + (isFinite(ood) ? ' · 离群 ' + ood.toFixed(1) : '')
                + '</div>'
                + '</div>'
                + '<span class="review-item-tier ' + tierClass + '">' + tierText + '</span>'
                + '</div>'
                + '<div class="review-item-actions" data-key="' + k + '">'
                + '<button type="button" class="review-action is-attack" data-verdict="malicious">确认攻击</button>'
                + '<button type="button" class="review-action is-benign" data-verdict="benign">确认误报</button>'
                + '<button type="button" class="review-action is-ignore" data-verdict="ignore">忽略</button>'
                + '</div>'
                + '</div>';
        }
        list.innerHTML = html;
    }

    // 复核结论落库后立刻重画队列。事件用委托绑在 #liveReviewList 上：
    // 每次 renderReviewQueue 都会整体替换 innerHTML，绑在按钮上的监听会跟着丢。
    function bindReviewActions() {
        const list = document.getElementById('liveReviewList');
        if (!list || list.dataset.reviewBound === '1') return;
        list.dataset.reviewBound = '1';
        list.addEventListener('click', function (ev) {
            const btn = ev.target.closest('.review-action');
            if (!btn) return;
            const wrap = btn.closest('.review-item-actions');
            const key = wrap ? wrap.getAttribute('data-key') : '';
            const verdict = btn.getAttribute('data-verdict');
            if (key && verdict) execReview(key, verdict, btn);
        });
    }

    async function execReview(key, verdict, btn) {
        const wrap = btn ? btn.closest('.review-item-actions') : null;
        if (wrap) {
            wrap.querySelectorAll('button').forEach(function (b) { b.disabled = true; });
        }
        try {
            const body = new URLSearchParams();
            body.set('key', key);
            body.set('verdict', verdict);
            const r = await fetch('/api/live/review', { method: 'POST', body: body });
            if (!r.ok) throw new Error('HTTP ' + r.status);
            const j = await r.json();
            renderReviewQueue(j.review_queue, j.stats);
            bindReviewActions();
        } catch (e) {
            if (wrap) {
                wrap.querySelectorAll('button').forEach(function (b) { b.disabled = false; });
            }
            console.warn('live review failed', e);
        }
    }

    window.liveImpl.pollLive = async function() {        try {
            const hasCursor = lastFlowId > 0;
            const liveStatsUrl = hasCursor
                ? `/api/live/stats?since_id=${encodeURIComponent(String(lastFlowId))}`
                : '/api/live/stats';
            const r = await fetch(liveStatsUrl);
            if (!r.ok) return; const j = await r.json(); if (!j) return; const stats = j.stats || {};
            // 新契约的增/全量数组放在 flows；保留对旧字段 recent 的兼容
            const incoming = Array.isArray(j.flows) ? j.flows : (Array.isArray(j.recent) ? j.recent : []);
            const isRunning = !!stats.running;
            // 停止状态下也要先把顶栏状态同步一次，否则刷新页面后顶栏会一直停在初始文案
            try { if ('running' in stats && window.setLiveRunningUI) window.setLiveRunningUI(isRunning); } catch(e){}
            if (!isRunning) {
                // 抓包停止后刷新页面：轮询不会再拿到新流，辐射图会一直停在「等待实时流量…」，
                // 与记录表里仍在的历史行自相矛盾。这里用本地记录把它补画一次（只补一次，避免每个轮询周期重放）。
                if (!stoppedGraphRestored) {
                    stoppedGraphRestored = true;
                    try { restoreGraphFromRecords(); } catch (e) { console.warn('restore live graph failed', e); }
                }
                // 注意：这里**不能**直接 return。
                // 抓包停下之后，后端仍然保留本轮的 flows 和六分类 threat_summary
                // （/api/live/stats 在 running=false 时对全部流做汇总，而不是只看活跃流），
                // 刷新或从别的页面切回来，正需要靠这一轮响应把「实时抓包记录」和
                // 「威胁类型分布」补回来。原来这里一律 return，结果就是：
                // 抓包一停（或者刷新时抓包已经停了），整页记录和被清空，看起来像数据丢了。
            } else {
                // 一旦重新跑起来，下一次停止需要再补一次
                stoppedGraphRestored = false;
            }
            // 按 id 去重后并入缓存；本轮为空数组时不清空已有内容
            rememberIncomingFlows(incoming);
            // 用顶层 latest_id 推进游标；缺失/非法时退化为本轮最大 id，保证游标单调不回退
            let nextCursor = Number(j.latest_id);
            if (!Number.isFinite(nextCursor) || nextCursor <= 0) {
                nextCursor = incoming.reduce((max, f) => Math.max(max, Number(f && f.id) || 0), 0);
            }
            if (nextCursor > lastFlowId) lastFlowId = nextCursor;
            const clearCutoff = Number(window.liveLogsClearedAt || 0) || 0;
            const liveRows = knownFlows.filter((f) => {
                if (!clearCutoff) return true;
                const capturedAt = Number(f && f.captured_at ? f.captured_at : 0) || 0;
                return capturedAt > clearCutoff;
            }).slice(-MAX_LIVE_ROWS);
            const uniqueSessions = new Set(); const protoCounts = {}; let inferredBytes = 0; let inferredMalware = 0;
            liveRows.forEach((f) => { const src = f && (f.src || f.src_ip || f.src_addr || '0.0.0.0'); const dst = f && (f.dst || f.dst_ip || f.dst_addr || '0.0.0.0'); const sport = f && (f.sport || f.src_port || 0); const dport = f && (f.dport || f.dst_port || 0); const proto = f && (f.proto || f.protocol || 'unknown'); uniqueSessions.add(`${src}:${sport}->${dst}:${dport}`); protoCounts[String(proto)] = (protoCounts[String(proto)] || 0) + 1; inferredBytes += Number(f && (f.bytes || f.length || 0)) || 0; if (f && (f.is_malicious || (typeof f.predicted_label === 'string' && /malware|malicious/i.test(f.predicted_label)))) inferredMalware += 1; });
            const liveSessionParseReport = { total_packets: stats.total_packets || liveRows.length, accepted_packets: stats.accepted_packets || liveRows.length, session_count: stats.session_count || uniqueSessions.size, parse_errors: stats.parse_errors || 0, non_ipv4: stats.non_ipv4 || 0, malware_count: stats.malware_count || inferredMalware, benign_count: Math.max(0, liveRows.length - (stats.malware_count || inferredMalware)), total_bytes: stats.total_bytes || inferredBytes, protocol_summary: Object.entries(protoCounts).slice(0, 4).map(([k, v]) => `${k}:${v}`).join('，') };
            const livePayloadDist = Array.from({ length: 16 }, (_, i) => { const bucket = liveRows[i % Math.max(1, liveRows.length)] || {}; const seed = `${bucket.src || bucket.src_ip || ''}|${bucket.dst || bucket.dst_ip || ''}|${bucket.sport || bucket.src_port || ''}|${bucket.dport || bucket.dst_port || ''}|${bucket.proto || bucket.protocol || ''}|${bucket.malware_conf || bucket.confidence || 0}`; let hash = 0; for (let j = 0; j < seed.length; j++) hash = ((hash << 5) - hash + seed.charCodeAt(j)) >>> 0; return Math.max(1, ((hash % 11) + 1) * (1 + Math.round(liveRows.length / 10))); });
            // 实时这一路只更新自己的口径：会话数/恶意会话数由后端累计给出，取最大值保证单调不回退。
            // KPI 与饼图统一由 dlidsRenderStats 输出，不再直接写 statTotal/statMalware（那会让两个数字口径不同）。
            try {
                if (window.dlidsStats) {
                    const liveStats = window.dlidsStats.live = window.dlidsStats.live || { sessions: 0, malware: 0 };
                    const sessions = Number(stats.session_count);
                    const malware = Number(stats.malware_count);
                    if (Number.isFinite(sessions)) liveStats.sessions = Math.max(Number(liveStats.sessions) || 0, sessions);
                    if (Number.isFinite(malware)) liveStats.malware = Math.max(Number(liveStats.malware) || 0, malware);
                }
                if (window.dlidsRenderStats) window.dlidsRenderStats();
            } catch(e){}
            // 威胁类型分布：实时链路每个包都喂 78 维累加器，/api/live/stats 返回本批连接的分类。
            // 卡片计数取自逐流明细（window.dlidsThreatFlows），而明细按双向五元组去重并持久化，
            // 与下方「实时抓包记录」同一口径——刷新页面后两者都还在，条数也对得上。
            // 之前这里是「每轮整份覆盖、不落盘」，刷新后卡片立刻空掉，而记录表从
            // localStorage 恢复，于是卡片看起来「没有之前检测的记录」。
            try {
                if (typeof window.dlidsMergeThreatFlows === 'function') {
                    window.dlidsMergeThreatFlows(j.threat_flows, { key: 'live' });
                }
                // meta 也必须走 store 的写入口：window.dlidsThreatMeta 现在是只读 getter，
                // 直接赋值会被静默丢弃（这正是「上传页的数据跑到实时页」那类串页问题的根源）。
                window.dlidsLiveThreatMeta = j.threat_meta || null;
                if (typeof window.dlidsSetThreatMeta === 'function') {
                    window.dlidsSetThreatMeta(j.threat_meta || {}, { key: 'live' });
                }
                renderReviewQueue(j.review_queue, j.review_stats);
                bindExportCalib();
                bindReviewActions();
            } catch (e) {}
            window._liveSeen = window._liveSeen || new Set();
            // limit how many new rows we insert per poll to avoid blocking the UI
            // 注意：这个上限必须 ≥ 后端 /api/live/stats 一次返回的流数（MAX_RECENT=50），
            // 否则首轮只会插进前 25 条，而 since_id 已经跳到最新值，剩下的永远补不上来
            // —— 表现就是「威胁类型分布」卡片说 41 条、下方记录表却只有 20 行。
            const MAX_NEW_ROWS_PER_POLL = 200;
            let newRowsInserted = 0;
            for (let i = incoming.length - 1; i >= 0; --i) {
                const f = incoming[i]; try {
                    const capturedAt = Number(f && f.captured_at ? f.captured_at : 0) || 0;
                    if (clearCutoff && capturedAt <= clearCutoff) continue;
                    const fid = `${f.src || '0'}:${f.sport||0}->${f.dst||'0'}:${f.dport||0}@${Math.floor((f.captured_at||0)*1000)}`;
                    if (window._liveSeen.has(fid)) continue; window._liveSeen.add(fid);
                    const name = `${f.src || '-'}:${f.sport||'-'} -> ${f.dst || '-'}:${f.dport||'-'}`; const time = f.captured_at ? (new Date(f.captured_at * 1000)).toLocaleString('zh-CN') : new Date().toLocaleString('zh-CN'); const norm = Object.assign({}, f);
                    norm.model = f.model || f.model_name || f.detected_model || f.detected_model_name || f.predicted_model || f.pred_label || window.getModelDisplayName(window.liveCaptureModel || window.currentActiveModel);
                    norm.is_malicious = !!f.is_malicious || (typeof f.predicted_label === 'string' && /malware|malicious/i.test(f.predicted_label)) || false;
                    // 置信度必须是「判定那一类的概率」。后端只下发 malware_conf（恶意概率），
                    // 判 Benign 时它 0.16 实际意味着良性置信度 0.84 —— 原样当置信度显示，
                    // 就把「模型很确定这是正常」读成了「模型拿不准」，实时页那些 16% 全是这个误会。
                    const _malConf = Number(f.malware_conf);
                    if (f.malware_conf !== null && f.malware_conf !== undefined && isFinite(_malConf)) {
                        norm.malware_conf = _malConf;
                        norm.confidence = norm.is_malicious ? _malConf : (1 - _malConf);
                    } else {
                        norm.confidence = f.confidence || f.conf || f.score || null;
                    }
                    norm.execution_time = f.execution_time || f.exec_time || f.executionTime || f.duration_ms || f.processing_time_ms || f.elapsed_ms || f.time_ms || null;
                    norm.size = f.size || f.bytes || f.length || f.packet_count || null;
                    // 每条记录只携带自己那一条流：原先赋 liveRows（本轮全部流）会让所有记录的详情完全相同。
                    // payloadDist 置空，交给 modal 按本条流自身派生分布图，不再全体共用同一张。
                    norm.flows = [f]; norm.sessionParseReport = Object.assign({}, liveSessionParseReport, { total_packets: Number(f.packets || f.packet_count || 1) || 1, accepted_packets: Number(f.packets || f.packet_count || 1) || 1, session_count: 1, malware_count: norm.is_malicious ? 1 : 0, benign_count: norm.is_malicious ? 0 : 1, total_bytes: Number(f.bytes || f.length || 0) || 0 }); norm.payloadDist = null; norm.liveCapture = true;
                    window.cacheRecordedRow && window.cacheRecordedRow('live', { name, size: norm.size || '-', time, data: norm, options: { recordKind: 'live' } });
                    if (newRowsInserted < MAX_NEW_ROWS_PER_POLL) {
                        // 总览/记录页（mode=all）也要能看到实时行，只有「上传检测」页不显示实时流量
                        if ((window.recordTableMode || 'upload') !== 'upload') { window.addResultRow && window.addResultRow(name, norm.size || '-', time, norm, { recordKind: 'live' }); }
                        newRowsInserted += 1;
                    }
                    try { if (window.updateTrendChart) window.updateTrendChart(!!norm.is_malicious); } catch(e){}
                } catch (e) { console.warn('pollLive row append failed', e); }
            }
            // 记录表收敛成「每个连接一条」，与「威胁类型分布」卡片同一口径；真的删了行才重渲染。
            // 检测工作台把上传与实时两路记录放在同一页：这里必须按当前 recordTableMode 重画，
            // 并且上传子视图下直接跳过 —— renderRecordedRows 是整表重建，传 'live' 会把
            // 上传记录表刷成实时行。
            try {
                const tableMode = window.recordTableMode || 'upload';
                if (tableMode !== 'upload' && compactLiveRecordStore() && typeof window.renderRecordedRows === 'function') {
                    window.renderRecordedRows(tableMode);
                }
            } catch (e) { console.warn('compact live records failed', e); }
            try { if (typeof window.renderLiveFlowGraph === 'function') window.renderLiveFlowGraph(liveRows); } catch(e){}
        } catch (e) { console.warn('liveImpl.pollLive failed', e); }
    };

    window.liveImpl.renderLiveFlowGraph = function(flows) { if (window.liveGraph && typeof window.liveGraph.renderLiveFlowGraph === 'function') return window.liveGraph.renderLiveFlowGraph(flows); console.warn('liveGraph module not loaded; cannot render flow graph'); };
    window.liveImpl.clearLiveLogs = function() {
        window.liveLogsClearedAt = Date.now() / 1000;
        try { window.localStorage && window.localStorage.setItem && window.localStorage.setItem('liveLogsClearedAt', String(window.liveLogsClearedAt)); } catch(e){}
        if (window.liveGraph && typeof window.liveGraph.clearLiveLogs === 'function') return window.liveGraph.clearLiveLogs();
        if (window.recordStores && Array.isArray(window.recordStores.live)) window.recordStores.live = [];
        if (window._liveSeen) window._liveSeen = new Set();
        const listDom = document.getElementById('liveFlowList');
        if (listDom) listDom.innerHTML = '<p class="text-sm text-gray-400">暂无实时流量</p>';
        // 清的是实时那一路；上传子视图里的表格与它无关，别顺手重画（整表重建）
        try {
            const tableMode = window.recordTableMode || 'upload';
            if (tableMode !== 'upload' && typeof window.renderRecordedRows === 'function') {
                window.renderRecordedRows(tableMode);
            }
        } catch(e){}
    };

    // 进场自动恢复轮询。
    // 抓包状态在后端，前端刷新或从别的页面跳回来时并不知道它还在跑：原来只有点
    // 「开始抓包」那一刻才 setInterval，于是刷新之后记录表和右栏「威胁类型分布」
    // 就停在旧数据上永远不动 —— 表现就是「我明明抓着包，卡片里却没内容」。
    // 这里进场先探一次后端状态，运行中就接着轮询；已经停了也拉一次，
    // 把上一轮留下的流重新算一遍六分类，别让刷新把已有结果清零。
    (function bootLivePolling() {
        // 合并成「检测工作台」后落地地址是 /detect（/live 会 302 过去），两种都要认。
        if (location.pathname !== '/live' && location.pathname !== '/detect') return;
        var start = function () {
            if (typeof window.pollLive !== 'function') { window.setTimeout(start, 200); return; }
            if (window.livePoller) return;
            try { window.pollLive(); } catch (e) { console.warn('initial pollLive failed', e); }
            window.livePoller = setInterval(window.pollLive, LIVE_POLL_INTERVAL_MS);
        };
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', start, { once: true });
        } else {
            window.setTimeout(start, 0);
        }
    })();

})();
