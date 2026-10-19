// main.js - 精简入口，只负责模块加载、事件绑定和委托

(function() {
    'use strict';

    // ---------- 图表动效规范 ----------
    // ECharts 默认入场动画 1000ms：趋势图每 1.5s 轮询刷新一次，数据一更新整张图就重新
    // "长"一遍，看起来像在抽搐。这里统一压到 220ms（与 CSS 的 --mo-base 同档），
    // 缓动固定 cubicOut，并让所有 echarts.init 显式带上这个主题。
    try {
        if (window.echarts && typeof window.echarts.registerTheme === 'function') {
            window.DLIDS_CHART_THEME = 'dlids';
            window.echarts.registerTheme('dlids', {
                animationDuration: 220,
                animationEasing: 'cubicOut',
                animationDurationUpdate: 220,
                animationEasingUpdate: 'cubicOut'
            });
        }
    } catch (e) { console.warn('registerTheme failed', e); }

    // KPI 数字变化时做一次短促高亮（styles.css 的 .mo-flash）；值没变就不动，
    // 避免轮询每秒把同一个数字闪一遍。
    window.dlidsFlash = function (el) {
        if (!el) return;
        el.classList.remove('mo-flash');
        void el.offsetWidth; // 强制重排，让同一个动画可以被连续触发
        el.classList.add('mo-flash');
        setTimeout(function () { el.classList.remove('mo-flash'); }, 460);
    };
    window.dlidsSetKpi = function (el, value) {
        if (!el) return;
        const next = String(value === null || value === undefined ? 0 : value);
        if (el.innerText === next) return;
        el.innerText = next;
        window.dlidsFlash(el);
    };

    // ---------- 全站统一统计口径 ----------
    // 为什么需要这一层：KPI 带与「威胁类型分布」原先各自记账——上传流程写 window.totalScans，
    // 实时轮询又直接把 KPI 改成"抓包累计包数"，于是同一屏里两个数字口径不同、切页面就变。
    // 现在只有一份数据、只有一个渲染函数；任何地方都不许再直接写 statTotal/statMalware/#threatBars。
    window.dlidsStats = window.dlidsStats || {
        upload: { total: 0, malware: 0 },   // 上传检测：按文件计
        live: { sessions: 0, malware: 0 },  // 实时抓包：按会话计（后端给的是累计值，取最大值保证单调）
        threats: {}                         // 六分类威胁累计：{类名: 条数}，按流计
    };

    // OOD 单列项的名字。必须与后端 class_summary['未知攻击'] 一字不差，
    // 否则这里认不出来，会把 OOD 也当六类之一并进总数（重复计数）。
    const OOD_CLASS_NAME = '未知攻击';
    // 界面上的显示名。内部匹配键不变，只有用户看到的字变了：
    // 「未知攻击」会被读成「检测到未知的攻击」，而它其实是「离所有类中心都太远、归类不可信」。
    const OOD_DISPLAY_NAME = '域外流量';

    // 六类的固定顺序与配色。顺序固定让卡片在三页之间的行位置不跳动；
    // 一类一色，同一类不会换个页面就换颜色。
    // 相邻两行色相拉开：绿 160 → 蓝 217 → 紫 258 → 红 350 → 橙 38 → 天蓝 199。
    const THREAT_CLASSES = [
        { name: 'Benign',         color: '#10B981' },
        { name: 'Bot',            color: '#3B82F6' },
        { name: 'DoS-GoldenEye',  color: '#A78BFA' },
        { name: 'DoS-Slowloris',  color: '#F43F5E' },
        { name: 'DDoS-HOIC',      color: '#F59E0B' },
        { name: 'DDoS-LOIC-UDP',  color: '#22D3EE' }
    ];

    function escapeHtml(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    // 威胁类型分布：横向条形（六类 + OOD 单列），不用饼图。
    // 六类的数量常常跨两个数量级（Benign 几十万、LOIC-UDP 几百），饼图在长尾分布下
    // 会挤成一条线；条形既能读出绝对数量也能读出相对占比。
    // 条长按「本行相对最大值」而不是相对总数：Benign 占九成时按总数算其余五类全是零宽，
    // 等于看不见（和饼图同一个毛病）。
    // 卡片 hint 文案：三页各自说明数据来源，避免「这张卡到底在统计哪一路」的歧义。
    // 上传页只统计上传检测的流，实时页只统计实时抓包的流，两边的明细各存各的 localStorage key。
    // 末尾统一标「流级」：这张卡片走 78 维流统计 + 六分类模型，与下方记录表的包级二分类
    // 是两套独立判定，不标出来就会有人问「记录表全是 Benign，这里怎么有攻击类」。
    function dlidsThreatHintText(total, info) {
        // 合并成「检测工作台」之后，/upload 与 /live 都不再是落地地址（后端 302 到
        // /detect?tab=...）。当前在看哪一路由 detectImpl 写在 window.dlidsDetectScope 上，
        // 这里优先读它，其次才退回旧地址判断（兼容直接访问旧链接的缓存页）。
        const p = (window.location && window.location.pathname) || '';
        const scope = window.dlidsDetectScope || (p === '/live' ? 'live' : (p === '/upload' ? 'upload' : ''));
        const unit = total > 0 ? total + ' 个连接' : '暂无连接';
        let base;
        if (scope === 'live') base = '本页实时抓包 · 流级 · ' + unit;
        else if (scope === 'upload') base = '本页上传检测 · 流级 · ' + unit;
        else base = '本次运行累计 · 上传与实时合计 · 流级 · ' + unit;
        info = info || {};
        // 这里只放短标记：卡片标题是「威胁类型分布」，hint 一长就会把标题挤成两行
        // （实测出现过标题折行）。长说明统一进 dlidsThreatHintTitle 拼的 title 提示。
        if (info.truncated) {
            base += ' · 前 ' + info.shown + '/' + info.classified + ' 条';
        }
        if (info.disagree) {
            base += ' · 冲突 ' + info.conflict + ' 条';
        }
        return base;
    }

    // hint 的完整说明（鼠标悬停可见）。把「明细被截断」与「两套模型结论不一致」这两件
    // 必须说清的事放在 title 里，既保证信息不丢，又不占卡片标题那一行的空间。
    function dlidsThreatHintTitle(total, info) {
        info = info || {};
        const lines = ['统计口径：' + dlidsThreatHintText(total, null)];
        // 缺陷 D-05 / L2：明细是按上限截断的（后端 THREAT_FLOW_ROWS_LIMIT=200，
        // 分类总数却可能是 4000）。卡片按明细统计，就必须说清它统计的是哪一部分。
        if (info.truncated) {
            lines.push('明细仅保留前 ' + info.shown + ' 条（共 ' + info.classified + ' 条已分类），卡片按这 ' + info.shown + ' 条统计。');
        }
        // 缺陷 B2/B3：两套模型的结论不一致时必须在同一处写明。
        if (info.disagree) {
            lines.push('包级二分类与流级六分类结论不一致：' + info.conflict + ' 条冲突，请以记录页明细为准。');
        }
        return lines.join('\n');
    }

    window.dlidsRenderThreatBars = function () {
        const host = document.getElementById('threatBars');
        if (!host) return;
        const empty = document.getElementById('threatEmpty');

        // 计数只认逐流明细：卡片必须和下方记录表是同一批连接，
        // 明细按双向五元组合并去重（与后端 _conn_sort_key 同规则），记录表也是同口径。
        //
        // 刻意不再用 dlidsStats.threats 做兜底。以前 summary 兜底会盖住「明细已清空」的事实：
        // 点了清除日志，明细干干净净，下一轮轮询又把 summary 写回统计层，卡片立刻长出数字
        // 却点不开任何一条 —— 用户看到的就是「清除根本没用」。
        // summary 与明细本来就是同一次推理的产物，没有明细说明这批流不该出现在卡片上。
        const flows = (dlidsCurrentBucket() || { flows: [] }).flows;
        // 后端的 threat_meta：分类总数、OOD 是否生效、两套模型的分歧量。
        // 卡片只能统计「拿得到的明细」，但必须把「还有多少条没进明细」写出来（D-05）。
        const tmeta = (window.dlidsThreatMeta || {});
        const _classified = Number(tmeta.flows_classified || 0);
        const _consistency = tmeta.consistency || null;
        const _hintInfo = {
            classified: _classified,
            shown: flows.length,
            truncated: _classified > flows.length,
            disagree: !!(_consistency && _consistency.disagree),
            conflict: _consistency ? Number(_consistency.conflict || 0) : 0
        };
        // 包级二分类明确判为「正常」、六分类却给出攻击类的行 —— 不算它是攻击类。
        //
        // 两条链路用的是完全独立的模型：下方记录表是 28×28 字节图 + 6 个二分类模型投票，
        // 这张卡是 78 维流特征 + 单个六分类模型。同一条连接两边结论打架是常态，
        // 但两套数字并排显示在同一条折叠线上，用户只会读成「系统自相矛盾」。
        // 所以：二分类说正常时，不再把六分类的攻击类展示出来，改计入域外流量。
        //
        // 只在 two_class_malicious 明确为 false 时压制。null 表示这条连接没被二分类
        // 采样过（拿不到结论），拿不到结论不等于结论是「正常」，此时保持六分类原样。
        const suppressed = function (r) {
            return !!(r && r.two_class_malicious === false
                && r.predicted && r.predicted !== 'Benign');
        };
        // 反方向的矛盾（缺陷 B2）：二分类说恶意、六分类却说 Benign。
        // 这些行以前既不进任何攻击类、也不进域外流量，于是在「六类之和」里凭空消失，
        // 卡片看上去自洽、实际上漏了一批二分类认定的恶意连接。
        //
        // 它们同样不能算进 Benign —— 记录表上那条连接写着 Malware 99.95%，
        // 卡片却把它算进正常，就是 B2 那次「Miuref 99.95% 与 Benign 100% 同屏」。
        // 计入域外（模型归类不可信）是这里唯一不撒谎的处理。
        const contradicted = function (r) {
            return !!(r && r.two_class_malicious === true
                && (!r.predicted || r.predicted === 'Benign'));
        };
        const countOf = function (name) {
            if (name === OOD_CLASS_NAME) {
                return flows.filter(function (r) {
                    return !!r.is_unknown || suppressed(r) || contradicted(r);
                }).length;
            }
            if (name === 'Benign') {
                return flows.filter(function (r) {
                    return r.predicted === 'Benign' && !contradicted(r);
                }).length;
            }
            return flows.filter(function (r) { return r.predicted === name && !suppressed(r); }).length;
        };

        const rows = THREAT_CLASSES.map(function (cls) {
            return { name: cls.name, display: cls.name, count: countOf(cls.name), color: cls.color, unknown: false };
        });
        const ood = countOf(OOD_CLASS_NAME);
        // 未知攻击为零时不显示这一行：一行永远为 0 的灰条只会让人以为系统坏了
        if (ood > 0) {
            // 显示名改成「域外流量」：原来的「未知攻击」容易被读成「检测到了未知的攻击」，
            // 但它的实际语义是「这条流离六个类中心都太远，模型对它的归类不可信」。
            // 内部仍用 OOD_CLASS_NAME 做匹配键（后端 class_summary 的键名不变）。
            rows.push({ name: OOD_CLASS_NAME, display: OOD_DISPLAY_NAME, count: ood, color: '#94A3B8', unknown: true });
        }

        // 分母用「本条明细里的连接总数」，不是六类之和。
        // 压制的那些流从攻击类挪到了域外，六类之和会比连接数少，
        // 拿六类之和当分母会让占比虚高；而 hint 上写的「N 个连接」取的是连接总数，
        // 那一格必须和下方记录表的行数对得上。
        const connTotal = flows.length;
        const dom = connTotal > 0 ? connTotal : 0;
        const hint = document.getElementById('threatChartHint');

        if (!connTotal) {
            host.innerHTML = '';
            if (empty) empty.style.display = '';
            if (hint) {
                hint.textContent = dlidsThreatHintText(0, _hintInfo);
                hint.title = dlidsThreatHintTitle(0, _hintInfo);
            }
            return;
        }
        if (empty) empty.style.display = 'none';
        if (hint) {
            hint.textContent = dlidsThreatHintText(connTotal, _hintInfo);
            hint.title = dlidsThreatHintTitle(connTotal, _hintInfo);
        }

        const maxCount = rows.reduce(function (a, r) { return Math.max(a, r.count); }, 0) || 1;
        const html = rows.map(function (r) {
            const width = Math.max(r.count > 0 ? 2 : 0, Math.round((r.count / maxCount) * 100));
            // 只有非零行可点：点开是空清单比不给点更让人困惑
            const clickable = r.count > 0;
            const rowClass = 'threat-row' + (r.unknown ? ' is-unknown' : '') + (clickable ? ' is-clickable' : '');
            const value = r.unknown
                ? r.count.toLocaleString('en-US') + ' 条 · 域外'
                : r.count.toLocaleString('en-US') + ' · ' + (dom > 0 ? ((r.count / dom) * 100).toFixed(2) : '0.00') + '%';
            return '<div class="' + rowClass + '"'
                + ' data-threat-class="' + escapeHtml(r.name) + '"'
                + ' data-threat-count="' + r.count + '"'
                + (clickable ? ' title="点击查看这 ' + r.count + ' 个连接的明细"' : '')
                + '>'
                + '<div class="threat-row-head">'
                + '<span class="threat-row-name">' + escapeHtml(r.display || r.name) + '</span>'
                + '<span class="threat-row-value">' + value + '</span>'
                + '</div>'
                + '<div class="threat-track"><div class="threat-fill" style="width:' + width + '%;background:' + r.color + '"></div></div>'
                + '</div>';
        }).join('');

        // 内容没变就不重设 innerHTML。
        // 实时页每 1.5 秒调一次 dlidsRenderStats，而无条件重写会让浏览器每轮重建这七行 DOM：
        // 鼠标停在某行上时 hover 态先丢再恢复，看起来就是卡片一直在闪；
        // 弹窗开着的时候这层重建还会带着上面的 backdrop-blur 合成层一起重排，更明显。
        if (host.innerHTML !== html) host.innerHTML = html;

        // 事件委托只绑一次：卡片在三个页面上都会被反复重绘，逐行绑会把监听重复挂上去
        if (!host.dataset.threatClickBound) {
            host.dataset.threatClickBound = '1';
            host.addEventListener('click', function (ev) {
                const row = ev.target && ev.target.closest ? ev.target.closest('[data-threat-class]') : null;
                if (!row) return;
                if (Number(row.dataset.threatCount) <= 0) return;
                window.dlidsShowThreatFlows(row.dataset.threatClass);
            });
        }
    };

    // ==================== 六分类数据写入与明细弹窗 ====================
    // 卡片点击后要看的逐条连接清单。与 threat_summary 同一次推理产出（后端 threat_flows），
    // 所以点开某一类的条数一定等于那一行显示的 count。
    //
    // ---- 两页完全独立的两套存储 ----
    // 历史上这里是一个全局数组 window.dlidsThreatFlows + 两个 localStorage key，靠
    // 「按 location.pathname 决定读写哪个 key」来区分。问题是：只要有一处调用忘记传
    // scope、或者一处兜底逻辑走了默认值，数据就写进同一个槽，用户看到的就是
    // 「上传检测的记录跑到实时抓包页去了」。
    //
    // 现在改成物理隔离：live 与 upload 各自一个 bucket 对象，互不相见；
    // 当前页面只认自己那一个（dlidsThreatScope()），别的 scope 的写入直接被拒绝；
    // /dashboard、/records 这类页面 scope 为 null，卡片永远空态。
    var THREAT_SCOPE_DEFS = {
        live:   { storeKey: 'dlids_threat_flows_live',   clearedKey: 'dlids_threat_cleared_live',   max: 200 },
        upload: { storeKey: 'dlids_threat_flows_upload', clearedKey: 'dlids_threat_cleared_upload', max: 1500 }
    };

    // 一次性存储迁移：v1 那套把两页的数据都塞进同一个全局数组，任何一处漏判 scope
    // 就会写串（上传检测的记录出现在实时抓包页就是这么来的）。已经写脏的 localStorage
    // 不会因为代码修好就自动变干净 —— 这里直接作废一遍，从下一次检测重新积累。
    // 记录表（recordStores）不受影响，仍按原样保留。
    try {
        if (window.localStorage && !window.localStorage.getItem('dlids_threat_store_v2')) {
            ['dlids_threat_flows_live', 'dlids_threat_flows_upload', 'dlids_threat_cleared_at'].forEach(function (k) {
                try { window.localStorage.removeItem(k); } catch (e) {}
            });
            window.localStorage.setItem('dlids_threat_store_v2', '1');
        }
    } catch (e) { /* 隐私模式等场景跳过 */ }

    function dlidsThreatScope() {
        // 检测工作台把「上传检测」与「实时抓包」合成同一页的两个子视图，地址不随子视图
        // 变化（两边都渲染 detect.html）。所以先看页面显式写下的覆盖值，否则切到实时
        // 子视图后威胁卡仍然在读上传那份数据。
        const override = window.dlidsDetectScope;
        if (override === 'upload' || override === 'live') return override;
        const p = (window.location && window.location.pathname) || '';
        if (p === '/live') return 'live';
        if (p === '/upload') return 'upload';
        // 概览 / 检测记录 这类页面给一个只读的聚合视图：把 live 与 upload 两套 bucket 合起来看。
        // 过去这里返回 null，导致「威胁类型分布」在总览页永远是空态。
        if (p === '/dashboard' || p === '/' || p === '/records') return 'all';
        return null;
    }
    window.dlidsThreatScope = dlidsThreatScope;

    var _threatBuckets = { live: null, upload: null };

    // 聚合视图：不落盘、不接受写入，只把两套独立 bucket 拼成一份给总览页看。
    function dlidsAggregateBucket() {
        const live = dlidsBucket('live');
        const upload = dlidsBucket('upload');
        const flows = [];
        if (upload && upload.flows && upload.flows.length) flows.push.apply(flows, upload.flows);
        if (live && live.flows && live.flows.length) flows.push.apply(flows, live.flows);
        return {
            scope: 'all',
            def: null,
            virtual: true,
            flows: flows,
            // 两路数据来自不同批次，meta 里的 flows_classified / consistency 无法相加，
            // 硬拼会让 hint 出现「明细仅保留前 N 条（共 M 条已分类）」这种错位说明，
            // 所以聚合视图不带 meta，只按合并后的明细统计。
            meta: {},
            rowsTotal: ((upload && upload.rowsTotal) || 0) + ((live && live.rowsTotal) || 0),
            clearedAt: 0
        };
    }

    function dlidsBucket(scope) {
        if (!scope) return null;
        if (scope === 'all') return dlidsAggregateBucket();
        if (!THREAT_SCOPE_DEFS[scope]) return null;
        // 本段代码在文件里位置偏后，而 dlidsRenderStats 可能在更早的地方就被调用一次，
        // 那时 _threatBuckets 还是 var 提升后的 undefined —— 这里补一行自举。
        if (!_threatBuckets) _threatBuckets = { live: null, upload: null };
        if (_threatBuckets[scope]) return _threatBuckets[scope];
        const def = THREAT_SCOPE_DEFS[scope];
        const b = { scope: scope, def: def, flows: [], meta: {}, rowsTotal: 0, clearedAt: 0 };
        try {
            const raw = window.localStorage && window.localStorage.getItem(def.storeKey);
            const arr = raw ? JSON.parse(raw) : null;
            if (Array.isArray(arr)) b.flows = arr.slice(-def.max);
            const c = window.localStorage && window.localStorage.getItem(def.clearedKey);
            b.clearedAt = c ? Number(c) || 0 : 0;
        } catch (e) { /* localStorage 不可用就只留内存 */ }
        _threatBuckets[scope] = b;
        return b;
    }

    function dlidsCurrentBucket() { return dlidsBucket(dlidsThreatScope()); }

    function dlidsPersistBucket(b) {
        if (!b || !b.def) return;   // 聚合视图（def 为 null）不落盘
        try {
            if (!window.localStorage) return;
            window.localStorage.setItem(b.def.storeKey, JSON.stringify(b.flows));
            window.localStorage.setItem(b.def.clearedKey, String(b.clearedAt || 0));
        } catch (e) { /* 配额满就只留内存，不影响显示 */ }
    }

    // 只读镜像：老代码（渲染、弹窗、results_impl）都按 window.dlidsThreatFlows 读。
    // 用 getter 暴露当前 bucket 的数组，写操作无法落到别人那一份上。
    try {
        Object.defineProperty(window, 'dlidsThreatFlows', {
            configurable: true,
            get: function () { const b = dlidsCurrentBucket(); return b ? b.flows : []; },
            set: function () { /* 只允许通过 dlidsMergeThreatFlows 写入，直接赋值一律忽略 */ }
        });
        Object.defineProperty(window, 'dlidsThreatMeta', {
            configurable: true,
            get: function () { const b = dlidsCurrentBucket(); return b ? b.meta : {}; },
            set: function () {}
        });
        Object.defineProperty(window, 'dlidsThreatRowsTotal', {
            configurable: true,
            get: function () { const b = dlidsCurrentBucket(); return b ? b.rowsTotal : 0; },
            set: function () {}
        });
        Object.defineProperty(window, 'dlidsThreatClearedAt', {
            configurable: true,
            get: function () { const b = dlidsCurrentBucket(); return b ? b.clearedAt : 0; },
            set: function () {}
        });
    } catch (e) { /* 非浏览器环境：退化成普通属性 */ }

    // 一条连接一个键：IP 字典序小的一侧固定为前向，正反向归同一条。
    // 字段名要同时兼容两套来源：/api/live/stats 的 threat_flows 用 src_ip/src_port/dst_ip/dst_port/proto，
    // 而 flows/recent 与 recordStores.live[].data 用 src/sport/dst/dport/protocol。
    // 不兼容的话「按记录表过滤卡片明细」永远匹配不上。
    function dlidsField(r, longName, shortName) {
        if (!r) return undefined;
        if (r[longName] !== undefined && r[longName] !== null && r[longName] !== '') return r[longName];
        return r[shortName];
    }

    function dlidsConnKey(r, ignoreProto) {
        const a = String(dlidsField(r, 'src_ip', 'src') || '');
        const b = String(dlidsField(r, 'dst_ip', 'dst') || '');
        const ap = Number(dlidsField(r, 'src_port', 'sport') || 0);
        const bp = Number(dlidsField(r, 'dst_port', 'dport') || 0);
        const proto = ignoreProto ? '' : String(Number(dlidsField(r, 'proto', 'protocol') || 0));
        return (a <= b)
            ? a + ':' + ap + '>' + b + ':' + bp + '|' + proto
            : b + ':' + bp + '>' + a + ':' + ap + '|' + proto;
    }
    window.dlidsConnKey = dlidsConnKey;

    // 逐流明细的唯一写入口。只写「当前页面自己」那一份：
    //   * scope 为 null 的页面（概览 / 检测记录）直接丢弃，卡片保持空态；
    //   * 显式传入的 options.key 与当前 scope 不一致时也丢弃 —— 跨页写不进来。
    // options.persist = false 时只更新内存。
    // 时间窗口：实时页每 1.5 秒轮询一次，后端每次都返回「当前全部流」，
    // 不按清除时刻过滤的话，点完「清除日志」下一轮就把同一批流灌回来，按钮看上去完全无效。
    window.dlidsMergeThreatFlows = function (incoming, options) {
        const opts = options || {};
        const scope = dlidsThreatScope();
        // 聚合视图（概览 / 检测记录）只读：不接受任何写入，只负责把两路数据合并展示
        if (!scope || scope === 'all') return 0;
        if (opts.key && opts.key !== scope) return 0;
        const b = dlidsBucket(scope);
        if (!b) return 0;

        const cap = Number(opts.max) || b.def.max;
        const rows = Array.isArray(incoming) ? incoming : [];
        const cutoff = Number(b.clearedAt) || 0;

        const keep = function (arr) {
            if (!cutoff) return arr.slice();
            return arr.filter(function (r) {
                const raw = r && r.captured_at;
                // 上传链路（pcap → 78 维特征）没有逐包时间戳，后端回的是 null。
                // 不能直接 Number(raw)：Number(null) === 0 且 isFinite(0) 为真，
                // 会被当成「1970 年的流」——只要用户点过一次清除（cutoff > 0），
                // 之后每一次上传的明细都会被全量过滤掉，卡片永远不动。
                if (raw === null || raw === undefined || raw === '') return true;
                const ts = Number(raw);
                return !isFinite(ts) || ts > cutoff;
            });
        };

        // Map 保留首次插入位置、值取最新 —— 和 results_impl.js 收敛 recordStores.live 用的是同一招
        const byConn = new Map();
        keep(b.flows).forEach(function (r) { byConn.set(dlidsConnKey(r), r); });
        keep(rows).forEach(function (r) { byConn.set(dlidsConnKey(r), r); });

        let list = Array.from(byConn.values());
        if (list.length > cap) list = list.slice(list.length - cap);
        b.flows = list;

        if (opts.persist !== false) dlidsPersistBucket(b);
        if (typeof window.dlidsRenderStats === 'function') window.dlidsRenderStats();
        return list.length;
    };

    // 记录卡片附带的 meta（后端 threat_meta：流总数 / OOD 阈值 / 是否被截断）
    window.dlidsSetThreatMeta = function (meta, options) {
        const opts = options || {};
        const scope = dlidsThreatScope();
        if (!scope) return;
        if (opts.key && opts.key !== scope) return;
        const b = dlidsBucket(scope);
        if (!b) return;
        b.meta = meta || {};
        b.rowsTotal = Number((meta || {}).flow_rows || (meta || {}).flows || 0) || 0;
        dlidsPersistBucket(b);
        if (typeof window.dlidsRenderStats === 'function') window.dlidsRenderStats();
    };

    // 清空当前页面的卡片明细（清除日志时调）。
    // 同时记下清除时刻 —— 轮询会把早于它的流滤掉，刷新后也照样生效（落盘）。
    window.dlidsClearThreatFlows = function (scopeArg) {
        const scope = dlidsThreatScope();
        if (!scope) return;
        if (scopeArg && scopeArg !== scope) return;
        const b = dlidsBucket(scope);
        if (!b) return;
        b.flows = [];
        b.meta = {};
        b.rowsTotal = 0;
        b.clearedAt = Date.now() / 1000;
        dlidsPersistBucket(b);
        if (window.dlidsStats) window.dlidsStats.threats = {};
        if (typeof window.dlidsRenderStats === 'function') window.dlidsRenderStats();
    };

    // 页面进场时载入自己那一份（bucket 惰性构造时已经读过 localStorage，这里只是触发一次渲染）
    window.dlidsRestoreThreatFlows = function (scopeArg) {
        const scope = dlidsThreatScope();
        if (!scope) return 0;
        if (scopeArg && scopeArg !== scope) return 0;
        const b = dlidsBucket(scope);
        if (!b) return 0;
        if (b.flows.length && typeof window.dlidsRenderStats === 'function') window.dlidsRenderStats();
        return b.flows.length;
    };

    // 让卡片明细收敛到「本页记录表实际还在的那些连接」。
    // 记录表（recordStores[scope]）是很早就开始累积并持久化的，而卡片明细是后加的，
    // 两者起点不同：卡片会留着记录表里早已不存在的连接（清空过、被上限挤掉过）。
    // 这里按记录表的连接集合过滤一次并把结果落盘，保证「卡片只显示本页面检测记录里的流量」。
    // 不用 proto 参与匹配：recordStores 里的行不一定带 protocol 字段，带上会导致全不匹配。
    // 记录表为空时直接返回 0：那多半是还没恢复完，不能据此把明细全删掉。
    window.dlidsPruneThreatFlowsToConnections = function (scopeArg) {
        const scope = dlidsThreatScope();
        if (!scope) return 0;
        if (scopeArg && scopeArg !== scope) return 0;
        const b = dlidsBucket(scope);
        if (!b) return 0;
        const store = (window.recordStores && window.recordStores[scope]) || [];
        if (!store.length) return 0;
        const keep = new Set();
        store.forEach(function (row) {
            const d = (row && row.data) ? row.data : row;
            if (d) keep.add(dlidsConnKey(d, true));
        });
        const before = b.flows.length;
        const list = b.flows.filter(function (r) { return keep.has(dlidsConnKey(r, true)); });
        if (list.length !== before) {
            b.flows = list;
            dlidsPersistBucket(b);
            if (typeof window.dlidsRenderStats === 'function') window.dlidsRenderStats();
        }
        return before - list.length;
    };

    // 进场即载入自己那一份存储
    try {
        if (dlidsThreatScope()) {
            const b = dlidsCurrentBucket();
            if (b && b.flows.length && typeof window.dlidsRenderStats === 'function') {
                window.dlidsRenderStats();
            }
        }
    } catch (e) { /* 非浏览器环境跳过 */ }


    // 六分类汇总进入统计层的唯一入口。
    // options.reset = true 用于实时链路：后端每次给的是「当前这批连接」的全量快照，
    // 必须覆盖而不是累加，否则连接数会随着轮询一路涨上去。
    window.dlidsAbsorbThreatSummary = function (summary, options) {
        if (!window.dlidsStats) return;
        const reset = !!(options && options.reset);
        const bag = reset ? {} : (window.dlidsStats.threats || {});
        Object.keys(summary || {}).forEach(function (name) {
            const entry = summary[name];
            const count = Number(entry && entry.count != null ? entry.count : entry) || 0;
            bag[name] = count;
        });
        window.dlidsStats.threats = bag;
        if (typeof window.dlidsRenderStats === 'function') window.dlidsRenderStats();
    };

    function dlidsProtoName(proto) {
        const p = Number(proto);
        if (p === 6) return 'TCP';
        if (p === 17) return 'UDP';
        if (p === 1) return 'ICMP';
        return p ? String(p) : '-';
    }

    function dlidsTierLabel(tier) {
        const t = String(tier || '');
        if (t === 'auto') return '自动';
        if (t === 'review') return '复核';
        if (t === 'manual') return '人工';
        return t || '-';
    }

    // 某一类该列出哪些行。
    // 关键：'未知攻击' 不是模型的输出类别（预测值永远是六类之一），它是「六类中
    // 到类中心距离超过 p99.9 的那些流」的另一个视角，所以只能按 is_unknown 过滤。
    //
    // 压制规则必须和 dlidsRenderThreatBars 的 countOf 完全一致，否则会出现
    // 「卡片写 3 条、点开只有 1 条」这种更难受的错位：包级二分类判正常的流
    // 从攻击类挪到域外行，所以这里也要跟着挪。
    function dlidsIsSuppressed(r) {
        return !!(r && r.two_class_malicious === false
            && r.predicted && r.predicted !== 'Benign');
    }

    // 与 dlidsIsSuppressed 反向的同一类矛盾（缺陷 B2）：包级二分类判恶意、
    // 流级六分类却说 Benign。这些行同样要从「Benign」挪进域外，否则记录表说
    // Malware、卡片把它算进正常，就是 B2 那次 99.95% 与 100% 同屏打架。
    function dlidsIsContradicted(r) {
        return !!(r && r.two_class_malicious === true
            && (!r.predicted || r.predicted === 'Benign'));
    }

    function dlidsFlagsFor(className) {
        const rows = window.dlidsThreatFlows || [];
        if (className === OOD_CLASS_NAME) {
            return rows.filter(function (r) {
                return !!r.is_unknown || dlidsIsSuppressed(r) || dlidsIsContradicted(r);
            });
        }
        if (className === 'Benign') {
            return rows.filter(function (r) {
                return r.predicted === 'Benign' && !dlidsIsContradicted(r);
            });
        }
        return rows.filter(function (r) {
            return r.predicted === className && !dlidsIsSuppressed(r);
        });
    }

    // 弹窗开着的时候把实时轮询停掉，关掉再恢复。
    // 理由：/live 每 1.5 秒会重绘辐射图、往记录表插行、重算卡片，屏幕上的弹窗是 fixed 层，
    // 底下这些重排每轮都要重新合成一次，看起来就是「点开之后画面一直在刷新」。
    // 明细是一次快照，本来也不需要边看边变。
    function pauseLivePollingForModal() {
        if (!window.livePoller) return false;
        try { clearInterval(window.livePoller); } catch (e) {}
        window.livePoller = null;
        return true;
    }

    function resumeLivePollingAfterModal() {
        // /live 现在只是旧地址（后端 302 到 /detect?tab=live），真实的"正在看实时子视图"
        // 由 detectImpl 写在 window.dlidsDetectScope 上，两个都要认。
        const path = (window.location && window.location.pathname) || '';
        const isLive = path === '/live' || window.dlidsDetectScope === 'live';
        if (!isLive) return;
        if (typeof window.pollLive !== 'function') return;
        if (window.livePoller) return;
        try {
            window.pollLive();
            window.livePoller = window.setInterval(window.pollLive, 1500);
        } catch (e) { console.warn('resume live polling failed', e); }
    }

    window.closeThreatFlowModal = function () {
        const modal = document.getElementById('threatFlowModal');
        if (modal) modal.classList.add('hidden');
        resumeLivePollingAfterModal();
    };

    window.dlidsShowThreatFlows = function (className) {
        const modal = document.getElementById('threatFlowModal');
        if (!modal) return;
        const title = document.getElementById('threatFlowTitle');
        const subtitle = document.getElementById('threatFlowSubtitle');
        const countEl = document.getElementById('threatFlowCount');
        const body = document.getElementById('threatFlowTableBody');
        const emptyEl = document.getElementById('threatFlowEmpty');
        const footEl = document.getElementById('threatFlowFootnote');

        const rows = dlidsFlagsFor(className);
        const shownName = (className === OOD_CLASS_NAME) ? OOD_DISPLAY_NAME : className;
        if (title) title.textContent = shownName;
        if (subtitle) {
            // 这两句是说给「为什么记录表全是 Benign、这张卡片却有攻击类」这个疑问的：
            // 卡片走的是流级 78 维统计特征 + 六分类模型，下方记录表走的是包级 28×28
            // 字节图 + 六模型二分类投票，是两个不同的模型，同一条连接给出不同结论是正常的。
            subtitle.textContent = (className === OOD_CLASS_NAME)
                ? '到所属类中心的距离超过 OOD 阈值，判定为域外流量（不是「检测到攻击」，而是「归类不可信」）'
                : '按双向五元组归并的连接，正反向计一条。与下方记录表的包级二分类是两套独立模型，结论可能不同';
        }
        if (countEl) countEl.textContent = rows.length + ' 个连接';

        if (body) {
            body.innerHTML = rows.map(function (r) {
                const ood = (r.ood_distance == null) ? '-' : Number(r.ood_distance).toFixed(2);
                const conf = (r.confidence == null) ? '-' : (Number(r.confidence) * 100).toFixed(1) + '%';
                // 三个标记互相独立：一条流可能既域外、又被二分类判正常或判恶意。
                // 标出来是让人知道「这个攻击类是六分类单独给的，包级模型不同意」，
                // 否则点开域外行看到一串 predicted=DoS-xxx 会更困惑。
                const flagParts = [];
                if (r.is_unknown) flagParts.push('<span class="text-xs" style="color:#e879f9">域外</span>');
                if (dlidsIsSuppressed(r)) flagParts.push('<span class="text-xs" style="color:#64748b">二分类判正常</span>');
                if (dlidsIsContradicted(r)) flagParts.push('<span class="text-xs" style="color:#f43f5e">二分类判恶意</span>');
                const flag = flagParts.length ? ' ' + flagParts.join(' ') : '';
                return '<tr class="border-b border-cyber-border/40">'
                    + '<td class="px-3 py-1.5 font-mono">' + escapeHtml(r.src_ip) + ':' + escapeHtml(r.src_port) + '</td>'
                    + '<td class="px-3 py-1.5 font-mono">' + escapeHtml(r.dst_ip) + ':' + escapeHtml(r.dst_port) + '</td>'
                    + '<td class="px-3 py-1.5">' + dlidsProtoName(r.proto) + '</td>'
                    + '<td class="px-3 py-1.5">' + escapeHtml(r.predicted) + flag + '</td>'
                    + '<td class="px-3 py-1.5 text-right font-mono">' + conf + '</td>'
                    + '<td class="px-3 py-1.5">' + dlidsTierLabel(r.tier) + '</td>'
                    + '<td class="px-3 py-1.5 text-right font-mono">' + ood + '</td>'
                    + '<td class="px-3 py-1.5 text-right font-mono">' + (r.packets == null ? '-' : r.packets) + '</td>'
                    + '</tr>';
            }).join('');
        }
        if (emptyEl) emptyEl.style.display = rows.length ? 'none' : '';

        if (footEl) {
            const meta = window.dlidsThreatMeta || {};
            const isLive = meta.source === 'live';
            const cap = Number(meta.flow_rows) || 0;
            const flows = Number(meta.flows) || 0;
            if (isLive) {
                footEl.textContent = (cap && flows && cap < flows)
                    ? '本次共 ' + flows + ' 个连接，明细只列出前 ' + cap + ' 条'
                    : '数据与上方卡片同一次推理，条数一一对应';
            } else {
                // 上传页的卡片 count 是多次上传累加，明细只保留最近 1500 条，
                // 两者不是同一次推理，不能说「一一对应」。
                footEl.textContent = '明细保留最近上传的连接，累计条数以卡片为准';
            }
        }

        modal.classList.remove('hidden');
        pauseLivePollingForModal();
    };

    // KPI 与威胁卡片共用同一套页面归属：/upload 只认上传、/live 只认实时，
    // 概览页没有「本页」的概念，才把两路相加。否则在实时页会看到上传的检测次数
    // 混进来（累计检测量 69 = 上传 9 + 实时 60），同一格数字换个页面就变。
    window.dlidsStatsSnapshot = function () {
        const s = window.dlidsStats || {};
        const upload = s.upload || {};
        const live = s.live || {};
        const upTotal = Number(upload.total) || 0;
        const upMalware = Number(upload.malware) || 0;
        const liveTotal = Number(live.sessions) || 0;
        const liveMalware = Number(live.malware) || 0;

        const scope = (typeof dlidsThreatScope === 'function') ? dlidsThreatScope() : null;
        let total;
        let malware;
        if (scope === 'upload') {
            total = upTotal;
            malware = upMalware;
        } else if (scope === 'live') {
            total = liveTotal;
            malware = liveMalware;
        } else {
            total = upTotal + liveTotal;
            malware = upMalware + liveMalware;
        }
        if (malware > total) malware = total;
        return {
            total: total,
            malware: malware,
            benign: Math.max(0, total - malware),
            scope: scope,
            uploadTotal: upTotal,
            liveTotal: liveTotal
        };
    };

    // KPI 三格下方的说明文字也跟着页面走，否则「本次会话处理的检测次数」在实时页
    // 会和实际显示的数字（本页抓包）对不上号。
    function dlidsKpiFootText(scope) {
        if (scope === 'upload') {
            return { total: '本页上传检测的次数', malware: '判定为 Malware 的次数', rate: '恶意 ÷ 本页上传检测量' };
        }
        if (scope === 'live') {
            // 实时这一路的「检测量」其实是后端累计的会话数（capture_stats.session_count），
            // 不是「点了几次检测」，文案要跟着改，否则读数和说明对不上。
            return { total: '本页实时抓包累计的会话数', malware: '其中判定为 Malware 的会话', rate: '恶意会话 ÷ 本页会话总数' };
        }
        return { total: '上传 + 实时合计', malware: '判定为 Malware 的数量', rate: '恶意流量 ÷ 累计检测量' };
    }

    function dlidsSetFoot(id, text) {
        const el = document.getElementById(id);
        if (el && el.textContent !== text) el.textContent = text;
    }

    // 唯一允许写 KPI 与威胁分布的地方
    window.dlidsRenderStats = function () {
        const snap = window.dlidsStatsSnapshot();
        // 兼容既有读取方（详情弹窗、模型评估等处仍在读这两个全局）
        window.totalScans = snap.total;
        window.malwareCount = snap.malware;
        try { if (window.dlidsSetKpi) window.dlidsSetKpi(document.getElementById('statTotal'), snap.total); } catch (e) {}
        try { if (window.dlidsSetKpi) window.dlidsSetKpi(document.getElementById('statMalware'), snap.malware); } catch (e) {}

        // 恶意占比：原先没有任何代码写这一格，所以它永远停在模板里的「--」。
        // 分母为 0 时保持「--」而不是 0.00%，0/0 说成 0% 会被读成「测了但都没事」。
        try {
            const rateEl = document.getElementById('statRate');
            const rateUnit = document.getElementById('statRateUnit');
            if (rateEl) {
                if (snap.total > 0) {
                    rateEl.textContent = ((snap.malware / snap.total) * 100).toFixed(2);
                    if (rateUnit) rateUnit.style.display = '';
                } else {
                    rateEl.textContent = '--';
                    if (rateUnit) rateUnit.style.display = 'none';
                }
            }
        } catch (e) {}

        try {
            const foot = dlidsKpiFootText(snap.scope);
            dlidsSetFoot('kpiTotalFoot', foot.total);
            dlidsSetFoot('kpiMalwareFoot', foot.malware);
            dlidsSetFoot('kpiRateFoot', foot.rate);
        } catch (e) {}

        try { window.dlidsRenderThreatBars(); } catch (e) { console.warn('dlidsRenderThreatBars failed', e); }
        return snap;
    };

    const UI_STATE_KEY = 'dlids_ui_state';
    // 实时抓包轮询间隔（毫秒）。pollLive 已改为按 since_id 增量拉取，这里只负责节奏。
    const LIVE_POLL_INTERVAL_MS = 1500;

    function loadUiState() {
        try {
            const raw = window.localStorage && window.localStorage.getItem && window.localStorage.getItem(UI_STATE_KEY);
            if (!raw) return {};
            const parsed = JSON.parse(raw);
            return parsed && typeof parsed === 'object' ? parsed : {};
        } catch (e) {
            console.warn('loadUiState failed', e);
            return {};
        }
    }

    function persistUiState() {
        try {
            const votingModeToggle = document.getElementById('votingModeToggle');
            const payload = {
                selectedInterface: window.liveUiState && window.liveUiState.selectedInterface ? window.liveUiState.selectedInterface : null,
                selectedInterfaceLabel: window.liveUiState && window.liveUiState.selectedInterfaceLabel ? window.liveUiState.selectedInterfaceLabel : null,
                tabMode: (window.recordTableMode === 'live' || window.recordTableMode === 'upload') ? window.recordTableMode : 'upload',
                votingModeEnabled: !!(votingModeToggle && votingModeToggle.checked),
                currentFilter: ['all', 'benign', 'malware'].includes(String(window.currentRecordFilter || '').toLowerCase())
                    ? String(window.currentRecordFilter).toLowerCase()
                    : 'all'
            };
            window.localStorage && window.localStorage.setItem && window.localStorage.setItem(UI_STATE_KEY, JSON.stringify(payload));
        } catch (e) {
            console.warn('persistUiState failed', e);
        }
    }
    window.persistUiState = persistUiState;

    const savedUiState = loadUiState();

    // ==================== 错误处理 ====================
    window.addEventListener('error', (ev) => {
        try {
            const msg = ev?.message || String(ev);
            let banner = document.getElementById('jsErrorBanner');
            if (!banner) {
                banner = document.createElement('div');
                banner.id = 'jsErrorBanner';
                banner.style.cssText = 'position:fixed;right:12px;bottom:12px;z-index:99999;max-width:420px;padding:10px 12px;background:rgba(255,46,99,0.95);color:#fff;border-radius:8px;font-size:13px;box-shadow:0 6px 18px rgba(0,0,0,0.4)';
                document.body.appendChild(banner);
            }
            banner.innerText = msg;
            setTimeout(() => banner.remove?.(), 8000);
        } catch (_) {}
    });

    // ==================== 全局状态初始化 ====================
    window.liveUiState = window.liveUiState || { running: false, selectedFilter: 'all', selectedInterface: null, selectedInterfaceLabel: null };
    if (savedUiState && savedUiState.selectedInterface) {
        window.liveUiState.selectedInterface = savedUiState.selectedInterface;
    }
    if (savedUiState && savedUiState.selectedInterfaceLabel) {
        window.liveUiState.selectedInterfaceLabel = savedUiState.selectedInterfaceLabel;
    }
    window.persistedVotingModeEnabled = !!(savedUiState && savedUiState.votingModeEnabled);
    window.recordStores = window.recordStores || { live: [], upload: [] };
    // 页面模式由当前页面模板上的 data-mode 决定（upload / live / all）。
    // 不再从 localStorage 的 tabMode 推断：页面之间是整页跳转，模式不该被上一次浏览的页面带过来。
    function resolvePageMode() {
        const panel = document.getElementById('resultCardPanel');
        const mode = panel && panel.dataset ? String(panel.dataset.mode || '') : '';
        return (mode === 'live' || mode === 'upload' || mode === 'all') ? mode : 'upload';
    }
    // 无刷新换页后内容区是新的 DOM，data-mode 可能变了，必须重算一次
    window.dlidsResolvePageMode = resolvePageMode;
    window.recordTableMode = resolvePageMode();

    const savedFilter = savedUiState && ['all', 'benign', 'malware'].includes(String(savedUiState.currentFilter || '').toLowerCase())
        ? String(savedUiState.currentFilter).toLowerCase()
        : 'all';
    window.currentRecordFilter = savedFilter;
    window._liveSeen = window._liveSeen || new Set();
    window.__detailsDataMap = window.__detailsDataMap || {};

    // ==================== 委托函数 ====================
    function delegate(name, fn) {
        return function(...args) {
            try {
                const impl = window[name];
                if (impl && typeof impl[fn] === 'function') {
                    return impl[fn].apply(impl, args);
                }
            } catch(e) {
                console.warn('delegate', name, fn, e);
            }
            return null;
        };
    }

    // 导出委托API（兼容现有模块调用）
    window.initModelEvalCharts = delegate('modelEvalImpl', 'initModelEvalCharts');
    window.initCharts = delegate('uploadTrendImpl', 'initCharts');
    window.updateTrendChart = delegate('liveTrendImpl', 'updateTrendChart');
    window.updateStats = delegate('uploadTrendImpl', 'updateStats');
    window.initModalChart = delegate('uploadTrendImpl', 'initModalChart');
    window.renderLiveFlowGraph = delegate('liveTrendImpl', 'renderLiveFlowGraph');
    // ensure a robust fallback: if liveTrendImpl doesn't provide renderLiveFlowGraph,
    // delegate to liveImpl or the liveGraph module directly so the canvas gets drawn.
    (function(){
        const orig = window.renderLiveFlowGraph;
        window.renderLiveFlowGraph = function(flows) {
            try {
                // try liveTrendImpl first (orig delegate)
                if (typeof orig === 'function') {
                    const res = orig.apply(null, arguments);
                    if (res !== null) return res;
                }
                } catch (e) { console.warn('liveTrendImpl.renderLiveFlowGraph failed', e); }
            try {
                if (window.liveImpl && typeof window.liveImpl.renderLiveFlowGraph === 'function') return window.liveImpl.renderLiveFlowGraph.apply(window.liveImpl, arguments);
            } catch (e) { console.warn('liveImpl.renderLiveFlowGraph failed', e); }
            try {
                if (window.liveGraph && typeof window.liveGraph.renderLiveFlowGraph === 'function') return window.liveGraph.renderLiveFlowGraph.apply(window.liveGraph, arguments);
            } catch (e) { console.warn('liveGraph.renderLiveFlowGraph failed', e); }
            console.warn('renderLiveFlowGraph: no implementation available');
            return null;
        };
    })();

    window.handleFiles = delegate('uploadImpl', 'handleFiles');
    window.uploadFilesSequentially = delegate('uploadImpl', 'uploadFilesSequentially');
    window.uploadFile = delegate('uploadImpl', 'uploadFile');
    window.readResponsePayload = delegate('uploadImpl', 'readResponsePayload');
    window.selectModel = delegate('uploadImpl', 'selectModel');
    window.getModelDisplayName = delegate('uploadImpl', 'getModelDisplayName');

    window.startLive = delegate('liveImpl', 'startLive');
    window.stopLive = delegate('liveImpl', 'stopLive');
    window.toggleLiveCapture = delegate('liveImpl', 'toggleLiveCapture');
    window.setLiveRunningUI = delegate('liveImpl', 'setLiveRunningUI');
    window.pollLive = delegate('liveImpl', 'pollLive');
    window.clearLiveResultTable = delegate('liveImpl', 'clearLiveResultTable');

    window.addResultRow = delegate('resultsImpl', 'addResultRow');
    window.openDetails = delegate('resultsImpl', 'openDetails');
    window.cacheRecordedRow = delegate('resultsImpl', 'cacheRecordedRow');
    window.renderRecordedRows = delegate('resultsImpl', 'renderRecordedRows');
    window.applyRecordFilter = delegate('resultsImpl', 'applyRecordFilter');
    window.activateUploadRecordCard = delegate('uploadRecordCardImpl', 'activate');
    window.activateLiveRecordCard = delegate('liveRecordCardImpl', 'activate');
    // fallback in case an older cached results_impl.js is loaded without applyRecordFilter
    (function() {
        const delegated = window.applyRecordFilter;
        window.applyRecordFilter = function(filterType) {
            const delegatedResult = (typeof delegated === 'function') ? delegated.apply(null, arguments) : null;
            if (delegatedResult !== null) return delegatedResult;
            try {
                const normalized = String(filterType || 'all').toLowerCase();
                const finalFilter = ['all', 'benign', 'malware'].includes(normalized) ? normalized : 'all';
                window.currentRecordFilter = finalFilter;
                const rows = document.querySelectorAll('#resultTableBody tr');
                rows.forEach((row) => {
                    const type = row && row.dataset ? String(row.dataset.recordType || '').toLowerCase() : '';
                    let show = finalFilter === 'all';
                    if (!show) show = (finalFilter === 'benign' && type === 'benign');
                    if (!show) show = (finalFilter === 'malware' && type === 'malware');
                    row.style.display = show ? '' : 'none';
                });
                ['recordFilterAll', 'recordFilterBenign', 'recordFilterMalware'].forEach((id) => {
                    const btn = document.getElementById(id);
                    if (!btn) return;
                    const isActive = (id === 'recordFilterAll' && finalFilter === 'all')
                        || (id === 'recordFilterBenign' && finalFilter === 'benign')
                        || (id === 'recordFilterMalware' && finalFilter === 'malware');
                    btn.classList.toggle('bg-cyber-primary', isActive);
                    btn.classList.toggle('text-black', isActive);
                    btn.classList.toggle('text-gray-300', !isActive);
                });
                try { if (typeof window.updateRecordFilterIndicator === 'function') window.updateRecordFilterIndicator(finalFilter); } catch (e) {}
                try { persistUiState(); } catch (e) {}
            } catch (e) {
                console.warn('fallback applyRecordFilter failed', e);
            }
            return null;
        };
    })();

    // ==================== 辅助函数 ====================
    function findButtonByText(text) {
        return Array.from(document.querySelectorAll('button')).find(b => 
            (b.innerText || '').trim().indexOf(text) !== -1
        );
    }

    function getSelectedInterface() {
        return window.liveUiState?.selectedInterface || null;
    }
    window.getSelectedInterface = getSelectedInterface;

    function initRecordFilterIndicator() {
        const btnAll = document.getElementById('recordFilterAll');
        const btnBenign = document.getElementById('recordFilterBenign');
        const btnMalware = document.getElementById('recordFilterMalware');
        const container = btnAll && btnAll.parentElement;
        if (!container || !btnAll || !btnBenign || !btnMalware) return;

        container.classList.add('relative', 'overflow-hidden');
        [btnAll, btnBenign, btnMalware].forEach((btn) => btn.classList.add('relative', 'z-10'));

        let indicator = container.querySelector('.record-filter-indicator');
        if (!indicator) {
            indicator = document.createElement('span');
            indicator.className = 'record-filter-indicator';
            indicator.setAttribute('aria-hidden', 'true');
            indicator.style.position = 'absolute';
            indicator.style.top = '4px';
            indicator.style.bottom = '4px';
            indicator.style.left = '0';
            indicator.style.width = '0px';
            indicator.style.borderRadius = '9999px';
            indicator.style.background = 'rgb(34 211 238)';
            indicator.style.transition = 'transform 220ms ease, width 220ms ease, opacity 180ms ease';
            indicator.style.willChange = 'transform, width';
            indicator.style.zIndex = '0';
            container.insertBefore(indicator, container.firstChild);
        }

        window.updateRecordFilterIndicator = function(filterType) {
            const f = String(filterType || window.currentRecordFilter || 'all').toLowerCase();
            const target = f === 'benign' ? btnBenign : (f === 'malware' ? btnMalware : btnAll);
            if (!target || !container || !indicator) return;
            const cRect = container.getBoundingClientRect();
            const tRect = target.getBoundingClientRect();
            const x = Math.max(0, tRect.left - cRect.left);
            indicator.style.width = `${tRect.width}px`;
            indicator.style.transform = `translateX(${x}px)`;
            indicator.style.opacity = '1';
        };

        try { window.updateRecordFilterIndicator(window.currentRecordFilter || 'all'); } catch (e) {}
        // resize 监听只能挂一次：这个函数在无刷新换页后每页都会重新跑一遍，写在函数体里会让
        // 换页次数等于监听器个数。回调每次都重新查 window.updateRecordFilterIndicator，
        // 所以换页后它自动指向新页面的滑块。
        if (!window.__recordFilterResizeBound) {
            window.__recordFilterResizeBound = true;
            window.addEventListener('resize', () => {
                // 代码审查第 49 条：拖拽窗口会连续触发 resize，用 rAF 把同一帧内的多次回调合并成一次
                if (window.__recordFilterResizeRaf) cancelAnimationFrame(window.__recordFilterResizeRaf);
                window.__recordFilterResizeRaf = requestAnimationFrame(() => {
                    try { window.updateRecordFilterIndicator(window.currentRecordFilter || 'all'); } catch (e) {}
                });
            });
        }
    }

    // ==================== 网卡加载 ====================
    async function loadInterfaces() {
        const dropdownMenu = document.getElementById('dropdownMenu');
        const selectedText = document.getElementById('selectedText');
        if (!dropdownMenu) return;
        
        dropdownMenu.innerHTML = '<li class="px-3 py-2 text-sm text-gray-400">加载中...</li>';
        try {
            const r = await fetch('/api/interfaces');
            if (!r.ok) throw new Error('Failed to load');
            const body = await r.json();
            dropdownMenu.innerHTML = '';
            const list = body?.interfaces || [];
            
            if (!list.length) {
                dropdownMenu.innerHTML = '<li class="px-3 py-2 text-sm text-gray-400">无可用网卡</li>';
                return;
            }
            
            list.forEach((iface) => {
                const li = document.createElement('li');
                li.className = 'px-3 py-2 cursor-pointer hover:bg-cyber-card/30';
                const value = typeof iface === 'object' ? (iface.value || iface.name) : iface;
                li.dataset.ifaceValue = value;
                
                let display = '';
                if (typeof iface === 'object') {
                    display = (iface.name || value) + (iface.description && iface.description !== (iface.name || value) ? (' - ' + iface.description) : '');
                } else {
                    display = String(iface);
                }
                li.innerText = display;
                
                li.addEventListener('click', () => {
                    window.liveUiState = window.liveUiState || {};
                    window.liveUiState.selectedInterface = value;
                    window.liveUiState.selectedInterfaceLabel = li.innerText;
                    if (selectedText) selectedText.innerText = li.innerText;
                    dropdownMenu.classList.add('hidden');
                    try { persistUiState(); } catch(e) {}
                    try { persistLiveUiState?.(); } catch(e) {}
                });
                dropdownMenu.appendChild(li);
            });
            
            // 恢复上次选择
            const pref = window.liveUiState?.selectedInterface;
            if (pref) {
                const node = Array.from(dropdownMenu.children).find(c => c.dataset?.ifaceValue === pref);
                if (node && selectedText) {
                    selectedText.innerText = node.innerText;
                    window.liveUiState.selectedInterfaceLabel = node.innerText;
                } else if (selectedText && window.liveUiState?.selectedInterfaceLabel) {
                    selectedText.innerText = window.liveUiState.selectedInterfaceLabel;
                }
            } else if (dropdownMenu.children.length > 0) {
                const first = dropdownMenu.children[0];
                if (first && selectedText) {
                    selectedText.innerText = first.innerText;
                    window.liveUiState = window.liveUiState || {};
                    window.liveUiState.selectedInterface = first.dataset.ifaceValue;
                    window.liveUiState.selectedInterfaceLabel = first.innerText;
                    try { persistUiState(); } catch(e) {}
                }
            }
        } catch(e) {
            dropdownMenu.innerHTML = '<li class="px-3 py-2 text-sm text-red-400">加载失败</li>';
        }
    }
    window.loadInterfaces = loadInterfaces;

    // ==================== 模型指标加载 ====================
    async function loadModelMetrics() {
        try {
            // 代码审查疑点 3：走 /api/metrics（唯一数据源），响应形如 {status, data}
            const r = await fetch('/api/metrics');
            if (!r.ok) return;
            const payload = await r.json();
            const body = (payload && payload.data) ? payload.data : payload;
            const map = {
                'CNN_BiLSTM': 'm1',
                'Classic_CNN': 'm2',
                'Lightweight_CNN_BiLSTM': 'm3',
                'Pure_BiLSTM': 'm4',
                'MLP': 'm5',
                'Transformer': 'm6'
            };
            const fieldMap = {
                pre: 'precision', rec: 'recall', f1: 'f1_score', f2: 'f2_score',
                dr: 'detection_rate', fpr: 'fpr', fnr: 'fnr', auc: 'auc',
                lat: 'latency_ms_per_sample', tps: 'throughput_samples_per_sec'
            };
            
            const formatValue = (field, value) => {
                if (value == null || value === '') return '--';
                if (field === 'auc') return String(value);
                if (field === 'lat') return value + ' ms/条';
                if (field === 'tps') return value + ' 条/s';
                const text = String(value);
                return text.endsWith('%') ? text : text + '%';
            };
            
            Object.keys(map).forEach((key) => {
                const node = body[key];
                const prefix = map[key];
                Object.entries(fieldMap).forEach(([suffix, field]) => {
                    const el = document.getElementById(prefix + '-' + suffix);
                    if (el) el.innerText = formatValue(suffix, node?.[field]);
                });
                const accEl = document.getElementById(prefix + '-acc');
                if (accEl) {
                    const val = node?.test_accuracy || node?.best_val_accuracy || node?.best_accuracy;
                    accEl.innerText = formatValue('acc', val);
                }
            });
            
            if (!window.currentActiveModel) {
                const selector = document.getElementById('modelSelector');
                window.currentActiveModel = window.persistedVotingModeEnabled ? 'ensemble_vote' : (selector?.value || 'pure_bilstm');
            }
            if (!window.persistedVotingModeEnabled) {
                try { window.selectModel?.(window.currentActiveModel); } catch(e) {}
            }
        } catch(e) { console.warn('loadModelMetrics failed', e); }
    }
    window.loadModelMetrics = loadModelMetrics;

    // ==================== 事件绑定 ====================
    function bindEventHandlers() {
        const elements = {
            fileInput: document.getElementById('fileInput'),
            chooseFileBtn: document.getElementById('chooseFileBtn'),
            liveToggleBtn: document.getElementById('liveToggleBtn'),
            selectedDisplay: document.getElementById('selectedDisplay'),
            dropdownMenu: document.getElementById('dropdownMenu'),
            refreshIfaces: document.getElementById('refreshIfaces'),
            recordFilterAll: document.getElementById('recordFilterAll'),
            recordFilterBenign: document.getElementById('recordFilterBenign'),
            recordFilterMalware: document.getElementById('recordFilterMalware')
        };

        if (elements.fileInput) {
            elements.fileInput.onchange = (e) => window.handleFiles?.(e.target.files);
        }
        if (elements.chooseFileBtn) {
            elements.chooseFileBtn.onclick = () => document.getElementById('fileInput')?.click();
        }

        // 上传区拖拽。页面上写着"拖拽文件到此处"，但之前只绑了"选择文件"按钮，
        // 拖进去没反应。注意 accept=".pcap,.pcapng" 只过滤文件选择框，拖拽不受限制，
        // 所以拖入非 pcap 文件会走后端 400 校验，界面能给出提示。
        const dropZoneEl = document.getElementById('dropZone');
        if (dropZoneEl) {
            const HILITE = 'var(--ds-accent)';
            const clearDrag = () => { dropZoneEl.style.outline = ''; dropZoneEl.style.outlineOffset = ''; };
            const showDrag = () => { dropZoneEl.style.outline = '2px dashed ' + HILITE; dropZoneEl.style.outlineOffset = '-6px'; };
            dropZoneEl.addEventListener('dragenter', (e) => { e.preventDefault(); showDrag(); });
            dropZoneEl.addEventListener('dragover', (e) => {
                e.preventDefault();
                if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
                showDrag();
            });
            dropZoneEl.addEventListener('dragleave', (e) => {
                if (!dropZoneEl.contains(e.relatedTarget)) clearDrag();
            });
            dropZoneEl.addEventListener('drop', (e) => {
                e.preventDefault();
                clearDrag();
                const dropped = e.dataTransfer && e.dataTransfer.files;
                if (dropped && dropped.length) window.handleFiles?.(dropped);
            });
            dropZoneEl.addEventListener('click', (e) => {
                // 点「选择文件」按钮时不要再冒泡触发一次，避免弹两次选择框
                if (e.target && e.target.closest && e.target.closest('#chooseFileBtn')) return;
                document.getElementById('fileInput')?.click();
            });
        }
        if (elements.liveToggleBtn) {
            elements.liveToggleBtn.onclick = () => window.toggleLiveCapture?.();
        }
        
        if (elements.selectedDisplay && elements.dropdownMenu) {
            elements.selectedDisplay.onclick = () => {
                elements.dropdownMenu.classList.toggle('hidden');
                const arrow = document.getElementById('dropdownArrow');
                if (arrow) arrow.style.transform = elements.dropdownMenu.classList.contains('hidden') ? '' : 'rotate(180deg)';
            };
        }
        if (elements.refreshIfaces) {
            elements.refreshIfaces.onclick = () => loadInterfaces();
        }
        // Model selector dropdown change -> update active model and UI
        const modelSelectorEl = document.getElementById('modelSelector');
        if (modelSelectorEl) {
            modelSelectorEl.onchange = (e) => {
                try { window.selectModel?.(e.target.value); } catch (err) { console.warn('modelSelector onchange failed', err); }
            };
        }
        if (elements.recordFilterAll) {
            elements.recordFilterAll.onclick = () => window.applyRecordFilter?.('all');
        }
        if (elements.recordFilterBenign) {
            elements.recordFilterBenign.onclick = () => window.applyRecordFilter?.('benign');
        }
        if (elements.recordFilterMalware) {
            elements.recordFilterMalware.onclick = () => window.applyRecordFilter?.('malware');
        }

        initRecordFilterIndicator();
    }

    // ==================== 模块加载器 ====================
    const assetVersion = '20261103f';
    const modulePaths = [
        '/static/js/core/state.js?v=' + assetVersion,
        '/static/js/core/utils.js?v=' + assetVersion,
        '/static/js/core/api.js?v=' + assetVersion,
        '/static/js/modules/upload_impl.js?v=' + assetVersion,
        '/static/js/modules/upload_record_card_impl.js?v=' + assetVersion,
        '/static/js/modules/live_record_card_impl.js?v=' + assetVersion,
        '/static/js/modules/upload_trend_impl.js?v=' + assetVersion,
        '/static/js/modules/live_trend_impl.js?v=' + assetVersion,
        '/static/js/modules/modelEval_impl.js?v=' + assetVersion,
        '/static/js/modules/results_impl.js?v=' + assetVersion,
        '/static/js/modules/flow_impl.js?v=' + assetVersion,
        '/static/js/modules/detect_impl.js?v=' + assetVersion,
        '/static/js/modules/live_impl.js?v=' + assetVersion,
        '/static/js/modules/review_impl.js?v=' + assetVersion,
        '/static/js/modules/ensemble_impl.js?v=' + assetVersion,
        '/static/js/modules/live_views_impl.js?v=' + assetVersion,
        '/static/js/modules/router_impl.js?v=' + assetVersion
    ];

    function loadScript(src) {
        return new Promise((resolve, reject) => {
            if (document.querySelector(`script[src="${src}"]`)) {
                resolve();
                return;
            }
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = () => reject(new Error(`Failed: ${src}`));
            document.head.appendChild(script);
        });
    }

    /* ==================== 每页初始化 / 拆卸 ====================
       这两个函数是无刷新换页的接口：router_impl 在替换内容区之前调 teardownPage()，
       替换完成后调 initPage()。首屏也走同一套路径，所以「整页加载」和「站内换页」
       之后的状态是一致的，不存在两套逻辑。 */
    /* 顶栏「运行配置」浮层：开合、点外部关闭、Esc 关闭。

       绑在这里而不是模板的内联 <script>：站内换页只替换 .app-content，
       顶栏是外壳的一部分，内联脚本不会重新执行；反过来，一旦把开合押在
       「用户是直接打开页面还是从侧栏点进来」上，就会出现「刷新一下就好、
       点进来就打不开」这种页面级随机故障。

       幂等靠 data-cfg-bound：首屏和每次换页都会走到 initPage，不能叠加监听器。
       注意 document 上的两个监听器只在首次绑定时挂载，btn 本身就是长期存在的
       外壳节点，所以它们也不会随着换页越积越多。 */
    function bindRuntimePopover() {
        var btn = document.getElementById('runtimeCfgBtn');
        var pop = document.getElementById('runtimeCfgPop');
        if (!btn || !pop) return;
        if (btn.getAttribute('data-cfg-bound') === '1') return;
        btn.setAttribute('data-cfg-bound', '1');

        function setOpen(open) {
            pop.hidden = !open;
            pop.classList.toggle('is-open', open);
            btn.setAttribute('aria-expanded', open ? 'true' : 'false');
            btn.classList.toggle('is-active', open);
        }

        setOpen(false);

        btn.addEventListener('click', function (e) {
            e.stopPropagation();
            setOpen(pop.hidden);
        });
        // 点浮层内部不该关掉它（里面是下拉框和复选框）
        pop.addEventListener('click', function (e) { e.stopPropagation(); });
        document.addEventListener('click', function () {
            if (!pop.hidden) setOpen(false);
        });
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && !pop.hidden) setOpen(false);
        });
    }

    function initPage() {
        // 页面模式由当前页面的 data-mode 决定，换页后必须重算（上传页 / 实时页 / 合并页）
        window.recordTableMode = resolvePageMode();

        bindEventHandlers();
        // 顶栏浮层：外壳不随换页重建，但首屏和每次换页都要确保它可用
        bindRuntimePopover();
        // 绑定结果表详情委托（由 results_impl 提供；内部按 tbody 上的标记做了幂等）
        try { window.resultsImpl && typeof window.resultsImpl.bindDetailButtonEvents === 'function' && window.resultsImpl.bindDetailButtonEvents(); } catch(e) { console.warn('bindDetailButtonEvents failed', e); }
        // 清除按钮：同样幂等，换页后要绑到新的那颗按钮上
        try { window.resultsImpl && typeof window.resultsImpl.bindClearButtons === 'function' && window.resultsImpl.bindClearButtons(); } catch(e) { console.warn('bindClearButtons failed', e); }
        // 这个模块的初始化挂在 DOMContentLoaded 上，换页不会再触发，必须手动补一次
        try { window.uploadImpl && typeof window.uploadImpl.initVotingModeControls === 'function' && window.uploadImpl.initVotingModeControls(); } catch(e) { console.warn('initVotingModeControls failed', e); }
        // 主机辐射图：只有实时页有 #liveFlowGraph，其它页面它会立刻返回
        try { window.liveGraph && typeof window.liveGraph.init === 'function' && window.liveGraph.init(); } catch(e) { console.warn('liveGraph.init failed', e); }
        // 流量分类页（/flow）：选择文件、阈值、筛选、AI 解释、环境校准全部在这个模块里。
        // 它只对 #flowDropZone 存在时生效，其它页面直接返回；内部按 DOM 标记做了幂等。
        try { window.flowImpl && typeof window.flowImpl.init === 'function' && window.flowImpl.init(); } catch(e) { console.warn('flowImpl.init failed', e); }
        // 检测工作台（/upload 与 /live 共用 detect.html）：按地址/上次选择展开子视图，
        // 并把 recordTableMode 与威胁卡 scope 一起切过去。必须在下面的 setTimeout 之前，
        // 那段定时任务才会按新的 recordTableMode 重画记录表。
        try { window.detectImpl && typeof window.detectImpl.init === 'function' && window.detectImpl.init(); } catch(e) { console.warn('detectImpl.init failed', e); }
        // 复核中心（/review）：上传检测与实时抓包的待复核流量统一处置台。
        // 只有 #reviewQueueList 存在时才启动轮询，其它页面直接返回。
        try { window.reviewImpl && typeof window.reviewImpl.init === 'function' && window.reviewImpl.init(); } catch(e) { console.warn('reviewImpl.init failed', e); }
        // 模型评估页的「分域投票权重」：只在页面里有该面板时读接口并绑定开关。
        try { window.ensembleImpl && typeof window.ensembleImpl.init === 'function' && window.ensembleImpl.init(); } catch(e) { console.warn('ensembleImpl.init failed', e); }
        // 实时抓包检测卡片的视图切换（辐射图 / 桑基图）：只在 #liveFlowGraphWrap 存在时生效。
        try { window.liveViewsImpl && typeof window.liveViewsImpl.init === 'function' && window.liveViewsImpl.init(); } catch(e) { console.warn('liveViewsImpl.init failed', e); }

        setTimeout(() => {
            loadInterfaces();
            loadModelMetrics();
            window.initCharts?.();
            window.initModelEvalCharts?.();
            try {
                const uploadRecords = window.recordStores && Array.isArray(window.recordStores.upload) ? window.recordStores.upload : [];
                const liveRecords = window.recordStores && Array.isArray(window.recordStores.live) ? window.recordStores.live : [];
                // 总览页（mode='all'）的折线要画「上传 + 实时」合并曲线：
                // 它的记录表本来就是两路合并的，趋势只画上传就会出现「表里有几十条、曲线一个点都没有」。
                // 合并数据只用于画线，不写回 dlidsStats.upload（否则 KPI 会把实时会话重复算一次）。
                const mergedTrend = String(window.recordTableMode || '') === 'all';
                const trendRecords = mergedTrend ? uploadRecords.concat(liveRecords) : uploadRecords;
                if (window.uploadTrendImpl && typeof window.uploadTrendImpl.restoreFromRecords === 'function') {
                    window.uploadTrendImpl.restoreFromRecords(trendRecords, { mergeStats: mergedTrend, statsRecords: uploadRecords });
                }
                // 合并模式下实时数据已经在上面那条线里，不再单独喂给 liveTrendImpl
                if (!mergedTrend && window.liveTrendImpl && typeof window.liveTrendImpl.restoreFromRecords === 'function') {
                    window.liveTrendImpl.restoreFromRecords(liveRecords);
                }
                if (window.liveGraph && typeof window.liveGraph.restoreFromRecords === 'function') {
                    window.liveGraph.restoreFromRecords(liveRecords);
                }
            } catch (e) { console.warn('restore chart state failed', e); }

            // 顶栏状态胶囊在内容区之外，但实时页卡片上的「运行中 / 停止」在内容区里，
            // 换页后是新节点，必须强制重画一次（setLiveRunningUI 默认在状态没变时会直接返回）。
            try { if (typeof window.setLiveRunningUI === 'function') window.setLiveRunningUI(!!(window.liveUiState && window.liveUiState.running), { force: true }); } catch (e) {}

            // 按当前页面的 mode 把落盘的历史记录画进表格：上传页只画上传、实时页只画实时、
            // 总览与记录页把两路合并。
            try {
                if (typeof window.renderRecordedRows === 'function') window.renderRecordedRows(window.recordTableMode);
            } catch (e) { console.warn('restore record rows failed', e); }
            try {
                if (typeof window.applyRecordFilter === 'function') {
                    window.applyRecordFilter(window.currentRecordFilter || 'all');
                }
            } catch (e) { console.warn('restore record filter failed', e); }

            // KPI 带与饼图的新 DOM 也是刚生成的（模板里是初始的 0），而 window.dlidsStats
            // 一直活在窗口上没被换页清掉 —— 统一统计层是唯一写这两处的地方，所以最后必须
            // 按它重画一次，否则换页后 KPI 会「回到 0」，与记录表里的行数自相矛盾。
            try { if (typeof window.dlidsRenderStats === 'function') window.dlidsRenderStats(); } catch (e) { console.warn('repaint stats failed', e); }
        }, 100);
    }

    function teardownPage() {
        // 内容区即将被替换。这里不能只依赖 upload_trend_impl 的 isChartInstance()：
        // ECharts 的 dispose() 之后 setOption / getOption 仍然是函数，旧实例会被当成
        // 有效实例复用 → 新页面上 setOption 抛错被 catch 吞掉，图表永久空白。
        // 所以必须显式 dispose 并把全局引用清零。
        const dispose = (inst) => {
            try { if (inst && typeof inst.dispose === 'function' && typeof inst.setOption === 'function') inst.dispose(); } catch (e) {}
        };
        dispose(window.uploadTrendChart);
        dispose(window.uploadPieChart);
        dispose(window.liveTrendChart);
        dispose(window.lineChart);
        dispose(window.pieChart);
        window.uploadTrendChart = null;
        window.uploadPieChart = null;
        window.liveTrendChart = null;
        // 这两个名字会被 HTML 命名元素访问解析成 <div>，置 null 是让 getLineChart / getPieChart
        // 回落到 echarts.getInstanceByDom（新容器没有实例 → 重新 init）。
        window.lineChart = null;
        window.pieChart = null;

        // 模型评估页的实例登记在模块内部的 instances 表里，必须让它自己清
        try { window.modelEvalImpl && typeof window.modelEvalImpl.disposeAll === 'function' && window.modelEvalImpl.disposeAll(); } catch (e) { console.warn('modelEval disposeAll failed', e); }
        // 主机辐射图是逐帧循环画的：旧 canvas 一走就要停，否则一直空转，而且重进实时页时
        // startLoop 会被 _animLoopBound 挡住，新 canvas 永远不画。
        try { window.liveGraph && typeof window.liveGraph.stopLoop === 'function' && window.liveGraph.stopLoop(); } catch (e) { console.warn('liveGraph stopLoop failed', e); }
        // 检测工作台留下的临时状态（威胁卡 scope 覆盖、顶栏抓包按钮显隐）要收干净，
        // 否则记录页/模型评估页会带着实时子视图的 scope 去画威胁类型分布。
        try { window.detectImpl && typeof window.detectImpl.teardown === 'function' && window.detectImpl.teardown(); } catch (e) { console.warn('detectImpl.teardown failed', e); }
        // 复核中心的定时刷新必须随页面卸载停掉，否则换页后仍在打 /api/review/queue
        try { window.reviewImpl && typeof window.reviewImpl.teardown === 'function' && window.reviewImpl.teardown(); } catch (e) { console.warn('reviewImpl.teardown failed', e); }
        // 分域权重面板的低频轮询同理：离开模型评估页就停。
        try { window.ensembleImpl && typeof window.ensembleImpl.teardown === 'function' && window.ensembleImpl.teardown(); } catch (e) { console.warn('ensembleImpl.teardown failed', e); }
        // 桑基图实例与 resize 定时器随页面一起销毁，避免换页后仍在计算。
        try { window.liveViewsImpl && typeof window.liveViewsImpl.teardown === 'function' && window.liveViewsImpl.teardown(); } catch (e) { console.warn('liveViewsImpl.teardown failed', e); }
    }

    window.dlidsPages = { initPage: initPage, teardownPage: teardownPage };

    async function init() {
        // 加载核心模块
        for (const path of modulePaths) {
            try {
                await loadScript(path);
                console.log(`✅ Loaded: ${path.split('/').pop()}`);
            } catch (e) {
                console.error(`❌ Failed: ${path}`, e);
            }
        }

        initPage();

        // 站内换页改成无刷新（见 modules/router_impl.js）。直接输网址与刷新仍是完整加载。
        try { window.routerImpl && typeof window.routerImpl.init === 'function' && window.routerImpl.init(); } catch (e) { console.warn('routerImpl init failed', e); }

        // 启动轮询（幂等：先清理可能已存在的定时器，避免 init 与 startLive 叠加出多个轮询器重复拉取）
        // 注意：这是全站唯一的轮询器，跨页面一直活着，不随换页重建 —— 否则换页会丢一帧抓包数据。
        if (window.pollLive) {
            if (window.livePoller) clearInterval(window.livePoller);
            window.pollLive();
            window.livePoller = setInterval(window.pollLive, LIVE_POLL_INTERVAL_MS);
        }

        console.log('✅ Main initialized');
    }

    // 启动
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();