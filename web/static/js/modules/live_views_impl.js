/* 实时抓包检测卡片的「视图切换」：辐射图（canvas，默认）与桑基图（ECharts）。
 *
 * 为什么不把辐射图换掉：两者回答的是不同问题。
 *   辐射图 —— 对端在哪个方位、哪些会话正在跑、有没有异常方向；
 *   桑基图 —— 谁占了多少流量、恶意与正常各占多少，一眼能看出「量」。
 * 桑基图的连线宽度就是字节数，这是辐射图给不出的信息。
 *
 * 数据源：liveGraph.js 每次轮询会派发 dlids:live-flows 事件（detail 是 flows 数组），
 * 这里只做聚合与渲染，不额外请求后端。
 */
(function () {
    var STORAGE_KEY = 'dlids_live_view';
    var VIEWS = ['radial', 'sankey'];
    var state = {
        view: 'radial',
        chart: null,
        flows: [],
        bound: false,
        timer: null
    };

    function el(id) { return document.getElementById(id); }

    function resolveView() {
        try {
            var saved = window.localStorage && window.localStorage.getItem(STORAGE_KEY);
            if (saved && VIEWS.indexOf(saved) !== -1) return saved;
        } catch (e) { /* 隐私模式下没有 localStorage */ }
        return 'radial';
    }

    function applyView(view) {
        if (VIEWS.indexOf(view) === -1) view = 'radial';
        state.view = view;

        var radial = el('liveFlowGraph');
        var sankey = el('liveSankeyGraph');
        var tooltip = el('liveFlowTooltip');
        if (radial) radial.classList.toggle('hidden', view !== 'radial');
        if (sankey) sankey.classList.toggle('hidden', view !== 'sankey');
        if (tooltip && view !== 'radial') tooltip.classList.add('hidden');

        Array.prototype.forEach.call(document.querySelectorAll('[data-live-view]'), function (btn) {
            btn.classList.toggle('is-active', btn.getAttribute('data-live-view') === view);
        });

        // 缩放按钮只对辐射图有意义
        ['liveFlowZoomOut', 'liveFlowZoomReset', 'liveFlowZoomIn'].forEach(function (id) {
            var b = el(id);
            if (b) b.classList.toggle('hidden', view !== 'radial');
        });

        try {
            if (window.localStorage) window.localStorage.setItem(STORAGE_KEY, view);
        } catch (e) { /* ignore */ }

        if (view === 'sankey') renderSankey();
        else disposeChart();
    }

    function disposeChart() {
        if (!state.chart) return;
        try { state.chart.dispose(); } catch (e) { /* ignore */ }
        state.chart = null;
    }

    /* 把 flows 聚合成「本机 → 对端」的两级桑基：节点是主机，连线的值是该主机的流量字节。 */
    function buildSankeyData(flows) {
        var peers = {};
        var total = 0;
        var localCount = 0;
        (flows || []).forEach(function (f) {
            var bytes = Number(f.bytes) || 0;
            if (!bytes) return;
            var src = f.src || f.src_ip;
            var dst = f.dst || f.dst_ip;
            if (!src || !dst) return;
            // 与辐射图同一套判定：私网一侧视作「本机」
            var srcPrivate = isPrivate(src);
            var peerIp = srcPrivate ? dst : src;
            if (srcPrivate) localCount += 1;
            var key = peerIp;
            if (!peers[key]) {
                peers[key] = { ip: peerIp, bytes: 0, sessions: 0, malicious: false, protocols: {} };
            }
            var item = peers[key];
            item.bytes += bytes;
            item.sessions += 1;
            if (f.malicious) item.malicious = true;
            var proto = f.protocol || f.proto || '?';
            item.protocols[proto] = (item.protocols[proto] || 0) + bytes;
            total += bytes;
        });

        var items = Object.keys(peers).map(function (k) { return peers[k]; });
        items.sort(function (a, b) { return b.bytes - a.bytes; });
        items = items.slice(0, 14);   // 桑基图节点太多就成了一堵墙

        var nodes = [{ name: localLabel(localCount), itemStyle: { color: '#22D3EE' } }];
        var links = [];
        var visibleTotal = 0;
        items.forEach(function (p) {
            nodes.push({
                name: p.ip,
                itemStyle: { color: p.malicious ? '#F43F5E' : '#10B981' }
            });
            links.push({
                source: nodes[0].name,
                target: p.ip,
                value: p.bytes,
                // 恶意链路的连线也染色，避免只能靠节点颜色区分
                lineStyle: { color: p.malicious ? 'rgba(244,63,94,0.55)' : 'rgba(16,185,129,0.35)' }
            });
            visibleTotal += p.bytes;
        });

        return { nodes: nodes, links: links, total: total, visibleTotal: visibleTotal, peers: items };
    }

    function localLabel(count) {
        return count > 0 ? '本机' : '本机';
    }

    function isPrivate(ip) {
        if (!ip) return false;
        var m = /^(\d+)\.(\d+)\./.exec(ip);
        if (!m) return false;
        var a = Number(m[1]);
        var b = Number(m[2]);
        if (a === 10) return true;
        if (a === 192 && b === 168) return true;
        if (a === 172 && b >= 16 && b <= 31) return true;
        if (a === 127) return true;
        if (a === 169 && b === 254) return true;
        return false;
    }

    function formatBytes(n) {
        n = Number(n) || 0;
        if (n < 1024) return n + ' B';
        if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
        if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(2) + ' MB';
        return (n / 1024 / 1024 / 1024).toFixed(2) + ' GB';
    }

    function ensureChart() {
        var host = el('liveSankeyGraph');
        if (!host || !window.echarts) return null;
        if (state.chart) return state.chart;
        state.chart = window.echarts.init(host, null, { renderer: 'canvas' });
        return state.chart;
    }

    function renderSankey() {
        var chart = ensureChart();
        if (!chart) return;
        var data = buildSankeyData(state.flows);

        if (!data.nodes.length || data.nodes.length < 2) {
            chart.clear();
            chart.setOption({
                title: {
                    text: '暂无实时流量',
                    left: 'center',
                    top: 'middle',
                    textStyle: { color: '#64748B', fontSize: 12, fontWeight: 'normal' }
                }
            });
            return;
        }

        chart.setOption({
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(15,23,42,0.95)',
                borderColor: '#2E3B4D',
                textStyle: { color: '#E6EDF3', fontSize: 11 },
                formatter: function (p) {
                    if (p.dataType === 'edge') {
                        var share = data.visibleTotal ? (p.value / data.visibleTotal * 100) : 0;
                        return p.data.source + ' → ' + p.data.target
                            + '<br/>流量 ' + formatBytes(p.value)
                            + '<br/>占已展示 ' + share.toFixed(1) + '%';
                    }
                    return p.name;
                }
            },
            series: [{
                type: 'sankey',
                left: 12,
                right: 120,
                top: 16,
                bottom: 16,
                nodeWidth: 12,
                nodeGap: 9,
                draggable: false,
                emphasis: { focus: 'adjacency' },
                label: {
                    color: '#94A3B8',
                    fontSize: 10.5,
                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace'
                },
                lineStyle: { color: 'gradient', curveness: 0.5, opacity: 0.42 },
                data: data.nodes,
                links: data.links
            }]
        }, true);
    }

    function onFlows(ev) {
        state.flows = (ev && ev.detail) || [];
        if (state.view === 'sankey') renderSankey();
    }

    function bindOnce() {
        // window 上的监听器只需登记一次：window 不随换页消失，重复添加只会重复渲染。
        if (!state.windowBound) {
            state.windowBound = true;
            window.addEventListener('dlids:live-flows', onFlows);
            // 容器尺寸变化（侧栏折叠、窗口缩放）时桑基图要重新排
            window.addEventListener('resize', function () {
                if (state.view !== 'sankey' || !state.chart) return;
                clearTimeout(state.timer);
                state.timer = setTimeout(function () {
                    if (state.chart) { try { state.chart.resize(); } catch (e) { /* ignore */ } }
                }, 160);
            });
        }
        // 按钮是这一页的 DOM，router 换页后会被替换掉，必须重绑 ——
        // 之前用模块级 state.bound 做幂等，换页回来按钮就点了没反应。
        var wrap = el('liveFlowGraphWrap');
        if (!wrap || wrap.dataset.lvBound === '1') return;
        wrap.dataset.lvBound = '1';
        Array.prototype.forEach.call(document.querySelectorAll('[data-live-view]'), function (btn) {
            btn.addEventListener('click', function () {
                applyView(btn.getAttribute('data-live-view'));
            });
        });
    }

    function init() {
        var wrap = el('liveFlowGraphWrap');
        if (!wrap) return;
        bindOnce();
        applyView(resolveView());
        if (state.view === 'sankey') renderSankey();
    }

    function teardown() {
        disposeChart();
        state.flows = [];
    }

    window.liveViewsImpl = {
        init: init,
        teardown: teardown,
        applyView: applyView,
        getView: function () { return state.view; }
    };
})();
