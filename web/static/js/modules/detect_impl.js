/* ============================================================================
   检测工作台：上传检测 / 实时抓包 两个子视图
   ----------------------------------------------------------------------------
   背景：这两块原先各占一个整屏页面（/upload 与 /live），用户比对"同一个检测
   能力换一种输入方式"要来回跳。现在侧栏只保留一个「检测工作台」入口（/detect），
   两个子视图是同页里的两个 pane；旧的 /upload 与 /live 由后端 302 到
   /detect?tab=upload|live，书签不会失效。

   切换时必须同时做四件事，漏一件就会出现"视图换了但数据还是另一路"的错位：
     1. pane 的 hidden 与按钮 is-active（纯视觉）
     2. window.recordTableMode → 结果卡片重画哪一路记录
     3. window.dlidsDetectScope → 威胁类型分布读哪个数据桶
     4. 结果卡标题与顶栏「开始抓包」的显隐（页面标题保持「检测工作台」不变）
   地址故意不改：改了会触发路由判定，而且两个子视图本来就是同一页。
   ========================================================================== */
(function () {
    'use strict';

    var STORAGE_KEY = 'dlids_detect_tab';
    var NAV_HREF = '/detect';

    var META = {
        upload: {
            recordTitle: '上传检测记录',
            recordHint: '只记录上传文件的结果',
            tabsHint: '拖入 PCAP 文件，批量分析加密流量'
        },
        live: {
            recordTitle: '实时抓包记录',
            recordHint: '只记录本次会话中抓到的流量',
            tabsHint: '选择网卡后，用右上角的「开始抓包」启动'
        }
    };

    var currentTab = 'upload';
    var tabsBound = false;

    function el(id) { return document.getElementById(id); }

    function queryTab() {
        // 旧链接 /upload、/live 被后端重定向成 /detect?tab=...，这里读一次
        try {
            var params = new URLSearchParams(window.location.search || '');
            var tab = params.get('tab');
            if (tab === 'live' || tab === 'upload') return tab;
        } catch (e) {}
        return null;
    }

    function resolveInitialTab() {
        var fromQuery = queryTab();
        if (fromQuery) return fromQuery;
        try {
            var saved = window.localStorage.getItem(STORAGE_KEY);
            if (saved === 'live' || saved === 'upload') return saved;
        } catch (e) {}
        return 'upload';
    }

    function syncPanes(tab) {
        var panes = document.querySelectorAll('[data-detect-pane]');
        Array.prototype.forEach.call(panes, function (pane) {
            var mine = pane.getAttribute('data-detect-pane') === tab;
            pane.classList.toggle('hidden', !mine);
        });
        var buttons = document.querySelectorAll('[data-detect-tab]');
        Array.prototype.forEach.call(buttons, function (btn) {
            var mine = btn.getAttribute('data-detect-tab') === tab;
            btn.classList.toggle('is-active', mine);
            btn.setAttribute('aria-selected', mine ? 'true' : 'false');
        });
    }

    function syncChrome(tab) {
        var meta = META[tab];
        var recordTitle = el('resultTableTitle');
        if (recordTitle) recordTitle.textContent = meta.recordTitle;
        var recordHint = el('resultTableHint');
        if (recordHint) recordHint.textContent = meta.recordHint;
        var tabsHint = el('detectTabsHint');
        if (tabsHint) tabsHint.textContent = meta.tabsHint;

        // 「开始抓包」与抓包状态只在实时子视图里说得通；留在上传子视图上会让人以为
        // 点了就能看到当前页的东西。
        var toggle = el('liveToggleBtn');
        var captureState = el('captureState');
        if (toggle) toggle.classList.toggle('hidden', tab !== 'live');
        if (captureState) captureState.classList.toggle('hidden', tab !== 'live');

        // 顶栏标题固定为「检测工作台」，不跟着子视图改：侧栏只有一项，
        // 标题再跳成「上传检测 / 实时抓包」会让人以为又换页了。
        // 侧栏高亮也只需保证这一项是亮的（合并后两种子视图共用同一项）。
        var navItems = document.querySelectorAll('.app-nav-item');
        Array.prototype.forEach.call(navItems, function (item) {
            var href = String(item.getAttribute('href') || '');
            if (href === NAV_HREF) item.classList.add('is-active');
        });
    }

    function repaintRecords(tab) {
        window.recordTableMode = tab;
        try {
            if (typeof window.renderRecordedRows === 'function') window.renderRecordedRows(tab);
        } catch (e) { console.warn('detect: renderRecordedRows failed', e); }
        try {
            if (typeof window.applyRecordFilter === 'function') {
                window.applyRecordFilter(window.currentRecordFilter || 'all');
            }
        } catch (e) { console.warn('detect: applyRecordFilter failed', e); }
    }

    function refreshReviewHint() {
        var hint = el('reviewEntryHint');
        if (!hint) return;
        fetch('/api/review/queue', {
            credentials: 'same-origin',
            headers: { 'X-Requested-With': 'fetch' }
        }).then(function (resp) {
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            return resp.json();
        }).then(function (data) {
            var pending = 0;
            if (data && data.stats && typeof data.stats.pending === 'number') pending = data.stats.pending;
            else if (data && Array.isArray(data.items)) pending = data.items.length;
            hint.textContent = pending + ' 条';
            hint.classList.toggle('text-cyber-danger', pending > 0);
        }).catch(function () {
            hint.textContent = '—';
        });
    }

    function resizeCharts(tab) {
        // 图表在隐藏容器里初始化时量到的是 0×0（ECharts 只在 init 时记一次尺寸），
        // 从另一个子视图切回来必须让它重新测量；否则折线图会空着或只画左上角一块。
        // 放在 rAF 里：此时 pane 刚从 hidden 变成可见，布局还没落到元素上。
        window.requestAnimationFrame(function () {
            if (tab !== 'upload') return;
            try {
                if (window.uploadTrendImpl && typeof window.uploadTrendImpl.activate === 'function') {
                    window.uploadTrendImpl.activate();
                }
            } catch (e) { console.warn('detect: uploadTrendImpl.activate failed', e); }
        });
    }

    function switchTab(tab, options) {
        var opts = options || {};
        if (tab !== 'upload' && tab !== 'live') return;
        currentTab = tab;

        syncPanes(tab);
        syncChrome(tab);

        // 威胁类型分布的取数范围（upload / live 两个数据桶），由 main.js 的
        // dlidsThreatScope 优先读这个覆盖值。
        window.dlidsDetectScope = tab;

        if (!opts.skipRecords) repaintRecords(tab);

        try { window.localStorage.setItem(STORAGE_KEY, tab); } catch (e) {}

        // 切到实时子视图时补一次轮询：否则要等下一个 1.5s 周期才看到当前状态。
        if (tab === 'live' && !opts.initial) {
            try { if (typeof window.pollLive === 'function') window.pollLive(); } catch (e) {}
        }
        // 威胁卡的内容其实由 dlidsStats 统一画，这里补一次重画保证切完即变。
        try { if (typeof window.dlidsRenderStats === 'function') window.dlidsRenderStats(); } catch (e) {}

        resizeCharts(tab);
    }

    function bindTabs() {
        var tabs = el('detectTabs');
        // 幂等标记必须挂在元素上，不能用模块级变量：站内换页时 router 会把
        // .app-content 整块替换掉，旧的 #detectTabs 连同监听器一起消失，
        // 而模块变量还是 true —— 新 DOM 就永远绑不上，点 tab 毫无反应（刷新才恢复）。
        if (!tabs || tabs.dataset.detectBound === '1') return;
        tabs.dataset.detectBound = '1';
        tabs.addEventListener('click', function (ev) {
            var btn = ev.target.closest ? ev.target.closest('[data-detect-tab]') : null;
            if (!btn || !tabs.contains(btn)) return;
            switchTab(btn.getAttribute('data-detect-tab'), {});
        });
    }

    function init() {
        var tabs = el('detectTabs');
        if (!tabs) return;                    // 不是检测工作台页
        bindTabs();
        switchTab(resolveInitialTab(), { initial: true });
        refreshReviewHint();
    }

    function teardown() {
        // 覆盖值与顶栏的显隐都是"这一页的临时状态"，离开页面必须收干净，
        // 否则记录页/模型页会带着实时子视图的 scope 去画威胁卡。
        window.dlidsDetectScope = null;
        var toggle = el('liveToggleBtn');
        var captureState = el('captureState');
        if (toggle) toggle.classList.remove('hidden');
        if (captureState) captureState.classList.remove('hidden');
    }

    window.detectImpl = {
        init: init,
        teardown: teardown,
        switchTab: switchTab,
        refreshReviewHint: refreshReviewHint,
        getTab: function () { return currentTab; }
    };
})();
