/* 站内无刷新切换（pjax 式）

   五个页面仍是真实网址、仍由后端 Jinja 渲染：直接输网址、刷新、从外部点进来，都是完整加载。
   只有「已经在站内、点导航」这一种情况改成：
       fetch 目标页 → 只替换 <main class="app-content"> 的内容 → history.pushState

   为什么不整页跳：侧栏、顶栏，以及顶栏上那颗「开始 / 停止抓包」按钮，全都在内容区之外。
   只换内容区，它们就天然不会被重画，正在跑的抓包也不会被打断 —— 这正是整页跳做不到的事。

   为什么不做成纯前端单页应用：五个模板、模型评估页的数据直出、以及表单这些都已经按
   「服务端渲染 + 真实网址」写好了，换页只需替换一小块 DOM，没必要把整套渲染搬进 JS。 */
(function () {
    'use strict';

    const CONTENT_SELECTOR = '.app-content';
    const NAV_SELECTOR = '.app-nav-item';
    const SLOW_HINT_MS = 150;    // 超过这个时长才显示进度条：够快的时候一点痕迹都不该有
    const CACHE_TTL_MS = 30000;  // 预取结果的保鲜期

    const cache = new Map();     // url -> { doc, at }
    let seq = 0;                 // 请求序号：连点两个导航时，只让最后一次的结果落地
    let bound = false;
    let progressEl = null;
    let progressTimer = null;
    let progressDoneTimer = null;

    // 只有侧栏里挂着的这几个页面走无刷新；其余链接（退出登录、外部链接、以后可能加的接口链接）
    // 一律交回浏览器按正常方式处理。
    function pagePaths() {
        return Array.from(document.querySelectorAll(NAV_SELECTOR)).map(function (a) {
            return new URL(a.getAttribute('href'), location.href).pathname.replace(/\/+$/, '') || '/';
        });
    }

    function resolvePageLink(node) {
        const a = node && node.closest ? node.closest('a[href]') : null;
        if (!a) return null;
        const raw = a.getAttribute('href') || '';
        if (!raw || raw.charAt(0) === '#') return null;
        if (a.hasAttribute && (a.hasAttribute('download') || a.hasAttribute('data-no-router'))) return null;
        if (a.target && a.target !== '_self') return null;
        let url;
        try { url = new URL(a.href, location.href); } catch (e) { return null; }
        if (url.origin !== location.origin) return null;
        const path = url.pathname.replace(/\/+$/, '') || '/';
        // /logout 必须走真实跳转：它要在服务端把会话 cookie 清掉
        if (path === '/login' || path === '/logout') return null;
        if (pagePaths().indexOf(path) === -1) return null;
        return url;
    }

    /* ----------------------------- 拉取 ----------------------------- */

    async function fetchDoc(url) {
        // 五个页面都要求会话 cookie，必须带凭据；不带的话拿回来的是登录页
        const res = await fetch(url, {
            credentials: 'same-origin',
            headers: { 'X-Requested-With': 'fetch' }
        });
        // 会话过期时后端会 302 到 /login，而 fetch 默认跟随重定向 —— 于是 200 + 登录页 HTML。
        // 必须认出这种情况，真的把浏览器送去登录页，而不是把登录页塞进内容区。
        if (res.redirected) {
            let path = '';
            try { path = new URL(res.url, location.href).pathname.replace(/\/+$/, ''); } catch (e) { path = ''; }
            if (path === '/login') { const err = new Error('session expired'); err.redirectTo = '/login'; throw err; }
        }
        if (!res.ok) { const err = new Error('HTTP ' + res.status); err.status = res.status; throw err; }
        const doc = new DOMParser().parseFromString(await res.text(), 'text/html');
        if (!doc.querySelector(CONTENT_SELECTOR)) { const err = new Error('missing content region'); err.fallback = true; throw err; }
        return doc;
    }

    function prefetch(url) {
        if (cache.has(url)) return;
        fetchDoc(url).then(function (doc) {
            cache.set(url, { doc: doc, at: Date.now() });
        }).catch(function () { /* 预取失败无所谓，真点的时候会再拉一次 */ });
    }

    /* ----------------------------- 进度条 ----------------------------- */

    function ensureProgressEl() {
        if (progressEl && document.body.contains(progressEl)) return progressEl;
        progressEl = document.createElement('div');
        progressEl.className = 'route-progress';
        progressEl.setAttribute('aria-hidden', 'true');
        document.body.appendChild(progressEl);
        return progressEl;
    }

    function showProgress() {
        clearTimeout(progressTimer);
        clearTimeout(progressDoneTimer);
        progressTimer = setTimeout(function () {
            const el = ensureProgressEl();
            el.classList.remove('is-done');
            el.classList.add('is-on');
        }, SLOW_HINT_MS);
    }

    function hideProgress() {
        clearTimeout(progressTimer);
        if (!progressEl) return;
        progressEl.classList.remove('is-on');
        progressEl.classList.add('is-done');
        progressDoneTimer = setTimeout(function () {
            if (progressEl) progressEl.classList.remove('is-done');
        }, 220);
    }

    /* ----------------------------- 落地 ----------------------------- */

    function syncNav(url) {
        const path = new URL(url, location.href).pathname.replace(/\/+$/, '') || '/';
        document.querySelectorAll(NAV_SELECTOR).forEach(function (a) {
            const ap = new URL(a.getAttribute('href'), location.href).pathname.replace(/\/+$/, '') || '/';
            const active = ap === path;
            a.classList.toggle('is-active', active);
            if (active) a.setAttribute('aria-current', 'page');
            else a.removeAttribute('aria-current');
        });
    }

    // innerHTML 插入的 <script> 不会执行。内容区里唯一的那个是统计面板的（KPI 带上的准确率），
    // 不重放的话换页后那一格会一直停在初始值。
    function runInlineScripts(root) {
        Array.from(root.querySelectorAll('script')).forEach(function (old) {
            const fresh = document.createElement('script');
            Array.from(old.attributes).forEach(function (attr) { fresh.setAttribute(attr.name, attr.value); });
            fresh.text = old.textContent;
            if (old.parentNode) old.parentNode.replaceChild(fresh, old);
        });
    }

    function closeModalIfOpen() {
        try {
            const modal = document.getElementById('detailModal');
            if (!modal || modal.classList.contains('hidden')) return;
            if (typeof window.closeModal === 'function') window.closeModal();
            else { modal.classList.add('hidden'); modal.classList.remove('flex'); }
        } catch (e) {}
    }

    function applyDoc(doc, url, options) {
        const current = document.querySelector(CONTENT_SELECTOR);
        const incoming = doc.querySelector(CONTENT_SELECTOR);
        if (!current || !incoming) { location.href = url; return false; }

        closeModalIfOpen();

        // 内容区即将被换掉：图表实例必须显式销毁并清掉全局引用，辐射图的逐帧循环必须停，
        // 否则旧实例会被当成有效实例复用（新页面的图永远空白），或者白白烧 CPU。
        try { window.dlidsPages && window.dlidsPages.teardownPage && window.dlidsPages.teardownPage(); } catch (e) { console.warn('teardownPage failed', e); }

        current.innerHTML = incoming.innerHTML;

        // 顶栏标题与副标题也在内容区之外，单独更新
        const newTitle = doc.querySelector('.app-page-title');
        const curTitle = document.querySelector('.app-page-title');
        if (newTitle && curTitle) curTitle.textContent = newTitle.textContent;
        const newSub = doc.querySelector('.app-page-sub');
        const curSub = document.querySelector('.app-page-sub');
        if (newSub && curSub) curSub.textContent = newSub.textContent;

        syncNav(url);
        runInlineScripts(current);
        if (doc.title) document.title = doc.title;

        if (options.push) {
            // 把当前滚动位置存进当前这条历史记录，后退时才能回到原处
            try {
                history.replaceState(Object.assign({}, history.state, { dlids: true, scrollY: window.scrollY }), '', location.href);
                history.pushState({ dlids: true, scrollY: 0 }, '', url);
            } catch (e) { console.warn('pushState failed', e); }
        }
        const targetY = options.push ? 0 : ((history.state && Number(history.state.scrollY)) || 0);
        window.scrollTo(0, targetY);

        try { window.dlidsPages && window.dlidsPages.initPage && window.dlidsPages.initPage(); } catch (e) { console.warn('initPage failed', e); }

        // 与整页跳转时一致的短淡入（沿用全站动效规范里的 .mo-in）
        current.classList.remove('mo-in');
        void current.offsetWidth;
        current.classList.add('mo-in');
        return true;
    }

    async function navigate(url, opts) {
        const options = opts || {};
        const mySeq = ++seq;
        showProgress();

        let doc;
        try {
            const hit = cache.get(url);
            if (hit && (Date.now() - hit.at) < CACHE_TTL_MS) { doc = hit.doc; cache.delete(url); }
            else doc = await fetchDoc(url);
        } catch (e) {
            hideProgress();
            if (e && e.redirectTo) { location.href = e.redirectTo; return; }
            // 拉不到就把这次点击交回浏览器：真实网址还在，用户按整页导航继续走，不会卡住
            console.warn('routerImpl: 回退为整页导航', e);
            location.href = url;
            return;
        }
        if (mySeq !== seq) return;   // 又被点了别的导航，这次结果作废

        const ok = applyDoc(doc, url, options);
        hideProgress();
        if (!ok) return;
    }

    /* ----------------------------- 事件 ----------------------------- */

    function onClick(ev) {
        if (ev.defaultPrevented) return;
        if (ev.button !== 0 || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
        const url = resolvePageLink(ev.target);
        if (!url) return;
        ev.preventDefault();
        const next = url.pathname + url.search;
        if (next === location.pathname + location.search) return;   // 点的就是当前页
        navigate(next, { push: true });
    }

    function onHover(ev) {
        // mouseover 会冒泡，正好一次覆盖整条导航项；键盘 Tab 到链接时用 focusin 再兜一次
        const url = resolvePageLink(ev.target);
        if (!url) return;
        prefetch(url.pathname + url.search);
    }

    function onPopState() {
        navigate(location.pathname + location.search, { push: false });
    }

    function init() {
        if (bound) return;
        bound = true;
        document.addEventListener('click', onClick);
        document.addEventListener('mouseover', onHover);
        document.addEventListener('focusin', onHover);
        window.addEventListener('popstate', onPopState);
        // 未支持 pushState 的老浏览器：什么都不做，就是原来的整页导航
        window.routerImpl.enabled = !!(window.history && history.pushState && window.DOMParser);
    }

    window.routerImpl = window.routerImpl || {};
    window.routerImpl.init = init;
    window.routerImpl.navigate = navigate;
    window.routerImpl.prefetch = prefetch;
    window.routerImpl.enabled = !!(window.history && history.pushState && window.DOMParser);
})();
