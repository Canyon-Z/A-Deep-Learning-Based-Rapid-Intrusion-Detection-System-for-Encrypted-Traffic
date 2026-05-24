/* live record card behavior (size + scenario specific) */
(function(){
    window.liveRecordCardImpl = window.liveRecordCardImpl || {};

    window.liveRecordCardImpl.activate = function() {
        try {
            const panel = document.getElementById('resultCardPanel');
            const viewport = document.getElementById('resultTableViewport');
            const resultTitle = document.getElementById('resultTableTitle');
            const resultHint = document.getElementById('resultTableHint');

            if (resultTitle) resultTitle.innerText = '实时抓包记录';
            if (resultHint) resultHint.innerText = '仅记录当前实时抓包会话中的流量';

            if (panel) panel.style.minHeight = '520px';
            if (viewport) viewport.style.maxHeight = '620px';

            window.recordTableMode = 'live';
            if (typeof window.renderRecordedRows === 'function') window.renderRecordedRows('live');
        } catch (e) {
            console.warn('liveRecordCardImpl.activate failed', e);
        }
    };

    // animate similarly to top card when activating
    (function(){
        const origActivate = window.liveRecordCardImpl.activate;
        window.liveRecordCardImpl.activate = function() {
            const panel = document.getElementById('resultCardPanel');
            try {
                if (typeof origActivate === 'function') origActivate.apply(this, arguments);
            } catch(e) { console.warn('liveRecordCardImpl.activate wrapper failed', e); }
            try {
                if (typeof window.setPanelVisible === 'function') {
                    window.setPanelVisible(panel, true, { enterTransform: 'translateX(28px) scale(0.985)' });
                } else if (panel) {
                    panel.classList.remove('hidden'); panel.style.opacity = '1'; panel.style.transform = '';
                }
                if (typeof window.animateMovedElement === 'function') {
                    requestAnimationFrame(() => window.animateMovedElement(panel));
                }
            } catch(e) {}
        };
    })();

    window.liveRecordCardImpl.deactivate = function() {
        try {
            const panel = document.getElementById('resultCardPanel');
            if (!panel) return;
            try {
                if (typeof window.setPanelVisible === 'function') {
                    window.setPanelVisible(panel, false, { exitTransform: 'translateX(-28px) scale(0.985)' });
                } else {
                    panel.style.opacity = '0';
                    panel.style.pointerEvents = 'none';
                    setTimeout(() => { panel.style.opacity = ''; panel.style.pointerEvents = ''; }, 320);
                }
            } catch(e) {}
        } catch (e) {
            console.warn('liveRecordCardImpl.deactivate failed', e);
        }
    };
})();
