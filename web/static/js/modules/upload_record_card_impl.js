/* upload record card behavior (size + scenario specific) */
(function(){
    window.uploadRecordCardImpl = window.uploadRecordCardImpl || {};

    window.uploadRecordCardImpl.activate = function() {
        try {
            const panel = document.getElementById('resultCardPanel');
            const viewport = document.getElementById('resultTableViewport');
            const resultTitle = document.getElementById('resultTableTitle');
            const resultHint = document.getElementById('resultTableHint');

            if (resultTitle) resultTitle.innerText = '上传检测记录';
            if (resultHint) resultHint.innerText = '只记录上传检测文件的结果';

            if (panel) panel.style.minHeight = '600px';
            if (viewport) viewport.style.maxHeight = '700px';

            window.recordTableMode = 'upload';
            if (typeof window.renderRecordedRows === 'function') window.renderRecordedRows('upload');

            // ensure visible then run the micro-move animation to match top-card panels
            try {
                if (typeof window.setPanelVisible === 'function') {
                    window.setPanelVisible(panel, true, { enterTransform: 'translateX(28px) scale(0.985)' });
                } else if (panel) {
                    panel.classList.remove('hidden'); panel.style.opacity = '1'; panel.style.transform = '';
                }
                if (typeof window.animateMovedElement === 'function') {
                    // run on next frame so setPanelVisible has applied
                    requestAnimationFrame(() => window.animateMovedElement(panel));
                }
            } catch (e) { /* non-fatal */ }
        } catch (e) {
            console.warn('uploadRecordCardImpl.activate failed', e);
        }
    };

    // provide a deactivate hook for symmetry (not required but useful)
    window.uploadRecordCardImpl.deactivate = function() {
        try {
            const panel = document.getElementById('resultCardPanel');
            if (!panel) return;
            try {
                if (typeof window.setPanelVisible === 'function') {
                    window.setPanelVisible(panel, false, { exitTransform: 'translateX(-28px) scale(0.985)' });
                } else if (typeof window.animateMovedElement === 'function') {
                    // quick fade out then restore
                    panel.style.opacity = '0';
                    panel.style.pointerEvents = 'none';
                    setTimeout(() => { panel.style.opacity = ''; panel.style.pointerEvents = ''; }, 320);
                }
            } catch (e) {}
        } catch (e) {
            console.warn('uploadRecordCardImpl.deactivate failed', e);
        }
    };
})();
