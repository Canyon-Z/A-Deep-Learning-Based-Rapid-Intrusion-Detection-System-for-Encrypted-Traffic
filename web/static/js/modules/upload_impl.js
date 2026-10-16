/* Full implementations of upload-related functions moved from main.js
   They attach implementations onto `window.uploadImpl` so main.js can delegate. */
(function(){
    window.uploadImpl = window.uploadImpl || {};

    // keep references to DOM elements lazily
    function getEls(){
        return {
            uploadContent: document.getElementById('uploadContent'),
            loadingState: document.getElementById('loadingState'),
            dropZone: document.getElementById('dropZone'),
            processStatus: document.getElementById('processStatus'),
            modelSelector: document.getElementById('modelSelector')
        };
    }

    window.uploadImpl.handleFiles = function(files) {
        const fileList = Array.isArray(files) ? files : Array.from(files || []);
        if (!fileList.length) return;
        return window.uploadImpl.uploadFilesSequentially(fileList);
    };

    window.uploadImpl.readResponsePayload = async function(response) {
        const rawText = await response.text();
        if (!rawText) return null;
        try { return JSON.parse(rawText); } catch (err) { return rawText; }
    };

    window.uploadImpl.formatErrorPayload = function(payload, statusCode) {
        if (payload === null || payload === undefined) return `HTTP ${statusCode}: Analyze failed`;
        if (typeof payload === 'string') return `HTTP ${statusCode}: ${payload.trim() || 'Analyze failed'}`;
        if (Array.isArray(payload)) return `HTTP ${statusCode}: ${payload.map(item => window.uploadImpl.formatErrorPayload(item, statusCode).replace(/^HTTP\s+\d+:\s*/i, '')).join('; ')}`;
        if (typeof payload === 'object') {
            if (payload.error) return `HTTP ${statusCode}: ${payload.error}`;
            if (payload.message) return `HTTP ${statusCode}: ${payload.message}`;
            if (payload.detail) return `HTTP ${statusCode}: ${window.uploadImpl.formatErrorPayload(payload.detail, statusCode).replace(/^HTTP\s+\d+:\s*/i, '')}`;
            try { return `HTTP ${statusCode}: ${JSON.stringify(payload)}`; } catch (e) { return `HTTP ${statusCode}: Analyze failed`; }
        }
        return `HTTP ${statusCode}: ${String(payload)}`;
    };

    window.uploadImpl.syncModelSelection = function(modelId, modelName) {
        window.currentActiveModel = modelId;
        const modelSelector = document.getElementById('modelSelector');
        if (modelSelector) {
            modelSelector.value = modelId;
            Array.from(modelSelector.options).forEach((option) => { option.selected = option.value === modelId; });
        }
        // 选了具体模型就等于退出投票模式。这里不能只写 .checked —— 那样勾选框、文案、
        // 存档会各说各话（用户报的「勾选刷新后没保持」就是这么来的）。统一交给
        // setVotingModeEnabled(false) 处理；先判断当前是否勾着，避免与它内部回调互相递归。
        const votingModeToggle = document.getElementById('votingModeToggle');
        if (votingModeToggle && modelId !== 'ensemble_vote' && votingModeToggle.checked) {
            window.uploadImpl.setVotingModeEnabled(false);
        }
        const displayName = modelName || window.getModelDisplayName(modelId);
        const activeModelBadge = document.getElementById('activeModelBadge');
        if (activeModelBadge) activeModelBadge.innerText = displayName + ' 当前使用';
        const modelToPrefix = { cnn_bilstm: 'm1', classic_cnn: 'm2', lightweight_cnn: 'm3', pure_bilstm: 'm4', mlp: 'm5', transformer: 'm6' };
        const prefix = modelToPrefix[modelId] || ('m' + (['cnn_bilstm', 'classic_cnn', 'lightweight_cnn', 'pure_bilstm', 'mlp', 'transformer'].indexOf(modelId) + 1));
        // 准确率优先走 stats_panel 缓存下来的 model_metrics.json：
        // mN-acc 元素只在「模型评估」页存在，直接读它会让其他页面切模型后准确率不更新。
        if (typeof window.dlidsRenderActiveAcc === 'function') {
            window.dlidsRenderActiveAcc(modelId);
        } else {
            const accEl = document.getElementById(prefix + '-acc');
            const statActiveAcc = document.getElementById('statActiveAcc');
            if (accEl && statActiveAcc) {
                const accText = (accEl.innerText || '--').replace('%', '').trim() || '--';
                statActiveAcc.innerHTML = accText + '<span class="text-sm text-gray-500">%</span>';
            }
        }
        return displayName;
    };

    window.uploadImpl.setVotingModeEnabled = function(enabled) {
        const selectorBox = document.getElementById('modelSelectorBox');
        const selector = document.getElementById('modelSelector');
        const activeModelBadge = document.getElementById('activeModelBadge');
        const statActiveAcc = document.getElementById('statActiveAcc');
        const votingModeToggle = document.getElementById('votingModeToggle');
        const votingModeLabelText = document.getElementById('votingModeLabelText');

        if (votingModeToggle) votingModeToggle.checked = !!enabled;
        if (votingModeLabelText) {
            votingModeLabelText.innerText = enabled ? '6 模型投票已启用' : '单模型选择已启用';
        }
        if (selectorBox) selectorBox.classList.remove('hidden');
        if (selector) {
            selector.disabled = !!enabled;
            if (enabled) {
                selector.setAttribute('disabled', 'disabled');
            } else {
                selector.removeAttribute('disabled');
            }
            selector.setAttribute('aria-disabled', enabled ? 'true' : 'false');
            selector.classList.toggle('opacity-50', !!enabled);
            selector.classList.toggle('cursor-not-allowed', !!enabled);
        }

        if (enabled) {
            window.currentActiveModel = 'ensemble_vote';
            if (activeModelBadge) activeModelBadge.innerText = '6 模型投票';
            if (statActiveAcc) {
                statActiveAcc.innerHTML = '--<span class="text-sm text-gray-500">%</span>';
            }
            if (selector) selector.value = 'cnn_bilstm';
        } else {
            const fallbackModel = (selector && selector.value) || 'cnn_bilstm';
            window.uploadImpl.syncModelSelection(fallbackModel, window.getModelDisplayName(fallbackModel));
        }
        // 勾选状态是用户设置：两个分支都要落盘，不能只落「取消」那一边
        try { window.persistUiState && window.persistUiState(); } catch (e) {}
    };

    /* 全站唯一的「当前是不是投票模式」判据。
       模型评估页要拿它决定「默认展开哪张卡的明细」：投票模式下 currentActiveModel 是
       ensemble_vote，不在 6 个单模型里，若照旧回退到 MODEL_LIST[0] 并调 selectModel，
       就会把用户刚勾上的投票清掉（并把 KPI 换成单模型准确率）。 */
    window.dlidsIsEnsembleMode = function() {
        if (window.currentActiveModel === 'ensemble_vote') return true;
        const toggle = document.getElementById('votingModeToggle');
        return !!(toggle && toggle.checked);
    };

    function initVotingModeControls() {
        const votingModeToggle = document.getElementById('votingModeToggle');
        const selector = document.getElementById('modelSelector');
        if (votingModeToggle && typeof window.persistedVotingModeEnabled === 'boolean') {
            votingModeToggle.checked = !!window.persistedVotingModeEnabled;
        }
        if (votingModeToggle) {
            window.uploadImpl.setVotingModeEnabled(!!votingModeToggle.checked);
        }
        if (votingModeToggle && !votingModeToggle._boundVotingMode) {
            votingModeToggle._boundVotingMode = true;
            votingModeToggle.addEventListener('change', function() {
                window.uploadImpl.setVotingModeEnabled(!!this.checked);
                try { window.persistUiState && window.persistUiState(); } catch (e) {}
            });
        }
        if (selector && !selector._boundVotingModeChange) {
            selector._boundVotingModeChange = true;
            selector.addEventListener('change', function() {
                const toggle = document.getElementById('votingModeToggle');
                if (toggle && toggle.checked) return;
                window.uploadImpl.syncModelSelection(this.value, window.getModelDisplayName(this.value));
            });
        }
    }

    if (document.readyState === 'loading') {
        window.addEventListener('DOMContentLoaded', initVotingModeControls);
    } else {
        initVotingModeControls();
    }
    // 无刷新换页不会再触发 DOMContentLoaded，所以由 main.js 的 initPage 显式再调一次
    // （函数体里对每个控件都有 _boundXxx 标记，重复调用是安全的）。
    window.uploadImpl.initVotingModeControls = initVotingModeControls;

    window.uploadImpl.selectModel = function(modelId, modelName) {
        const displayName = window.uploadImpl.syncModelSelection(modelId, modelName);
        const models = ['cnn_bilstm', 'classic_cnn', 'lightweight_cnn', 'pure_bilstm', 'mlp', 'transformer'];
        models.forEach(id => {
            const card = document.getElementById('modelCard-' + id);
            const tag = document.getElementById('tag-' + id);
            if (!card || !tag) return;
            const active = (id === modelId);
            card.classList.toggle('is-active', active);
            tag.innerText = active ? '当前使用' : '待命';
        });
        try { window.modelEvalImpl && window.modelEvalImpl.showDetail(modelId); } catch (e) {}
        return displayName;
    };

    window.uploadImpl.getModelDisplayName = function(modelId) { const map = { cnn_bilstm: 'CNN-BiLSTM', classic_cnn: 'Classic-CNN', lightweight_cnn: 'Lightweight CNN-BiLSTM', pure_bilstm: 'Pure BiLSTM', mlp: 'MLP', transformer: 'Transformer', ensemble_vote: '6-model Majority Vote Ensemble' }; return map[modelId] || modelId; };

    window.uploadImpl.uploadFilesSequentially = async function(files) {
        const els = getEls();
        try {
            els.uploadContent && els.uploadContent.classList.add('hidden');
            els.loadingState && els.loadingState.classList.remove('hidden');
            els.dropZone && els.dropZone.classList.add('scanning');
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                els.processStatus && (els.processStatus.innerText = `正在分析第 ${i + 1}/${files.length} 个文件...`);
                await window.uploadImpl.uploadFile(file);
            }
        } finally {
            els.loadingState && els.loadingState.classList.add('hidden');
            els.uploadContent && els.uploadContent.classList.remove('hidden');
            els.dropZone && els.dropZone.classList.remove('scanning');
            if (els.processStatus) {
                // Clear status after run; keep errors shown by uploadFile catch
                if (!els.processStatus.innerText || els.processStatus.innerText.indexOf('失败') === -1) els.processStatus.innerText = '';
            }
        }
    };

    window.uploadImpl.uploadFile = async function(file) {
        const activeModel = window.currentActiveModel || (document.getElementById('modelSelector') && document.getElementById('modelSelector').value) || 'cnn_bilstm';
        window.currentActiveModel = activeModel;
        const formData = new FormData(); formData.append('file', file); formData.append('model_type', activeModel);
        // 代码审查第 35 条：包级判定阈值可选下发（留空/非法值不发，后端用默认值）；
        // 这里与后端一样把值夹到 0~1，避免手输 5 之类让服务端 422。
        const appendThreshold = (id, field) => {
            const el = document.getElementById(id);
            if (!el) return;
            const raw = String(el.value == null ? '' : el.value).trim();
            if (raw === '') return;
            const num = Number(raw);
            if (!Number.isFinite(num)) return;
            formData.append(field, String(Math.min(1, Math.max(0, num))));
        };
        appendThreshold('pkgThresholdInput', 'package_threshold');
        appendThreshold('pkgConfThresholdInput', 'package_conf_threshold');
        const statusTimers = [];
        const scheduleStatus = (text, delay) => { const timerId = setTimeout(() => { const ps = document.getElementById('processStatus'); if (ps) ps.innerText = text; }, delay); statusTimers.push(timerId); };
        try {
            const size = (file.size / 1024).toFixed(2) + ' KB';
            const timestamp = new Date().toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit' });
            scheduleStatus("读取 PCAP 文件...", 500);
            scheduleStatus("提取五元组 Session...", 1500);
            scheduleStatus("生成灰度图张量...", 2500);
            scheduleStatus('输入 6 模型投票引擎...', 3500);

            const response = await fetch('/analyze', { method: 'POST', body: formData });
            const payload = await window.uploadImpl.readResponsePayload(response);

            if (!response.ok) {
                throw new Error(window.uploadImpl.formatErrorPayload(payload, response.status));
            }

            if (!payload || typeof payload !== 'object') throw new Error(typeof payload === 'string' && payload.trim() ? payload.trim() : '服务器返回了非 JSON 响应');

            const data = payload;
            if (data && data.status === 'No valid sessions found in pcap') {
                const ps = document.getElementById('processStatus'); if (ps) ps.innerText = '已检测到上传文件，但未解析出有效 Session。';
            }
            const rowId = window.addResultRow ? window.addResultRow(file.name, size, timestamp, data, { recordKind: 'upload', skipStatsUpdate: true }) : ('row_' + Date.now());
            try {
                const hasValidResult = !!data && data.status && data.status !== 'No valid sessions found in pcap';
                const hasSessionCount = typeof data.total_sessions === 'number' && data.total_sessions >= 0;
                if ((hasValidResult || hasSessionCount) && typeof window.updateStats === 'function') {
                    window.updateStats(data);
                }
            } catch(se) { console.error('Stats Update Error:', se); }
            return rowId;
        } catch (error) {
            console.error(error);
            const ps = document.getElementById('processStatus'); if (ps) ps.innerText = '检测失败: ' + (error && error.message ? error.message : String(error));
            alert("检测失败: " + (error && error.message ? error.message : String(error)));
            throw error;
        } finally { statusTimers.forEach((timerId) => clearTimeout(timerId)); }
    };

})();
