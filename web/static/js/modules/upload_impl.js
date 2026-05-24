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
        const votingModeToggle = document.getElementById('votingModeToggle');
        if (votingModeToggle && modelId !== 'ensemble_vote') {
            votingModeToggle.checked = false;
        }
        const displayName = modelName || window.getModelDisplayName(modelId);
        const activeModelBadge = document.getElementById('activeModelBadge');
        if (activeModelBadge) activeModelBadge.innerText = displayName + ' Active';
        const modelToPrefix = { cnn_bilstm: 'm1', classic_cnn: 'm2', lightweight_cnn: 'm3', pure_bilstm: 'm4', mlp: 'm5', transformer: 'm6' };
        const prefix = modelToPrefix[modelId] || ('m' + (['cnn_bilstm', 'classic_cnn', 'lightweight_cnn', 'pure_bilstm', 'mlp', 'transformer'].indexOf(modelId) + 1));
        const accEl = document.getElementById(prefix + '-acc');
        const statActiveAcc = document.getElementById('statActiveAcc');
        if (accEl && statActiveAcc) {
            const accText = (accEl.innerText || '--').replace('%', '').trim() || '--';
            statActiveAcc.innerHTML = accText + '<span class="text-sm text-gray-500">%</span>';
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
            if (activeModelBadge) activeModelBadge.innerText = '投票模式';
            if (statActiveAcc) {
                statActiveAcc.innerHTML = '--<span class="text-sm text-gray-500">%</span>';
            }
            if (selector) selector.value = 'cnn_bilstm';
            return;
        }

        const fallbackModel = (selector && selector.value) || 'cnn_bilstm';
        window.uploadImpl.syncModelSelection(fallbackModel, window.getModelDisplayName(fallbackModel));
        try { window.persistUiState && window.persistUiState(); } catch (e) {}
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

    window.uploadImpl.selectModel = function(modelId, modelName) {
        const displayName = window.uploadImpl.syncModelSelection(modelId, modelName);
        const models = ['cnn_bilstm', 'classic_cnn', 'lightweight_cnn', 'pure_bilstm', 'mlp', 'transformer'];
        models.forEach(id => {
            const card = document.getElementById('modelCard-' + id);
            const tag = document.getElementById('tag-' + id);
            if (!card || !tag) return;
            if (id === modelId) {
                card.classList.remove('border-cyber-border', 'opacity-80', 'hover:opacity-100');
                card.classList.add('border-2', 'border-cyber-success', 'shadow-[0_0_15px_rgba(0,208,132,0.3)]', 'transform', 'scale-105', 'z-10', 'opacity-100');
                tag.innerText = 'Active 当前使用';
                tag.className = 'px-2 py-0.5 rounded bg-cyber-success/10 text-cyber-success text-[10px] border border-cyber-success/20 font-bold';
            } else {
                card.classList.add('border-cyber-border', 'opacity-80', 'hover:opacity-100');
                card.classList.remove('border-2', 'border-cyber-success', 'shadow-[0_0_15px_rgba(0,208,132,0.3)]', 'transform', 'scale-105', 'z-10', 'opacity-100');
                tag.innerText = 'Standby 待命中';
                tag.className = 'px-2 py-0.5 rounded bg-gray-500/10 text-gray-400 text-[10px] border border-gray-500/20';
            }
        });
        const modelToPrefix = { cnn_bilstm: 'm1', classic_cnn: 'm2', lightweight_cnn: 'm3', pure_bilstm: 'm4', mlp: 'm5', transformer: 'm6' };
        const prefix = modelToPrefix[modelId] || ('m' + (models.indexOf(modelId) + 1));
        const accEl = document.getElementById(prefix + '-acc');
        if (accEl) {
            const accText = accEl.innerText.replace('%', '').trim() || '--';
            const statActiveAcc = document.getElementById('statActiveAcc');
            if (statActiveAcc) {
                statActiveAcc.innerHTML = accText + '<span class="text-sm text-gray-500">%</span>';
            }
        }
        window.scrollTo({ top: 0, behavior: 'smooth' });
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
