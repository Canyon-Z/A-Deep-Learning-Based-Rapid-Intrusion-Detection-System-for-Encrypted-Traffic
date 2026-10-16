// core/api.js - API调用封装

(function() {
    'use strict';

    const API = {
        async request(endpoint, options = {}) {
            const response = await fetch(endpoint, options);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
        },

        async uploadFile(file, modelType) {
            const fd = new FormData();
            fd.append('file', file);
            fd.append('model_type', modelType);
            return this.request('/api/upload', { method: 'POST', body: fd });
        },

        async startLive(iface, modelType) {
            const fd = new FormData();
            fd.append('iface', iface);
            fd.append('model_type', modelType);
            return this.request('/api/live/start', { method: 'POST', body: fd });
        },

        async stopLive() {
            return this.request('/api/live/stop', { method: 'POST' });
        },

        async getLiveStats() {
            return this.request('/api/live/stats');
        },

        async getInterfaces() {
            return this.request('/api/interfaces');
        },

        async getModelMetrics() {
            // 代码审查疑点 3：指标只有一份真源 —— 后端 /api/metrics 读的是
            // checkpoints/model_metrics.json；这里不再直接取 /static 下的副本。
            const body = await this.request('/api/metrics');
            return (body && body.data) ? body.data : body;
        }
    };

    window.ApiModule = API;
})();