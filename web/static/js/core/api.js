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
            const response = await fetch('/static/checkpoints/model_metrics.json');
            if (!response.ok) throw new Error('Failed to load metrics');
            return response.json();
        }
    };

    window.ApiModule = API;
})();