// core/utils.js - 工具函数

(function() {
    'use strict';

    const Utils = {
        formatDuration(data) {
            if (!data || typeof data !== 'object') return '-';
            if (typeof data.execution_time === 'string' && data.execution_time.trim()) return data.execution_time;
            if (typeof data.elapsed_time === 'string' && data.elapsed_time.trim()) return data.elapsed_time;
            const msKeys = ['elapsed_ms', 'processing_time_ms', 'duration_ms', 'elapsed', 'time_ms'];
            for (const k of msKeys) {
                if (k in data && typeof data[k] === 'number') {
                    return data[k] >= 1000 ? (data[k] / 1000).toFixed(2) + ' s' : data[k].toFixed(2) + ' ms';
                }
            }
            const sKeys = ['duration_s', 'processing_time_s', 'elapsed_s'];
            for (const k of sKeys) {
                if (k in data && typeof data[k] === 'number') return data[k].toFixed(2) + ' s';
            }
            return '-';
        },

        formatConfidence(value) {
            if (value == null) return '0.00%';
            const num = typeof value === 'string' ? parseFloat(value) : value;
            if (!Number.isFinite(num)) return String(value);
            return num <= 1 ? (num * 100).toFixed(2) + '%' : num.toFixed(2) + '%';
        },

        formatTime(timestamp) {
            if (!timestamp) return '-';
            if (typeof timestamp === 'number') {
                const ms = timestamp < 1e12 ? timestamp * 1000 : timestamp;
                const d = new Date(ms);
                if (!isNaN(d.getTime())) return d.toLocaleString('zh-CN');
            }
            const d = new Date(timestamp);
            if (!isNaN(d.getTime())) return d.toLocaleString('zh-CN');
            return String(timestamp);
        },

        formatSize(bytes) {
            if (!bytes || bytes === '-') return '-';
            const num = parseInt(bytes);
            if (isNaN(num)) return String(bytes);
            if (num < 1024) return num + ' B';
            if (num < 1024 * 1024) return (num / 1024).toFixed(1) + ' KB';
            return (num / (1024 * 1024)).toFixed(1) + ' MB';
        },

        protocolToName(proto) {
            const map = { 1: 'ICMP', 6: 'TCP', 17: 'UDP', 41: 'IPv6', 47: 'GRE', 50: 'ESP', 51: 'AH', 89: 'OSPF' };
            if (proto == null) return '-';
            const num = Number(proto);
            if (isNaN(num)) return String(proto);
            return map[num] || String(num);
        },

        escapeSvgText(text) {
            return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&apos;');
        },

        generateSessionId(src, sport, dst, dport, proto) {
            if (src <= dst) return `${src}:${sport}->${dst}:${dport}:${proto}`;
            return `${dst}:${dport}->${src}:${sport}:${proto}`;
        }
    };

    window.Utils = Utils;
})();