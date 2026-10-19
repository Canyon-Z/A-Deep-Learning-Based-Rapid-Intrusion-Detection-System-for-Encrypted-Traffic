/* Modal & details helpers module (extracted from main.js)
   Exposes: window.modalHelpers with showDetails, initModalChart, normalizeDetailPayload, buildDetailVisualizationDataUri
*/
(function(){
    const MAX_FLOW_ROWS_RENDER = window.MAX_FLOW_ROWS_RENDER || 300;
    let modalBarChart = null;
    let currentPayloadDist = null;

    // HTML 转义：优先复用 core/utils.js 的 Utils.escapeHtml（本文件在 core/utils.js 之前加载，
    // 所以必须在调用时惰性查找，不能在建栈时缓存），未就绪时本地兜底。
    // 任何把外部数据（IP / 端口 / 协议 / 模型名 / 状态文本等）拼进 innerHTML 的位置都必须先过这里。
    function escapeHtml(value) {
        if (value === null || value === undefined) return '';
        if (window.Utils && typeof window.Utils.escapeHtml === 'function') return window.Utils.escapeHtml(value);
        if (typeof window.escapeHtml === 'function' && window.escapeHtml !== escapeHtml) return window.escapeHtml(value);
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function normalizeDetailPayload_impl(data) {
        const d = data && typeof data === 'object' ? data : {};
        return {
            ...d,
            name: d.name || d.file_name || d.filename || d.file || d.row_name || '',
            statusText: d.statusText || d.status || d.confidence_label || d.result || '',
            conf: d.conf || d.confidence || d.score || d.malware_confidence || d.predicted_confidence || '0',
            execTime: d.execTime || d.execution_time || d.processing_time || d.elapsed || d.time_ms || d.duration_ms,
            captureTime: d.captureTime || d.capture_time || d.timestamp || d.created_at || d.time,
            imageData: d.imageData || d.image_data || d.image || d.visualization || d.visualization_data || '',
            payloadDist: d.payloadDist || d.payload_dist || d.payload_distribution || d.histogram || null,
            flows: Array.isArray(d.flows) ? d.flows : (Array.isArray(d.session_flows) ? d.session_flows : (Array.isArray(d.records) ? d.records : [])),
            // 代码审查第 55 条：后端会把 flows 截断到 DLIDS_MAX_RESPONSE_FLOWS 条，
            // 这里把「总数 / 是否被后端截断」一起带出来，明细表就不会谎报条数。
            flowsTotal: Number.isFinite(Number(d.flows_total)) ? Number(d.flows_total) : (Array.isArray(d.flows) ? d.flows.length : 0),
            flowsTruncated: !!d.flows_truncated,
            sessionParseReport: d.sessionParseReport || d.session_parse_report || d.parse_report || d.session_report || null,
        };
    }

    function formatConfidenceText(value) {
        if (value === null || value === undefined || value === '') return '0.00%';
        const num = typeof value === 'string' ? parseFloat(value) : value;
        if (!Number.isFinite(num)) return String(value);
        return num <= 1 ? (num * 100).toFixed(2) + '%' : num.toFixed(2) + '%';
    }

    function formatExecTime(value) {
        if (value === null || value === undefined || value === '') return '-';
        if (typeof value === 'number') return value >= 1000 ? (value / 1000).toFixed(2) + ' s' : value.toFixed(2) + ' ms';
        return String(value);
    }

    function formatCaptureTime(value) {
        if (value === null || value === undefined || value === '') return '-';
        if (typeof value === 'number') {
            const ms = value < 1e12 ? value * 1000 : value;
            const d = new Date(ms);
            return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString('zh-CN');
        }
        const d = new Date(value);
        return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString('zh-CN');
    }

    function formatSessionParseReport(report) {
        if (!report || typeof report !== 'object') return '';
        const parts = [];
        if (typeof report.total_packets === 'number') parts.push(`总包数 ${report.total_packets}`);
        if (typeof report.accepted_packets === 'number') parts.push(`已纳入 ${report.accepted_packets}`);
        if (typeof report.session_count === 'number') parts.push(`session 数 ${report.session_count}`);
        if (typeof report.too_short === 'number') parts.push(`过短 ${report.too_short}`);
        if (typeof report.non_ipv4 === 'number') parts.push(`非 IPv4 ${report.non_ipv4}`);
        if (typeof report.vlan_too_short === 'number') parts.push(`VLAN 过短 ${report.vlan_too_short}`);
        if (typeof report.parse_errors === 'number') parts.push(`解析异常 ${report.parse_errors}`);
        return parts.length > 0 ? parts.join('，') : '';
    }

    function buildFallbackSessionReport(flows) {
        const list = Array.isArray(flows) ? flows : [];
        if (!list.length) {
            return '实时抓包未提供 session 解析统计，当前也没有可展示的流记录。';
        }
        const mal = list.filter((f) => f && f.is_malicious).length;
        const benign = list.length - mal;
        const sessions = new Set();
        const protoCounts = {};
        let totalBytes = 0;
        list.forEach((f) => {
            if (!f) return;
            const src = f.src_ip || f.src || '-';
            const dst = f.dst_ip || f.dst || '-';
            const sport = f.src_port || f.sport || '-';
            const dport = f.dst_port || f.dport || '-';
            const proto = f.protocol || f.proto || 'unknown';
            sessions.add(`${src}:${sport}->${dst}:${dport}`);
            protoCounts[String(proto)] = (protoCounts[String(proto)] || 0) + 1;
            totalBytes += Number(f.bytes || f.length || 0) || 0;
        });
        const protoSummary = Object.entries(protoCounts).slice(0, 4).map(([k, v]) => `${k}:${v}`).join('，');
        return `实时抓包会话统计：${list.length} 条流，${sessions.size} 个会话，恶意 ${mal}，正常 ${benign}，总字节 ${totalBytes || 0}${protoSummary ? `，协议分布 ${protoSummary}` : ''}`;
    }

    function escapeSvgText(text) {
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&apos;');
    }

    function buildDetailVisualizationDataUri(name, result, conf, imageData, payloadDist, flows) {
        if (imageData && typeof imageData === 'string' && imageData.length > 0) {
            return imageData.startsWith('data:') ? imageData : ('data:image/png;base64,' + imageData);
        }
        const list = Array.isArray(flows) ? flows : [];
        const mal = list.filter((f) => f && f.is_malicious).length;
        const benign = list.length - mal;
        const labels = ['0x00','0x10','0x20','0x30','0x40','0x50','0x60','0x70','0x80','0x90','0xA0','0xB0','0xC0','0xD0','0xE0','0xF0'];
        const bins = Array.isArray(payloadDist) && payloadDist.length === 16
            ? payloadDist
            : (() => {
                const arr = Array.from({ length: 16 }, () => 0);
                list.forEach((f, idx) => {
                    const seed = `${f && (f.src_ip || f.src || '')}|${f && (f.dst_ip || f.dst || '')}|${f && (f.src_port || f.sport || '')}|${f && (f.dst_port || f.dport || '')}|${f && (f.protocol || f.proto || '')}|${f && (f.malware_conf || f.confidence || 0)}`;
                    let hash = 0;
                    for (let i = 0; i < seed.length; i++) hash = ((hash << 5) - hash + seed.charCodeAt(i)) >>> 0;
                    const base = hash % 16;
                    const weight = 1 + Math.round((Number(f && (f.bytes || f.length || 0)) || 0) / 512) + (f && f.is_malicious ? 2 : 0) + (idx % 3);
                    arr[base] += weight;
                    arr[(base + 5) % 16] += Math.max(1, Math.ceil(weight / 3));
                });
                return arr.every((v) => v === 0) ? labels.map((_, i) => Math.max(4, Math.round((i % 4 + 1) * (1 + list.length / 6)))) : arr;
            })();
        const maxBin = Math.max(...bins, 1);
        const barWidth = 10;
        const gap = 3;
        const chartX = 26;
        const chartY = 126;
        const chartH = 60;
        const bars = bins.map((v, i) => {
            const h = Math.max(2, Math.round((v / maxBin) * chartH));
            const x = chartX + i * (barWidth + gap);
            const y = chartY + (chartH - h);
            return `<rect x="${x}" y="${y}" width="${barWidth}" height="${h}" rx="2" fill="url(#barGrad)" opacity="0.95"/>`;
        }).join('');

        const svg = `\n<svg xmlns="http://www.w3.org/2000/svg" width="560" height="260" viewBox="0 0 560 260">\n    <defs>\n        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">\n            <stop offset="0%" stop-color="#0b0f1e"/>\n            <stop offset="100%" stop-color="#101a35"/>\n        </linearGradient>\n        <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">\n            <stop offset="0%" stop-color="#22d3ee"/>\n            <stop offset="100%" stop-color="#2563eb"/>\n        </linearGradient>\n        <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">\n            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>\n            <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>\n        </filter>\n    </defs>\n    <rect width="560" height="260" fill="url(#bg)" rx="18"/>\n    <circle cx="110" cy="82" r="34" fill="rgba(34,211,238,0.15)" stroke="#22d3ee" stroke-width="2"/>\n    <circle cx="450" cy="82" r="34" fill="rgba(37,99,235,0.15)" stroke="#60a5fa" stroke-width="2"/>\n    <line x1="145" y1="82" x2="415" y2="82" stroke="#22d3ee" stroke-width="4" stroke-linecap="round" filter="url(#glow)"/>\n    <polygon points="415,82 402,74 402,90" fill="#22d3ee"/>\n    <text x="110" y="78" text-anchor="middle" fill="#e5e7eb" font-size="14" font-family="sans-serif">SRC</text>\n    <text x="110" y="98" text-anchor="middle" fill="#9ca3af" font-size="11" font-family="monospace">${escapeSvgText((list[0] && (list[0].src_ip || list[0].src)) || 'unknown')}</text>\n    <text x="450" y="78" text-anchor="middle" fill="#e5e7eb" font-size="14" font-family="sans-serif">DST</text>\n    <text x="450" y="98" text-anchor="middle" fill="#9ca3af" font-size="11" font-family="monospace">${escapeSvgText((list[0] && (list[0].dst_ip || list[0].dst)) || 'unknown')}</text>\n    <text x="280" y="58" text-anchor="middle" fill="#e5e7eb" font-size="18" font-weight="700" font-family="sans-serif">${escapeSvgText(name || 'Traffic Detail')}</text>\n    <text x="280" y="80" text-anchor="middle" fill="#7dd3fc" font-size="12" font-family="monospace">${escapeSvgText(result || '-') } · ${escapeSvgText(conf || '0.00%')}</text>\n    <text x="280" y="118" text-anchor="middle" fill="#cbd5e1" font-size="12" font-family="sans-serif">Flow Summary: ${list.length} records · Malware ${mal} · Benign ${benign}</text>\n    ${bars}\n    <text x="26" y="204" fill="#9ca3af" font-size="10" font-family="monospace">Payload histogram fallback when original image is unavailable</text>\n</svg>`;
        return 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(svg);
    }

    function initModalChart_impl() {
        const chartDom = document.getElementById('modalBarChart');
        if (!chartDom || typeof echarts === 'undefined') return;
        if (!modalBarChart) modalBarChart = echarts.init(chartDom, window.DLIDS_CHART_THEME || null);
        const labels = ['0x00','0x10','0x20','0x30','0x40','0x50','0x60','0x70','0x80','0x90','0xA0','0xB0','0xC0','0xD0','0xE0','0xF0'];
        let data = currentPayloadDist;
        if (!Array.isArray(data) || data.length !== 16) data = Array.from({ length: 16 }, () => 0);
        const option = {
            backgroundColor: 'transparent',
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            grid: { top: '10%', left: '0%', right: '0%', bottom: '5%', containLabel: true },
            xAxis: { type: 'category', data: labels, axisLabel: { color: '#9ca3af', fontSize: 10 }, axisLine: { lineStyle: { color: '#374151' } } },
            yAxis: { type: 'value', axisLabel: { color: '#9ca3af', fontSize: 10 }, splitLine: { lineStyle: { color: '#1f2937' } } },
            series: [{ name: 'Bytes', type: 'bar', data: data, itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#22d3ee' }, { offset: 1, color: '#2563eb' }]) } }]
        };
        modalBarChart.setOption(option, true);
        try { modalBarChart.resize(); } catch(e){}
    }

    function showDetails_impl(name, result, conf, execTime, captureTime, imageData, payloadDist, flows, sessionParseReport, extra) {
        try {
            const modal = document.getElementById('detailModal');
            if (!modal) return;
            modal.classList.remove('hidden');

            // ensure modal content can receive pointer events (avoid accidental overlay blocking)
            try {
                const glass = modal.querySelector('.glass-card');
                if (glass) {
                    glass.style.pointerEvents = 'auto';
                    glass.style.zIndex = 101;
                }
            } catch (e) { console.warn('ensure modal pointer failed', e); }

            // 一次性淡入（styles.css 的 .mo-in）。关闭时故意不做动画：
            // 让"点关闭"立刻生效，比等一段淡出更跟手。
            try {
                const card = modal.querySelector('.glass-card');
                if (card) {
                    card.classList.remove('mo-in');
                    void card.offsetWidth; // 强制重排，保证连续打开时动画能重播
                    card.classList.add('mo-in');
                    setTimeout(() => card.classList.remove('mo-in'), 300);
                }
            } catch (e) { /* 动画失败不影响弹窗功能 */ }

            const normalizedPayload = normalizeDetailPayload_impl({ name, statusText: result, conf, execTime, captureTime, imageData, payloadDist, flows, sessionParseReport });

            name = normalizedPayload.name;
            result = normalizedPayload.statusText;
            conf = normalizedPayload.conf;
            execTime = normalizedPayload.execTime;
            captureTime = normalizedPayload.captureTime;
            imageData = normalizedPayload.imageData;
            payloadDist = normalizedPayload.payloadDist;
            flows = normalizedPayload.flows;
            sessionParseReport = normalizedPayload.sessionParseReport;

            currentPayloadDist = payloadDist;

            const fileNameEl = document.getElementById('modalFileName');
            const execTimeEl = document.getElementById('modalExecTime');
            const captureTimeEl = document.getElementById('modalCaptureTime');
            const parseReportEl = document.getElementById('modalSessionParseReport');
            const imgEl = document.getElementById('modalImage');
            const noImgEl = document.getElementById('modalNoImage');
            const flowTbody = document.getElementById('modalFlowTableBody');
            const flowCountEl = document.getElementById('modalFlowCount');
            const resultEl = document.getElementById('modalResult');
            const confTextEl = document.getElementById('modalConfText');
            const confBarEl = document.getElementById('modalConfBar');

            if (fileNameEl) fileNameEl.innerText = name || '-';
            if (execTimeEl) execTimeEl.innerText = execTime || '-';
            if (captureTimeEl) captureTimeEl.innerText = captureTime || '-';
            const summaryText = formatSessionParseReport(sessionParseReport) || buildFallbackSessionReport(flows);
            if (parseReportEl) parseReportEl.innerText = summaryText;

            if (imgEl && noImgEl) {
                imgEl.src = buildDetailVisualizationDataUri(name, result, conf, imageData, payloadDist, flows);
                // prefer a larger, responsive image inside modal
                imgEl.style.maxWidth = '100%';
                imgEl.style.height = 'auto';
                imgEl.style.maxHeight = '60vh';
                imgEl.classList.remove('hidden');
                noImgEl.classList.add('hidden');
                // bind expand click to show a simple lightbox for larger view
                try {
                    const expandBtn = (modal && modal.querySelector) ? modal.querySelector('.fa-expand') : null;
                    if (expandBtn) {
                        expandBtn._boundClick = expandBtn._boundClick || function(ev) {
                            ev && ev.stopPropagation && ev.stopPropagation();
                            try {
                                // create overlay
                                let overlay = document.getElementById('imageLightbox');
                                if (!overlay) {
                                    overlay = document.createElement('div');
                                    overlay.id = 'imageLightbox';
                                    overlay.style.position = 'fixed';
                                    overlay.style.inset = '0';
                                    overlay.style.zIndex = '200';
                                    overlay.style.background = 'rgba(1,6,15,0.9)';
                                    overlay.style.display = 'flex';
                                    overlay.style.alignItems = 'center';
                                    overlay.style.justifyContent = 'center';
                                    overlay.style.cursor = 'zoom-out';
                                    const img = document.createElement('img');
                                    img.src = imgEl.src || '';
                                    img.style.maxWidth = '92%';
                                    img.style.maxHeight = '92%';
                                    img.style.boxShadow = '0 8px 40px rgba(0,0,0,0.6)';
                                    img.style.imageRendering = 'pixelated';
                                    overlay.appendChild(img);
                                    overlay.addEventListener('click', () => { try { overlay.remove(); } catch(e){} });
                                    document.body.appendChild(overlay);
                                }
                            } catch (e) { console.warn('expand preview failed', e); }
                        };
                        expandBtn.removeEventListener('click', expandBtn._boundClick);
                        expandBtn.addEventListener('click', expandBtn._boundClick);
                    }
                } catch(e) { console.warn('bind expand failed', e); }
            }

            const isMalware = String(result || '').includes('Malware') || String(result || '').includes('Malicious');
            if (resultEl) {
                // 状态文本可能来自后端/文件名，先转义再拼；<i> 图标标签是固定结构，保持原样。
                resultEl.innerHTML = isMalware ? '<i class="fa-solid fa-bug"></i> ' + escapeHtml(result) : '<i class="fa-solid fa-shield-check"></i> ' + escapeHtml(result);
                resultEl.className = isMalware
                    ? 'px-3 py-1.5 rounded bg-cyber-danger/10 text-cyber-danger border border-cyber-danger/50 text-sm font-bold shadow-neon-red flex items-center gap-2'
                    : 'px-3 py-1.5 rounded bg-cyber-success/10 text-cyber-success border border-cyber-success/50 text-sm font-bold shadow-[0_0_10px_rgba(0,208,132,0.2)] flex items-center gap-2';
            }
            if (confTextEl) confTextEl.innerText = conf || '0.00%';
            if (confBarEl) {
                const pct = String(conf || '0').replace('%', '');
                confBarEl.style.width = '0%';
                setTimeout(() => { confBarEl.style.width = (parseFloat(pct) || 0) + '%'; }, 50);
            }

            if (flowTbody && flowCountEl) {
                flowTbody.innerHTML = '';
                flowCountEl.innerText = '加载中...';
                requestAnimationFrame(() => {
                    const list = Array.isArray(flows) ? flows : [];
                    if (!list.length) {
                        flowCountEl.innerText = '0 Session';
                        flowTbody.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-gray-500 font-mono">No Flow Context Available</td></tr>';
                    } else {
                        const payloadTotal = Number(normalizedPayload && normalizedPayload.flowsTotal);
                        const totalFlows = Math.max(list.length, Number.isFinite(payloadTotal) ? payloadTotal : 0);
                        const serverTruncated = !!(normalizedPayload && normalizedPayload.flowsTruncated);
                        const renderCount = Math.min(totalFlows, MAX_FLOW_ROWS_RENDER);
                        const fragment = document.createDocumentFragment();
                        for (let i = 0; i < renderCount; i++) {
                            const flow = list[i] || {};
                            const tr = document.createElement('tr');
                            tr.className = 'border-b border-cyber-border/40 hover:bg-white/5 transition-colors';
                            const isMal = !!flow.is_malicious;
                            const predictedVal = Number.isFinite(parseFloat(flow.predicted_confidence)) ? (parseFloat(flow.predicted_confidence) * 100).toFixed(1) : '0.0';
                            const malwareVal = Number.isFinite(parseFloat(flow.malware_confidence)) ? (parseFloat(flow.malware_confidence) * 100).toFixed(1) : '0.0';
                            const labelText = flow.predicted_label || (isMal ? 'Malware' : 'Benign');
                            const voteMap = flow && flow.vote_summary && flow.vote_summary.model_votes && typeof flow.vote_summary.model_votes === 'object'
                                ? flow.vote_summary.model_votes
                                : null;
                            let voteDetailHtml = '';
                            if (voteMap) {
                                const voteLines = Object.entries(voteMap).map(([modelId, detail]) => {
                                    const modelName = (window.getModelDisplayName && typeof window.getModelDisplayName === 'function')
                                        ? window.getModelDisplayName(modelId)
                                        : modelId;
                                    const predicted = detail && detail.predicted_label ? detail.predicted_label : '-';
                                    // 置信度必须是「判定那一类」的概率。原来恒取 malware_confidence，
                                    // 于是判 Benign 的模型全都显示成 Benign (0.0%) —— 读起来像「拿不准」，
                                    // 实际是「很确定是良性」（benign_confidence 接近 1）。
                                    const malRaw = detail ? parseFloat(detail.malware_confidence) : NaN;
                                    const benRaw = detail ? parseFloat(detail.benign_confidence) : NaN;
                                    const votedMalware = /^mal/i.test(String(predicted)) || String(predicted) === 'Malicious';
                                    const confRaw = votedMalware
                                        ? malRaw
                                        : (Number.isFinite(benRaw) ? benRaw : (Number.isFinite(malRaw) ? 1 - malRaw : NaN));
                                    const confText = Number.isFinite(confRaw) ? `${(confRaw * 100).toFixed(1)}%` : '-';
                                    // malware 概率仍然带出来：判良性时它才是「这一票有多不情愿」的依据。
                                    const malText = Number.isFinite(malRaw) ? `${(malRaw * 100).toFixed(1)}%` : '-';
                                    const title = `恶意概率 ${malText} · 阈值 ${detail && detail.threshold != null ? detail.threshold : '-'}`;
                                    return `<div title="${escapeHtml(title)}">${escapeHtml(modelName)}: ${escapeHtml(predicted)} (${confText})</div>`;
                                }).join('');
                                voteDetailHtml = `<div class="text-[10px] text-cyan-300 mt-1">投票明细:${voteLines}</div>`;
                            }
                            const flowStatusHtml = isMal
                                ? `<span class="text-cyber-danger font-medium"><i class="fa-solid fa-triangle-exclamation"></i> Malware (${malwareVal}%)</span>`
                                : `<span class="text-cyber-success font-medium"><i class="fa-solid fa-check"></i> Benign</span>`;
                            // 抓包内容（src/dst IP、端口、协议、字节数、预测标签）全部先 escapeHtml；
                            // flowStatusHtml / voteDetailHtml 是已转义好的 HTML 片段，不能再整体转义。
                            tr.innerHTML = `\n                                <td class="px-3 py-2 font-mono">${escapeHtml(flow.src_ip || flow.src || '-')}:${escapeHtml(flow.src_port || flow.sport || '-')}</td>\n                                <td class="px-3 py-2 font-mono">${escapeHtml(flow.dst_ip || flow.dst || '-')}:${escapeHtml(flow.dst_port || flow.dport || '-')}</td>\n                                <td class="px-3 py-2 text-cyber-primary">${escapeHtml(flow.protocol || '-')}</td>\n                                <td class="px-3 py-2 font-mono text-gray-400">${escapeHtml(flow.bytes || flow.length || 0)} B</td>\n                                <td class="px-3 py-2">${flowStatusHtml}<div class="text-[10px] text-gray-500 mt-1">Pred: ${escapeHtml(labelText)} (${predictedVal}%)</div>${voteDetailHtml}</td>\n                            `;
                            fragment.appendChild(tr);
                        }
                        flowTbody.appendChild(fragment);
                        if (totalFlows > renderCount) {
                            const more = document.createElement('tr');
                            more.innerHTML = `<td colspan="5" class="text-center py-3 text-xs text-gray-500 font-mono">仅展示前 ${renderCount} / ${totalFlows} 条会话，已自动截断以提升交互性能</td>`;
                            flowTbody.appendChild(more);
                        }
                        if (serverTruncated) {
                            const note = document.createElement('tr');
                            note.innerHTML = `<td colspan="5" class="text-center py-3 text-xs text-gray-500 font-mono">后端仅回传前 ${list.length} / ${totalFlows} 条会话明细（上限见 DLIDS_MAX_RESPONSE_FLOWS）</td>`;
                            flowTbody.appendChild(note);
                        }
                        flowCountEl.innerText = `${totalFlows} Sessions Found`;
                    }
                });
            }

            setTimeout(initModalChart_impl, 120);
            renderEvidence(extra);
        } catch (e) {
            console.warn('showDetails failed', e);
            alert('显示详情失败: ' + (e && e.message ? e.message : String(e)));
        }
    }

    // 加权取证：回答「这次到底有没有用域加权、域分数从哪来、票僵持到什么程度」。
    // 这些值早就在 /analyze 的响应里（ensemble_weighting / distribution / reliability），
    // 但界面上一直没有出口，用户只能靠「恶意占比没变」去猜，等于不可观测。
    function renderEvidence(extra) {
        const box = document.getElementById('modalEvidence');
        const body = document.getElementById('modalEvidenceBody');
        const modeEl = document.getElementById('modalEvidenceMode');
        if (!box || !body) return;
        const source = (extra && typeof extra === 'object') ? extra : {};
        const ew = source.ensemble_weighting && typeof source.ensemble_weighting === 'object' ? source.ensemble_weighting : null;
        const dist = source.distribution && typeof source.distribution === 'object' ? source.distribution : null;
        if (!ew && !dist) {
            box.classList.add('hidden');
            body.innerHTML = '';
            return;
        }
        box.classList.remove('hidden');

        // 一行「键 → 值」，值里不放 HTML，统一走 escapeHtml。
        const rows = [];
        const push = (label, value) => rows.push(
            '<div class="flex justify-between gap-3"><span class="text-gray-500">' + escapeHtml(label) +
            '</span><span class="text-gray-300 break-all text-right">' + escapeHtml(value) + '</span></div>'
        );

        if (ew) {
            const weighting = String(ew.weighting || 'uniform');
            const score = Number(ew.domain_score);
            const signals = ew.domain_signals && typeof ew.domain_signals === 'object' ? ew.domain_signals : {};
            const weights = ew.weights_used && typeof ew.weights_used === 'object' ? ew.weights_used : null;
            const sessions = Array.isArray(ew.sessions) ? ew.sessions : [];

            if (modeEl) {
                const tag = weighting === 'domain' ? '按域加权' : '等权（未加权）';
                modeEl.textContent = Number.isFinite(score) ? (tag + ' · 域外分数 ' + score.toFixed(3)) : tag;
            }

            const portSig = Number(signals.port);
            const distSig = Number(signals.dist);
            if (Number.isFinite(portSig) || Number.isFinite(distSig)) {
                const parts = [];
                if (Number.isFinite(portSig)) parts.push('端口 ' + portSig.toFixed(2));
                if (Number.isFinite(distSig)) parts.push('距离 ' + distSig.toFixed(2));
                push('域外信号', parts.join(' · ') || '-');
            }
            const batchDistance = Number(ew.batch_distance);
            if (Number.isFinite(batchDistance)) {
                const threshold = Number(ew.distance_threshold);
                push('批马氏距离', threshold > 0
                    ? batchDistance.toFixed(3) + ' / 阈值 ' + threshold.toFixed(3) + '（' + (batchDistance / threshold).toFixed(2) + '×）'
                    : batchDistance.toFixed(3));
            }

            // 权重是否真的偏离 1.0 —— 这是「加权有没有生效」最直接的证据。
            if (weights) {
                const entries = Object.keys(weights).map((k) => [k, Number(weights[k])]).filter(([, v]) => Number.isFinite(v));
                if (entries.length) {
                    const maxDev = entries.reduce((m, [, v]) => Math.max(m, Math.abs(v - 1)), 0);
                    const top = entries.slice().sort((a, b) => b[1] - a[1])[0];
                    const low = entries.slice().sort((a, b) => a[1] - b[1])[0];
                    push('实际权重', maxDev < 1e-6
                        ? '全为 1.000（等同等权）'
                        : (low[0] + ' ' + low[1].toFixed(3) + ' … ' + top[0] + ' ' + top[1].toFixed(3)));
                }
            } else {
                push('实际权重', '全为 1.000（等同等权）');
            }

            if (sessions.length) {
                const maxContest = sessions.reduce((m, s) => Math.max(m, Number((s && s.contest) || 0)), 0);
                const adjusted = sessions.filter((s) => s && s.adjust && Number(s.adjust.blend) > 0).length;
                const strength = Number(ew.contest_strength);
                push('模型分歧', maxContest.toFixed(2)
                    + (Number.isFinite(strength) ? '（介入门槛 ' + strength.toFixed(2) + '）' : '')
                    + ' · 调权会话 ' + adjusted + '/' + sessions.length);
            }
        }

        if (dist) {
            const parts = [];
            const iec = Number(dist.industrial_ratio);
            const fec = Number(dist.foreign_ratio);
            if (Number.isFinite(iec)) parts.push('工控 ' + iec.toFixed(1) + '%');
            if (Number.isFinite(fec)) parts.push('外部协议 ' + fec.toFixed(1) + '%');
            if (parts.length) push('协议占比', parts.join(' · '));
        }

        const reliability = String(source.reliability || '').trim();
        if (reliability && reliability !== 'ok') {
            push('判定可信度', reliability === 'out_of_domain' ? 'out_of_domain（分布外，结论不可采信）' : reliability);
        }

        body.innerHTML = rows.join('');
    }

    // export
    window.modalHelpers = window.modalHelpers || {};
    window.modalHelpers.normalizeDetailPayload = normalizeDetailPayload_impl;
    window.modalHelpers.formatConfidenceText = formatConfidenceText;
    window.modalHelpers.formatExecTime = formatExecTime;
    window.modalHelpers.formatCaptureTime = formatCaptureTime;
    window.modalHelpers.formatSessionParseReport = formatSessionParseReport;
    window.modalHelpers.buildFallbackSessionReport = buildFallbackSessionReport;
    window.modalHelpers.buildDetailVisualizationDataUri = buildDetailVisualizationDataUri;
    window.modalHelpers.escapeSvgText = escapeSvgText;
    window.modalHelpers.escapeHtml = escapeHtml;
    window.modalHelpers.initModalChart = initModalChart_impl;
    window.modalHelpers.showDetails = showDetails_impl;
    // global helper to close the details modal (used by inline onclick handlers in template)
    window.closeModal = function() {
        try {
            const modal = document.getElementById('detailModal');
            if (!modal) return;
            modal.classList.add('hidden');
            // allow clicks to pass through when hidden
            try { modal.style.pointerEvents = 'none'; } catch(e){}
            // remove any existing lightbox overlay
            try { const lb = document.getElementById('imageLightbox'); if (lb) lb.remove(); } catch(e){}
            // clear modal image src to free memory (optional)
            try { const img = document.getElementById('modalImage'); if (img) { img.src = ''; } } catch(e){}
        } catch (e) { console.warn('closeModal failed', e); }
    };
})();
