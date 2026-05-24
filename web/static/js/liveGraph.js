/* Live graph rendering module (extract from main.js)
   Exposes: window.liveGraph.renderLiveFlowGraph(flows)
*/
(function(){
    function hashString(s) {
        let h = 2166136261 >>> 0;
        for (let i = 0; i < s.length; i++) {
            h = Math.imul(h ^ s.charCodeAt(i), 16777619) >>> 0;
        }
        return h >>> 0;
    }

    // ensure persisted clear timestamp is respected across reloads
    window.liveLogsClearedAt = Number(window.localStorage && window.localStorage.getItem && window.localStorage.getItem('liveLogsClearedAt')) || window.liveLogsClearedAt || 0;

    function renderLiveFlowGraph_impl(flows) {
        try {
            // helper to map protocol numbers to names for tooltip/display
            function protoToName(p) {
                if (p === null || p === undefined || p === '') return '-';
                const n = Number(p);
                const map = { 1: 'ICMP', 6: 'TCP', 17: 'UDP', 41: 'IPv6', 47: 'GRE', 50: 'ESP', 51: 'AH', 89: 'OSPF' };
                if (Number.isFinite(n) && map[n]) return map[n] + ' (' + n + ')';
                return String(p);
            }
            const chartDom = document.getElementById('liveFlowGraph');
            const listDom = document.getElementById('liveFlowList');
            if (!chartDom) return;
            // store latest flows; the actual drawing is driven by an animation loop
            window._lastLiveFlows = Array.isArray(flows) ? flows : [];
            flows = window._lastLiveFlows;

            if (listDom) {
                if (flows.length === 0) {
                    listDom.innerHTML = '<p class="text-sm text-gray-400">暂无实时流量</p>';
                } else {
                    listDom.innerHTML = '';
                    for (let i = flows.length - 1; i >= 0; --i) {
                        const f = flows[i];
                        const el = document.createElement('div');
                        el.className = 'px-2 py-1 text-xs text-gray-300 border-b border-cyber-border/20';
                        const src = f.src || '0.0.0.0';
                        const dst = f.dst || '0.0.0.0';
                        const t = f.captured_at ? (new Date(f.captured_at * 1000)).toLocaleTimeString('zh-CN') : new Date().toLocaleTimeString('zh-CN');
                        el.innerText = `${t}  ${src}:${f.sport||'-'} → ${dst}:${f.dport||'-'}  ${f.is_malicious ? 'Malicious' : ''}`;
                        listDom.appendChild(el);
                    }
                }
            }

            let canvas = chartDom.querySelector('canvas');
            const cw = chartDom.clientWidth || 600;
            const ch = chartDom.clientHeight || 300;
            const dpr = window.devicePixelRatio || 1;
            if (!canvas) {
                canvas = document.createElement('canvas');
                // set CSS size once; avoid toggling each render to prevent layout thrash
                canvas.style.width = cw + 'px';
                canvas.style.height = ch + 'px';
                chartDom.appendChild(canvas);
            }
            // compute target backing store size
            const targetW = Math.max(1, Math.floor(cw * dpr));
            const targetH = Math.max(1, Math.floor(ch * dpr));
            // only update canvas pixel buffer when size actually changed to avoid flicker/jump
            if (canvas.width !== targetW || canvas.height !== targetH) {
                canvas.width = targetW;
                canvas.height = targetH;
                // ensure visible CSS size remains stable
                canvas.style.width = cw + 'px';
                canvas.style.height = ch + 'px';
                // record size for debugging
                canvas.dataset.livegraphLastW = targetW;
                canvas.dataset.livegraphLastH = targetH;
                // only log backing-size updates when debug is enabled to avoid console spam
                try {
                    window.liveGraph = window.liveGraph || {};
                    if (window.liveGraph._debugEnabled) {
                        const lastSizeLog = window.liveGraph._lastSizeLogTs || 0;
                        if (Date.now() - lastSizeLog > 5000) {
                            console.log('liveGraph: updated canvas backing size', {targetW, targetH, dpr});
                            window.liveGraph._lastSizeLogTs = Date.now();
                        }
                    }
                } catch (e) { /* ignore */ }
            }
            const ctx = canvas.getContext('2d');
            // diagnostic: if context unavailable, add visible fallback and bail
            if (!ctx) {
                console.warn('liveGraph: canvas.getContext returned null — rendering disabled');
                try {
                    chartDom.style.position = chartDom.style.position || 'relative';
                    canvas.style.backgroundColor = 'rgba(255,0,0,0.03)';
                    if (!chartDom.querySelector('.liveGraph-fallback')) {
                        const note = document.createElement('div');
                        note.className = 'liveGraph-fallback';
                        note.style.position = 'absolute';
                        note.style.left = '8px';
                        note.style.top = '8px';
                        note.style.color = '#b91c1c';
                        note.style.fontSize = '12px';
                        note.style.pointerEvents = 'none';
                        note.innerText = 'Canvas context unavailable';
                        chartDom.appendChild(note);
                    }
                } catch (e) {
                    console.error('liveGraph: error creating fallback UI', e);
                }
                return;
            }
            try {
                ctx.setTransform(1,0,0,1,0,0);
                ctx.scale(dpr, dpr);
            } catch (e) {
                console.error('liveGraph: ctx transformation failed', e);
            }

            // diagnostic logging of sizes to help debug DPR / layout issues
            try {
                const dims = {clientW: cw, clientH: ch, canvasW: canvas.width, canvasH: canvas.height, dpr: dpr};
                window.liveGraph = window.liveGraph || {};
                window.liveGraph._lastLoggedDims = window.liveGraph._lastLoggedDims || {dims: null, ts: 0};
                const last = window.liveGraph._lastLoggedDims;
                const now = Date.now();
                const changed = !last.dims || last.dims.clientW !== dims.clientW || last.dims.clientH !== dims.clientH || last.dims.canvasW !== dims.canvasW || last.dims.canvasH !== dims.canvasH || last.dims.dpr !== dims.dpr;
                // only log when changed or at most once every 2s
                // only log dims when debug enabled; throttle to once every 5s
                if (window.liveGraph && window.liveGraph._debugEnabled) {
                    if (changed || (now - last.ts) > 5000) {
                        console.log('liveGraph: canvas dims', dims);
                        last.dims = dims; last.ts = now;
                    }
                } else {
                    last.dims = dims; last.ts = now; // still update last to avoid repeated change detection
                }
            } catch (e) { /* ignore */ }

            // diagnostic probe: draw a small visible dot once so user can tell if drawing works
            try {
                window.liveGraph = window.liveGraph || {};
                if (!window.liveGraph._probeDrawn) {
                    ctx.save();
                    ctx.fillStyle = 'rgba(0,200,132,0.95)';
                    ctx.fillRect(8, 8, 6, 6);
                    ctx.restore();
                    window.liveGraph._probeDrawn = true;
                }
            } catch (e) {
                console.error('liveGraph: diagnostic probe draw failed', e);
            }

            window.liveGraphState = window.liveGraphState || { panX: 0, panY: 0, scale: 1, isPanning: false, lastX: 0, lastY: 0 };
            const state = window.liveGraphState;

            // persistent stores to avoid flicker and to keep encrypted/malicious items
            window.liveGraph._nodeStore = window.liveGraph._nodeStore || {};
            window.liveGraph._edgeStore = window.liveGraph._edgeStore || {};
            const nodeStore = window.liveGraph._nodeStore;
            const edgeStore = window.liveGraph._edgeStore;
            const nodes = {};
            const nowTs = Date.now();
            const NODE_TTL = 20 * 1000; // ms - keep nodes for 20s after last seen
            const EDGE_TTL = 8 * 1000; // ms - non-persistent edge lifetime
            const PERSISTENT_TTL = 24 * 3600 * 1000; // 24h for encrypted/malicious

            // update edgeStore with incoming flows (directional keys), keep a sample metadata for hover
            flows.forEach((f) => {
                const s = f.src || f.src_ip || f.src_addr || '0.0.0.0';
                const d = f.dst || f.dst_ip || f.dst_addr || '0.0.0.0';
                const dirKey = `${s}-->${d}`;
                // Determine encrypted and malicious flags from explicit fields only.
                const encrypted = !!f.encrypted || !!f.is_encrypted;
                const malicious = !!f.is_malicious || (typeof f.predicted_label === 'string' && /malware/i.test(f.predicted_label)) || (typeof f.predicted_label === 'string' && /malicious/i.test(f.predicted_label));
                edgeStore[dirKey] = edgeStore[dirKey] || { src: s, dst: d, lastSeen: nowTs, count: 0, encrypted: false, malicious: false };
                edgeStore[dirKey].lastSeen = nowTs;
                edgeStore[dirKey].count = (edgeStore[dirKey].count || 0) + 1;
                edgeStore[dirKey].encrypted = edgeStore[dirKey].encrypted || encrypted;
                edgeStore[dirKey].malicious = edgeStore[dirKey].malicious || malicious;
                // keep last sample metadata for hover (macs, protocol, bytes)
                edgeStore[dirKey].sample = edgeStore[dirKey].sample || {};
                edgeStore[dirKey].sample.src_mac = edgeStore[dirKey].sample.src_mac || f.src_mac || f.srcMac || f.src_mac_addr || '';
                edgeStore[dirKey].sample.dst_mac = edgeStore[dirKey].sample.dst_mac || f.dst_mac || f.dstMac || f.dst_mac_addr || '';
                edgeStore[dirKey].sample.protocol = edgeStore[dirKey].sample.protocol || f.protocol || f.proto || '';
                edgeStore[dirKey].sample.bytes = edgeStore[dirKey].sample.bytes || f.bytes || f.length || 0;

                // update node presence
                if (!nodeStore[s]) nodeStore[s] = { id: s, lastSeen: nowTs, count: 0 };
                if (!nodeStore[d]) nodeStore[d] = { id: d, lastSeen: nowTs, count: 0 };
                nodeStore[s].lastSeen = nowTs;
                nodeStore[d].lastSeen = nowTs;
                nodeStore[s].count = Math.max(1, Math.round((nodeStore[s].count || 0) * 0.98)) + 1;
            });

            // expire non-persistent edges and nodes
            Object.keys(edgeStore).forEach((k) => {
                const e = edgeStore[k];
                const ttl = (e.encrypted || e.malicious) ? PERSISTENT_TTL : EDGE_TTL;
                if (nowTs - (e.lastSeen || 0) > ttl) delete edgeStore[k];
            });

            Object.keys(nodeStore).forEach((k) => {
                // keep node if connected to any persistent edge
                const stillConnected = Object.values(edgeStore).some((e) => e.src === k || e.dst === k);
                if (stillConnected) { nodeStore[k].lastSeen = nowTs; }
                if (nowTs - (nodeStore[k].lastSeen || 0) > NODE_TTL) delete nodeStore[k];
            });

            // build edges array from edgeStore for rendering
            const edges = Object.keys(edgeStore).map((k) => ({ src: edgeStore[k].src, dst: edgeStore[k].dst, malicious: !!edgeStore[k].malicious, encrypted: !!edgeStore[k].encrypted, count: edgeStore[k].count || 1 }));

            // Layout: put all source nodes on the far left and all destination nodes on the far right.
            const srcSet = new Set();
            const dstSet = new Set();
            edges.forEach((e) => { if (e && e.src) srcSet.add(e.src); if (e && e.dst) dstSet.add(e.dst); });
            const srcList = Array.from(srcSet).sort();
            const dstList = Array.from(dstSet).sort();
            const leftX = 60;
            const rightX = cw - 60;
            const pad = 40;
            // vertical spacing per side
            const leftStep = Math.max(1, (ch - pad*2) / (srcList.length + 1));
            const rightStep = Math.max(1, (ch - pad*2) / (dstList.length + 1));

            // ensure visual nodes are created with side-prefixed keys to allow same IP to appear on both sides
            srcList.forEach((ip, i) => { const key = 'src:' + ip; nodes[key] = nodes[key] || { id: ip, count: (nodeStore[ip] && nodeStore[ip].count) || 1 }; nodes[key].x = leftX; nodes[key].y = pad + leftStep * (i + 1); });
            dstList.forEach((ip, i) => { const key = 'dst:' + ip; nodes[key] = nodes[key] || { id: ip, count: (nodeStore[ip] && nodeStore[ip].count) || 1 }; nodes[key].x = rightX; nodes[key].y = pad + rightStep * (i + 1); });

            // For any nodes seen in nodeStore but not in src/dst lists, place them centered vertically
            Object.keys(nodeStore).forEach((ip) => {
                const srcKey = 'src:' + ip; const dstKey = 'dst:' + ip;
                if (!nodes[srcKey] && !nodes[dstKey]) {
                    const key = 'src:' + ip; nodes[key] = { id: ip, count: (nodeStore[ip] && nodeStore[ip].count) || 1, x: (leftX + rightX) / 2, y: ch/2 };
                }
            });

            const nodeKeys = Object.keys(nodes).sort();

            // --- animated nodes: maintain per-node animated state for enter/exit transitions ---
            window.liveGraph._animNodes = window.liveGraph._animNodes || {};
            const animNodes = window.liveGraph._animNodes;
            const animNow = Date.now();
            const ENTER_DUR = 400; // ms
            const EXIT_DUR = 350; // ms
            const SIDE_OFFSET = 40;
            const easeOut = (t) => 1 - Math.pow(1 - t, 3);

            // create or update animated entries for current nodes
            Object.keys(nodes).forEach((k) => {
                const targetX = nodes[k].x;
                const targetY = nodes[k].y;
                const side = (targetX <= cw/2) ? 'left' : 'right';
                if (!animNodes[k]) {
                    // new node: start slightly offset horizontally and fade in
                    const startX = (side === 'left') ? (targetX - SIDE_OFFSET) : (targetX + SIDE_OFFSET);
                    animNodes[k] = { x: startX, y: targetY, startX: startX, startY: targetY, targetX: targetX, targetY: targetY, startTs: animNow, duration: ENTER_DUR, opacity: 0, targetOpacity: 1, removing: false, count: nodes[k].count || 1 };
                } else {
                    const an = animNodes[k];
                    // if returning from removing state, reset
                    if (an.removing) {
                        an.removing = false;
                        an.startTs = animNow; an.duration = ENTER_DUR; an.startX = an.x; an.startY = an.y; an.targetOpacity = 1; an.targetX = targetX; an.targetY = targetY;
                    }
                    // if target moved, animate move
                    if (an.targetX !== targetX || an.targetY !== targetY) {
                        an.startX = an.x; an.startY = an.y; an.targetX = targetX; an.targetY = targetY; an.startTs = animNow; an.duration = 300;
                    }
                    an.count = nodes[k].count || 1;
                }
                // ensure nodes positions are driven by anim state below
            });

            // mark removed nodes to animate out
            Object.keys(animNodes).forEach((k) => {
                if (!(k in nodes) && !animNodes[k].removing) {
                    const an = animNodes[k];
                    an.removing = true;
                    an.startTs = animNow;
                    an.duration = EXIT_DUR;
                    an.startX = an.x; an.startY = an.y;
                    // push outwards when removing
                    an.targetX = (an.x <= cw/2) ? (an.x - SIDE_OFFSET) : (an.x + SIDE_OFFSET);
                    an.targetY = an.y;
                    an.targetOpacity = 0;
                }
            });

            // advance animation state and apply back to nodes map for rendering
            Object.keys(animNodes).forEach((k) => {
                const an = animNodes[k];
                const progress = Math.max(0, Math.min(1, (animNow - (an.startTs || 0)) / (an.duration || 300)));
                const t = easeOut(progress);
                // ensure startX/startY exist
                an.startX = (typeof an.startX === 'number') ? an.startX : an.x;
                an.startY = (typeof an.startY === 'number') ? an.startY : an.y;
                const sx = an.startX, sy = an.startY, tx = an.targetX, ty = an.targetY;
                an.x = (sx * (1 - t)) + (tx * t);
                an.y = (sy * (1 - t)) + (ty * t);
                const so = (typeof an.opacity === 'number') ? an.opacity : 1;
                const to = (typeof an.targetOpacity === 'number') ? an.targetOpacity : 1;
                an.opacity = so * (1 - t) + to * t;
                // when exit animation finished, remove animNodes entry
                if (an.removing && progress >= 1) {
                    delete animNodes[k];
                    return;
                }
                // reflect animated position back into nodes map so existing rendering code uses it
                if (nodes[k]) {
                    nodes[k].x = an.x; nodes[k].y = an.y; nodes[k].opacity = an.opacity; nodes[k].count = an.count || nodes[k].count;
                }
            });

            // Clear entire backing buffer to avoid residual artifacts during zoom/pan.
            // Use device-pixel backing size (canvas.width/height) for a full clear,
            // then reapply DPR scaling and user pan/zoom transforms.
            try {
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                // reapply DPR scale
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            } catch (e) {
                // fallback to previous clear if setTransform fails
                ctx.save();
                ctx.translate(state.panX, state.panY);
                ctx.scale(state.scale, state.scale);
                ctx.clearRect(-state.panX, -state.panY, cw / Math.max(1, state.scale), ch / Math.max(1, state.scale));
                ctx.restore();
            }

            // apply user pan/zoom
            ctx.save();
            ctx.translate(state.panX, state.panY);
            ctx.scale(state.scale, state.scale);

            // group edges by unordered pair but separate forward/backward directions
            const pairMap = {};
            edges.forEach((e) => {
                const a = e.src, b = e.dst;
                const key = a < b ? `${a}|${b}` : `${b}|${a}`;
                pairMap[key] = pairMap[key] || { forward: [], backward: [] };
                if (a < b) pairMap[key].forward.push(e);
                else pairMap[key].backward.push(e);
            });

            const time = Date.now();
            // Draw curved arcs with moving 'flight' particles to mimic a flight map
            Object.keys(pairMap).forEach((key) => {
                const { forward, backward } = pairMap[key];
                const hasF = forward.length > 0;
                const hasB = backward.length > 0;
                const baseSpacing = 28;
                const offsets = [];
                if (hasF && hasB) offsets.push(-baseSpacing/2, baseSpacing/2);
                else if (forward.length > 1) {
                    for (let i = 0; i < forward.length; i++) offsets.push((i - (forward.length - 1) / 2) * 8);
                } else if (backward.length > 1) {
                    for (let i = 0; i < backward.length; i++) offsets.push((i - (backward.length - 1) / 2) * 8);
                } else offsets.push(0);

                function drawEdge(e, idx, directionSign) {
                    const a = nodes['src:' + (e.src || '')] || nodes[e.src];
                    const b = nodes['dst:' + (e.dst || '')] || nodes[e.dst];
                    if (!a || !b) return;
                    const dx = b.x - a.x, dy = b.y - a.y; const len = Math.sqrt(dx*dx+dy*dy) || 1;
                    const ux = dx/len, uy = dy/len;
                    const pxn = -uy, pyn = ux;
                    const offset = offsets[idx % offsets.length] || 0;
                    const ax = a.x + pxn * offset, ay = a.y + pyn * offset;
                    const bx = b.x + pxn * offset, by = b.y + pyn * offset;

                    // control point - push perpendicular to create an arc
                    const midX = (ax + bx) / 2;
                    const midY = (ay + by) / 2;
                    const distance = Math.max(1, Math.sqrt((bx-ax)*(bx-ax) + (by-ay)*(by-ay)));
                    const curvature = Math.min(160, Math.max(30, distance * 0.35));
                    const ctrlX = midX + pxn * curvature * directionSign;
                    const ctrlY = midY + pyn * curvature * directionSign;

                    const isMaliciousEdge = !!e.malicious;
                    const isEncrypted = !!e.encrypted || false;
                    const dirKey = `${e.src}-->${e.dst}`;
                    const hover = chartDom && chartDom._hover ? chartDom._hover : null;
                    const isHoveredEdge = hover && hover.edgeKey === dirKey;
                    // base stroke color depending on malicious flag
                    let strokeColor = isMaliciousEdge ? 'rgba(255,80,90,0.98)' : 'rgba(34,197,94,0.92)';
                    if (isHoveredEdge) {
                        // brighten on hover
                        strokeColor = isMaliciousEdge ? 'rgba(255,100,100,1)' : 'rgba(34,210,80,1)';
                    }
                    ctx.beginPath();
                    ctx.moveTo(ax, ay);
                    ctx.quadraticCurveTo(ctrlX, ctrlY, bx, by);
                    ctx.strokeStyle = strokeColor;
                    // thicker when hovered
                    const baseWidth = isEncrypted ? 3.2 : 2.4;
                    ctx.lineWidth = baseWidth + (isHoveredEdge ? 1.8 : 0);
                    ctx.shadowColor = isMaliciousEdge ? 'rgba(255,80,90,0.14)' : 'rgba(0,210,255,0.10)';
                    ctx.shadowBlur = (isHoveredEdge ? 14 : (isEncrypted ? 10 : 8));
                    const dashLen = isEncrypted ? 14 : 10;
                    ctx.setLineDash([dashLen, dashLen]);
                    ctx.lineDashOffset = -((time / 80) % (dashLen * 2));
                    ctx.stroke();
                    ctx.shadowBlur = 0;

                    // draw arrowhead at destination (small triangle oriented along tangent)
                    const tArrow = 0.92;
                    const axp = ax, ayp = ay, cxp = ctrlX, cyp = ctrlY, bxp = bx, byp = by;
                    // quadratic bezier tangent derivative: B'(t) = 2(1-t)(C-A) + 2t(B-C)
                    const tx = 2*(1 - tArrow)*(cxp - axp) + 2*tArrow*(bxp - cxp);
                    const ty = 2*(1 - tArrow)*(cyp - ayp) + 2*tArrow*(byp - cyp);
                    const tlen = Math.sqrt(tx*tx + ty*ty) || 1;
                    const tux = tx / tlen, tuy = ty / tlen;
                    const perpX = -tuy, perpY = tux;
                    const tipX = (1 - tArrow)*(1 - tArrow)*axp + 2*(1 - tArrow)*tArrow*cxp + tArrow*tArrow*bxp;
                    const tipY = (1 - tArrow)*(1 - tArrow)*ayp + 2*(1 - tArrow)*tArrow*cyp + tArrow*tArrow*byp;
                    ctx.beginPath();
                    ctx.moveTo(tipX, tipY);
                    ctx.lineTo(tipX - tux * 10 + perpX * 6, tipY - tuy * 10 + perpY * 6);
                    ctx.lineTo(tipX - tux * 10 - perpX * 6, tipY - tuy * 10 - perpY * 6);
                    ctx.closePath();
                    ctx.fillStyle = strokeColor;
                    ctx.fill();

                    // moving flight particle along the bezier
                    // slow down particle animation: larger base and gentler reduction by count
                    const speedBase = 3000; // ms per lap for base (larger == slower)
                    const jitterSeed = Math.abs(hashString(e.src + '->' + e.dst)) % 9999;
                    const speed = Math.max(1500, speedBase - Math.min(1000, (e.count || 1) * 60));
                    const phase = ((time + jitterSeed) % speed) / speed; // 0..1
                    const t = phase;
                    const inv = 1 - t;
                    // point on quadratic bezier
                    const px = inv*inv*axp + 2*inv*t*cxp + t*t*bxp;
                    const py = inv*inv*ayp + 2*inv*t*cyp + t*t*byp;
                    // draw particle (larger when its edge is hovered)
                    ctx.beginPath(); ctx.arc(px, py, (isMaliciousEdge ? 3.6 : 2.8) + (isHoveredEdge ? 1.2 : 0), 0, Math.PI*2);
                    ctx.fillStyle = isMaliciousEdge ? 'rgba(255,120,120,1)' : 'rgba(34,197,94,1)';
                    if (isHoveredEdge) {
                        ctx.shadowColor = 'rgba(255,255,255,0.06)'; ctx.shadowBlur = 8;
                    }
                    ctx.fill();
                    ctx.shadowBlur = 0;
                }

                // forward edges (directionSign = 1)
                forward.forEach((e, idx) => drawEdge(e, idx, 1));
                // backward edges (directionSign = -1)
                backward.forEach((e, idx) => drawEdge(e, idx, -1));
            });

            // highlight info from mouse interactions
            const hover = chartDom._hover || null;

            nodeKeys.forEach((k) => {
                const nd = nodes[k];
                // use a fixed node radius so all nodes stay visually uniform
                const isHoveredNode = hover && hover.nodeKey === k;
                const baseRadius = 5;
                const radius = isHoveredNode ? Math.max(8, baseRadius + 4) : baseRadius;
                const alpha = (typeof nd.opacity === 'number') ? nd.opacity : 1;
                ctx.save();
                ctx.globalAlpha = alpha;
                ctx.beginPath();
                ctx.arc(nd.x, nd.y, radius, 0, Math.PI*2);
                ctx.fillStyle = isHoveredNode ? (nd.id && (String(nd.id).includes('mal') || String(nd.id).includes('Mal') ) ? 'rgba(255,120,120,1)' : 'rgba(34,197,94,1)') : (nd.opacity < 1 ? 'rgba(0,229,255,0.55)' : 'rgba(0,229,255,0.92)');
                ctx.fill();
                ctx.lineWidth = isHoveredNode ? 2.4 : 1.2;
                ctx.strokeStyle = isHoveredNode ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.30)';
                if (isHoveredNode) { ctx.shadowColor = 'rgba(255,255,255,0.06)'; ctx.shadowBlur = 8; }
                ctx.stroke();
                // label below node
                ctx.font = '11px Inter, sans-serif';
                ctx.fillStyle = '#E6EEF8';
                const label = nd.id || k;
                const metrics = ctx.measureText(label);
                const pad = 8;
                const lx = nd.x - metrics.width/2 - (pad/2);
                const ly = nd.y + radius + 10; // slightly closer
                ctx.fillStyle = 'rgba(0,0,0,0.5)';
                ctx.fillRect(lx, ly-12, metrics.width + pad, 16);
                ctx.fillStyle = '#E6EEF8';
                if (isHoveredNode) { ctx.font = '12px Inter, sans-serif'; }
                ctx.fillText(label, lx + (pad/2), ly);
                ctx.restore();
            });

            ctx.restore();

            if (!canvas.dataset.panHandlersBound) {
                canvas.dataset.panHandlersBound = '1';
                canvas.style.cursor = 'grab';
                canvas.addEventListener('mousedown', (ev) => {
                    ev.preventDefault();
                    state.isPanning = true;
                    canvas.style.cursor = 'grabbing';
                    state.lastX = ev.clientX;
                    state.lastY = ev.clientY;
                });
                window.addEventListener('mousemove', (ev) => {
                    if (!state.isPanning) return;
                    const dx = ev.clientX - state.lastX;
                    const dy = ev.clientY - state.lastY;
                    state.lastX = ev.clientX; state.lastY = ev.clientY;
                    state.panX += dx; state.panY += dy;
                    try { renderLiveFlowGraph_impl(window._lastLiveFlows || []); } catch(e){}
                });
                window.addEventListener('mouseup', (ev) => {
                    if (!state.isPanning) return;
                    state.isPanning = false;
                    canvas.style.cursor = 'grab';
                });
                canvas.addEventListener('wheel', (ev) => {
                    ev.preventDefault();
                    const rect = canvas.getBoundingClientRect();
                    const mx = ev.clientX - rect.left;
                    const my = ev.clientY - rect.top;
                    const prev = state.scale;
                    const delta = -ev.deltaY * 0.0015;
                    const next = Math.min(3, Math.max(0.5, prev * (1 + delta)));
                    const ratio = next / prev;
                    state.scale = next;
                    state.panX = mx - ratio * (mx - state.panX);
                    state.panY = my - ratio * (my - state.panY);
                    try { renderLiveFlowGraph_impl(window._lastLiveFlows || []); } catch(e){}
                }, { passive: false });
                let touchLast = null;
                canvas.addEventListener('touchstart', (ev) => {
                    if (ev.touches.length === 1) { state.isPanning = true; touchLast = { x: ev.touches[0].clientX, y: ev.touches[0].clientY }; }
                }, { passive: true });
                canvas.addEventListener('touchmove', (ev) => {
                    if (ev.touches.length === 1 && state.isPanning && touchLast) {
                        const dx = ev.touches[0].clientX - touchLast.x;
                        const dy = ev.touches[0].clientY - touchLast.y;
                        touchLast = { x: ev.touches[0].clientX, y: ev.touches[0].clientY };
                        state.panX += dx; state.panY += dy;
                        try { renderLiveFlowGraph_impl(window._lastLiveFlows || []); } catch(e){}
                    }
                }, { passive: true });
                canvas.addEventListener('touchend', (ev) => { state.isPanning = false; touchLast = null; });
            }
            // ensure an animation loop runs to animate dash offset even when flows don't update
            if (!window.liveGraph._animLoopBound) {
                window.liveGraph._animLoopBound = true;
                (function animLoop(){
                    try { renderLiveFlowGraph_impl(window._lastLiveFlows || []); } catch(e){}
                    window.requestAnimationFrame(animLoop);
                })();
            }

            // tooltip handling: create a floating div for details
            if (!chartDom._lgTooltip) {
                const tip = document.createElement('div');
                tip.style.position = 'absolute';
                tip.style.pointerEvents = 'none';
                tip.style.zIndex = 9999;
                tip.style.background = 'rgba(2,6,23,0.9)';
                tip.style.color = '#E6EEF8';
                tip.style.padding = '6px 8px';
                tip.style.borderRadius = '6px';
                tip.style.fontSize = '12px';
                tip.style.boxShadow = '0 4px 14px rgba(2,6,23,0.6)';
                tip.style.display = 'none';
                chartDom.style.position = chartDom.style.position || 'relative';
                chartDom.appendChild(tip);
                chartDom._lgTooltip = tip;

                canvas.addEventListener('mousemove', (ev) => {
                    const rect = canvas.getBoundingClientRect();
                    const mx = ev.clientX - rect.left;
                    const my = ev.clientY - rect.top;
                    const tx = (mx - state.panX) / state.scale;
                    const ty = (my - state.panY) / state.scale;
                    // find nearest node
                    let nearestNode = null; let bestNodeD = 999999; let nearestNodeKey = null;
                    nodeKeys.forEach((k) => {
                        const nd = nodes[k];
                        if (!nd) return;
                        const dx = nd.x - tx; const dy = nd.y - ty; const d = Math.sqrt(dx*dx+dy*dy);
                        if (d < bestNodeD) { bestNodeD = d; nearestNode = nd; nearestNodeKey = k; }
                    });
                    // find nearest edge (consider offsets when bidirectional)
                    let nearestEdge = null; let bestEdgeD = 999999;
                    const spacing = 10;
                    edges.forEach((e) => {
                        const a = nodes['src:' + (e.src || '')] || nodes[e.src] || nodes['dst:' + (e.src || '')];
                        const b = nodes['dst:' + (e.dst || '')] || nodes[e.dst] || nodes['src:' + (e.dst || '')];
                        if (!a || !b) return;
                        const dx = b.x - a.x, dy = b.y - a.y; const len = Math.sqrt(dx*dx+dy*dy)||1; const ux = dx/len, uy = dy/len; const pxn = -uy, pyn = ux;
                        // detect if reverse exists
                        const revKey = `${e.dst}-->${e.src}`;
                        const hasRev = !!edgeStore[revKey];
                        const dirKey = `${e.src}-->${e.dst}`;
                        const offset = hasRev ? (spacing/2) : 0;
                        const ax = a.x + pxn * offset, ay = a.y + pyn * offset;
                        const bx = b.x + pxn * offset, by = b.y + pyn * offset;
                        // approximate distance from point (tx,ty) to the quadratic bezier curve
                        // control point computed to match drawing curvature
                        const midX = (ax + bx) / 2;
                        const midY = (ay + by) / 2;
                        const segDist = Math.max(1, Math.sqrt((bx-ax)*(bx-ax) + (by-ay)*(by-ay)));
                        const curvature = Math.min(160, Math.max(30, segDist * 0.35));
                        // choose sign based on natural perpendicular direction (use positive for detection)
                        const ctrlX = midX + pxn * curvature;
                        const ctrlY = midY + pyn * curvature;
                        // sample the bezier at several t values to find closest point (cheap and robust)
                        let localBest = 999999; let bestSampleT = 0; let bestPx = ax, bestPy = ay;
                        const STEPS = 18; // ~19 samples
                        for (let si = 0; si <= STEPS; si++) {
                            const tS = si / STEPS;
                            // quadratic bezier point: B(t) = (1-t)^2*A + 2(1-t)t*C + t^2*B
                            const omt = (1 - tS);
                            const px = omt*omt*ax + 2*omt*tS*ctrlX + tS*tS*bx;
                            const py = omt*omt*ay + 2*omt*tS*ctrlY + tS*tS*by;
                            const ddx = tx - px, ddy = ty - py; const dS = Math.sqrt(ddx*ddx + ddy*ddy);
                            if (dS < localBest) { localBest = dS; bestSampleT = tS; bestPx = px; bestPy = py; }
                        }
                        if (localBest < bestEdgeD) { bestEdgeD = localBest; nearestEdge = { edge: e, ax, ay, bx, by, dirKey: dirKey, sample: (edgeStore[dirKey] && edgeStore[dirKey].sample) || {}, ctrlX, ctrlY, hitPoint: {x: bestPx, y: bestPy, t: bestSampleT} }; }
                    });

                    // decide whether to show edge or node tooltip (edge prioritized)
                    if (nearestEdge && bestEdgeD <= 12) {
                        const e = nearestEdge.edge;
                        const s = e.src || '-'; const d = e.dst || '-';
                        const sample = nearestEdge.sample || {};
                        const protoText = protoToName(sample.protocol || sample.proto || sample.protocol_number || sample.proto_number || '-');
                        const mal = e.malicious ? 'yes' : 'no';
                        const encryptedText = e.encrypted ? 'yes' : (e.malicious ? 'suspicious' : 'no');
                        const predicted = sample.predicted_label || sample.predicted || '-';
                        const malConf = (typeof sample.malware_confidence === 'number') ? `${(sample.malware_confidence*100).toFixed(1)}%` : (sample.malware_conf || sample.malware_confidence || '-');
                        const lines = [`${s} → ${d}`, `Proto: ${protoText}`, `Bytes: ${sample.bytes || 0}`, `Malicious: ${mal}`, `Malware Conf: ${malConf}`, `Predicted: ${predicted}`, `Src MAC: ${sample.src_mac || '-'}`, `Dst MAC: ${sample.dst_mac || '-'}`, `Encrypted: ${encryptedText}`];
                        chartDom._lgTooltip.innerText = lines.join('\n');
                        chartDom._lgTooltip.style.left = (mx + 12) + 'px';
                        chartDom._lgTooltip.style.top = (my + 12) + 'px';
                        chartDom._lgTooltip.style.display = 'block';
                        // store hover state for rendering highlights
                        chartDom._hover = { edgeKey: nearestEdge.dirKey, nodeKey: null };
                        return;
                    }

                    if (nearestNode && bestNodeD <= 24) {
                        const count = nearestNode.count || 0;
                        chartDom._lgTooltip.innerText = `${nearestNode.id}\nSessions: ${count}`;
                        chartDom._lgTooltip.style.left = (mx + 12) + 'px';
                        chartDom._lgTooltip.style.top = (my + 12) + 'px';
                        chartDom._lgTooltip.style.display = 'block';
                        chartDom._hover = { nodeKey: nearestNodeKey, edgeKey: null };
                    } else {
                        chartDom._lgTooltip.style.display = 'none';
                        // clear hover state
                        chartDom._hover = null;
                    }
                });
                canvas.addEventListener('mouseleave', () => { if (chartDom._lgTooltip) chartDom._lgTooltip.style.display = 'none'; });
            }
        } catch (e) { console.warn('liveGraph render error', e); }
    }

    // expose
    window.liveGraph = window.liveGraph || {};
    window.liveGraph.renderLiveFlowGraph = renderLiveFlowGraph_impl;
    window.liveGraph.zoomIn = function() { try { window.liveGraphState = window.liveGraphState || { scale: 1 }; window.liveGraphState.scale = Math.min(3, (window.liveGraphState.scale || 1) * 1.2); renderLiveFlowGraph_impl(window._lastLiveFlows || []); } catch(e){} };
    window.liveGraph.zoomOut = function() { try { window.liveGraphState = window.liveGraphState || { scale: 1 }; window.liveGraphState.scale = Math.max(0.4, (window.liveGraphState.scale || 1) / 1.2); renderLiveFlowGraph_impl(window._lastLiveFlows || []); } catch(e){} };
    window.liveGraph.resetView = function() { try { window.liveGraphState = window.liveGraphState || { scale: 1, panX:0, panY:0 }; window.liveGraphState.scale = 1; window.liveGraphState.panX = 0; window.liveGraphState.panY = 0; renderLiveFlowGraph_impl(window._lastLiveFlows || []); } catch(e){} };
    window.liveGraph.clearLiveLogs = function() {
        try {
            window.liveLogsClearedAt = Date.now() / 1000;
            try { window.localStorage && window.localStorage.setItem && window.localStorage.setItem('liveLogsClearedAt', String(window.liveLogsClearedAt)); } catch(e){}
            window._lastLiveFlows = [];
            if (window.liveGraph && window.liveGraph._nodeStore) window.liveGraph._nodeStore = {};
            if (window.liveGraph && window.liveGraph._edgeStore) window.liveGraph._edgeStore = {};
            if (window.liveGraph && window.liveGraph._animNodes) window.liveGraph._animNodes = {};
            if (window.recordStores && Array.isArray(window.recordStores.live)) window.recordStores.live = [];
            if (window._liveSeen) window._liveSeen = new Set();
            try { if (window.resultsImpl && typeof window.resultsImpl.persistRecordStores === 'function') window.resultsImpl.persistRecordStores(); } catch(e){}
            try { if (window.liveTrendImpl && typeof window.liveTrendImpl.resetTrendChart === 'function') window.liveTrendImpl.resetTrendChart(); } catch(e){}
            const listDom = document.getElementById('liveFlowList');
            if (listDom) listDom.innerHTML = '<p class="text-sm text-gray-400">暂无实时流量</p>';
            const chartDom = document.getElementById('liveFlowGraph');
            if (chartDom) {
                const canvas = chartDom.querySelector('canvas');
                if (canvas) {
                    const ctx = canvas.getContext('2d');
                    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
            }
            renderLiveFlowGraph_impl([]);
            try { if (typeof window.renderRecordedRows === 'function') window.renderRecordedRows('live'); } catch(e){}
        } catch (e) { console.warn('clearLiveLogs failed', e); }
    };

    window.liveGraph.restoreFromRecords = function(records) {
        try {
            const list = Array.isArray(records) ? records : [];
            const flows = list
                .map((item) => item && item.data ? item.data : null)
                .filter((item) => item && typeof item === 'object');
            if (flows.length === 0) {
                renderLiveFlowGraph_impl([]);
                return;
            }
            window._lastLiveFlows = flows;
            renderLiveFlowGraph_impl(flows);
        } catch (e) { console.warn('restoreFromRecords failed', e); }
    };

    function bindToolbarButtons() {
        const clearBtn = document.getElementById('clearLiveLogsBtn');
        const zoomOutBtn = document.getElementById('liveFlowZoomOut');
        const zoomResetBtn = document.getElementById('liveFlowZoomReset');
        const zoomInBtn = document.getElementById('liveFlowZoomIn');
        if (clearBtn && !clearBtn._boundClick) {
            clearBtn._boundClick = () => {
                if (window.liveImpl && typeof window.liveImpl.clearLiveLogs === 'function') return window.liveImpl.clearLiveLogs();
                return window.liveGraph.clearLiveLogs && window.liveGraph.clearLiveLogs();
            };
            clearBtn.addEventListener('click', clearBtn._boundClick);
        }
        if (zoomOutBtn && !zoomOutBtn._boundClick) {
            zoomOutBtn._boundClick = () => window.liveGraph.zoomOut && window.liveGraph.zoomOut();
            zoomOutBtn.addEventListener('click', zoomOutBtn._boundClick);
        }
        if (zoomResetBtn && !zoomResetBtn._boundClick) {
            zoomResetBtn._boundClick = () => window.liveGraph.resetView && window.liveGraph.resetView();
            zoomResetBtn.addEventListener('click', zoomResetBtn._boundClick);
        }
        if (zoomInBtn && !zoomInBtn._boundClick) {
            zoomInBtn._boundClick = () => window.liveGraph.zoomIn && window.liveGraph.zoomIn();
            zoomInBtn.addEventListener('click', zoomInBtn._boundClick);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bindToolbarButtons);
    } else {
        bindToolbarButtons();
    }
})();
