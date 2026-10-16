/* ==========================================================================
   实时流量拓扑：以本机为中心的辐射图（host-centric radial graph）
   --------------------------------------------------------------------------
   为什么不是地图：把内网 IP 映射到城市是编造出来的信息；数据的本质只有四件事
   ——谁和谁、什么方向、多少量、是否恶意——辐射图直接表达这四件事：
     · 圆心是本机，对端按"私网内环 / 公网外环"环绕；
     · 连线粗细表示字节量；
     · 脉冲沿连线往返，向外=出站，向内=入站；
     · 恶意会话整条连线转为告警色，并在首次出现时扩散一次。

   性能约定（本次重写的核心）：
     · 高频（每次轮询，约 1.5s）：ingest() 合并数据、做过期清理、标记 layoutDirty。
     · 低频（仅当数据或画布尺寸变化）：rebuildLayout() 一次性算好节点坐标、半径、
       边宽、脉冲周期与相位，存成快照。
     · 逐帧 drawFrame() 只读快照画线画点，不做排序、filter、哈希与碰撞检测。
   旧实现每帧都跑一整套 renderLiveFlowGraph_impl（含底图重建、O(n²) 连接判断、
   标签重排），掉帧和跳变是必然的。

   对外契约（与 live_impl.js / results_impl.js 的既有约定，不可改）：
     window.liveGraph.renderLiveFlowGraph(flows) / zoomIn / zoomOut / resetView /
     clearLiveLogs / restoreFromRecords；window.renderLiveFlowGraph(flows)；
     window.liveGraphState{panX,panY,scale,isPanning,lastX,lastY}；
     window.liveGraph._nodeStore/_edgeStore/_animNodes/_animLoopBound；
     window.liveLogsClearedAt 与 localStorage['liveLogsClearedAt']；
     canvas.dataset.panHandlersBound / livegraphLastW / livegraphLastH；
     chartDom._lgTooltip 与 chartDom._hover{edgeKey,nodeKey}；#liveFlowList 列表。
   ========================================================================== */

(function () {
    'use strict';

    /* ------------------------------ 常量 ------------------------------ */
    // 保留策略：图上内容不再按时间自动删除。停止抓包后记录仍然留在图上，只有点「清空」才消失。
    // FRESH_* 只决定「活跃」外观（亮线 + 流动脉冲），超过它的事务降级为静态轨迹。
    var FRESH_NODE_MS = 20000;             // 20s 内到达过的节点算活跃
    var FRESH_EDGE_MS = 8000;              // 8s 内到达过的会话算活跃
    var RETAIN_MS = 24 * 3600 * 1000;      // 兜底上限：页面极长时间不清理时最多保留 24h
    var MAX_RENDER_NODES = 60;
    var MAX_RENDER_EDGES = 90;
    var MAX_FLOW_LIST_ROWS = 60;
    var HIT_NODE = 14;
    var HIT_EDGE = 8;
    var IDLE_FRAME_MS = 1000 / 30;        // 无数据时 30fps 足够（只有静态空态）
    var ALERT_MS = 620;                   // 恶意告警扩散环只画一次，持续 620ms
    var TAU = Math.PI * 2;

    var COLOR_OK = '34, 211, 238';
    var COLOR_BAD = '244, 63, 94';

    /* ------------------------------ 状态 ------------------------------ */
    var nodeStore = {};   // ip -> {ip, local, malicious, bytes, count, lastSeen, alertAt}
    var edgeStore = {};   // key -> {src, dst, sport, dport, protocol, bytes, count, malicious, encrypted, lastSeen, confidence, alertAt}
    var animNodes = {};   // 'n:'+ip -> 快照节点（供外部读取）
    var layout = null;    // {nodes:[], edges:[], center:{x,y}, radius, builtAt}
    var layoutDirty = true;
    var trackedFlows = [];  // 最近一次 ingest 的规范化流，用于列表
    var fps = 60;
    var fpsFrames = 0;
    var fpsWindowStart = 0;
    var lastDrawAt = 0;

    /* --------------------------- 小工具 --------------------------- */
    function hashString(str) {
        var h = 0;
        var s = String(str == null ? '' : str);
        for (var i = 0; i < s.length; i++) {
            h = (h * 31 + s.charCodeAt(i)) | 0;
        }
        return Math.abs(h);
    }

    function isPrivateIp(ip) {
        var s = String(ip == null ? '' : ip).trim().toLowerCase();
        if (!s) return true;
        if (s === '::1' || s === 'localhost') return true;
        if (s.indexOf('.') === -1) {
            // IPv6：链路本地与唯一本地地址视为内网
            return s.indexOf('fe80') === 0 || s.indexOf('fc') === 0 || s.indexOf('fd') === 0;
        }
        var parts = s.split('.');
        if (parts.length !== 4) return true;
        var a = parseInt(parts[0], 10);
        var b = parseInt(parts[1], 10);
        if (!isFinite(a) || !isFinite(b)) return true;
        if (a === 10 || a === 127 || a === 0) return true;
        if (a === 192 && b === 168) return true;
        if (a === 172 && b >= 16 && b <= 31) return true;
        if (a === 169 && b === 254) return true;
        if (a >= 224 && a <= 239) return true;
        if (a === 255) return true;
        return false;
    }

    function formatBytes(n) {
        var v = Number(n) || 0;
        if (v < 1024) return v + ' B';
        if (v < 1024 * 1024) return (v / 1024).toFixed(1) + ' KB';
        if (v < 1024 * 1024 * 1024) return (v / 1024 / 1024).toFixed(1) + ' MB';
        return (v / 1024 / 1024 / 1024).toFixed(2) + ' GB';
    }

    function protoToName(p) {
        var s = String(p == null ? '' : p).trim();
        if (!s) return '未知';
        if (/^\d+$/.test(s)) {
            var n = parseInt(s, 10);
            if (n === 6) return 'TCP';
            if (n === 17) return 'UDP';
            if (n === 1) return 'ICMP';
            return '协议 ' + n;
        }
        return s.toUpperCase();
    }

    function normalizeFlow(raw) {
        if (!raw || typeof raw !== 'object') return null;
        var src = raw.src || raw.src_ip || raw.src_addr || '';
        var dst = raw.dst || raw.dst_ip || raw.dst_addr || '';
        if (!src || !dst) return null;
        var label = String(raw.predicted_label == null ? '' : raw.predicted_label);
        var malicious = (raw.is_malicious === true) || /malware|malicious/i.test(label);
        var conf = raw.malware_confidence;
        if (conf === undefined || conf === null) conf = raw.confidence;
        conf = parseFloat(conf);
        if (!isFinite(conf)) conf = NaN;
        var capturedAt = parseFloat(raw.captured_at);
        if (!isFinite(capturedAt)) capturedAt = Date.now() / 1000;
        return {
            id: raw.id !== undefined ? raw.id : null,
            src: String(src),
            dst: String(dst),
            sport: raw.sport !== undefined && raw.sport !== null ? raw.sport : (raw.src_port !== undefined ? raw.src_port : ''),
            dport: raw.dport !== undefined && raw.dport !== null ? raw.dport : (raw.dst_port !== undefined ? raw.dst_port : ''),
            protocol: raw.protocol !== undefined ? raw.protocol : raw.proto,
            bytes: Number(raw.bytes || raw.length || 0) || 0,
            malicious: malicious,
            encrypted: raw.encrypted === true || raw.is_encrypted === true,
            confidence: conf,
            capturedAt: capturedAt,
            raw: raw
        };
    }

    /* --------------------------- 数据聚合（高频，无需逐帧） --------------------------- */
    function ingest(flows) {
        var now = Date.now();
        var list = Array.isArray(flows) ? flows : [];
        var normalized = [];

        for (var i = 0; i < list.length; i++) {
            var f = normalizeFlow(list[i]);
            if (!f) continue;
            normalized.push(f);

            var pairKey = f.src < f.dst ? f.src + '|' + f.dst : f.dst + '|' + f.src;
            var key = pairKey + '|' + f.sport + '|' + f.dport + '|' + (f.protocol == null ? '' : f.protocol);
            var edge = edgeStore[key];
            if (!edge) {
                edge = edgeStore[key] = {
                    key: key, pairKey: pairKey,
                    src: f.src, dst: f.dst,
                    sport: f.sport, dport: f.dport,
                    protocol: f.protocol,
                    bytes: 0, count: 0,
                    malicious: f.malicious, encrypted: f.encrypted,
                    confidence: f.confidence,
                    firstSeen: now, lastSeen: now, alertAt: 0,
                    sample: f.raw
                };
            }
            edge.bytes += f.bytes;
            edge.count += 1;
            edge.lastSeen = now;
            edge.sample = f.raw;
            if (f.malicious) {
                edge.malicious = true;
                if (!edge.alertAt) edge.alertAt = now;   // 每次"由正常转恶意"只响一次
            }
            if (f.encrypted) edge.encrypted = true;
            if (isFinite(f.confidence)) edge.confidence = f.confidence;

            bumpNode(f.src, f.bytes, f.malicious, now);
            bumpNode(f.dst, f.bytes, f.malicious, now);
        }

        if (!normalized.length) return;

        // 过期清理：低频做，逐帧不做
        pruneStores(now);

        var merged = trackedFlows.concat(normalized);
        if (merged.length > 400) merged = merged.slice(merged.length - 400);
        trackedFlows = merged;

        layoutDirty = true;
        renderFlowList(trackedFlows);
    }

    function bumpNode(ip, bytes, malicious, now) {
        var nd = nodeStore[ip];
        if (!nd) {
            nd = nodeStore[ip] = {
                ip: ip, local: isPrivateIp(ip),
                malicious: false, bytes: 0, count: 0,
                lastSeen: now, alertAt: 0
            };
        }
        nd.bytes += bytes;
        nd.count += 1;
        nd.lastSeen = now;
        if (malicious) {
            nd.malicious = true;
            if (!nd.alertAt) nd.alertAt = now;
        }
    }

    function pruneStores(now) {
        // 只做兜底清理：普通会话不再 8s/20s 后被删掉，否则停止抓包后图上什么都不剩。
        var ip;
        for (ip in nodeStore) {
            if (!Object.prototype.hasOwnProperty.call(nodeStore, ip)) continue;
            if (now - nodeStore[ip].lastSeen > RETAIN_MS) delete nodeStore[ip];
        }
        var key;
        for (key in edgeStore) {
            if (!Object.prototype.hasOwnProperty.call(edgeStore, key)) continue;
            if (now - edgeStore[key].lastSeen > RETAIN_MS) delete edgeStore[key];
        }
    }

    /* --------------------------- 布局快照（低频） --------------------------- */
    function rebuildLayout(cw, ch) {
        var center = { x: cw / 2, y: ch / 2 };
        var R = Math.max(40, Math.min(cw, ch) * 0.44);

        var locals = [];
        var remotes = [];
        var ips = Object.keys(nodeStore);
        for (var i = 0; i < ips.length; i++) {
            var nd = nodeStore[ips[i]];
            if (nd.local) locals.push(nd); else remotes.push(nd);
        }

        // 会话数多的优先保留，避免节点过多时画面糊成一团
        function byWeight(a, b) { return (b.count + b.bytes / 4096) - (a.count + a.bytes / 4096); }
        locals.sort(byWeight);
        remotes.sort(byWeight);
        locals = locals.slice(0, MAX_RENDER_NODES);
        remotes = remotes.slice(0, MAX_RENDER_NODES);

        var nodes = [];
        var anim = {};

        function place(store, ringIndex) {
            var n = store.length;
            if (!n) return;
            var radius = ringIndex === 0 ? R * 0.46 : R * 0.82;
            // 角度锚定在 IP 哈希上：同一台主机始终出现在同一方向，重新布局时不跳。
            var slots = [];
            for (var k = 0; k < n; k++) {
                var ip = store[k].ip;
                var angle = (hashString(ip) % 10000) / 10000 * TAU;
                slots.push({ nd: store[k], angle: angle });
            }
            slots.sort(function (a, b) { return a.angle - b.angle; });

            // 同环相邻角度太近时做一次最小推开，避免节点重叠
            var minGap = TAU / Math.max(n, 1) * 0.55;
            for (var m = 1; m < slots.length; m++) {
                var gap = slots[m].angle - slots[m - 1].angle;
                if (gap < minGap) slots[m].angle = slots[m - 1].angle + minGap;
            }

            for (var j = 0; j < slots.length; j++) {
                var item = slots[j];
                var nd2 = item.nd;
                var x = center.x + Math.cos(item.angle) * radius;
                var y = center.y + Math.sin(item.angle) * radius;
                var nodeRadius = 3 + Math.min(5, Math.log10(nd2.bytes + 10) * 1.3);
                var node = {
                    ip: nd2.ip,
                    x: x, y: y,
                    radius: nodeRadius,
                    local: nd2.local,
                    malicious: nd2.malicious,
                    bytes: nd2.bytes,
                    count: nd2.count,
                    alertAt: nd2.alertAt,
                    lastSeen: nd2.lastSeen,
                    angle: item.angle,
                    ring: ringIndex
                };
                nodes.push(node);
                anim['n:' + nd2.ip] = node;
            }
        }

        place(locals, 0);
        place(remotes, 1);

        var nodeByIp = {};
        for (var q = 0; q < nodes.length; q++) nodeByIp[nodes[q].ip] = nodes[q];

        // 边：本机（圆心）↔ 对端
        var edges = [];
        var keys = Object.keys(edgeStore);
        keys.sort(function (a, b) { return (edgeStore[b].bytes + edgeStore[b].count * 512) - (edgeStore[a].bytes + edgeStore[a].count * 512); });
        keys = keys.slice(0, MAX_RENDER_EDGES);

        for (var r = 0; r < keys.length; r++) {
            var e = edgeStore[keys[r]];
            var srcLocal = isPrivateIp(e.src);
            var peerIp = srcLocal ? e.dst : e.src;
            var peer = nodeByIp[peerIp];
            if (!peer) continue;
            var weight = Math.log10(e.bytes + 10);
            var period = Math.max(900, 2400 - weight * 260);   // 数据量越大，脉冲跑得越快
            edges.push({
                key: e.key,
                lastSeen: e.lastSeen,
                src: e.src, dst: e.dst,
                sport: e.sport, dport: e.dport,
                protocol: e.protocol,
                bytes: e.bytes,
                count: e.count,
                malicious: e.malicious,
                encrypted: e.encrypted,
                confidence: e.confidence,
                sample: e.sample,
                outbound: srcLocal,
                width: Math.max(1, Math.min(4, 1 + weight * 0.55)),
                period: period,
                phase: (hashString(e.key) % 1000) / 1000,
                x1: center.x, y1: center.y,
                x2: peer.x, y2: peer.y,
                peerIp: peerIp,
                alertAt: e.alertAt
            });
        }

        layout = {
            center: center,
            radius: R,
            nodes: nodes,
            edges: edges,
            nodeByIp: nodeByIp,
            builtAt: Date.now()
        };
        layoutDirty = false;

        // 保持 window.liveGraph._animNodes 的既有语义（'n:'+ip -> 节点运行时状态）
        for (var oldKey in animNodes) {
            if (Object.prototype.hasOwnProperty.call(animNodes, oldKey)) delete animNodes[oldKey];
        }
        for (var nk in anim) {
            if (Object.prototype.hasOwnProperty.call(anim, nk)) animNodes[nk] = anim[nk];
        }
    }

    /* --------------------------- 绘制（逐帧，只读快照） --------------------------- */
    var ctx = null;
    var chartDom = null;
    var canvas = null;
    var cw = 600;
    var ch = 300;
    var dpr = 1;

    function ensureCanvas() {
        chartDom = document.getElementById('liveFlowGraph');
        if (!chartDom) return false;
        canvas = chartDom.tagName === 'CANVAS' ? chartDom : chartDom.querySelector('canvas');
        if (!canvas) {
            canvas = document.createElement('canvas');
            canvas.style.width = '100%';
            canvas.style.height = '100%';
            canvas.style.display = 'block';
            chartDom.appendChild(canvas);
        }
        ctx = canvas.getContext ? canvas.getContext('2d') : null;
        if (!ctx) {
            if (!chartDom.querySelector('.liveGraph-fallback')) {
                var tip = document.createElement('div');
                tip.className = 'liveGraph-fallback text-xs text-gray-500 p-3';
                tip.innerText = '当前浏览器不支持 Canvas 2D 渲染';
                chartDom.appendChild(tip);
            }
            return false;
        }
        resizeCanvas();
        return true;
    }

    function resizeCanvas() {
        if (!chartDom || !canvas) return false;
        var w = chartDom.clientWidth || 600;
        var h = chartDom.clientHeight || 300;
        var ratio = window.devicePixelRatio || 1;
        var pxW = Math.max(1, Math.floor(w * ratio));
        var pxH = Math.max(1, Math.floor(h * ratio));
        var changed = canvas.width !== pxW || canvas.height !== pxH;

        cw = w;
        ch = h;
        dpr = ratio;
        canvas.width = pxW;
        canvas.height = pxH;
        canvas.dataset.livegraphLastW = String(w);
        canvas.dataset.livegraphLastH = String(h);

        if (changed) layoutDirty = true;
        return changed;
    }

    function drawFrame(ts) {
        if (!ctx || !chartDom) return;

        var w = chartDom.clientWidth || 600;
        var h = chartDom.clientHeight || 300;
        if (w !== cw || h !== ch || (window.devicePixelRatio || 1) !== dpr) {
            resizeCanvas();
        }

        if (layoutDirty || !layout) rebuildLayout(cw, ch);

        var hasData = layout.nodes.length > 0;
        // 空态是静态的，30fps 足够，别为了一个静止的点跑满帧
        var minInterval = hasData ? 0 : IDLE_FRAME_MS;
        if (minInterval && ts - lastDrawAt < minInterval) return;
        lastDrawAt = ts;

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cw, ch);

        // 背景：中心到两环的同心参考环（静态，无动画）
        ctx.save();
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.10)';
        ctx.lineWidth = 1;
        for (var ring = 1; ring <= 4; ring++) {
            ctx.beginPath();
            ctx.arc(layout.center.x, layout.center.y, layout.radius * ring / 4, 0, TAU);
            ctx.stroke();
        }
        ctx.restore();

        ctx.save();
        ctx.translate(state.panX, state.panY);
        ctx.scale(state.scale, state.scale);

        drawEdges(ts);
        drawNodes(ts);

        ctx.restore();

        drawLegend(hasData);
        if (!hasData) drawEmptyState();
    }

    function drawEdges(ts) {
        var hover = chartDom._hover || {};
        for (var i = 0; i < layout.edges.length; i++) {
            var e = layout.edges[i];
            var isHover = hover.edgeKey === e.key;
            var rgb = e.malicious ? COLOR_BAD : COLOR_OK;
            // 活跃会话亮、有脉冲；更早的会话降级成静态轨迹线，但不会消失
            var fresh = (ts - (e.lastSeen || 0)) < FRESH_EDGE_MS;
            var idleAlpha = e.malicious ? 0.55 : 0.32;
            var alpha = isHover ? 0.95 : (fresh ? idleAlpha : idleAlpha * 0.34);

            ctx.strokeStyle = 'rgba(' + rgb + ', ' + alpha + ')';
            ctx.lineWidth = isHover ? e.width + 0.8 : (fresh ? e.width : Math.max(1, e.width - 0.6));
            ctx.beginPath();
            ctx.moveTo(e.x1, e.y1);
            ctx.lineTo(e.x2, e.y2);
            ctx.stroke();

            // 方向：连线靠近对端处画一个静态箭头，表示这条会话的主方向
            drawDirectionMark(e, rgb, alpha);

            // 往返脉冲：向外=出站，向内=入站；只有活跃会话才跑，避免几十条旧会话同时闪
            if (fresh) {
                var t = ((ts - layout.builtAt) / e.period + e.phase) % 1;
                if (t < 0) t += 1;
                var p = e.outbound ? t : 1 - t;
                drawPulse(e, p, rgb, isHover);
            }

            // 恶意会话首次出现时扩散一次，不做无限脉冲
            if (e.alertAt) {
                var age = ts - e.alertAt;
                if (age >= 0 && age < ALERT_MS) {
                    var k = age / ALERT_MS;
                    drawAlertRing(e.x2, e.y2, 10 + 26 * k, 0.42 * (1 - k));
                }
            }
        }
    }

    function drawDirectionMark(e, rgb, alpha) {
        var dx = e.x2 - e.x1;
        var dy = e.y2 - e.y1;
        var len = Math.sqrt(dx * dx + dy * dy) || 1;
        var ux = dx / len;
        var uy = dy / len;
        var tipX = e.x1 + dx * 0.68;
        var tipY = e.y1 + dy * 0.68;
        var size = 4 + e.width;

        ctx.fillStyle = 'rgba(' + rgb + ', ' + Math.min(0.9, alpha + 0.25) + ')';
        ctx.beginPath();
        if (e.outbound) {
            ctx.moveTo(tipX + ux * size, tipY + uy * size);
            ctx.lineTo(tipX - uy * size * 0.55, tipY + ux * size * 0.55);
            ctx.lineTo(tipX + uy * size * 0.55, tipY - ux * size * 0.55);
        } else {
            ctx.moveTo(tipX - ux * size, tipY - uy * size);
            ctx.lineTo(tipX - uy * size * 0.55, tipY + ux * size * 0.55);
            ctx.lineTo(tipX + uy * size * 0.55, tipY - ux * size * 0.55);
        }
        ctx.closePath();
        ctx.fill();
    }

    function drawPulse(e, p, rgb, isHover) {
        var headX = e.x1 + (e.x2 - e.x1) * p;
        var headY = e.y1 + (e.y2 - e.y1) * p;
        var baseAlpha = isHover ? 0.95 : (e.malicious ? 0.85 : 0.7);

        // 拖尾 3 点，只能靠位置表达速度，不能靠闪烁
        for (var k = 3; k >= 1; k--) {
            var tp = e.outbound ? p - k * 0.035 : p + k * 0.035;
            if (tp < 0 || tp > 1) continue;
            var tx = e.x1 + (e.x2 - e.x1) * tp;
            var ty = e.y1 + (e.y2 - e.y1) * tp;
            ctx.fillStyle = 'rgba(' + rgb + ', ' + (baseAlpha * (0.30 - k * 0.07)) + ')';
            ctx.beginPath();
            ctx.arc(tx, ty, 2.6 - k * 0.4, 0, TAU);
            ctx.fill();
        }

        ctx.fillStyle = 'rgba(' + rgb + ', ' + baseAlpha + ')';
        ctx.beginPath();
        ctx.arc(headX, headY, 2.6, 0, TAU);
        ctx.fill();
    }

    function drawAlertRing(x, y, r, alpha) {
        if (alpha <= 0) return;
        ctx.strokeStyle = 'rgba(' + COLOR_BAD + ', ' + alpha + ')';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, TAU);
        ctx.stroke();
    }

    function drawNodes(ts) {
        var hover = chartDom._hover || {};
        var showAllLabels = layout.nodes.length <= 12;

        // 本机
        ctx.fillStyle = 'rgba(34, 211, 238, 0.95)';
        ctx.beginPath();
        ctx.arc(layout.center.x, layout.center.y, 5.5, 0, TAU);
        ctx.fill();
        ctx.strokeStyle = 'rgba(34, 211, 238, 0.35)';
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(layout.center.x, layout.center.y, 10, 0, TAU);
        ctx.stroke();

        for (var i = 0; i < layout.nodes.length; i++) {
            var n = layout.nodes[i];
            var isHover = hover.nodeKey === 'n:' + n.ip;
            var rgb = n.malicious ? COLOR_BAD : COLOR_OK;
            var fresh = (ts - (n.lastSeen || 0)) < FRESH_NODE_MS;

            ctx.fillStyle = 'rgba(' + rgb + ', ' + (isHover ? 1 : (fresh ? 0.88 : 0.48)) + ')';
            ctx.beginPath();
            ctx.arc(n.x, n.y, NODE_HIT_VISUAL(n, isHover, fresh), 0, TAU);
            ctx.fill();

            if (n.malicious) {
                // 恶意节点用静态双环标识，不用无限脉冲
                ctx.strokeStyle = 'rgba(' + COLOR_BAD + ', 0.55)';
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.radius + 3.5, 0, TAU);
                ctx.stroke();
            } else if (isHover) {
                ctx.strokeStyle = 'rgba(' + COLOR_OK + ', 0.6)';
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.radius + 3.5, 0, TAU);
                ctx.stroke();
            }

            if (n.alertAt) {
                var age = ts - n.alertAt;
                if (age >= 0 && age < ALERT_MS) {
                    var k = age / ALERT_MS;
                    drawAlertRing(n.x, n.y, n.radius + 4 + 20 * k, 0.45 * (1 - k));
                }
            }

            // 陈旧节点仍保留标签（记录要看得见），但节点多时只标活跃与悬停项，避免糊成一片
            if (showAllLabels || isHover || (fresh && layout.nodes.length <= 24)) drawNodeLabel(n, isHover, fresh);
        }
    }

    function NODE_HIT_VISUAL(n, isHover, fresh) {
        var base = fresh === false ? Math.max(2.2, n.radius - 1.4) : n.radius;
        return isHover ? base + 1.5 : base;
    }

    function drawNodeLabel(n, isHover, fresh) {
        var outward = n.ring === 0 ? 14 : 12;
        var lx = n.x + Math.cos(n.angle) * outward;
        var ly = n.y + Math.sin(n.angle) * outward;
        var align = Math.cos(n.angle) >= 0 ? 'left' : 'right';
        if (align === 'right') lx -= 4; else lx += 4;

        ctx.font = (isHover ? '600 ' : '') + '11px -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif';
        ctx.textAlign = align;
        ctx.textBaseline = 'middle';
        var dim = fresh === false;
        ctx.fillStyle = n.malicious
            ? 'rgba(244, 63, 94, ' + (dim ? 0.62 : 0.95) + ')'
            : (isHover ? 'rgba(230, 237, 243, 0.95)' : ('rgba(148, 163, 184, ' + (dim ? 0.55 : 0.85) + ')'));
        ctx.fillText(n.ip, lx, ly);
        ctx.textAlign = 'left';
    }

    function drawLegend(hasData) {
        var x = 14;
        var y = 14;
        ctx.font = '11px -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif';
        ctx.textBaseline = 'middle';

        ctx.fillStyle = 'rgba(' + COLOR_OK + ', 0.9)';
        ctx.beginPath();
        ctx.arc(x + 4, y, 3.5, 0, TAU);
        ctx.fill();
        ctx.fillStyle = 'rgba(148, 163, 184, 0.9)';
        ctx.textAlign = 'left';
        ctx.fillText('正常流量', x + 14, y);

        ctx.fillStyle = 'rgba(' + COLOR_BAD + ', 0.9)';
        ctx.beginPath();
        ctx.arc(x + 4, y + 16, 3.5, 0, TAU);
        ctx.fill();
        ctx.fillStyle = 'rgba(148, 163, 184, 0.9)';
        ctx.fillText('恶意流量', x + 14, y + 16);

        if (hasData && layout) {
            ctx.fillStyle = 'rgba(100, 116, 139, 0.95)';
            ctx.fillText('会话 ' + layout.edges.length + ' · 节点 ' + layout.nodes.length, x + 4, y + 34);
        }
    }

    function drawEmptyState() {
        ctx.save();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = 'rgba(34, 211, 238, 0.45)';
        ctx.beginPath();
        ctx.arc(cw / 2, ch / 2 - 10, 4, 0, TAU);
        ctx.fill();
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
        ctx.lineWidth = 1;
        for (var ring = 1; ring <= 3; ring++) {
            ctx.beginPath();
            ctx.arc(cw / 2, ch / 2 - 10, 22 * ring, 0, TAU);
            ctx.stroke();
        }
        ctx.font = '12px -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif';
        ctx.fillStyle = 'rgba(148, 163, 184, 0.7)';
        ctx.fillText('等待实时流量…', cw / 2, ch / 2 + 78);
        ctx.restore();
    }

    /* --------------------------- 列表 --------------------------- */
    function renderFlowList(flows) {
        var listDom = document.getElementById('liveFlowList');
        if (!listDom) return;
        var list = Array.isArray(flows) ? flows : [];
        if (!list.length) {
            listDom.innerHTML = '<p class="text-sm text-gray-400 px-2 py-1">暂无实时流量</p>';
            return;
        }
        var recent = list.slice().reverse().slice(0, MAX_FLOW_LIST_ROWS);
        var html = '';
        for (var i = 0; i < recent.length; i++) {
            var f = normalizeFlow(recent[i]);
            if (!f) continue;
            var time = new Date(f.capturedAt * 1000).toLocaleTimeString('zh-CN');
            var cls = f.malicious ? 'text-cyber-danger' : 'text-gray-300';
            var sport = f.sport === '' ? '' : ':' + f.sport;
            var dport = f.dport === '' ? '' : ':' + f.dport;
            html += '<div class="px-2 py-1 text-xs ' + cls + ' border-b border-cyber-border/20">'
                + time + '  ' + escapeHtml(f.src + sport) + ' → ' + escapeHtml(f.dst + dport)
                + (f.malicious ? '  <span class="font-bold">Malware</span>' : '')
                + '</div>';
        }
        if (list.length > MAX_FLOW_LIST_ROWS) {
            html += '<div class="px-2 py-1 text-xs text-gray-500">仅显示最近 ' + MAX_FLOW_LIST_ROWS + ' / ' + list.length + ' 条</div>';
        }
        listDom.innerHTML = html;
    }

    function escapeHtml(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    /* --------------------------- 对外主入口 --------------------------- */
    function renderLiveFlowGraph_impl(flows) {
        if (!ensureCanvas()) return;
        ingest(flows);
        var list = Array.isArray(flows) ? flows : [];
        if (!list.length) {
            // 显式传入空数组＝清空视图（但保留 TTL 语义外的既有数据）
            trackedFlows = [];
            renderFlowList([]);
        }
        // 数据可能已过期，标记一次让下一帧重建快照
        layoutDirty = true;
    }

    function restoreFromRecords(records) {
        if (!Array.isArray(records)) return;
        var flows = [];
        for (var i = 0; i < records.length; i++) {
            var r = records[i];
            if (!r) continue;
            if (Array.isArray(r.flows)) flows = flows.concat(r.flows);
            else if (r.src || r.src_ip) flows.push(r);
        }
        if (!flows.length) return;
        renderLiveFlowGraph_impl(flows);
    }

    function clearLiveLogs() {
        window.liveLogsClearedAt = Date.now() / 1000;
        try { localStorage.setItem('liveLogsClearedAt', String(window.liveLogsClearedAt)); } catch (e) {}

        window._lastLiveFlows = [];
        trackedFlows = [];
        var k;
        for (k in nodeStore) if (Object.prototype.hasOwnProperty.call(nodeStore, k)) delete nodeStore[k];
        for (k in edgeStore) if (Object.prototype.hasOwnProperty.call(edgeStore, k)) delete edgeStore[k];
        for (k in animNodes) if (Object.prototype.hasOwnProperty.call(animNodes, k)) delete animNodes[k];
        layout = null;
        layoutDirty = true;

        if (window.recordStores && window.recordStores.live) window.recordStores.live = [];
        if (window._liveSeen && typeof window._liveSeen.clear === 'function') window._liveSeen.clear();
        try { if (window.resultsImpl && window.resultsImpl.persistRecordStores) window.resultsImpl.persistRecordStores(); } catch (e) {}
        try { if (window.liveTrendImpl && window.liveTrendImpl.resetTrendChart) window.liveTrendImpl.resetTrendChart(); } catch (e) {}
        renderFlowList([]);
        try { if (typeof window.renderRecordedRows === 'function') window.renderRecordedRows('live'); } catch (e) {}
    }

    /* --------------------------- 视图状态与交互 --------------------------- */
    var state = window.liveGraphState = window.liveGraphState || {
        panX: 0, panY: 0, scale: 1, isPanning: false, lastX: 0, lastY: 0
    };

    function zoomBy(factor, anchorX, anchorY) {
        var prev = state.scale || 1;
        var next = Math.min(4, Math.max(0.5, prev * factor));
        if (next === prev) return;
        if (anchorX !== undefined && anchorY !== undefined) {
            state.panX = anchorX - (anchorX - state.panX) * (next / prev);
            state.panY = anchorY - (anchorY - state.panY) * (next / prev);
        }
        state.scale = next;
    }

    function screenToWorld(mx, my) {
        var s = state.scale || 1;
        return { x: (mx - state.panX) / s, y: (my - state.panY) / s };
    }

    function bindInteractions() {
        if (!canvas || canvas.dataset.panHandlersBound === '1') return;
        canvas.dataset.panHandlersBound = '1';
        canvas.style.cursor = 'grab';

        canvas.addEventListener('mousedown', function (ev) {
            state.isPanning = true;
            state.lastX = ev.clientX;
            state.lastY = ev.clientY;
            canvas.style.cursor = 'grabbing';
            hideTooltip();
        });

        window.addEventListener('mousemove', function (ev) {
            if (state.isPanning) {
                state.panX += ev.clientX - state.lastX;
                state.panY += ev.clientY - state.lastY;
                state.lastX = ev.clientX;
                state.lastY = ev.clientY;
            }
        });

        window.addEventListener('mouseup', function () {
            if (!state.isPanning) return;
            state.isPanning = false;
            if (canvas) canvas.style.cursor = 'grab';
        });

        canvas.addEventListener('wheel', function (ev) {
            ev.preventDefault();
            var rect = canvas.getBoundingClientRect();
            zoomBy(1 - ev.deltaY * 0.0015, ev.clientX - rect.left, ev.clientY - rect.top);
        }, { passive: false });

        canvas.addEventListener('mousemove', function (ev) {
            var rect = canvas.getBoundingClientRect();
            updateHover(ev.clientX - rect.left, ev.clientY - rect.top);
        });

        canvas.addEventListener('mouseleave', function () {
            hideTooltip();
        });
    }

    function ensureTooltip() {
        if (chartDom._lgTooltip) return chartDom._lgTooltip;
        var tip = document.getElementById('liveFlowTooltip');
        if (!tip) {
            tip = document.createElement('div');
            tip.className = 'live-flow-tooltip hidden';
            (chartDom.parentNode || document.body).appendChild(tip);
        }
        tip.style.position = 'absolute';
        tip.style.whiteSpace = 'pre-line';
        chartDom._lgTooltip = tip;
        return tip;
    }

    function hideTooltip() {
        chartDom && (chartDom._hover = null);
        if (chartDom && chartDom._lgTooltip) chartDom._lgTooltip.style.display = 'none';
    }

    function updateHover(mx, my) {
        if (!layout) return;
        var w = screenToWorld(mx, my);

        var bestNodeKey = null;
        var bestNodeD = 99999;
        for (var i = 0; i < layout.nodes.length; i++) {
            var n = layout.nodes[i];
            var d = Math.sqrt((n.x - w.x) * (n.x - w.x) + (n.y - w.y) * (n.y - w.y));
            if (d < bestNodeD) { bestNodeD = d; bestNodeKey = 'n:' + n.ip; }
        }

        var bestEdge = null;
        var bestEdgeD = 99999;
        for (var j = 0; j < layout.edges.length; j++) {
            var e = layout.edges[j];
            var ex = e.x2 - e.x1;
            var ey = e.y2 - e.y1;
            var len2 = ex * ex + ey * ey || 1;
            var t = ((w.x - e.x1) * ex + (w.y - e.y1) * ey) / len2;
            if (t < 0) t = 0; else if (t > 1) t = 1;
            var px = e.x1 + ex * t;
            var py = e.y1 + ey * t;
            var ed = Math.sqrt((px - w.x) * (px - w.x) + (py - w.y) * (py - w.y));
            if (ed < bestEdgeD) { bestEdgeD = ed; bestEdge = e; }
        }

        var tip = ensureTooltip();

        if (bestEdge && bestEdgeD <= HIT_EDGE) {
            var sample = bestEdge.sample || {};
            var conf = parseFloat(bestEdge.confidence);
            var confText = isFinite(conf) ? (conf * 100).toFixed(1) + '%' : '-';
            var dir = bestEdge.outbound ? '出站' : '入站';
            tip.innerText = [
                bestEdge.src + ':' + bestEdge.sport + ' → ' + bestEdge.dst + ':' + bestEdge.dport,
                '方向 ' + dir + ' · 协议 ' + protoToName(bestEdge.protocol),
                '会话 ' + bestEdge.count + ' 次 · ' + formatBytes(bestEdge.bytes),
                '判定 ' + (bestEdge.malicious ? '恶意' : '正常') + (bestEdge.encrypted ? ' · 加密' : ''),
                '恶意置信 ' + confText
            ].join('\n');
            positionTooltip(tip, mx, my);
            chartDom._hover = { edgeKey: bestEdge.key, nodeKey: null };
            return;
        }

        if (bestNodeKey && bestNodeD <= HIT_NODE) {
            var ip = bestNodeKey.slice(2);
            var nd = layout.nodeByIp[ip] || {};
            tip.innerText = [
                ip,
                nd.local ? '本地网络' : '公网对端',
                '会话 ' + (nd.count || 0) + ' 次 · ' + formatBytes(nd.bytes || 0),
                nd.malicious ? '涉及恶意流量' : '仅正常流量'
            ].join('\n');
            positionTooltip(tip, mx, my);
            chartDom._hover = { nodeKey: bestNodeKey, edgeKey: null };
            return;
        }

        hideTooltip();
    }

    function positionTooltip(tip, mx, my) {
        tip.classList.remove('hidden');
        tip.style.display = 'block';
        tip.style.left = (mx + 14) + 'px';
        tip.style.top = (my + 14) + 'px';
    }

    /* --------------------------- 帧循环 --------------------------- */
    // 每一代循环带一个自己的编号。stopLoop 只把编号 +1，在跑的旧循环下一帧就会看到
    // 编号对不上而自行退出 —— 不需要保存 requestAnimationFrame 的句柄。
    var loopGeneration = 0;

    function startLoop() {
        if (window.liveGraph && window.liveGraph._animLoopBound) return;
        if (window.liveGraph) window.liveGraph._animLoopBound = true;
        var generation = ++loopGeneration;

        function loop(ts) {
            if (generation !== loopGeneration) return;
            if (!document.hidden) {
                drawFrame(ts);
                fpsFrames++;
                if (!fpsWindowStart) fpsWindowStart = ts;
                if (ts - fpsWindowStart >= 1000) {
                    fps = fpsFrames * 1000 / (ts - fpsWindowStart);
                    fpsFrames = 0;
                    fpsWindowStart = ts;
                }
            }
            requestAnimationFrame(loop);
        }
        requestAnimationFrame(loop);
    }

    // 无刷新换页时调用：内容区被替换后旧 canvas 已脱离文档，继续逐帧画是纯烧 CPU；
    // 而且 _animLoopBound 会一直停在 true，重进实时页时 startLoop 直接返回 —— 新 canvas
    // 永远不会被绘制（表现就是辐射图卡在「等待实时流量…」）。
    function stopLoop() {
        loopGeneration++;
        if (window.liveGraph) window.liveGraph._animLoopBound = false;
    }

    /* --------------------------- 工具栏 --------------------------- */
    function bindToolbarButtons() {
        function bind(id, fn) {
            var el = document.getElementById(id);
            if (!el || el._boundClick) return;
            el._boundClick = true;
            el.addEventListener('click', fn);
        }
        bind('clearLiveLogsBtn', function () {
            if (window.liveImpl && typeof window.liveImpl.clearLiveLogs === 'function') {
                window.liveImpl.clearLiveLogs();
            } else {
                clearLiveLogs();
            }
        });
        bind('liveFlowZoomOut', function () { zoomBy(1 / 1.2); });
        bind('liveFlowZoomReset', function () {
            state.scale = 1;
            state.panX = 0;
            state.panY = 0;
        });
        bind('liveFlowZoomIn', function () { zoomBy(1.2); });
    }

    /* --------------------------- 初始化 --------------------------- */
    function init() {
        // 换页后 canvas 是新的：必须先把旧引用清掉，否则 ensureCanvas 会拿到已经脱离文档的
        // 那个 canvas，尺寸读成 0，图还是画在旧对象上。
        chartDom = null;
        canvas = null;
        ctx = null;
        if (!ensureCanvas()) return;
        bindInteractions();
        bindToolbarButtons();
        startLoop();
        drawFrame(performance.now ? performance.now() : Date.now());
    }

    window.liveGraph = window.liveGraph || {};
    window.liveGraph.init = init;
    window.liveGraph.stopLoop = stopLoop;
    window.liveGraph.renderLiveFlowGraph = renderLiveFlowGraph_impl;
    window.liveGraph.restoreFromRecords = restoreFromRecords;
    window.liveGraph.clearLiveLogs = clearLiveLogs;
    window.liveGraph.renderFlowList = renderFlowList;
    window.liveGraph.zoomIn = function () { zoomBy(1.2); };
    window.liveGraph.zoomOut = function () { zoomBy(1 / 1.2); };
    window.liveGraph.resetView = function () { state.scale = 1; state.panX = 0; state.panY = 0; };
    window.liveGraph.getFps = function () { return Math.round(fps); };
    window.liveGraph._nodeStore = nodeStore;
    window.liveGraph._edgeStore = edgeStore;
    window.liveGraph._animNodes = animNodes;
    window.liveGraph._animLoopBound = false;
    // 兼容旧调用名
    window.renderLiveFlowGraph = renderLiveFlowGraph_impl;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
