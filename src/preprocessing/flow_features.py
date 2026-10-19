"""pcap -> CICFlowMeter 78 维流统计特征。

为什么需要这个文件：项目里有两条完全不同的检测链路。

* pcap 链路（``/analyze``）：28x28 字节图 -> 6 个二分类模型，只会说「正常/恶意」。
* CSV 链路（``/analyze-flow``）：CICFlowMeter 的 78 维流统计 -> 单个六分类模型，
  能说出 Benign / Bot / DDoS-HOIC / DoS-GoldenEye / DoS-Slowloris / DDoS-LOIC-UDP。
  ``checkpoints/flow_multiclass.pth`` 训练时用的 feature_names 就是这 78 列。

「威胁类型分布」卡片要做成真正的六分类，就必须让 pcap 也能喂进那个六分类模型，
而中间缺的正是这一层：从原始包算出与训练 CSV 同口径的 78 维。所以这里按
CICFlowMeter 的定义自己实现一遍统计，不引入 tsflow / cicflowmeter 这类外部依赖
（它们需要额外的流量分析器，requirements 里只有 scapy）。

单位口径（与 CICFrame 训练 CSV 一致，错一位模型就废）：
* Flow Duration、各 IAT、Active/Idle：**微秒**。
* 包长/窗口/头长：字节。

有 10 列在训练集里恒为 0（Bulk 系列 6 列、CWE Flag Count、Fwd/Bwd URG Flags、
Bwd PSH Flags），标准化参数是 mean=0/std=1，所以这里直接填 0 —— 与训练分布
完全一致，不需要实现 CICFlowMeter 里那套复杂的 bulk 切分逻辑。
"""

import os

import numpy as np

# ---- CICFlowMeter 的切分常量 ----
# 流空闲超过这个时长就切断，另起一条流（CICFlowMeter 默认 120 秒）。
FLOW_TIMEOUT_US = 120 * 1_000_000
# 相邻包间隔超过这个值就认为进入了 idle 段（Active/Idle 统计的边界）。
ACTIVE_TIMEOUT_US = 5 * 1_000_000
# 同向相邻包超过这个间隔就另起一条子流（Subflow 统计的边界）。
SUBFLOW_TIMEOUT_US = 1 * 1_000_000

FEATURE_NAMES = [
    'ACK Flag Cnt', 'Active Max', 'Active Mean', 'Active Min', 'Active Std',
    'Bwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Header Len', 'Bwd IAT Max',
    'Bwd IAT Mean', 'Bwd IAT Min', 'Bwd IAT Std', 'Bwd IAT Tot',
    'Bwd PSH Flags', 'Bwd Pkt Len Max', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Min',
    'Bwd Pkt Len Std', 'Bwd Pkts/b Avg', 'Bwd Pkts/s', 'Bwd Seg Size Avg',
    'Bwd URG Flags', 'CWE Flag Count', 'Down/Up Ratio', 'Dst Port',
    'ECE Flag Cnt', 'FIN Flag Cnt', 'Flow Byts/s', 'Flow Duration',
    'Flow IAT Max', 'Flow IAT Mean', 'Flow IAT Min', 'Flow IAT Std',
    'Flow Pkts/s', 'Fwd Act Data Pkts', 'Fwd Blk Rate Avg', 'Fwd Byts/b Avg',
    'Fwd Header Len', 'Fwd IAT Max', 'Fwd IAT Mean', 'Fwd IAT Min',
    'Fwd IAT Std', 'Fwd IAT Tot', 'Fwd PSH Flags', 'Fwd Pkt Len Max',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Min', 'Fwd Pkt Len Std', 'Fwd Pkts/b Avg',
    'Fwd Pkts/s', 'Fwd Seg Size Avg', 'Fwd Seg Size Min', 'Fwd URG Flags',
    'Idle Max', 'Idle Mean', 'Idle Min', 'Idle Std', 'Init Bwd Win Byts',
    'Init Fwd Win Byts', 'PSH Flag Cnt', 'Pkt Len Max', 'Pkt Len Mean',
    'Pkt Len Min', 'Pkt Len Std', 'Pkt Len Var', 'Pkt Size Avg', 'Protocol',
    'RST Flag Cnt', 'SYN Flag Cnt', 'Subflow Bwd Byts', 'Subflow Bwd Pkts',
    'Subflow Fwd Byts', 'Subflow Fwd Pkts', 'Tot Bwd Pkts', 'Tot Fwd Pkts',
    'TotLen Bwd Pkts', 'TotLen Fwd Pkts', 'URG Flag Cnt',
]

# 训练集里恒为 0 的列：标准化是 mean=0 / std=1，填 0 即等于训练分布。
ALWAYS_ZERO_FEATURES = {
    'Bwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd URG Flags',
    'Bwd PSH Flags', 'CWE Flag Count', 'Fwd Blk Rate Avg', 'Fwd Byts/b Avg',
    'Fwd Pkts/b Avg', 'Fwd URG Flags',
}

# ---- TCP 标志位（IPv4 头里 TCP 的第 13 字节）----
_FIN, _SYN, _RST, _PSH, _ACK, _URG, _ECE, _CWR = (1, 2, 4, 8, 16, 32, 64, 128)


def _stat(values, want_min=True):
    """返回 (min, mean, max, std)；空输入一律 0。"""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    a = np.asarray(values, dtype=np.float64)
    return (float(a.min()) if want_min else 0.0,
            float(a.mean()),
            float(a.max()),
            float(a.std()))


def _iat_stats(timestamps_us):
    """相邻包间隔的 (min, mean, max, std, tot)，单位微秒。"""
    if len(timestamps_us) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    diffs = np.diff(np.asarray(timestamps_us, dtype=np.float64))
    return (float(diffs.min()), float(diffs.mean()), float(diffs.max()),
            float(diffs.std()), float(diffs.sum()))


def _active_idle(timestamps_us):
    """按 ACTIVE_TIMEOUT 把流切成 active / idle 段，各返回 (min, mean, max, std)。

    CICFlowMeter 的语义：相邻包间隔 > 5s 就认为前一活跃段结束、随后是一段 idle；
    每段 idle 时长 = 相邻两包的间隔本身。
    """
    active, idle = [], []
    if len(timestamps_us) < 2:
        return (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)

    ts = np.asarray(timestamps_us, dtype=np.float64)
    seg_start = ts[0]
    prev = ts[0]
    for cur in ts[1:]:
        gap = cur - prev
        if gap > ACTIVE_TIMEOUT_US:
            active.append(prev - seg_start)
            idle.append(gap)
            seg_start = cur
        prev = cur
    active.append(prev - seg_start)

    if not active:
        active = [0.0]
    if not idle:
        idle = [0.0]
    a = np.asarray(active, dtype=np.float64)
    i = np.asarray(idle, dtype=np.float64)
    return ((float(a.min()), float(a.mean()), float(a.max()), float(a.std())),
            (float(i.min()), float(i.mean()), float(i.max()), float(i.std())))


def _subflow(packets):
    """同向相邻包间隔 > SUBFLOW_TIMEOUT 时另起一条子流，返回累计 (pkts, bytes)。

    这里按 CICFlowMeter 的用法取「最长的一条子流」，因为训练 CSV 里 Subflow
    系列是每条流一个值而不是列表；取最长子流比求和更接近原始实现。
    """
    if not packets:
        return 0, 0
    best_pkts = cur_pkts = 1
    best_bytes = cur_bytes = packets[0][1]
    for i in range(1, len(packets)):
        ts, length = packets[i][0], packets[i][1]
        if ts - packets[i - 1][0] > SUBFLOW_TIMEOUT_US:
            cur_pkts, cur_bytes = 1, length
        else:
            cur_pkts += 1
            cur_bytes += length
        if cur_pkts > best_pkts:
            best_pkts, best_bytes = cur_pkts, cur_bytes
    return best_pkts, best_bytes


class _Flow:
    """一条双向流（五元组）的累加器。"""

    __slots__ = ('key', 'fwd', 'bwd', 'last_ts', 'start_ts', 'flags',
                 'init_fwd_win', 'init_bwd_win', 'fwd_hdr', 'bwd_hdr',
                 'fwd_act_data')

    def __init__(self, key, ts):
        self.key = key
        self.start_ts = ts
        self.last_ts = ts
        # 每项：(ts_us, payload_len, header_len, flags, win, payload_bytes)
        self.fwd = []
        self.bwd = []
        self.flags = {'FIN': 0, 'SYN': 0, 'RST': 0, 'PSH': 0, 'ACK': 0,
                      'URG': 0, 'ECE': 0, 'CWR': 0}
        self.init_fwd_win = 0
        self.init_bwd_win = 0
        self.fwd_hdr = 0
        self.bwd_hdr = 0
        self.fwd_act_data = 0


def _parse_packet(pkt_data):
    """从以太网帧里抠出 (五元组, ts 无关的元信息)。解析不了返回 None。"""
    if len(pkt_data) < 34:
        return None

    eth_type = (pkt_data[12] << 8) | pkt_data[13]
    offset = 14
    if eth_type == 0x8100:          # 802.1Q VLAN
        if len(pkt_data) < 38:
            return None
        eth_type = (pkt_data[16] << 8) | pkt_data[17]
        offset = 18
    if eth_type != 0x0800:          # 只处理 IPv4
        return None
    if offset + 20 > len(pkt_data):
        return None

    ip_hl = (pkt_data[offset] & 0x0F) * 4
    if ip_hl < 20 or offset + ip_hl > len(pkt_data):
        return None
    proto = pkt_data[offset + 9]
    total_len = (pkt_data[offset + 2] << 8) | pkt_data[offset + 3]

    src_ip = '%d.%d.%d.%d' % tuple(pkt_data[offset + 12:offset + 16])
    dst_ip = '%d.%d.%d.%d' % tuple(pkt_data[offset + 16:offset + 20])

    sport = dport = 0
    flags = 0
    win = 0
    trans_offset = offset + ip_hl
    payload_len = 0
    header_len = ip_hl

    if proto in (6, 17) and len(pkt_data) >= trans_offset + 4:
        sport = (pkt_data[trans_offset] << 8) | pkt_data[trans_offset + 1]
        dport = (pkt_data[trans_offset + 2] << 8) | pkt_data[trans_offset + 3]

    if proto == 6 and len(pkt_data) >= trans_offset + 14:
        tcp_hl = (pkt_data[trans_offset + 12] >> 4) * 4
        if tcp_hl < 20:
            tcp_hl = 20
        flags = pkt_data[trans_offset + 13]
        win = (pkt_data[trans_offset + 14] << 8) | pkt_data[trans_offset + 15]
        header_len = ip_hl + tcp_hl
        payload_len = max(0, total_len - header_len)
    elif proto == 17 and len(pkt_data) >= trans_offset + 8:
        header_len = ip_hl + 8
        payload_len = max(0, total_len - header_len)
    else:
        payload_len = max(0, total_len - ip_hl)

    return {
        'src_ip': src_ip, 'dst_ip': dst_ip, 'sport': sport, 'dport': dport,
        'proto': proto, 'flags': flags, 'win': win,
        'payload_len': payload_len, 'header_len': header_len,
    }


def _flow_key(info):
    """双向五元组：IP 字典序小的一侧固定为「前向」，保证正反向归同一条流。"""
    if info['src_ip'] <= info['dst_ip']:
        return (info['src_ip'], info['sport'], info['dst_ip'], info['dport'], info['proto']), True
    return (info['dst_ip'], info['dport'], info['src_ip'], info['sport'], info['proto']), False


def _packet_timestamp(pkt_meta):
    """取包时间戳（秒，float）。

    两种格式都要认：
    * pcap：``PacketMetadata(sec, usec)``。
    * pcapng：``PacketMetadataNg(tshigh, tslow, tsresol)``，64 位时间戳按
      ``(tshigh<<32 | tslow) / tsresol`` 还原。不处理这一支会让整条 pcapng
      的流时长恒为 0，速率类特征全被压成 0（实测 ics_b_00021_ABB1.pcap）。

    实现统一在 ``feature_extraction.packet_timestamp`` 里，这里只做转发：
    同一条流在二分类与六分类两条链路上必须用同一套时间戳换算，否则空闲超时切分
    会把同一条流切成不同段数，两张卡片的基数天然对不上。
    """
    from .feature_extraction import packet_timestamp

    return packet_timestamp(pkt_meta)


def pcap_to_flow_features(pcap_file, max_flows=5000, max_packets=200_000, diagnostics=None):
    """读取 pcap，返回 [(flow_key, {78 维特征})]，按包数降序、最多 max_flows 条。

    只做统计、不做推理；解析不了的包静默跳过（与 FeatureExtractor 的策略一致，
    大 pcap 里混杂的非 IPv4 帧不该让整次分析失败）。

    参数 ``diagnostics``（可选 dict）：传进来就填入 ``capture_format`` /
    ``parsed_packets`` / ``error``。原先读取失败只 print 一行，接口层拿到的是一份
    「空但成功」的流列表——前端只能显示 0 条攻击流，看不出解析其实失败了。
    """
    from .feature_extraction import detect_capture_format, open_capture_reader

    flows = {}
    order = []
    parsed = 0
    capture_format = detect_capture_format(pcap_file)
    if diagnostics is not None:
        diagnostics['capture_format'] = capture_format

    try:
        # 代码审查：pcapng 走 PcapNgReader。原先无条件用 RawPcapReader，
        # 它读 pcapng 不报错而是产出假包，六分类特征因此整体失真。
        with open_capture_reader(pcap_file) as reader:
            for pkt_data, pkt_ts in reader:
                if parsed >= max_packets:
                    break
                info = _parse_packet(pkt_data)
                if info is None:
                    continue
                parsed += 1

                ts = pkt_ts if pkt_ts else 0.0
                ts_us = ts * 1e6

                key, is_fwd = _flow_key(info)
                flow = flows.get(key)
                # 空闲超时：这条流已经断掉，先把旧的留在结果里，再起一条新的。
                if flow is not None and ts_us - flow.last_ts > FLOW_TIMEOUT_US:
                    order.append(flow)
                    flow = None
                if flow is None:
                    flow = _Flow(key, ts_us)
                    flows[key] = flow

                _accumulate(flow, info, ts_us, is_fwd)
    except Exception as e:  # noqa: BLE001 - 损坏的 pcap 不该抛到接口层
        # 不吞异常：print 给人看，diagnostics 给接口层看（响应里带 flow_parse_error）
        print('pcap_to_flow_features failed:', e)
        if diagnostics is not None:
            diagnostics['error'] = f'{type(e).__name__}: {e}'

    order.extend(f for f in flows.values() if f not in order)

    results = []
    for flow in order:
        feats = _compute_features(flow)
        if feats is not None:
            results.append((flow.key, feats))

    if diagnostics is not None:
        diagnostics['parsed_packets'] = parsed
        diagnostics['flow_count'] = len(results)

    results.sort(key=lambda kv: kv[1]['Tot Fwd Pkts'] + kv[1]['Tot Bwd Pkts'],
                 reverse=True)
    return results[:max_flows]


def _compute_features(flow):
    """把一条流的累加状态换算成 78 维。"""
    fwd, bwd = flow.fwd, flow.bwd
    if not fwd and not bwd:
        return None

    all_pkts = sorted(fwd + bwd, key=lambda e: e[0])
    ts_all = [e[0] for e in all_pkts]
    payload_all = [e[1] for e in all_pkts]

    duration_us = max(0.0, ts_all[-1] - ts_all[0])
    duration_s = duration_us / 1e6

    tot_fwd = len(fwd)
    tot_bwd = len(bwd)
    tot_pkts = tot_fwd + tot_bwd
    totlen_fwd = sum(e[1] for e in fwd)
    totlen_bwd = sum(e[1] for e in bwd)
    totlen = totlen_fwd + totlen_bwd

    fwd_payload = [e[1] for e in fwd]
    bwd_payload = [e[1] for e in bwd]

    # 速率类统一用 duration_s；duration 为 0（单包流）时按 0 处理，避免除零放大成 inf。
    def rate(total):
        return float(total / duration_s) if duration_s > 0 else 0.0

    fl_min, fl_mean, fl_max, fl_std, fl_tot = _iat_stats(ts_all)
    fw_min, fw_mean, fw_max, fw_std, fw_tot = _iat_stats([e[0] for e in fwd])
    bw_min, bw_mean, bw_max, bw_std, bw_tot = _iat_stats([e[0] for e in bwd])

    act, idle = _active_idle(ts_all)
    sf_fwd_pkts, sf_fwd_bytes = _subflow([(e[0], e[1]) for e in fwd])
    sf_bwd_pkts, sf_bwd_bytes = _subflow([(e[0], e[1]) for e in bwd])

    p_min, p_mean, p_max, p_std = _stat(payload_all, want_min=True)
    p_var = float(np.var(np.asarray(payload_all, dtype=np.float64))) if payload_all else 0.0
    f_min, f_mean, f_max, f_std = _stat(fwd_payload, want_min=False)
    b_min, b_mean, b_max, b_std = _stat(bwd_payload, want_min=False)

    dst_port = flow.key[3]
    proto = flow.key[4]

    feats = {
        'Flow Duration': duration_us,
        'Tot Fwd Pkts': tot_fwd,
        'Tot Bwd Pkts': tot_bwd,
        'TotLen Fwd Pkts': totlen_fwd,
        'TotLen Bwd Pkts': totlen_bwd,
        'Fwd Pkt Len Max': f_max,
        'Fwd Pkt Len Mean': f_mean,
        'Fwd Pkt Len Min': f_min,
        'Fwd Pkt Len Std': f_std,
        'Bwd Pkt Len Max': b_max,
        'Bwd Pkt Len Mean': b_mean,
        'Bwd Pkt Len Min': b_min,
        'Bwd Pkt Len Std': b_std,
        'Pkt Len Max': p_max,
        'Pkt Len Mean': p_mean,
        'Pkt Len Min': p_min,
        'Pkt Len Std': p_std,
        'Pkt Len Var': p_var,
        'Pkt Size Avg': float(totlen / tot_pkts) if tot_pkts else 0.0,
        'Flow Byts/s': rate(totlen),
        'Flow Pkts/s': rate(tot_pkts),
        'Fwd Pkts/s': rate(tot_fwd),
        'Bwd Pkts/s': rate(tot_bwd),
        'Flow IAT Min': fl_min,
        'Flow IAT Mean': fl_mean,
        'Flow IAT Max': fl_max,
        'Flow IAT Std': fl_std,
        'Fwd IAT Tot': fw_tot,
        'Fwd IAT Min': fw_min,
        'Fwd IAT Mean': fw_mean,
        'Fwd IAT Max': fw_max,
        'Fwd IAT Std': fw_std,
        'Bwd IAT Tot': bw_tot,
        'Bwd IAT Min': bw_min,
        'Bwd IAT Mean': bw_mean,
        'Bwd IAT Max': bw_max,
        'Bwd IAT Std': bw_std,
        'FIN Flag Cnt': flow.flags['FIN'],
        'SYN Flag Cnt': flow.flags['SYN'],
        'RST Flag Cnt': flow.flags['RST'],
        'PSH Flag Cnt': flow.flags['PSH'],
        'ACK Flag Cnt': flow.flags['ACK'],
        'URG Flag Cnt': flow.flags['URG'],
        'ECE Flag Cnt': flow.flags['ECE'],
        'Fwd PSH Flags': sum(1 for e in fwd if e[3] & _PSH),
        'Down/Up Ratio': (tot_bwd / tot_fwd) if tot_fwd else 0.0,
        'Fwd Header Len': flow.fwd_hdr,
        'Bwd Header Len': flow.bwd_hdr,
        'Fwd Seg Size Avg': f_mean,
        'Fwd Seg Size Min': (min(fwd_payload) if fwd_payload else 0),
        'Bwd Seg Size Avg': b_mean,
        'Fwd Act Data Pkts': flow.fwd_act_data,
        'Init Fwd Win Byts': flow.init_fwd_win,
        'Init Bwd Win Byts': flow.init_bwd_win,
        'Active Min': act[0], 'Active Mean': act[1],
        'Active Max': act[2], 'Active Std': act[3],
        'Idle Min': idle[0], 'Idle Mean': idle[1],
        'Idle Max': idle[2], 'Idle Std': idle[3],
        'Subflow Fwd Pkts': sf_fwd_pkts,
        'Subflow Fwd Byts': sf_fwd_bytes,
        'Subflow Bwd Pkts': sf_bwd_pkts,
        'Subflow Bwd Byts': sf_bwd_bytes,
        'Dst Port': dst_port,
        'Protocol': proto,
    }

    for name in ALWAYS_ZERO_FEATURES:
        feats[name] = 0

    # 严格按训练时的列顺序拼出向量；缺列说明统计漏了，宁可返回 None 也不许错位。
    try:
        vector = [float(feats[name]) for name in FEATURE_NAMES]
    except KeyError as e:
        print('flow feature missing column:', e)
        return None

    ordered = {name: float(feats[name]) for name in FEATURE_NAMES}
    ordered['__vector__'] = vector
    return ordered


def _accumulate(flow, info, ts_us, is_fwd):
    """把一个已解析的包计入流的累加器。

    pcap 离线扫描（pcap_to_flow_features）和实时抓包（LiveFlowTracker）共用这一段，
    保证两条路径算出来的 78 维口径完全一致——否则实时页和上传页对同一段流量会给出
    不同的类别，读的人只会以为模型不稳定。
    """
    flow.last_ts = ts_us
    flags = info['flags']
    mask = flow.flags
    if flags & _FIN: mask['FIN'] += 1
    if flags & _SYN: mask['SYN'] += 1
    if flags & _RST: mask['RST'] += 1
    if flags & _PSH: mask['PSH'] += 1
    if flags & _ACK: mask['ACK'] += 1
    if flags & _URG: mask['URG'] += 1
    if flags & _ECE: mask['ECE'] += 1
    if flags & _CWR: mask['CWR'] += 1

    entry = (ts_us, info['payload_len'], info['header_len'], flags, info['win'])
    if is_fwd:
        if not flow.fwd:
            flow.init_fwd_win = info['win']
        if info['payload_len'] > 0:
            flow.fwd_act_data += 1
        flow.fwd.append(entry)
        flow.fwd_hdr += info['header_len']
    else:
        if not flow.bwd:
            flow.init_bwd_win = info['win']
        flow.bwd.append(entry)
        flow.bwd_hdr += info['header_len']


class LiveFlowTracker:
    """实时抓包用的增量流特征表。

    为什么要有它：``/api/live/stats`` 原来只有二分类的 is_malicious，前端「威胁类型分布」
    卡片因此永远是空的。而抓包线程手里每个包都是完整的原始字节 + 时间戳，缺的只是把
    它们按五元组累加成 78 维这一步——也就是这个类。

    与 pcap 离线扫描的差别：
    * 离线是「扫完再算」，这里是「边收边算」，``snapshot()`` 随时可取当前快照；
    * 流空闲超时就重置该条流（实时只关心当下，不保留历史分段）；
    * ``version`` 每次收包递增，轮询接口据此判断要不要重算，避免 1.5 秒一次的无谓推理。
    """

    __slots__ = ('_flows', '_idle_us', 'version')

    def __init__(self, idle_timeout_us=FLOW_TIMEOUT_US):
        self._flows = {}
        self._idle_us = idle_timeout_us
        self.version = 0

    def reset(self):
        self._flows.clear()
        self.version += 1

    def add_packet(self, pkt_bytes, ts_seconds):
        """喂一个包。解析不了（非 IPv4/截断）时返回 None。"""
        info = _parse_packet(pkt_bytes)
        if info is None:
            return None
        try:
            ts_us = float(ts_seconds) * 1e6
        except (TypeError, ValueError):
            return None

        key, is_fwd = _flow_key(info)
        flow = self._flows.get(key)
        if flow is not None and ts_us - flow.last_ts > self._idle_us:
            flow = None        # 空闲超时：断开重来
        if flow is None:
            flow = _Flow(key, ts_us)
            self._flows[key] = flow

        _accumulate(flow, info, ts_us, is_fwd)
        self.version += 1
        return key

    def snapshot(self, active_only=True, with_ts=False):
        """返回 [(key, 78 维字典)]。

        active_only=True：只保留当前仍活跃（距最新包未超时）的流，抓包进行中用这个。
        active_only=False：把表里所有流都算进去。抓包停止后要用这个——否则停止超过
        两分钟再打开页面，所有流都被判成"已超时"，卡片会莫名其妙变回空。

        with_ts=True：每项变成 (key, 字典, last_ts_秒)。调用方据此判断某条流是
        "清空记录之前"还是"之后"产生的——否则前端点了「清除日志」，1.5 秒后
        轮询把同一批流又灌回来，看上去就是清不掉。
        """
        latest = max((f.last_ts for f in self._flows.values()), default=0.0)
        results = []
        for key, flow in self._flows.items():
            if active_only and latest - flow.last_ts > self._idle_us:
                continue
            feats = _compute_features(flow)
            if feats is not None:
                if with_ts:
                    results.append((key, feats, flow.last_ts / 1e6))
                else:
                    results.append((key, feats))
        results.sort(key=lambda kv: kv[1]['Tot Fwd Pkts'] + kv[1]['Tot Bwd Pkts'],
                     reverse=True)
        return results

    def prune(self):
        """清掉已超时的流，避免长时间抓包把内存撑大。"""
        latest = max((f.last_ts for f in self._flows.values()), default=0.0)
        stale = [k for k, f in self._flows.items() if latest - f.last_ts > self._idle_us]
        for k in stale:
            self._flows.pop(k, None)
        return len(stale)


def features_to_matrix(flow_features):
    """把 pcap_to_flow_features 的返回值整理成 (flows, X)。"""
    rows = [item[1]['__vector__'] for item in flow_features]
    if not rows:
        return [], np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return flow_features, np.asarray(rows, dtype=np.float32)

