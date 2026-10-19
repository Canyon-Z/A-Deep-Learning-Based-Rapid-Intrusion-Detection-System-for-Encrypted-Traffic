import os

import numpy as np
from tqdm import tqdm

# 代码审查第 23/41 条：抹头长度只在这里定义一处。
# MAC 头固定覆盖前 12 字节；IPv4 源/目的地址占 IP 头内偏移 12~19 共 8 字节
# （历史 docstring 写成「20 字节」，与实现不符，已按实际行为改正）。
MAC_MASK_LEN = 12
IP_MASK_LEN = 8


def detect_capture_format(pcap_file):
    """按文件头魔法判断抓包格式，返回 ``'pcap'`` / ``'pcapng'`` / ``'unknown'``。

    为什么必须显式判断：``scapy.utils.RawPcapReader`` 只认经典 pcap 的 24 字节
    全局头。喂给它一个 pcapng 文件时，它会把 section header block 当成全局头解析，
    得出错误的链路类型与字节序，然后一连串"包"被读出来——既不会抛异常，也不会
    记 ``parse_errors``，于是前端看到「解析成功、0 错误、若干假会话」。
    pcapng 必须走 ``scapy.utils.PcapNgReader``。

    pcapng 的 Section Header Block 类型固定为 0x0A0D0D0A（大端写死在规范里，
    字节序对称，所以按原始字节比较即可，不需要判断大小端）。
    """
    try:
        with open(pcap_file, 'rb') as fh:
            head = fh.read(4)
    except Exception:
        return 'unknown'
    if len(head) < 4:
        return 'unknown'
    if head == b'\x0a\x0d\x0d\x0a':
        return 'pcapng'
    # 经典 pcap：0xa1b2c3d4（微秒）/ 0xa1b23c4d（纳秒），各自都有两种字节序
    for order in ('little', 'big'):
        try:
            magic = int.from_bytes(head, order)
        except Exception:
            continue
        if magic in (0xA1B2C3D4, 0xA1B23C4D):
            return 'pcap'
    return 'unknown'


def open_capture_reader(pcap_file):
    """按格式返回统一的抓包读取器，迭代出 ``(packet_bytes, timestamp_seconds)``。

    为什么需要在 scapy 之上再包一层：两个 reader 的迭代协议不一样。
      * ``RawPcapReader.__next__`` -> ``(bytes, PacketMetadata(sec, usec))``
      * ``PcapNgReader.__next__``  -> ``Packet``（已解析，时间在 ``.time``）
    直接 ``for data, meta in reader`` 混用会在 pcapng 上抛
    ``ValueError: not enough values to unpack (expected 2, got 1)``；
    而若不换 reader 又用 RawPcapReader 硬读 pcapng，它不报错而是产出假包。
    两种失败方式都会让上层把「解析失败」误当成「解析成功、0 条流」。
    """
    return CaptureReader(pcap_file)


class CaptureReader:
    """pcap / pcapng 的统一读取器，见 ``open_capture_reader``。"""

    def __init__(self, pcap_file):
        self.path = pcap_file
        self.format = detect_capture_format(pcap_file)
        if self.format == 'pcapng':
            from scapy.utils import PcapNgReader

            self._reader = PcapNgReader(pcap_file)
            self._protocol = 'packet'
        else:
            from scapy.utils import RawPcapReader

            self._reader = RawPcapReader(pcap_file)
            self._protocol = 'raw'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        try:
            self._reader.close()
        except Exception:
            pass

    def __iter__(self):
        if self._protocol == 'packet':
            for packet in self._reader:
                try:
                    timestamp = float(packet.time)
                except Exception:
                    timestamp = 0.0
                yield bytes(packet), timestamp
        else:
            for data, meta in self._reader:
                yield data, packet_timestamp(meta)


def packet_timestamp(pkt_meta):
    """取包时间戳（秒，float），pcap 与 pcapng 两种元数据都认。

    * pcap：``PacketMetadata(sec, usec)``。
    * pcapng：``PacketMetadataNg(tshigh, tslow, tsresol)``，64 位计数按 tsresol 还原。
      不处理这一支会让整条 pcapng 的时间戳恒为 0。
    """
    if hasattr(pkt_meta, 'tshigh') and hasattr(pkt_meta, 'tslow'):
        tsresol = getattr(pkt_meta, 'tsresol', 1_000_000) or 1_000_000
        return ((int(pkt_meta.tshigh) << 32) | int(pkt_meta.tslow)) / float(tsresol)
    sec = getattr(pkt_meta, 'sec', None)
    if sec is not None:
        return float(sec) + float(getattr(pkt_meta, 'usec', 0) or 0) / 1e6
    try:
        return float(pkt_meta[0]) + float(pkt_meta[1]) / 1e6
    except Exception:
        return 0.0


class FeatureExtractor:
    """
    从 PCAP 原始文件中提取流量特征的处理流水线。

    处理流程:
    1. 读取 PCAP，按五元组 (src_ip, src_port, dst_ip, dst_port, protocol)
       拆分为双向会话
    2. 可选地抹掉 MAC/IP 头部字段（迫使模型学习应用层特征）
    3. 每个会话截取前 28 个包、每包前 28 字节，排列成 28×28 灰度图

    参数:
        mask_headers:     是否用填充值覆盖 MAC/IP 头部
        mask_fill:        填充值 (0-255)，默认 0
        max_packets:      每个会话最大包数，默认 28
        bytes_per_packet: 每包最大字节数，默认 28
        idle_timeout:     会话空闲超时（秒）。同一五元组停顿超过该时长即视为新会话。
                          默认 None = 不按时间切分（保持历史行为，已训练的 28x28 模型
                          权重与既有指标正是在这个口径下得到的）。
                          六分类那条链路（``flow_features.FLOW_TIMEOUT_US``）默认是
                          120 秒，两条链路的"流"因此不必是同一批对象——需要两边严格
                          对齐基数时，用 ``DLIDS_FLOW_IDLE_TIMEOUT`` 显式设成同一个值。
    """

    def __init__(self, mask_headers=True, mask_fill=0, max_packets=28, bytes_per_packet=28,
                 idle_timeout=None):
        self.mask_headers = mask_headers
        self.mask_fill = mask_fill & 0xFF
        self.max_packets = max_packets
        self.bytes_per_packet = bytes_per_packet
        self.idle_timeout = None if idle_timeout in (None, 0) else float(idle_timeout)

    def _mask_packet_headers(self, pkt_list, offset):
        """用填充字节替换 MAC 头部（前 12 字节）和 IPv4 源/目的地址
        （IP 头内偏移 + 12 开始的 8 字节），让模型不依赖 MAC/IP 地址做判断，
        专注于应用层特征。"""
        if not self.mask_headers:
            return pkt_list

        fill_byte = bytes([self.mask_fill])

        mac_end = min(MAC_MASK_LEN, len(pkt_list))
        if mac_end > 0:
            pkt_list[:mac_end] = fill_byte * mac_end

        ip_start = offset + MAC_MASK_LEN
        ip_end = min(ip_start + IP_MASK_LEN, len(pkt_list))
        if ip_start < ip_end:
            pkt_list[ip_start:ip_end] = fill_byte * (ip_end - ip_start)

        return pkt_list

    def pcap_to_sessions(self, pcap_file):
        """
        读取 PCAP 文件，按五元组拆分成双向会话。

        返回:
            sessions:   {五元组: (packet_bytes, ...)}
            timestamps: {五元组: 会话起始时间戳}
            skip_stats: 各阶段跳过的包统计
        """
        sessions = {}
        timestamps = {}
        # 空闲超时切分所需：五元组 -> 上次见到的时间戳 / 已切出的段序号
        last_seen_ts = {}
        segment_seq = {}
        skip_stats = {
            "total_packets": 0,
            "accepted_packets": 0,
            "too_short": 0,
            "non_ipv4": 0,
            "vlan_too_short": 0,
            "parse_errors": 0,
            "capture_format": detect_capture_format(pcap_file),
        }
        try:
            file_size = os.path.getsize(pcap_file)
            file_size_mb = file_size / (1024 * 1024)
            # 代码审查：pcapng 必须走 PcapNgReader。RawPcapReader 读 pcapng 不会抛异常，
            # 而是把 section header 当全局头，吐出一串"假包"（parse_errors 仍为 0）。
            with open_capture_reader(pcap_file) as reader:
                packet_iter = tqdm(
                    reader,
                    desc=f"packets:{os.path.basename(pcap_file)} [{file_size_mb:.1f}MB]",
                    leave=False,
                    unit="pkt",
                    mininterval=0.5,
                )
                packet_count = 0
                for pkt_data, pkt_ts in packet_iter:
                    skip_stats["total_packets"] += 1
                    packet_count += 1
                    if packet_count == 1 or packet_count % 1000 == 0:
                        packet_iter.set_postfix({"processed": packet_count})

                    try:
                        if len(pkt_data) < 34:
                            skip_stats["too_short"] += 1
                            continue

                        eth_type = (pkt_data[12] << 8) | pkt_data[13]
                        offset = 14

                        if eth_type == 0x8100:  # VLAN 标签
                            eth_type = (pkt_data[16] << 8) | pkt_data[17]
                            offset = 18
                            if len(pkt_data) < 38:
                                skip_stats["vlan_too_short"] += 1
                                continue

                        if eth_type != 0x0800:
                            skip_stats["non_ipv4"] += 1
                            continue

                        if offset + 20 > len(pkt_data):
                            skip_stats["too_short"] += 1
                            continue

                        ip_hl = (pkt_data[offset] & 0x0F) * 4
                        proto = pkt_data[offset + 9]

                        src_ip = f"{pkt_data[offset+12]}.{pkt_data[offset+13]}.{pkt_data[offset+14]}.{pkt_data[offset+15]}"
                        dst_ip = f"{pkt_data[offset+16]}.{pkt_data[offset+17]}.{pkt_data[offset+18]}.{pkt_data[offset+19]}"

                        sport = 0
                        dport = 0

                        if proto == 6:  # TCP
                            trans_offset = offset + ip_hl
                            if len(pkt_data) >= trans_offset + 4:
                                sport = (pkt_data[trans_offset] << 8) | pkt_data[trans_offset + 1]
                                dport = (pkt_data[trans_offset + 2] << 8) | pkt_data[trans_offset + 3]
                        elif proto == 17:  # UDP
                            trans_offset = offset + ip_hl
                            if len(pkt_data) >= trans_offset + 4:
                                sport = (pkt_data[trans_offset] << 8) | pkt_data[trans_offset + 1]
                                dport = (pkt_data[trans_offset + 2] << 8) | pkt_data[trans_offset + 3]

                        # 双向五元组：始终把较小的 IP 放前面，保证正反向包归入同一会话
                        if src_ip <= dst_ip:
                            base_key = (src_ip, sport, dst_ip, dport, proto)
                        else:
                            base_key = (dst_ip, dport, src_ip, sport, proto)

                        # 空闲超时（可选）：同一五元组停顿超过 idle_timeout 秒即另起一个会话。
                        # 默认关闭，保持历史口径；开启后与六分类链路（120s）一致，
                        # 两条链路的流基数才能对齐。
                        if self.idle_timeout is not None:
                            last_seen = last_seen_ts.get(base_key)
                            if last_seen is not None and pkt_ts - last_seen > self.idle_timeout:
                                key = (base_key, segment_seq[base_key])
                                segment_seq[base_key] += 1
                            else:
                                key = (base_key, segment_seq.setdefault(base_key, 0))
                            last_seen_ts[base_key] = pkt_ts
                        else:
                            key = base_key

                        if key not in sessions:
                            sessions[key] = []
                            timestamps[key] = pkt_ts

                        pkt_list = self._mask_packet_headers(bytearray(pkt_data), offset)

                        if len(sessions[key]) < self.max_packets:
                            sessions[key].append(bytes(pkt_list[: self.bytes_per_packet]))
                        skip_stats["accepted_packets"] += 1
                    except Exception as exc:
                        # 代码审查：逐包异常也要留证据。原先只加计数，事后无从判断
                        # 是哪种畸形包让整条 pcap 少了一半会话。
                        skip_stats["parse_errors"] += 1
                        skip_stats.setdefault("parse_error_samples", [])
                        if len(skip_stats["parse_error_samples"]) < 5:
                            skip_stats["parse_error_samples"].append(
                                f"{type(exc).__name__}: {exc}"
                            )
                        continue
        except Exception as exc:
            # 读文件级别的失败（文件损坏、格式不认、权限不足）必须显式暴露：
            # 打印 + 结构化错误，让调用方能把它回给前端，而不是返回一份"空但成功"的结果。
            message = f"{type(exc).__name__}: {exc}"
            print(f"Error reading {pcap_file}: {message}")
            skip_stats["parse_errors"] += 1
            skip_stats["read_error"] = message

        finalized_sessions = {k: tuple(v) for k, v in sessions.items()}
        skip_stats["session_count"] = len(finalized_sessions)
        return finalized_sessions, timestamps, skip_stats

    def process_session(self, session_packets):
        """
        把会话的包列表转成 28×28 的 numpy 数组（灰度图）。

        每行是一个包的前 28 字节，最多取 28 个包。
        不足的部分用 0 补齐。
        """
        img_array = np.zeros((self.max_packets, self.bytes_per_packet), dtype=np.uint8)

        if isinstance(session_packets, (bytes, bytearray)):
            session_packets = [
                bytes(session_packets[i : i + self.bytes_per_packet])
                for i in range(0, len(session_packets), self.bytes_per_packet)
            ]

        for packet_index, packet_bytes in enumerate(session_packets[: self.max_packets]):
            if packet_bytes is None:
                continue
            packet_array = np.frombuffer(bytes(packet_bytes[: self.bytes_per_packet]), dtype=np.uint8)
            if packet_array.size > 0:
                img_array[packet_index, : packet_array.size] = packet_array

        return img_array
