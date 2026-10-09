import os

import numpy as np
from tqdm import tqdm


class FeatureExtractor:
    """
    从 PCAP 原始文件中提取流量特征的处理流水线。

    处理流程:
    1. 读取 PCAP，按五元组 (src_ip, src_port, dst_ip, dst_port, protocol)
       拆分为双向会话
    2. 可选地抹掉 MAC/IP 头部字段（迫使模型学习应用层特征）
    3. 每个会话截取前 28 个包、每包前 28 字节，排列成 28×28 灰度图

    参数:
        truncate_len:     会话字节截断长度（暂未使用）
        mask_headers:     是否用填充值覆盖 MAC/IP 头部
        mask_fill:        填充值 (0-255)，默认 0
        max_packets:      每个会话最大包数，默认 28
        bytes_per_packet: 每包最大字节数，默认 28
    """

    def __init__(self, truncate_len=784, mask_headers=True, mask_fill=0, max_packets=28, bytes_per_packet=28):
        self.truncate_len = truncate_len
        self.mask_headers = mask_headers
        self.mask_fill = mask_fill & 0xFF
        self.max_packets = max_packets
        self.bytes_per_packet = bytes_per_packet

    def _mask_packet_headers(self, pkt_list, offset):
        """用填充字节替换 MAC 头部（前 12 字节）和 IP 头部（偏移 + 12 开始的 20 字节），
        让模型不依赖 MAC/IP 地址做判断，专注于应用层特征。"""
        if not self.mask_headers:
            return pkt_list

        fill_byte = bytes([self.mask_fill])

        mac_end = min(12, len(pkt_list))
        if mac_end > 0:
            pkt_list[:mac_end] = fill_byte * mac_end

        ip_start = offset + 12
        ip_end = min(offset + 20, len(pkt_list))
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
        skip_stats = {
            "total_packets": 0,
            "accepted_packets": 0,
            "too_short": 0,
            "non_ipv4": 0,
            "vlan_too_short": 0,
            "parse_errors": 0,
        }
        try:
            file_size = os.path.getsize(pcap_file)
            file_size_mb = file_size / (1024 * 1024)
            from scapy.utils import RawPcapReader

            with RawPcapReader(pcap_file) as reader:
                packet_iter = tqdm(
                    reader,
                    desc=f"packets:{os.path.basename(pcap_file)} [{file_size_mb:.1f}MB]",
                    leave=False,
                    unit="pkt",
                    mininterval=0.5,
                )
                packet_count = 0
                for pkt_data, pkt_meta in packet_iter:
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

                        if offset + 20 > len(pkt_data):
                            skip_stats["too_short"] += 1
                            continue

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
                            key = (src_ip, sport, dst_ip, dport, proto)
                        else:
                            key = (dst_ip, dport, src_ip, sport, proto)

                        if key not in sessions:
                            sessions[key] = []
                            try:
                                ts = pkt_meta.sec + pkt_meta.usec / 1e6 if hasattr(pkt_meta, 'sec') else pkt_meta[0] + pkt_meta[1] / 1e6
                                timestamps[key] = ts
                            except Exception:
                                timestamps[key] = 0.0

                        pkt_list = self._mask_packet_headers(bytearray(pkt_data), offset)

                        if len(sessions[key]) < self.max_packets:
                            sessions[key].append(bytes(pkt_list[: self.bytes_per_packet]))
                        skip_stats["accepted_packets"] += 1
                    except Exception:
                        skip_stats["parse_errors"] += 1
                        continue
        except Exception as e:
            print(f"Error reading {pcap_file}: {e}")
            skip_stats["parse_errors"] += 1

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
