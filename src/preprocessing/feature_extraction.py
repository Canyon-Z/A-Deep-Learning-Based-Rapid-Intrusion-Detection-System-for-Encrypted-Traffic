import numpy as np
from scapy.all import rdpcap
from scapy.layers.inet import IP
from scapy.layers.inet import TCP, UDP

class FeatureExtractor:
    def __init__(self, target_seq_len=100):
        self.target_seq_len = target_seq_len
    
    def extract_features(self, pcap_file):
        """
        Main function to extract features from a pcap file.
        Returns a processed numpy array suitable for model input.
        """
        packets = rdpcap(pcap_file)
        
        features = []
        prev_time = None
        for pkt in packets:
            if IP in pkt:
                # Raw features: length, protocol, inter-arrival time, TCP flags, src port, dst port.
                pkt_len = len(pkt)
                proto = pkt[IP].proto
                time = float(pkt.time)

                if prev_time is None:
                    iat = 0.0
                else:
                    iat = max(time - prev_time, 0.0)
                prev_time = time

                tcp_flags = 0
                src_port = 0
                dst_port = 0
                if TCP in pkt:
                    tcp_flags = int(pkt[TCP].flags)
                    src_port = int(pkt[TCP].sport)
                    dst_port = int(pkt[TCP].dport)
                elif UDP in pkt:
                    src_port = int(pkt[UDP].sport)
                    dst_port = int(pkt[UDP].dport)

                features.append([pkt_len, proto, iat, tcp_flags, src_port, dst_port])

                if len(features) >= self.target_seq_len:
                    break

        # Pad if necessary
        if len(features) < self.target_seq_len:
            pad_len = self.target_seq_len - len(features)
            for _ in range(pad_len):
                features.append([0, 0, 0, 0, 0, 0])

        arr = np.array(features, dtype=np.float32)
        return self._normalize_features(arr)

    def _normalize_features(self, features):
        """
        Normalize to a compact range so model logits are less likely to saturate.
        Output shape keeps (seq_len, 6):
        [pkt_len_norm, proto_norm, iat_norm, tcp_flags_norm, src_port_norm, dst_port_norm]
        """
        if features.size == 0:
            return features

        # Non-padded rows are treated as valid packets.
        valid_mask = features[:, 0] > 0
        normalized = np.zeros_like(features, dtype=np.float32)

        if not np.any(valid_mask):
            return normalized

        pkt_len = np.clip(features[:, 0] / 1500.0, 0.0, 1.0)
        proto = np.clip(features[:, 1] / 255.0, 0.0, 1.0)

        # Inter-arrival time log scale; 1s is used as practical cap for normalization.
        iat = np.log1p(np.maximum(features[:, 2], 0.0))
        iat = np.clip(iat / np.log1p(1.0), 0.0, 1.0)

        tcp_flags = np.clip(features[:, 3] / 255.0, 0.0, 1.0)
        src_port = np.clip(features[:, 4] / 65535.0, 0.0, 1.0)
        dst_port = np.clip(features[:, 5] / 65535.0, 0.0, 1.0)

        normalized[:, 0] = pkt_len
        normalized[:, 1] = proto
        normalized[:, 2] = iat
        normalized[:, 3] = tcp_flags
        normalized[:, 4] = src_port
        normalized[:, 5] = dst_port
        normalized[~valid_mask] = 0.0
        return normalized

    def get_byte_distribution(self, payload):
        # Placeholder for byte distribution logic
        pass
