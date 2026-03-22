import numpy as np
from scapy.all import PcapReader, IP, TCP, UDP

class FeatureExtractor:
    def __init__(self, truncate_len=784):
        self.truncate_len = truncate_len

    def pcap_to_sessions(self, pcap_file):
        """
        Phase 1: Traffic Splitting (Session + All Layers)
        Reads a pcap file and splits traffic into sessions based on 5-tuple.
        Returns a dictionary of sessions: {5-tuple: raw_bytes}
        """
        sessions = {}
        timestamps = {}
        try:
            # Using PcapReader to iterate packet by packet (memory efficient)
            with PcapReader(pcap_file) as packets:
                for pkt in packets:
                    if IP in pkt:
                        src_ip = pkt[IP].src
                        dst_ip = pkt[IP].dst
                        proto = pkt[IP].proto
                        
                        sport = 0
                        dport = 0
                        
                        if TCP in pkt:
                            sport = pkt[TCP].sport
                            dport = pkt[TCP].dport
                        elif UDP in pkt:
                            sport = pkt[UDP].sport
                            dport = pkt[UDP].dport
                        
                        # Five-tuple key for session (bidirectional)
                        # Sorting IP/Port pairs ensures A->B and B->A are mapped to the same session
                        # Session key: (LowIP, LowPort, HighIP, HighPort, Proto)
                        if src_ip <= dst_ip:
                             key = (src_ip, sport, dst_ip, dport, proto)
                        else:
                             key = (dst_ip, dport, src_ip, sport, proto)

                        if key not in sessions:
                            sessions[key] = b''
                            timestamps[key] = float(pkt.time) # Record start time of session
                        
                        # "Session + all layers": Concatenate raw packet bytes
                        # This includes headers (Ethernet/IP/TCP etc.) + Payload
                        sessions[key] += bytes(pkt)
        except Exception as e:
            print(f"Error reading {pcap_file}: {e}")
            
        return sessions, timestamps

    def process_session(self, session_bytes):
        """
        Phase 2: Traffic Cleaning/Truncation (784 bytes)
        Phase 3: Byte to Tensor Conversion (Preparation: numpy array)
        """
        # Truncate or Pad to 784 bytes
        if len(session_bytes) >= self.truncate_len:
            byte_data = session_bytes[:self.truncate_len]
        else:
            # Zero padding
            byte_data = session_bytes + b'\x00' * (self.truncate_len - len(session_bytes))
            
        # Convert to numpy array (uint8) -> 0-255
        img_array = np.frombuffer(byte_data, dtype=np.uint8)
        
        # Reshape to 28x28 (Grayscale image format)
        try:
            img_array = img_array.reshape((28, 28))
        except ValueError:
            # Fallback if something went wrong with buffer size (shouldn't happen with padding logic)
            img_array = np.zeros((28, 28), dtype=np.uint8)
            
        return img_array
