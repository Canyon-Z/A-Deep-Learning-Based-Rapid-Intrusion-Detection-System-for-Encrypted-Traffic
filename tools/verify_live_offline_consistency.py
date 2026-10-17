# -*- coding: utf-8 -*-
"""比对实时链路和离线链路生成的 28x28 输入是否一致。

模型是按"会话前 28 个包、每包前 28 字节、MAC/IP 抹掉"的图训练的。实时这边如果攒包方式
或者掩码跟离线不一样，喂进去的东西就变了，表现就是离线判恶意、实时的判成良性。
以前踩过这个坑，留个脚本方便以后回归。

  离线：FeatureExtractor.pcap_to_sessions 出会话，再 process_session 出图
  实时：把同一个 pcap 逐帧喂给 packet_handler（把 sniff 换成回放器，不用真网卡）
  最后逐个会话比矩阵

用法：
  python tools/verify_live_offline_consistency.py <某个 pcap 文件>
"""
import argparse
import base64
import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

import web.backend.main as M
from src.preprocessing.feature_extraction import FeatureExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap", help="用于比对的 PCAP 文件路径")
    ap.add_argument("--model-type", default="ensemble_vote")
    args = ap.parse_args()

    if not os.path.exists(args.pcap):
        print("找不到文件:", args.pcap)
        return 2

    print("[1/4] 加载模型 ...")
    M.load_model()

    print("[2/4] 离线链路：提取会话与 28x28 输入 ...")
    extractor = FeatureExtractor()
    sessions, _, _ = extractor.pcap_to_sessions(args.pcap)
    offline = {k: np.asarray(extractor.process_session(tuple(v)), dtype=np.uint8)
               for k, v in sessions.items()}
    print("      会话数 = %d" % len(offline))

    print("[3/4] 实时链路：把同一份 PCAP 逐帧回放进 packet_handler ...")
    state = {"passes": 0}

    def replay_sniff(iface=None, prn=None, store=False, timeout=1):
        if state["passes"] < 1:
            state["passes"] = 1
            from scapy.layers.l2 import Ether
            from scapy.utils import RawPcapReader
            with RawPcapReader(args.pcap) as reader:
                for data, _meta in reader:
                    prn(Ether(bytes(data)))
        time.sleep(0.03)

    M.resolve_iface = lambda v: v
    M.sniff = replay_sniff
    M.start_live_capture(iface="replay0", model_type=args.model_type, session_user="verify")
    for _ in range(200):
        time.sleep(0.05)
        if state["passes"] >= 1:
            break
    time.sleep(0.6)
    M.stop_live_capture(session_user="verify")
    time.sleep(0.3)

    resp = M.live_stats(session_user="verify")
    flows = (resp if isinstance(resp, dict) else
             json.loads(resp.body.decode("utf-8", "ignore"))).get("flows") or []
    print("      实时流数 = %d" % len(flows))

    print("[4/4] 逐会话比对 ...")
    same = diff = 0
    worst = None
    for f in flows:
        b64 = f.get("image_data")
        if not b64:
            continue
        # image_data 是 140x140 的 NEAREST 放大图，取每 5 像素还原 28x28
        live = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))[::5, ::5]
        src, dst = f.get("src"), f.get("dst")
        sport, dport, proto = f.get("sport"), f.get("dport"), f.get("proto")
        key = ((src, sport, dst, dport, proto) if src <= dst
               else (dst, dport, src, sport, proto))
        ref = offline.get(key)
        if ref is None:
            continue
        d = int((ref.astype(int) != live.astype(int)).sum())
        if d == 0:
            same += 1
        else:
            diff += 1
            if worst is None or d > worst[1]:
                worst = (key, d)

    print()
    print("逐格完全一致 : %d 条" % same)
    print("存在差异     : %d 条" % diff)
    if worst:
        print("最大差异     : %d/784 格  %s" % (worst[1], worst[0]))
    ok = (diff == 0 and same > 0)
    print("结论         :", "PASS —— 两条链路的输入逐格一致" if ok else "FAIL —— 请检查实时链路的输入构造")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
