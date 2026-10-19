"""端到端验证级联检测（cascade）：单模型优先，低置信自动升级 6 模型集成。

用法：
    python -u tools/verify_cascade.py
    python -u tools/verify_cascade.py --pcap data/test_ics/Benign/ics_b_00007_104pcap.pcap
"""

import argparse
import glob
import json
import os
import sys

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from web.backend.main import SESSION_COOKIE_NAME, _sign_session

BASE = 'http://127.0.0.1:8000'


def pick_pcap(explicit=None):
    if explicit:
        return explicit
    candidates = sorted(glob.glob(os.path.join(_PROJECT_ROOT, 'data', 'test_ics', '**', '*.pcap'),
                                  recursive=True), key=lambda p: os.path.getsize(p))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(_PROJECT_ROOT, 'data', 'test_ustc', '**', '*.pcap'),
                                      recursive=True), key=lambda p: os.path.getsize(p))
    return candidates[0] if candidates else None


def run(pcap, model_type, label):
    sess = requests.Session()
    sess.cookies.set(SESSION_COOKIE_NAME, _sign_session('admin'))
    print(f"\n=== {label}：model_type={model_type} | pcap={os.path.basename(pcap)} "
          f"({os.path.getsize(pcap) / 1024:.0f} KB) ===")
    with open(pcap, 'rb') as fh:
        files = {'file': (os.path.basename(pcap), fh, 'application/octet-stream')}
        data = {'model_type': model_type}
        r = sess.post(f'{BASE}/analyze', files=files, data=data, timeout=600)
    if r.status_code != 200:
        print(f"[FAIL] HTTP {r.status_code}: {r.text[:400]}")
        return None
    payload = r.json()
    keep = {k: payload.get(k) for k in
            ('model_type', 'model_name', 'ensemble', 'ensemble_models',
             'cascade', 'predicted_result', 'malware_ratio', 'total_sessions',
             'avg_confidence', 'execution_time')}
    print(json.dumps(keep, ensure_ascii=False, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description="验证级联检测")
    parser.add_argument('--pcap', type=str, default=None)
    parser.add_argument('--model-type', type=str, default='cascade')
    parser.add_argument('--compare', action='store_true',
                        help="同时跑单模型与集成，用于对照升级前后的差异")
    args = parser.parse_args()

    pcap = pick_pcap(args.pcap)
    if not pcap or not os.path.exists(pcap):
        print("找不到可用的 pcap")
        return 1

    result = run(pcap, args.model_type, '级联')
    if args.compare:
        run(pcap, 'pure_bilstm', '单模型对照')
        run(pcap, 'ensemble_vote', '集成对照')

    if result:
        cascade = result.get('cascade') or {}
        print("\n=== 级联判定 ===")
        print(f"  探针模型      : {cascade.get('probe_model')}")
        print(f"  批次平均置信度: {cascade.get('avg_confidence')}")
        print(f"  升级阈值      : {cascade.get('threshold')}")
        print(f"  是否升级      : {'是 → 已由 6 模型集成复核' if cascade.get('escalated') else '否 → 单模型直接判定'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
