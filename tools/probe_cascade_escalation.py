"""验证级联的「升级」路径会不会真的触发。

背景：单份 USTC 样本（Facetime.pcap，6000 会话）的批次平均置信度是 0.9904，
高于 0.90 阈值，所以走的是"不升级"。但那只验证了不升级分支。

这个脚本用**分布外**的样本（CSE-CIC-IDS2018 的工控协议 pcap，协议族与 USTC 的
HTTP/SMB/P2P 完全不同）批量跑级联，统计 avg_confidence 的分布：

  - 若大量样本低于 0.90  -> 升级路径在实际数据上会被触发，阈值合理
  - 若全部高于 0.90      -> 阈值 0.90 过松，级联等于永远不升级，应上调到 0.99/0.995

用法：
    python -u tools/probe_cascade_escalation.py
    python -u tools/probe_cascade_escalation.py --limit 12 --threshold 0.99
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


def run_one(sess, pcap):
    with open(pcap, 'rb') as fh:
        files = {'file': (os.path.basename(pcap), fh, 'application/octet-stream')}
        r = sess.post(f'{BASE}/analyze', files=files,
                      data={'model_type': 'cascade'}, timeout=600)
    if r.status_code != 200:
        return {'pcap': os.path.basename(pcap), 'error': f'HTTP {r.status_code}'}
    payload = r.json()
    cascade = payload.get('cascade') or {}
    return {
        'pcap': os.path.basename(pcap),
        'sessions': payload.get('total_sessions'),
        'avg_confidence': cascade.get('avg_confidence'),
        'escalated': cascade.get('escalated'),
        'seconds': payload.get('execution_time'),
        'probe': cascade.get('probe_model'),
    }


def main():
    parser = argparse.ArgumentParser(description="探测级联升级触发情况")
    parser.add_argument('--glob', type=str, default=None,
                        help="pcap 匹配模式，默认取 data/test_ics 下最小的若干 pcap")
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=None,
                        help="仅用于报告：按这个阈值重新判定会升级多少个（不改服务端）")
    args = parser.parse_args()

    if args.glob:
        pcaps = sorted(glob.glob(args.glob, recursive=True))
    else:
        pcaps = sorted(glob.glob(os.path.join(_PROJECT_ROOT, 'data', 'test_ics', '**', '*.pcap'),
                                 recursive=True), key=lambda p: os.path.getsize(p))
    pcaps = [p for p in pcaps if os.path.getsize(p) > 1024][:args.limit]
    if not pcaps:
        print("找不到可用的 pcap")
        return 1

    sess = requests.Session()
    sess.cookies.set(SESSION_COOKIE_NAME, _sign_session('admin'))

    print(f"{'pcap':<44}{'会话数':>8}{'平均置信':>11}{'升级':>7}{'耗时':>10}")
    print('-' * 82)
    rows = []
    for pcap in pcaps:
        info = run_one(sess, pcap)
        if 'error' in info:
            print(f"{info['pcap'][:43]:<44}{'-':>8}{'-':>11}{info['error']:>7}")
            continue
        conf = info['avg_confidence']
        conf_txt = f"{conf:.4f}" if isinstance(conf, (int, float)) else '-'
        esc = '是' if info['escalated'] else '否'
        print(f"{info['pcap'][:43]:<44}{str(info['sessions']):>8}{conf_txt:>11}{esc:>7}"
              f"{str(info['seconds']):>10}")
        rows.append(info)

    valid = [r for r in rows if isinstance(r.get('avg_confidence'), (int, float))]
    if not valid:
        print("\n没有任何样本返回 avg_confidence，无法判断")
        return 1

    confs = [r['avg_confidence'] for r in valid]
    confs_sorted = sorted(confs)
    n = len(confs_sorted)
    print(f"\n=== 置信度分布（{n} 个样本）===")
    print(f"  最小 {confs_sorted[0]:.4f} | 中位 {confs_sorted[n // 2]:.4f} | 最大 {confs_sorted[-1]:.4f}")
    print(f"  均值 {sum(confs) / n:.4f}")
    print(f"  低于服务端阈值 0.90 的样本：{sum(1 for c in confs if c < 0.90)} / {n}")
    for thr in (0.95, 0.99, 0.995, 0.999):
        cnt = sum(1 for c in confs if c < thr)
        print(f"  若阈值提到 {thr}: 会有 {cnt} / {n} 个样本升级")

    if args.threshold is not None:
        cnt = sum(1 for c in confs if c < args.threshold)
        print(f"\n[按 --threshold {args.threshold}] 升级数 {cnt} / {n}")

    out = os.path.join(_PROJECT_ROOT, 'logs', 'cascade_escalation_probe.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({'samples': rows, 'confidence_sorted': confs_sorted}, fh,
                  ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
