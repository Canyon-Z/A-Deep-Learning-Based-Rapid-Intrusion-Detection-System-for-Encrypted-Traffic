"""下载 CSE-CIC-IDS2018 官方已提取的 CICFlowMeter CSV 特征表。

为什么要下这个：项目里现有的 CSE-CIC-IDS2018 数据是 pcap2png.py 切出来的 28x28
图像，和训练用的 FeatureExtractor 会话矩阵不是同一套表示，拿它评估只能说明"换表示
就失效"，不能说明模型在真实跨域流量上的表现。官方发布的这套 CSV 是 CICFlowMeter
跑出来的**流统计特征**（80 列：包长统计、到达间隔、流持续时间、上下行字节数……），
正好用来验证"流特征路线是否比字节图像更耐跨域"。

数据来源：https://www.unb.ca/cic/datasets/ids-2018.html（AWS 公开镜像，无需鉴权）

用法：
    python tools/download_cse_csv.py --list                    # 只列可用文件
    python tools/download_cse_csv.py --days 15-02,21-02,02-03  # 下指定几天
    python tools/download_cse_csv.py --all                     # 下全部（约 6.4 GB）
"""

import argparse
import os
import re
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_DEFAULT_OUT = os.path.join(_PROJECT_ROOT, 'data', 'cse_csv')

_BUCKET = ('https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/'
           '?list-type=2&prefix=Processed%20Traffic%20Data%20for%20ML%20Algorithms/&max-keys=1000')
_PREFIX = ('https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/'
           'Processed%20Traffic%20Data%20for%20ML%20Algorithms/')

# 每天的攻击类型（官方 README 记载），只用于给用户选文件时做参考
DAY_ATTACKS = {
    '14-02': 'FTP/SSH Brute Force',
    '15-02': 'DoS/DDoS（GoldenEye、Hulk、Slowloris、Slowhttptest、LOIC-HTTP/HOIC）',
    '16-02': 'DoS/DDoS 续（LOIC-UDP、HOIC）',
    '20-02': 'DDoS + LOIC-UDP（该日文件异常大，3.8 GB）',
    '21-02': 'Brute Force、XSS、SQL Injection、Infiltration、DDoS',
    '22-02': 'Brute Force、XSS、SQL Injection、Infiltration',
    '23-02': 'Brute Force、XSS、SQL Injection、Infiltration',
    '28-02': 'Infiltration、DDoS',
    '01-03': 'Infiltration',
    '02-03': 'Bot（Botnet）',
}


def list_remote():
    """返回 [(文件名, 字节数)]，按名字排序。"""
    xml = urllib.request.urlopen(_BUCKET, timeout=60).read().decode('utf-8', 'ignore')
    items = re.findall(r'<Key>(.*?)</Key>.*?<Size>(\d+)</Size>', xml, re.S)
    out = []
    for key, size in items:
        name = key.split('/')[-1]
        if name.endswith('.csv'):
            out.append((name, int(size)))
    return sorted(out)


def day_of(filename):
    m = re.search(r'(\d{2}-\d{2}-\d{4})', filename)
    return m.group(1)[:5] if m else ''


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url, dest, name):
    """分块下载并打印进度；支持已存在文件的大小校验后跳过。"""
    tmp = dest + '.part'
    existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0

    req = urllib.request.Request(url)
    if existing:
        req.add_header('Range', f'bytes={existing}-')
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except Exception as exc:                       # noqa: BLE001
        print(f"[error] {name}: 打开连接失败 {exc}")
        return False

    total = existing + int(resp.headers.get('Content-Length') or 0)
    mode = 'ab' if existing and resp.status == 206 else 'wb'
    if mode == 'wb':
        existing = 0
        total = int(resp.headers.get('Content-Length') or 0)

    done = existing
    last_print = 0.0
    t0 = time.time()
    with open(tmp, mode) as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last_print >= 5:
                last_print = now
                speed = done / max(now - t0, 0.001)
                pct = (done / total * 100) if total else 0
                print(f"  {name}: {pct:5.1f}%  {human(done)}/{human(total)}  "
                      f"{human(speed)}/s", flush=True)

    if total and done != total:
        print(f"[warn] {name}: 下载不完整 {human(done)}/{human(total)}，保留 .part 以便续传")
        return False

    if os.path.exists(dest):
        os.remove(dest)
    os.rename(tmp, dest)
    print(f"[done] {name}: {human(done)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="下载 CSE-CIC-IDS2018 的 CICFlowMeter CSV")
    parser.add_argument('--out', type=str, default=_DEFAULT_OUT, help="输出目录")
    parser.add_argument('--list', action='store_true', help="只列出远端可用文件")
    parser.add_argument('--days', type=str, default=None,
                        help="要下载的日期，逗号分隔，如 15-02,21-02,02-03")
    parser.add_argument('--all', action='store_true', help="下载全部文件")
    args = parser.parse_args()

    print("正在读取远端文件清单…")
    try:
        files = list_remote()
    except Exception as exc:                       # noqa: BLE001
        print(f"[error] 无法读取清单：{exc}")
        return 1

    total_bytes = sum(s for _, s in files)
    print(f"远端共 {len(files)} 个 CSV，合计 {human(total_bytes)}\n")
    for name, size in files:
        day = day_of(name)
        print(f"  {human(size):>9}  {name}   [{DAY_ATTACKS.get(day, '?')}]")

    if args.list:
        return 0

    if args.all:
        wanted = files
    elif args.days:
        days = [d.strip() for d in args.days.split(',') if d.strip()]
        wanted = [(n, s) for n, s in files if day_of(n) in days]
        missing = [d for d in days if not any(day_of(n) == d for n, _ in files)]
        if missing:
            print(f"\n[warn] 没找到这些日期：{missing}")
    else:
        print("\n请指定 --days 或 --all（或 --list 先看清单）")
        return 1

    if not wanted:
        print("没有匹配的文件")
        return 1

    os.makedirs(args.out, exist_ok=True)
    print(f"\n开始下载 {len(wanted)} 个文件到 {args.out}\n")
    ok = 0
    for name, size in wanted:
        dest = os.path.join(args.out, name)
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            print(f"[skip] {name}: 已存在且大小一致")
            ok += 1
            continue
        print(f"[get ] {name}  ({human(size)})")
        if download(_PREFIX + urllib.parse.quote(name), dest, name):
            ok += 1

    print(f"\n完成 {ok}/{len(wanted)} 个文件 -> {args.out}")
    return 0 if ok == len(wanted) else 1


if __name__ == '__main__':
    import urllib.parse
    sys.exit(main())
