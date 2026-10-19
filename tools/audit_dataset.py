"""只读统计脚本：清点数据集规模、特征缓存里的会话数、processed 图像分布。

不加载 npz 的数组数据，只读 zip 内 .npy 的 header（shape/dtype），因此很快。
"""
from __future__ import annotations

import os
import sys
import zipfile
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 src/preprocessing/data_loader.py 的 FEATURE_CACHE_VERSION 保持一致。
# 版本 4 起缓存键 = 版本 + 绝对路径 + 内容指纹（大小 + 首/中/尾采样片段的 sha256），
# 不再包含 mtime；版本 3 及以前的 npz 是用含 mtime 的旧键命名的，
# 所以现在对现存 pcap 反查命中率必然是 0，这属于预期结果（那批缓存只能重建）。
CACHE_VERSION = 4
_FINGERPRINT_PROBE_BYTES = 64 * 1024

CACHE_DIRS = [
    "data/USTC-TFC2016-master/.feature_cache",
    "data/combined_train_pcap/.feature_cache",
    "data/mixed_train_pcap/.feature_cache",
    "data/test_ics/.feature_cache",
]


def npz_shape(path: str):
    """只读 .npy header，返回 (shape, dtype)。失败返回 None。"""
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".npy")]
            if not names:
                return None
            with z.open(names[0]) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(f)
                elif version == (3, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_3_0(f)
                else:
                    return ("ERR", f"unsupported npy version {version}")
        return tuple(shape), np.dtype(dtype).str
    except Exception as exc:  # noqa: BLE001
        return ("ERR", str(exc)[:80])


def audit_cache(rel: str) -> None:
    full = os.path.join(ROOT, rel)
    if not os.path.isdir(full):
        print(f"[cache] {rel}: MISSING")
        return
    files = sorted(f for f in os.listdir(full) if f.endswith(".npz"))
    total = 0
    bad = []
    empties = 0
    shapes = Counter()
    for name in files:
        info = npz_shape(os.path.join(full, name))
        if info is None:
            bad.append(name)
            continue
        if info[0] == "ERR":
            bad.append(f"{name}: {info[1]}")
            continue
        shape, _dtype = info
        n = int(shape[0]) if shape else 0
        if n == 0:
            empties += 1
        total += n
        shapes[shape[1:] if len(shape) > 1 else shape] += n
    print(f"[cache] {rel}: files={len(files)} sessions={total} empty_files={empties} bad={len(bad)}")
    if shapes:
        print(f"        per-file image shape -> sessions: {dict(shapes)}")
    for b in bad[:5]:
        print(f"        bad: {b}")


def _cache_key(pcap_path: str) -> str:
    """与 data_loader._build_cache_key 同算法：版本 + 绝对路径 + 内容指纹。"""
    import hashlib

    stat = os.stat(pcap_path)
    hasher = hashlib.sha256()
    with open(pcap_path, "rb") as handle:
        hasher.update(handle.read(_FINGERPRINT_PROBE_BYTES))
        if stat.st_size > _FINGERPRINT_PROBE_BYTES * 3:
            handle.seek(stat.st_size // 2)
            hasher.update(handle.read(_FINGERPRINT_PROBE_BYTES))
            handle.seek(-_FINGERPRINT_PROBE_BYTES, os.SEEK_END)
            hasher.update(handle.read(_FINGERPRINT_PROBE_BYTES))
    fingerprint = f"{stat.st_size}:{hasher.hexdigest()}"
    raw = f"{CACHE_VERSION}|{os.path.abspath(pcap_path)}|{fingerprint}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def audit_class_sessions(rel: str) -> None:
    """用缓存键反推每个 pcap 的会话数，再按类别汇总。

    缓存键算法与 src/preprocessing/data_loader.py 的 _build_cache_key 保持一致。
    注意：现有 .npz 全部由含 mtime 的旧键生成，因此 no_cache 通常等于文件总数 ——
    这是"缓存需要重建"的信号，不是数据缺失。
    """
    cache_dir = os.path.join(ROOT, rel, ".feature_cache")
    full = os.path.join(ROOT, rel)
    if not os.path.isdir(full):
        return
    for cls in sorted(os.listdir(full)):
        cls_path = os.path.join(full, cls)
        if not os.path.isdir(cls_path) or cls.startswith("."):
            continue
        sessions = 0
        cached = 0
        missing = 0
        for dirpath, _d, filenames in os.walk(cls_path):
            for f in filenames:
                if not f.lower().endswith(".pcap"):
                    continue
                pcap = os.path.join(dirpath, f)
                try:
                    key = _cache_key(pcap)
                except OSError:
                    missing += 1
                    continue
                npz = os.path.join(cache_dir, f"{key}.npz")
                if not os.path.isfile(npz):
                    missing += 1
                    continue
                info = npz_shape(npz)
                if info is None or info[0] == "ERR":
                    missing += 1
                    continue
                cached += 1
                sessions += int(info[0][0]) if info[0] else 0
        print(f"[sessions] {rel}/{cls}: sessions={sessions} cached_pcap={cached} no_cache={missing}")


def audit_processed(rel: str) -> None:
    full = os.path.join(ROOT, rel)
    if not os.path.isdir(full):
        print(f"[processed] {rel}: MISSING")
        return
    print(f"[processed] {rel}")
    for top in sorted(os.listdir(full)):
        top_path = os.path.join(full, top)
        if not os.path.isdir(top_path):
            continue
        # 顶层子目录可能还有一层（如 Benign/xxx/*.png）
        per_sub = {}
        total_png = 0
        other = Counter()
        for dirpath, _dirnames, filenames in os.walk(top_path):
            sub = os.path.relpath(dirpath, top_path)
            png = sum(1 for f in filenames if f.lower().endswith((".png", ".jpg")))
            total_png += png
            if png:
                per_sub[sub] = per_sub.get(sub, 0) + png
            for f in filenames:
                if not f.lower().endswith((".png", ".jpg")):
                    other[os.path.splitext(f)[1].lower()] += 1
        print(f"  {top}: images={total_png} subdirs={len(per_sub)}")
        for sub, cnt in sorted(per_sub.items(), key=lambda kv: -kv[1])[:8]:
            print(f"      {sub or '.'}: {cnt}")
        if other:
            print(f"      non-image files: {dict(other)}")


def main() -> int:
    print("=" * 60)
    print("特征缓存（.npz）里的会话数")
    print("=" * 60)
    for rel in CACHE_DIRS:
        audit_cache(rel)

    print()
    print("=" * 60)
    print("data/processed 图像分布")
    print("=" * 60)
    audit_processed("data/processed")

    print()
    print("=" * 60)
    print("pcap 文件数与大小")
    print("=" * 60)
    for rel in [
        "data/USTC-TFC2016-master",
        "data/combined_train_pcap",
        "data/mixed_train_pcap",
        "data/test_ustc",
        "data/test_ics",
    ]:
        full = os.path.join(ROOT, rel)
        if not os.path.isdir(full):
            print(f"[pcap] {rel}: MISSING")
            continue
        for cls in sorted(os.listdir(full)):
            cls_path = os.path.join(full, cls)
            if not os.path.isdir(cls_path) or cls.startswith("."):
                continue
            total = 0
            size = 0
            for dirpath, _d, filenames in os.walk(cls_path):
                for f in filenames:
                    if f.lower().endswith(".pcap"):
                        total += 1
                        size += os.path.getsize(os.path.join(dirpath, f))
            print(f"[pcap] {rel}/{cls}: files={total} size={size/1024/1024:.1f}MB")

    print()
    print("=" * 60)
    print("每个类别的会话样本数（用 pcap 反推特征缓存键）")
    print("=" * 60)
    for rel in [
        "data/USTC-TFC2016-master",
        "data/combined_train_pcap",
        "data/mixed_train_pcap",
        "data/test_ustc",
        "data/test_ics",
    ]:
        audit_class_sessions(rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
