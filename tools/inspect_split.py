import glob
import os
import random

root = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data_USTC-TK2016",
    "1_Pcap",
    "data/USTC-TFC2016-master",
)

for cls in sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]):
    files = glob.glob(os.path.join(root, cls, "**", "*.pcap"), recursive=True)
    files = sorted(files)
    rnd = random.Random(42)
    rnd.shuffle(files)

    n = len(files)
    if n <= 1:
        train, val, test = files, [], []
    else:
        train_end = max(1, int(n * 0.7))
        val_end = train_end + int(n * 0.15)
        if n >= 3 and val_end >= n:
            val_end = n - 1
        if n >= 3 and val_end <= train_end:
            val_end = min(n - 1, train_end + 1)
        train, val, test = files[:train_end], files[train_end:val_end], files[val_end:]

    print(f"[{cls}] total={n} train={len(train)} val={len(val)} test={len(test)}")
    print("  test files:")
    for f in test:
        print("   -", os.path.relpath(f, root))
