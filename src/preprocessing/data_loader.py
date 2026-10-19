import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import glob
from tqdm import tqdm
import sys
import os
import random
import hashlib
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.preprocessing.feature_extraction import FeatureExtractor


# 代码审查第 41 条：truncate_len 已从缓存键里去掉（它本来就没参与特征提取），
# 键的语义变了，因此把版本号 +1，避免继续命中按旧规则算出的缓存文件。
#
# 版本 4：缓存键不再包含 mtime（见下面 _build_cache_key 的说明）。
# 版本 3 及以前的键含文件 mtime，文件被复制、搬家或重新下载一次就整份失效 ——
# 实测仓库里 1GB+ 的 .feature_cache 对不上任何现存 pcap（键反推命中数全为 0），
# 重训时被迫重新解析 4.9GB 原始流量。键的语义变了，版本号必须 +1。
FEATURE_CACHE_VERSION = 4

# 计算内容指纹时采样的字节数（原始 pcap 动辄上百 MB，不能整体哈希）
_FINGERPRINT_PROBE_BYTES = 64 * 1024


def _file_content_fingerprint(pcap_file, probe_bytes=_FINGERPRINT_PROBE_BYTES):
    """对 pcap 内容做轻量指纹：文件大小 + 首/中/尾采样片段的 sha256。

    刻意不使用 mtime：文件在不同机器之间复制、目录搬家、重新下载都会改变 mtime，
    而内容其实没变。用 mtime 做键正是版本 3 缓存集体失效的原因。
    为了仍能识别"同样大小、内容被替换"的情况，这里额外对首、中、尾各采一段哈希。
    """
    stat = os.stat(pcap_file)
    hasher = hashlib.sha256()
    with open(pcap_file, "rb") as handle:
        hasher.update(handle.read(probe_bytes))
        if stat.st_size > probe_bytes * 3:
            handle.seek(stat.st_size // 2)
            hasher.update(handle.read(probe_bytes))
            handle.seek(-probe_bytes, os.SEEK_END)
            hasher.update(handle.read(probe_bytes))
    return f"{stat.st_size}:{hasher.hexdigest()}"


def _build_cache_key(pcap_file):
    """根据文件绝对路径与内容指纹生成缓存键（不依赖 mtime）"""
    abs_path = os.path.abspath(pcap_file)
    try:
        fingerprint = _file_content_fingerprint(abs_path)
    except OSError:
        fingerprint = "missing"
    raw = f"{FEATURE_CACHE_VERSION}|{abs_path}|{fingerprint}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cached_sessions(cache_path):
    """从 .npz 缓存加载已处理好的会话图像"""
    with np.load(cache_path, allow_pickle=False) as data:
        sessions = data["sessions"]
    if sessions.ndim == 2:
        sessions = np.expand_dims(sessions, axis=0)
    return sessions


def _save_cached_sessions(cache_path, sessions, compress=False):
    """把处理好的会话图像保存为 .npz 缓存，下次直接读取"""
    if sessions:
        arr = np.stack(sessions, axis=0).astype(np.uint8)
    else:
        arr = np.empty((0, 28, 28), dtype=np.uint8)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if compress:
        np.savez_compressed(cache_path, sessions=arr)
    else:
        np.savez(cache_path, sessions=arr)


class TrafficDataset(Dataset):
    """PyTorch Dataset，封装 28×28 流量灰度图和对应标签"""

    def __init__(self, data_list, labels_list, transform=None):
        self.data = data_list
        self.labels = labels_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = self.labels[idx]

        if isinstance(item, str):
            img = Image.open(item).convert("L")
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_array = np.array(img)
                img_tensor = torch.from_numpy(img_array).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)
        else:
            img_array = item
            img = Image.fromarray(img_array, mode="L")
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = torch.from_numpy(img_array).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)

        return img_tensor, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def load_data(data_dir):
        """从目录加载图片数据，子目录名作为类别标签"""
        if not os.path.exists(data_dir):
            print(f"Data directory {data_dir} does not exist.")
            return [], []

        data_list = []
        labels_list = []

        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        classes.sort()
        print(f"Found classes: {classes}")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for cls_name in classes:
            cls_dir = os.path.join(data_dir, cls_name)
            img_files = glob.glob(os.path.join(cls_dir, "*.png")) + glob.glob(os.path.join(cls_dir, "*.jpg"))

            label = class_to_idx[cls_name]
            print(f"Processing class '{cls_name}' from {cls_dir}...")

            for img_file in tqdm(img_files, desc=f"Loading {cls_name}"):
                data_list.append(img_file)
                labels_list.append(label)

        print(f"Loaded {len(data_list)} images from {data_dir}")
        return data_list, labels_list 


AUGMENT_PRESETS = {
    'light':  dict(p=0.6, packet_dropout=0.04, byte_cut=0.08, noise_std=0.01, max_shift=1),
    'medium': dict(p=0.8, packet_dropout=0.08, byte_cut=0.15, noise_std=0.02, max_shift=2),
    'strong': dict(p=0.9, packet_dropout=0.15, byte_cut=0.25, noise_std=0.04, max_shift=3),
}


class FlowSessionAugment:
    """28x28 流会话图像的增强。

    图像的语义是「行 = 包、列 = 该包的字节」，所以只能用保持这个结构的增强，
    不能用图像分类里常见的水平翻转/旋转（翻转会把以太网头搬到载荷位置，
    旋转会颠倒包序——那是在破坏标签，不是在增强数据）：

      - packet_dropout：逐行按概率整行清零，模拟丢包或采样窗口缺失
      - byte_cut：每行尾部按同一比例清零，模拟载荷被截断（TLS 记录长度变化）
      - noise：整幅图叠加小幅噪声，模拟字段位抖动与时间戳截断
      - row_shift：整体上下平移若干行，模拟会话起点的轻微偏移

    只对训练集使用；验证/测试集保持原始分布，否则指标不可比。
    """

    def __init__(self, p=0.8, packet_dropout=0.08, byte_cut=0.15,
                 noise_std=0.02, max_shift=2):
        self.p = float(p)
        self.packet_dropout = float(packet_dropout)
        self.byte_cut = float(byte_cut)
        self.noise_std = float(noise_std)
        self.max_shift = int(max_shift)

    def __call__(self, tensor):
        if self.p <= 0:
            return tensor
        if float(torch.rand(1).item()) >= self.p:
            return tensor

        out = tensor.clone()
        _, rows, cols = out.shape

        if self.packet_dropout > 0:
            keep = (torch.rand(rows, 1) >= self.packet_dropout).to(out.dtype)
            out = out * keep

        if self.byte_cut > 0:
            cut = int(round(cols * self.byte_cut * float(torch.rand(1).item())))
            if cut > 0:
                out[:, :, cols - cut:] = 0.0

        if self.noise_std > 0:
            out = torch.clamp(out + torch.randn_like(out) * self.noise_std, 0.0, 1.0)

        if self.max_shift > 0:
            shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
            if shift != 0:
                rolled = torch.zeros_like(out)
                if shift > 0:
                    rolled[:, shift:, :] = out[:, :rows - shift, :]
                else:
                    rolled[:, :rows + shift, :] = out[:, -shift:, :]
                out = rolled
        return out


def get_dataloaders(
    data_root,
    batch_size=32,
    cache_dir=None,
    use_cache=True,
    rebuild_cache=False,
    cache_compress=False,
    mask_headers=True,
    mask_fill=0,
    augment=False,
    augment_strength='medium',
):
    """
    数据加载总入口：从原始 PCAP 一路处理到 PyTorch DataLoader。

    处理流程:
    1. 遍历 data_root 下的子目录（每个子目录是一个类别）
    2. 把每个类别的 PCAP 文件按 7:1.5:1.5 拆成训练/验证/测试集
    3. 用 FeatureExtractor 将 PCAP 转成 28×28 灰度图
    4. 用 TrafficDataset + DataLoader 封装，传给训练脚本

    参数:
        data_root:      数据集根目录，下面有 Benign/Malware 等子目录
        cache_dir:      特征缓存目录，默认 data_root/.feature_cache
        use_cache:      启用缓存（跳过重复的特征提取）
        rebuild_cache:  强制重建缓存
        cache_compress: 使用压缩格式存储缓存
        mask_headers:   训练时 MAC/IP 头部字段
        mask_fill:      头部填充值 (0-255)

    返回:
        train_loader, val_loader, test_loader, class_to_idx
    """
    extractor = FeatureExtractor(
        mask_headers=mask_headers,
        mask_fill=mask_fill,
    )

    if not os.path.exists(data_root):
        print(f"Data root {data_root} does not exist.")
        return None, None, None, {}

    if cache_dir is None:
        cache_dir = os.path.join(data_root, ".feature_cache")
    cache_dir = os.path.abspath(cache_dir)

    classes = [
        d
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and not d.startswith(".")
        and d != os.path.basename(cache_dir)
    ]
    classes.sort()
    print(f"Found classes: {classes}")

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Feature cache enabled at: {cache_dir}")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    def split_pcap_files(file_list, seed=42, val_ratio=0.15, test_ratio=0.15, min_eval_files=2):
        """按文件级 7:1.5:1.5 拆分训练/验证/测试集（分层抽样）。

        旧实现是把文件列表洗牌后按 70/15/15 直接切段。当某个类别的 pcap 很少时
        （Benign 14 个、Malware 10 个），验证集只剩 1 个文件、测试集只剩 2 个文件，
        于是出现"验证集准确率 99.9%、测试集 87.8%"这种不可复现的落差 ——
        指标实际由一两个文件决定，模型选择也跟着失真。

        现在的做法：
        1. 每个类别的验证/测试集各至少保留 min_eval_files 个 pcap（文件总数不足时自动放宽）；
        2. 先按文件体积降序排列，再切成 (n_val + n_test) 个连续桶、逐桶随机取一个文件 ——
           单个 pcap 的会话数可能相差几个数量级，这样能让 eval 集覆盖大中小各类文件，
           而不是被随机洗牌后的某个超大文件主导；
        3. 随机过程由 seed 固定，划分结果可复现。
        """
        files = list(file_list)
        total = len(files)
        if total <= 1:
            return files, [], []
        if total == 2:
            return files[:1], files[1:], []

        rng = random.Random(seed)
        try:
            ordered = sorted(files, key=lambda p: os.path.getsize(p), reverse=True)
        except OSError:
            ordered = list(files)
            rng.shuffle(ordered)

        n_val = max(min_eval_files, int(round(total * val_ratio)))
        n_test = max(min_eval_files, int(round(total * test_ratio)))
        # 训练集至少留 1 个文件
        while n_val + n_test > total - 1:
            if n_val > 1 and n_val >= n_test:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

        n_eval = n_val + n_test
        eval_files = []
        if n_eval > 0:
            bucket_size = total / n_eval
            for i in range(n_eval):
                start = int(round(i * bucket_size))
                end = int(round((i + 1) * bucket_size))
                if end <= start:
                    end = min(start + 1, total)
                bucket = ordered[start:end]
                if bucket:
                    eval_files.append(rng.choice(bucket))

        # 极端比例下桶边界可能取到重复文件，去重后按体积补足到 n_eval 个
        eval_files = list(dict.fromkeys(eval_files))
        for candidate in ordered:
            if len(eval_files) >= n_eval:
                break
            if candidate not in eval_files:
                eval_files.append(candidate)

        rng.shuffle(eval_files)
        val_files = eval_files[:n_val]
        test_files = eval_files[n_val:n_val + n_test]
        eval_set = set(val_files) | set(test_files)
        train_files = [f for f in files if f not in eval_set]
        return train_files, val_files, test_files

    cache_stats = {"hit": 0, "miss": 0, "rebuilt": 0}

    def ingest_pcap_files(pcap_files, target_data, target_labels, label, split_name, cls_name):
        """从 PCAP 文件提取会话图片，支持缓存读/写"""
        for pcap_file in tqdm(pcap_files, desc=f"{split_name}:{cls_name}", leave=False):
            cached_images = None
            cache_path = None
            file_name = os.path.basename(pcap_file)
            tqdm.write(f"[{split_name}:{cls_name}] 处理 {file_name}")

            if use_cache:
                cache_key = _build_cache_key(pcap_file)
                cache_path = os.path.join(cache_dir, f"{cache_key}.npz")
                if not rebuild_cache and os.path.exists(cache_path):
                    try:
                        cached_images = _load_cached_sessions(cache_path)
                        cache_stats["hit"] += 1
                        tqdm.write(f"[{split_name}:{cls_name}] 缓存命中 {file_name}")
                    except Exception as e:
                        print(f"Corrupted cache skipped for {pcap_file}: {e}")
                        cached_images = None

            if cached_images is None:
                tqdm.write(f"[{split_name}:{cls_name}] 正在解析 {file_name}，这一步可能比较慢")
                extracted_images = []
                try:
                    result = extractor.pcap_to_sessions(pcap_file)
                    if isinstance(result, tuple):
                        sessions = result[0]
                    else:
                        sessions = result
                except Exception as e:
                    print(f"Skipping {pcap_file} due to error: {e}")
                    continue

                for session_bytes in sessions.values():
                    try:
                        extracted_images.append(extractor.process_session(session_bytes))
                    except Exception as e:
                        print(f"Error processing session in {pcap_file}: {e}")

                if use_cache and cache_path is not None:
                    try:
                        _save_cached_sessions(cache_path, extracted_images, compress=cache_compress)
                        if rebuild_cache:
                            cache_stats["rebuilt"] += 1
                        else:
                            cache_stats["miss"] += 1
                    except Exception as e:
                        print(f"Failed to write cache for {pcap_file}: {e}")

                if extracted_images:
                    cached_images = np.stack(extracted_images, axis=0).astype(np.uint8)
                else:
                    cached_images = np.empty((0, 28, 28), dtype=np.uint8)

            for img in cached_images:
                target_data.append(img)
                target_labels.append(label)

    split_data = {"train": [], "val": [], "test": []}
    split_labels = {"train": [], "val": [], "test": []}
    # split -> 类别 -> 文件名，用于划分概览打印（数据划分质量直接决定指标可信度）
    split_file_names = {"train": {}, "val": {}, "test": {}}

    for cls_name in classes:
        cls_dir = os.path.join(data_root, cls_name)
        pcap_files = glob.glob(os.path.join(cls_dir, "**", "*.pcap"), recursive=True)

        label = class_to_idx[cls_name]
        print(f"Processing class '{cls_name}' from {cls_dir} ...")
        train_files, val_files, test_files = split_pcap_files(pcap_files)
        split_file_names["train"][cls_name] = [os.path.basename(f) for f in train_files]
        split_file_names["val"][cls_name] = [os.path.basename(f) for f in val_files]
        split_file_names["test"][cls_name] = [os.path.basename(f) for f in test_files]
        ingest_pcap_files(train_files, split_data["train"], split_labels["train"], label, "train", cls_name)
        ingest_pcap_files(val_files, split_data["val"], split_labels["val"], label, "val", cls_name)
        ingest_pcap_files(test_files, split_data["test"], split_labels["test"], label, "test", cls_name)

    if not split_data["train"] and not split_data["val"] and not split_data["test"]:
        print("No data found.")
        return None, None, None, {}

    # 数据增强只作用于训练集：验证/测试集必须保持原始分布，否则指标不可比。
    eval_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = eval_transform
    if augment:
        preset = AUGMENT_PRESETS.get(str(augment_strength).lower(), AUGMENT_PRESETS['medium'])
        train_transform = transforms.Compose([transforms.ToTensor(), FlowSessionAugment(**preset)])
        print(f"数据增强已启用（强度 {augment_strength}）：{preset}")

    train_dataset = TrafficDataset(split_data["train"], split_labels["train"], transform=train_transform)
    val_dataset = TrafficDataset(split_data["val"], split_labels["val"], transform=eval_transform)
    test_dataset = TrafficDataset(split_data["test"], split_labels["test"], transform=eval_transform)

    use_cuda = torch.cuda.is_available()
    # DataLoader 工作进程数：默认 0，在主进程里加载。
    # 原因一：受限/沙箱环境下 multiprocessing 的命名管道会被系统拒绝
    #   （PermissionError: [WinError 5] 拒绝访问），num_workers>0 会在第一个 epoch 直接崩溃。
    # 原因二：28x28 的张量极小，主进程加载通常比多进程调度更快。
    # 确有多进程需求时用环境变量显式开启：DLIDS_NUM_WORKERS=4
    num_workers = 0
    _env_workers = os.environ.get("DLIDS_NUM_WORKERS")
    if _env_workers not in (None, ""):
        try:
            num_workers = max(0, int(_env_workers))
        except (TypeError, ValueError):
            num_workers = 0
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs) if len(train_dataset) > 0 else None
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs) if len(test_dataset) > 0 else None

    print(f"Data splitted by pcap. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 划分自检：验证/测试集的会话数与类别构成直接决定指标可不可信。
    # 旧划分下 Malware 的 val/test 可能只剩 1~2 个 pcap，就会报出"val 99.9% / test 87.8%"。
    print("\n--- 数据划分概览 ---")
    for split_name in ("train", "val", "test"):
        counts = split_file_names[split_name]
        detail = "，".join(f"{cls}={len(counts.get(cls, []))} 个" for cls in classes)
        print(f"[{split_name}] pcap 文件：{detail}")

    for split_name in ("train", "val", "test"):
        labels = split_labels[split_name]
        sample_total = len(labels)
        if sample_total == 0:
            print(f"[{split_name}] 会话样本：空（该 split 没有样本，对应指标不可用）")
            continue
        label_counts = Counter(labels)
        detail = "，".join(
            f"{cls}={label_counts.get(idx, 0)}（{label_counts.get(idx, 0) / sample_total * 100:.1f}%）"
            for idx, cls in enumerate(classes)
        )
        print(f"[{split_name}] 会话样本：共 {sample_total}，{detail}")

    for split_name in ("val", "test"):
        print(f"[{split_name}] 具体文件：")
        for cls in classes:
            for name in split_file_names[split_name].get(cls, []):
                print(f"    {cls}/{name}")

    if use_cache:
        print(
            "Feature cache stats -> "
            f"hit: {cache_stats['hit']}, miss: {cache_stats['miss']}, rebuilt: {cache_stats['rebuilt']}"
        )
        if cache_stats["hit"] == 0 and cache_stats["miss"] > 0:
            print(
                "[提示] 本次没有任何缓存命中，所有 pcap 都要重新解析（GB 级数据会非常慢）。\n"
                "        缓存键 = 缓存版本 + 文件绝对路径 + 内容指纹（首/中/尾采样 + 大小）。\n"
                "        路径改动或缓存版本升级都会导致失效；确认数据没变可以保留缓存目录等待重建。"
            )

    return train_loader, val_loader, test_loader, class_to_idx
