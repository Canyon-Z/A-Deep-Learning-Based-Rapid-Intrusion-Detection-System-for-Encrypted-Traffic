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

# Add project root to sys.path to allow running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.preprocessing.feature_extraction import FeatureExtractor


FEATURE_CACHE_VERSION = 2


def _build_cache_key(pcap_file, truncate_len):
    abs_path = os.path.abspath(pcap_file)
    try:
        stat = os.stat(abs_path)
        raw = f"{FEATURE_CACHE_VERSION}|{abs_path}|{truncate_len}|{stat.st_mtime_ns}|{stat.st_size}"
    except OSError:
        raw = f"{FEATURE_CACHE_VERSION}|{abs_path}|{truncate_len}|missing"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cached_sessions(cache_path):
    with np.load(cache_path, allow_pickle=False) as data:
        sessions = data["sessions"]
    if sessions.ndim == 2:
        sessions = np.expand_dims(sessions, axis=0)
    return sessions


def _save_cached_sessions(cache_path, sessions, compress=False):
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
    def __init__(self, data_list, labels_list, transform=None):
        """
        Custom PyTorch Dataset for traffic images.
        data_list: List of numpy arrays (28x28)
        labels_list: List of integer labels
        """
        self.data = data_list
        self.labels = labels_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Phase 3: Byte to Tensor Conversion (Tensor creation in Dataset)
        item = self.data[idx] 
        label = self.labels[idx] 
        item = self.data[idx] 
        label = self.labels[idx] 
        
        # Check if item is a file path or numpy array 
        if isinstance(item, str): 
            # Load image from file path using PIL 
            img = Image.open(item).convert('L')  # Convert to grayscale 
            if self.transform: 
                img_tensor = self.transform(img) 
            else: 
                # Basic conversion if no transform is provided 
                img_array = np.array(img) 
                img_tensor = torch.from_numpy(img_array).float() / 255.0 
                img_tensor = img_tensor.unsqueeze(0) 
        else: 
            # Handle numpy array 
            img_array = item 
            img = Image.fromarray(img_array, mode='L') 
            if self.transform: 
                img_tensor = self.transform(img) 
            else: 
                # Basic conversion if no transform is provided 
                img_tensor = torch.from_numpy(img_array).float() / 255.0 
                img_tensor = img_tensor.unsqueeze(0) 
        # Check if item is a file path or numpy array 
        if isinstance(item, str): 
            # Load image from file path using PIL 
            img = Image.open(item).convert('L')  # Convert to grayscale 
            if self.transform: 
                img_tensor = self.transform(img) 
            else: 
                # Basic conversion if no transform is provided 
                img_array = np.array(img) 
                img_tensor = torch.from_numpy(img_array).float() / 255.0 
                img_tensor = img_tensor.unsqueeze(0) 
        else: 
            # Handle numpy array 
            img_array = item 
            img = Image.fromarray(img_array, mode='L') 
            if self.transform: 
                img_tensor = self.transform(img) 
            else: 
                # Basic conversion if no transform is provided 
                img_tensor = torch.from_numpy(img_array).float() / 255.0 
                img_tensor = img_tensor.unsqueeze(0) 
            
        return img_tensor, torch.tensor(label, dtype=torch.long) 
    
    @staticmethod
    def load_data(data_dir): 
        """ 
        Load data from image files in the specified directory. 
        Reads images from data/processed/Png or data/USTC-TFC2016/4_Png 
        """ 
        if not os.path.exists(data_dir): 
            print(f"Data directory {data_dir} does not exist.") 
            return [], [] 
        
        data_list = [] 
        labels_list = [] 
        
        # Identify classes from subdirectories 
        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))] 
        classes.sort()  # Ensure consistent ordering 
        print(f"Found classes: {classes}") 
        
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)} 
        
        for cls_name in classes: 
            cls_dir = os.path.join(data_dir, cls_name) 
            img_files = glob.glob(os.path.join(cls_dir, '*.png')) + glob.glob(os.path.join(cls_dir, '*.jpg')) 
            
            label = class_to_idx[cls_name] 
            print(f"Processing class '{cls_name}' from {cls_dir}...") 
            
            for img_file in tqdm(img_files, desc=f"Loading {cls_name}"): 
                data_list.append(img_file) 
                labels_list.append(label) 
        
        print(f"Loaded {len(data_list)} images from {data_dir}") 
        return data_list, labels_list 


def get_dataloaders(
    data_root,
    batch_size=32,
    truncate_len=784,
    cache_dir=None,
    use_cache=True,
    rebuild_cache=False,
    cache_compress=False,
    mask_headers=True,
    mask_fill=0,
):
    """
    Phase 4: Dataset Splitting
    Phase 5: PyTorch DataLoader Encapsulation
    
    Reads data from folders, preprocesses it, splits it, and returns DataLoaders.
    Assumes directory structure: data_root/class_name/*.pcap
    """
    extractor = FeatureExtractor(
        truncate_len=truncate_len,
        mask_headers=mask_headers,
        mask_fill=mask_fill,
    )
    
    if not os.path.exists(data_root):
        print(f"Data root {data_root} does not exist.")
        # Return empty loaders or handle error better in production
        return None, None, None, {}

    if cache_dir is None:
        cache_dir = os.path.join(data_root, ".feature_cache")
    cache_dir = os.path.abspath(cache_dir)

    # Identify classes from real data subdirectories only.
    classes = [
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and not d.startswith('.')
        and d != os.path.basename(cache_dir)
    ]
    classes.sort() # Ensure consistent ordering
    print(f"Found classes: {classes}")

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Feature cache enabled at: {cache_dir}")
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    def split_pcap_files(file_list, seed=42):
        shuffled = list(file_list)
        random.Random(seed).shuffle(shuffled)
        total = len(shuffled)
        if total <= 1:
            return shuffled, [], []

        train_end = max(1, int(total * 0.7))
        val_end = train_end + int(total * 0.15)

        if total >= 3 and val_end >= total:
            val_end = total - 1
        if total >= 3 and val_end <= train_end:
            val_end = min(total - 1, train_end + 1)

        return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]

    cache_stats = {"hit": 0, "miss": 0, "rebuilt": 0}

    def ingest_pcap_files(pcap_files, target_data, target_labels, label, split_name, cls_name):
        for pcap_file in tqdm(pcap_files, desc=f"{split_name}:{cls_name}", leave=False):
            cached_images = None
            cache_path = None
            file_name = os.path.basename(pcap_file)
            tqdm.write(f"[{split_name}:{cls_name}] 处理 {file_name}")

            if use_cache:
                cache_key = _build_cache_key(pcap_file, truncate_len)
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
    
    for cls_name in classes:
        cls_dir = os.path.join(data_root, cls_name)
        # Search recursively under each class folder so both flat and nested layouts work.
        pcap_files = glob.glob(os.path.join(cls_dir, '**', '*.pcap'), recursive=True)
        
        label = class_to_idx[cls_name]
        print(f"Processing class '{cls_name}' from {cls_dir} ...")
        train_files, val_files, test_files = split_pcap_files(pcap_files)
        ingest_pcap_files(train_files, split_data['train'], split_labels['train'], label, 'train', cls_name)
        ingest_pcap_files(val_files, split_data['val'], split_labels['val'], label, 'val', cls_name)
        ingest_pcap_files(test_files, split_data['test'], split_labels['test'], label, 'test', cls_name)

    if not split_data['train'] and not split_data['val'] and not split_data['test']:
        print("No data found.")
        return None, None, None, {}

    # Define transforms
    # ToTensor() converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
    # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])
    
    train_dataset = TrafficDataset(split_data['train'], split_labels['train'], transform=transform)
    val_dataset = TrafficDataset(split_data['val'], split_labels['val'], transform=transform)
    test_dataset = TrafficDataset(split_data['test'], split_labels['test'], transform=transform)

    # Phase 5: DataLoader Encapsulation
    use_cuda = torch.cuda.is_available()
    num_workers = 2 if use_cuda else 0
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': use_cuda,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs) if len(train_dataset) > 0 else None
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs) if len(test_dataset) > 0 else None

    print(f"Data splitted by pcap. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    if use_cache:
        print(
            "Feature cache stats -> "
            f"hit: {cache_stats['hit']}, miss: {cache_stats['miss']}, rebuilt: {cache_stats['rebuilt']}"
        )
    
    return train_loader, val_loader, test_loader, class_to_idx
