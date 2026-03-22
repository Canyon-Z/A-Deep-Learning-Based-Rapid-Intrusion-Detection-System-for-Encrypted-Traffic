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

# Add project root to sys.path to allow running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.preprocessing.feature_extraction import FeatureExtractor

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
        img_array = self.data[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image (L = grayscale mode)
        img = Image.fromarray(img_array, mode='L')
        
        if self.transform:
            img_tensor = self.transform(img)
        else:
            # Basic conversion if no transform is provided
            # Converts [0, 255] to [0.0, 1.0] and adds channel dimension -> (1, 28, 28)
            img_tensor = torch.from_numpy(img_array).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
        return img_tensor, torch.tensor(label, dtype=torch.long)

def get_dataloaders(data_root, batch_size=32, truncate_len=784):
    """
    Phase 4: Dataset Splitting
    Phase 5: PyTorch DataLoader Encapsulation
    
    Reads data from folders, preprocesses it, splits it, and returns DataLoaders.
    Assumes directory structure: data_root/class_name/*.pcap
    """
    extractor = FeatureExtractor(truncate_len=truncate_len)
    all_data = []
    all_labels = []
    
    if not os.path.exists(data_root):
        print(f"Data root {data_root} does not exist.")
        # Return empty loaders or handle error better in production
        return None, None, {}

    # Identify classes from subdirectories
    classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    classes.sort() # Ensure consistent ordering
    print(f"Found classes: {classes}")
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    for cls_name in classes:
        cls_dir = os.path.join(data_root, cls_name)
        pcap_files = glob.glob(os.path.join(cls_dir, '*.pcap'))
        
        label = class_to_idx[cls_name]
        print(f"Processing class '{cls_name}' from {cls_dir}...")
        
        for pcap_file in tqdm(pcap_files, desc=f"Loading {cls_name}"):
            try:
                result = extractor.pcap_to_sessions(pcap_file)
                if isinstance(result, tuple):
                    sessions = result[0]
                else:
                    sessions = result
            except Exception as e:
                print(f"Skipping {pcap_file} due to error: {e}")
                continue

            # Iterate through sessions and process them
            for session_bytes in sessions.values():
                try:
                    img = extractor.process_session(session_bytes)
                    all_data.append(img)
                    all_labels.append(label)
                except Exception as e:
                    print(f"Error processing session in {pcap_file}: {e}")

    if not all_data:
        print("No data found.")
        return None, None, {}

    # Define transforms
    # ToTensor() converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
    # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])
    
    dataset = TrafficDataset(all_data, all_labels, transform=transform)
    
    # Phase 4: Dataset Splitting (Train: 70%, Val: 15%, Test: 15%)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Phase 5: DataLoader Encapsulation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Data splitted. Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    return train_loader, val_loader, test_loader, class_to_idx
