import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from pathlib import Path
from typing import List, Tuple

from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import numpy as np

from src.models.cnn_bilstm import CNN_BiLSTM
from src.preprocessing.feature_extraction import FeatureExtractor


class PcapFeatureDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], seq_len: int = 100):
        self.samples = samples
        self.extractor = FeatureExtractor(target_seq_len=seq_len)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pcap_path, label = self.samples[idx]
        features = self.extractor.extract_features(pcap_path)
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def collect_samples(benign_dir: str, malware_dir: str) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []

    for base_dir, label in [(benign_dir, 0), (malware_dir, 1)]:
        if not base_dir:
            continue
        base = Path(base_dir)
        if not base.exists():
            continue
        for pattern in ("*.pcap", "*.pcapng"):
            for p in base.rglob(pattern):
                samples.append((str(p), label))

    return samples


def stratified_split_indices(samples: List[Tuple[str, int]], val_ratio: float, seed: int):
    label_to_indices = {0: [], 1: []}
    for idx, (_, label) in enumerate(samples):
        if label in label_to_indices:
            label_to_indices[label].append(idx)

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for label, indices in label_to_indices.items():
        if not indices:
            continue
        indices = np.array(indices)
        rng.shuffle(indices)

        val_count = max(1, int(len(indices) * val_ratio)) if len(indices) > 1 else 0
        if val_count >= len(indices):
            val_count = len(indices) - 1

        if val_count > 0:
            val_indices.extend(indices[:val_count].tolist())
            train_indices.extend(indices[val_count:].tolist())
        else:
            train_indices.extend(indices.tolist())

    if not train_indices:
        train_indices = val_indices[:-1]
        val_indices = val_indices[-1:]

    return train_indices, val_indices


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += (preds == labels).sum().item()
            total += batch_size

    if total == 0:
        return 0.0, 0.0

    return running_loss / total, running_corrects / total


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda', save_path='checkpoints/cnn_bilstm_best.pth'):
    model.to(device)
    best_acc = 0.0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc='train'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            train_corrects += (preds == labels).sum().item()
            train_total += batch_size

        epoch_train_loss = train_loss / max(train_total, 1)
        epoch_train_acc = train_corrects / max(train_total, 1)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f'train_loss={epoch_train_loss:.4f}, train_acc={epoch_train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'Saved best model to: {save_path}')

    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train small 1D-CNN + BiLSTM on pcap files')
    parser.add_argument('--benign-dir', type=str, default='data_USTC-TK2016/1_Pcap/USTC-TFC2016-master/Benign')
    parser.add_argument('--malware-dir', type=str, default='data_USTC-TK2016/1_Pcap/USTC-TFC2016-master/Malware')
    parser.add_argument('--seq-len', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--save-path', type=str, default='checkpoints/cnn_bilstm_best.pth')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input-dim', type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    samples = collect_samples(args.benign_dir, args.malware_dir)
    if len(samples) < 2:
        raise RuntimeError('Not enough pcap samples found. Check benign/malware paths.')

    benign_count = sum(1 for _, label in samples if label == 0)
    malware_count = sum(1 for _, label in samples if label == 1)
    print(f'Total samples: {len(samples)} (benign={benign_count}, malware={malware_count})')
    if benign_count == 0 or malware_count == 0:
        raise RuntimeError('Both benign and malware samples are required for training.')

    dataset = PcapFeatureDataset(samples=samples, seq_len=args.seq_len)
    train_indices, val_indices = stratified_split_indices(samples, args.val_ratio, args.seed)
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    print(f'Train/Val split: {len(train_set)}/{len(val_set)}')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN_BiLSTM(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_classes=2)

    train_labels = [samples[i][1] for i in train_indices]
    train_benign = sum(1 for x in train_labels if x == 0)
    train_malware = sum(1 for x in train_labels if x == 1)
    class_weights = torch.tensor(
        [1.0 / max(train_benign, 1), 1.0 / max(train_malware, 1)],
        dtype=torch.float32,
        device=device,
    )
    class_weights = class_weights / class_weights.sum() * 2.0
    print(f'Class weights: benign={class_weights[0].item():.4f}, malware={class_weights[1].item():.4f}')

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        save_path=args.save_path,
    )

    print('Training finished.')
