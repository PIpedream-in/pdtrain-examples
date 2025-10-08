#!/usr/bin/env python3
"""
Minimal PyTorch training script for Script Mode.

Features:
- Reads dataset from SM_CHANNEL_DATASET (SageMaker) or --data_dir
- Saves artifacts to SM_MODEL_DIR or --model_dir
- Works with pdtrain in framework (script) mode
- Falls back to synthetic data when no dataset is available

Usage (local):
  python train.py --data_dir ../test_data --model_dir ./output --epochs 2
"""

import argparse
import json
import logging
import os
from pathlib import Path
import tarfile
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pytorch-script-minimal")


class SimpleImageFolder(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        if not self.root.exists():
            raise FileNotFoundError(f"Data dir not found: {self.root}")
        for cdir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            idx = len(self.classes)
            self.class_to_idx[cdir.name] = idx
            self.classes.append(cdir.name)
            for f in cdir.iterdir():
                if f.suffix.lower() in exts:
                    self.samples.append((f, idx))
        if not self.samples:
            raise RuntimeError(f"No images found under {self.root} (class-per-subfolder)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        fp, y = self.samples[i]
        img = Image.open(fp).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def resolve_data_dir(default: str) -> str:
    # Prefer explicit SM env when present
    sm_dataset = os.environ.get("SM_CHANNEL_DATASET")
    if sm_dataset and os.path.isdir(sm_dataset):
        logger.info(f"Using SM_CHANNEL_DATASET: {sm_dataset}")
        return sm_dataset
    # Generic channels listing
    channels = os.environ.get("SM_CHANNELS")
    if channels:
        chans = [c for c in channels.split(",") if c]
        for cand in ("dataset", "input0"):
            if cand in chans:
                p = f"/opt/ml/input/data/{cand}"
                if os.path.isdir(p):
                    logger.info(f"Using SageMaker channel: {p}")
                    return p
    return default


def maybe_extract_single_archive(data_dir: str) -> str:
    """If data_dir contains a single tar/tgz/tar.gz archive, extract it in place.
    Returns the potentially updated data_dir (e.g., subfolder after extraction)."""
    if not os.path.isdir(data_dir):
        return data_dir
    try:
        entries = os.listdir(data_dir)
        if len(entries) == 1:
            item = entries[0]
            path = os.path.join(data_dir, item)
            if os.path.isfile(path) and (item.endswith((".tar.gz", ".tgz", ".tar"))):
                logger.info(f"Found archive in dataset channel: {path}; extracting...")
                mode = "r:gz" if item.endswith((".tar.gz", ".tgz")) else "r:"
                with tarfile.open(path, mode) as tar:
                    tar.extractall(path=data_dir)
                try:
                    os.remove(path)
                except Exception:
                    pass
                # If a single directory was created, descend into it
                remaining = [e for e in os.listdir(data_dir)]
                if len(remaining) == 1:
                    new_root = os.path.join(data_dir, remaining[0])
                    if os.path.isdir(new_root):
                        logger.info(f"Updated dataset path to extracted folder: {new_root}")
                        return new_root
    except Exception as e:
        logger.warning(f"Dataset auto-extract skipped due to error: {e}")
    return data_dir


def load_data(data_dir: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, list]:
    # Auto-extract if channel only contains a single archive
    data_dir = maybe_extract_single_archive(data_dir)
    transform_train = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    ds = SimpleImageFolder(data_dir, transform=transform_train)
    logger.info(f"Loaded dataset from {data_dir}: {len(ds)} samples, classes={ds.classes}")
    n_train = max(1, int(0.8 * len(ds)))
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, ds.classes


def synthetic_data(num_classes=3, n=200) -> Tuple[DataLoader, DataLoader, list]:
    x = torch.rand(n, 3, 224, 224)
    y = torch.randint(0, num_classes, (n,))
    ds = torch.utils.data.TensorDataset(x, y)
    n_train = int(0.8 * n)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n - n_train])
    return (
        DataLoader(train_ds, batch_size=16, shuffle=True),
        DataLoader(val_ds, batch_size=16, shuffle=False),
        [f"class_{i}" for i in range(num_classes)],
    )


def train_one_epoch(model, dl, optim_, crit, device):
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        optim_.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        optim_.step()
        run_loss += loss.item()
        pred = out.argmax(1)
        total += yb.size(0)
        correct += (pred == yb).sum().item()
    return run_loss / max(1, len(dl)), (100.0 * correct / max(1, total))


@torch.no_grad()
def validate(model, dl, crit, device):
    model.eval()
    run_loss, correct, total = 0.0, 0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = crit(out, yb)
        run_loss += loss.item()
        pred = out.argmax(1)
        total += yb.size(0)
        correct += (pred == yb).sum().item()
    return run_loss / max(1, len(dl)), (100.0 * correct / max(1, total))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--data_dir", type=str, default="/opt/ml/input/data/dataset")
    p.add_argument("--model_dir", type=str, default="/opt/ml/model")
    args, extra = p.parse_known_args()
    if extra:
        logger.warning(f"Ignoring unknown arguments: {extra}")

    # Respect SageMaker env dirs when present
    model_dir = os.environ.get("SM_MODEL_DIR", args.model_dir)
    data_dir = resolve_data_dir(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Try real data; fallback to synthetic
    use_synth = False
    try:
        train_dl, val_dl, classes = load_data(data_dir, args.batch_size, args.num_workers)
        num_classes = len(classes)
    except Exception as e:
        logger.warning(f"Falling back to synthetic data: {e}")
        train_dl, val_dl, classes = synthetic_data()
        num_classes = len(classes)
        use_synth = True

    model = TinyCNN(num_classes).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, opt, crit, device)
        va_loss, va_acc = validate(model, val_dl, crit, device)
        best_val_acc = max(best_val_acc, va_acc)
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.2f}%"
        )

    # Save artifacts
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pth")

    (out_dir / "model_info.json").write_text(
        json.dumps({
            "model": "TinyCNN",
            "num_classes": num_classes,
            "classes": classes,
            "input_shape": [3, 224, 224],
            "framework": "pytorch",
        }, indent=2)
    )

    (out_dir / "metrics.json").write_text(
        json.dumps({
            "best_val_accuracy": best_val_acc,
            "used_synthetic_data": use_synth,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
        }, indent=2)
    )

    logger.info(f"Saved artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
