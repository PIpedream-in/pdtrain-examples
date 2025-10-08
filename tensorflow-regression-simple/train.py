#!/usr/bin/env python3
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tarfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tf-regression-simple")


def maybe_extract_single_archive(data_dir: str) -> str:
    if not os.path.isdir(data_dir):
        return data_dir
    try:
        entries = os.listdir(data_dir)
        if len(entries) == 1:
            item = entries[0]
            path = os.path.join(data_dir, item)
            if os.path.isfile(path) and item.endswith((".tar.gz", ".tgz", ".tar")):
                logger.info(f"Found archive in dataset channel: {path}; extracting...")
                mode = "r:gz" if item.endswith((".tar.gz", ".tgz")) else "r:"
                with tarfile.open(path, mode) as tar:
                    tar.extractall(path=data_dir)
                try:
                    os.remove(path)
                except Exception:
                    pass
                remain = [e for e in os.listdir(data_dir)]
                if len(remain) == 1:
                    new_root = os.path.join(data_dir, remain[0])
                    if os.path.isdir(new_root):
                        logger.info(f"Updated dataset path to extracted folder: {new_root}")
                        return new_root
    except Exception as e:
        logger.warning(f"Dataset auto-extract skipped due to error: {e}")
    return data_dir


def resolve_data_dir(default: str) -> str:
    sm_dataset = os.environ.get("SM_CHANNEL_DATASET")
    if sm_dataset and os.path.isdir(sm_dataset):
        return sm_dataset
    channels = os.environ.get("SM_CHANNELS")
    if channels:
        chans = [c for c in channels.split(",") if c]
        for cand in ("dataset", "input0"):
            p = f"/opt/ml/input/data/{cand}"
            if cand in chans and os.path.isdir(p):
                return p
    return default


def find_first_csv(root: str) -> str:
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".csv"):
                return os.path.join(dirpath, f)
    return ""


def load_regression_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, list]:
    csv_path = find_first_csv(data_dir)
    if csv_path:
        df = pd.read_csv(csv_path)
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            raise RuntimeError("CSV has no numeric columns for regression")
        cols = list(numeric.columns)
        if "target" in numeric.columns:
            y = numeric["target"].to_numpy().astype(np.float32)
            X = numeric.drop(columns=["target"]).to_numpy().astype(np.float32)
        else:
            # Use last numeric column as target
            y = numeric.iloc[:, -1].to_numpy().astype(np.float32)
            X = numeric.iloc[:, :-1].to_numpy().astype(np.float32)
        logger.info(f"Loaded CSV {csv_path}: X={X.shape}, y={y.shape}")
        return X, y, cols
    # Synthetic if not found
    logger.warning("No CSV found; generating synthetic regression data")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500, 8)).astype(np.float32)
    w = rng.normal(size=(8, 1)).astype(np.float32)
    y = (X @ w).squeeze() + 0.1 * rng.normal(size=(500,)).astype(np.float32)
    return X, y, [f"f{i}" for i in range(X.shape[1])] + ["target"]


def build_model(input_dim: int, lr: float) -> tf.keras.Model:
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--data_dir", type=str, default="/opt/ml/input/data/dataset")
    p.add_argument("--model_dir", type=str, default="/opt/ml/model")
    args, extra = p.parse_known_args()
    if extra:
        logger.warning(f"Ignoring unknown arguments: {extra}")

    model_dir = os.environ.get("SM_MODEL_DIR", args.model_dir)
    data_dir = resolve_data_dir(args.data_dir)
    data_dir = maybe_extract_single_archive(data_dir)

    X, y, cols = load_regression_data(data_dir)
    model = build_model(input_dim=X.shape[1], lr=args.learning_rate)
    history = model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, verbose=2, validation_split=0.2)

    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Keras 3+: use `export` for SavedModel format (for TF Serving/TFLite)
    model.export(str(out / "saved_model"))
    final_loss = float(history.history["loss"][-1])
    val_loss = float(history.history.get("val_loss", [final_loss])[-1])
    (out / "metrics.json").write_text(json.dumps({
        "final_loss": final_loss,
        "val_loss": val_loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "used_synthetic": not bool(find_first_csv(data_dir)),
        "feature_columns": cols,
    }, indent=2))
    logger.info(f"Saved artifacts to: {out}")


if __name__ == "__main__":
    main()
