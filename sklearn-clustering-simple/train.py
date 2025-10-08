#!/usr/bin/env python3
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import tarfile


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sklearn-clustering-simple")


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


def load_or_synthesize(data_dir: str) -> Tuple[np.ndarray, list]:
    csv_path = find_first_csv(data_dir)
    if csv_path:
        df = pd.read_csv(csv_path)
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            raise RuntimeError("CSV has no numeric columns for clustering")
        X = numeric.to_numpy()
        cols = list(numeric.columns)
        logger.info(f"Loaded CSV {csv_path}: shape={X.shape}, numeric_cols={cols}")
        return X, cols
    # Synthetic data
    logger.warning("No CSV found; generating synthetic blobs")
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=3, n_features=4, random_state=42)
    return X, [f"f{i}" for i in range(X.shape[1])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_clusters", type=int, default=3)
    p.add_argument("--max_iter", type=int, default=300)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--data_dir", type=str, default="/opt/ml/input/data/dataset")
    p.add_argument("--model_dir", type=str, default="/opt/ml/model")
    args, extra = p.parse_known_args()
    if extra:
        logger.warning(f"Ignoring unknown arguments: {extra}")

    model_dir = os.environ.get("SM_MODEL_DIR", args.model_dir)
    data_dir = resolve_data_dir(args.data_dir)
    data_dir = maybe_extract_single_archive(data_dir)

    X, cols = load_or_synthesize(data_dir)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=args.n_clusters, max_iter=args.max_iter, n_init=10, random_state=args.random_state)
    km.fit(Xs)

    inertia = float(km.inertia_)
    logger.info(f"Training complete: n_samples={len(X)}, n_clusters={args.n_clusters}, inertia={inertia:.4f}")

    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, out / "model.pkl")
    joblib.dump(scaler, out / "scaler.pkl")
    (out / "metrics.json").write_text(json.dumps({
        "inertia": inertia,
        "n_clusters": args.n_clusters,
        "max_iter": args.max_iter,
        "features": cols,
        "used_synthetic": not bool(find_first_csv(data_dir)),
    }, indent=2))
    logger.info(f"Saved artifacts to: {out}")


if __name__ == "__main__":
    main()

