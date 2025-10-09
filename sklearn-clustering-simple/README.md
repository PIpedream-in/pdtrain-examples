# Scikit-learn Clustering (Simple)

A minimal scikit-learn clustering example for SageMaker Script Mode via pdtrain. Uses KMeans on CSV data or synthetic blobs if no dataset is found. Saves `model.pkl` and `metrics.json` to `SM_MODEL_DIR`.

- Framework: `sklearn` (Script Mode)
- Entry point: `train.py`
- Input: CSV(s) in `dataset/` channel; uses all numeric columns
- Output: `model.pkl`, `metrics.json`

## Local Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --data_dir ./data --model_dir ./output --n_clusters 3 --max_iter 100
```

## Run with pdtrain CLI

```bash
# Install and configure CLI (once)
pip install pdtrain
pdtrain configure

# Upload bundle (exclude local data from code bundle)
cd sklearn-clustering-simple
pdtrain bundle upload . --name "sklearn-clustering-simple" \
  --exclude "data" \
  --wait

# Upload dataset (CSV or directory). Test data is uploaded as a dataset, not bundled with code.
pdtrain dataset upload ./data --name "skl-clustering-data" --wait

# Create run (framework mode)
pdtrain run create \
  --bundle <bundle_id> \
  --dataset <dataset_id> \
  --framework sklearn \
  --framework-version 1.2-1 \
  --python-version py3 \
  --entry train.py \
  --hyperparameter n_clusters=3 \
  --hyperparameter max_iter=100 \
  --submit --wait

# Watch and fetch logs
pdtrain logs <run-id> --follow
```

## Notes
- Auto-extracts a single `.tar`, `.tgz`, or `.tar.gz` if provided as dataset.
- Falls back to synthetic data with `make_blobs` when no CSV found.
- Ignores unknown hyperparameters safely.

## Important: SageMaker Dependencies

⚠️ **DO NOT** specify pandas, numpy, or scikit-learn in `requirements.txt` when using SageMaker Script Mode.

The SageMaker scikit-learn container has pre-installed versions that are tested together. Adding newer versions in requirements.txt causes numpy/pandas incompatibility errors.

For local development with different versions, use a separate `requirements-dev.txt`.

See [SAGEMAKER_DEPENDENCIES.md](./SAGEMAKER_DEPENDENCIES.md) for details.
