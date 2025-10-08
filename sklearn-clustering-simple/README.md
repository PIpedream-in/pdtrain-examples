# Scikit-learn Clustering (Simple)

A minimal scikit-learn clustering example for SageMaker Script Mode via pdtrain. Uses KMeans on CSV data or synthetic blobs if no dataset is found. Saves `model.pkl` and `metrics.json` to `SM_MODEL_DIR`.

- Framework: `sklearn` (Script Mode)
- Entry point: `train.py`
- Input: CSV(s) in `dataset/` channel; uses all numeric columns
- Output: `model.pkl`, `metrics.json`

## Local Run

```bash
cd orchestrator-api/examples/sklearn-clustering-simple
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --data_dir ./data --model_dir ./output --n_clusters 3 --max_iter 100
```

## Run with pdtrain CLI

```bash
# Install and configure CLI (once)
cd /Users/anish/git/codex-api/pdtrain
pip install -e .
pdtrain configure

# Upload bundle (exclude local data from code bundle)
cd /Users/anish/git/codex-api/orchestrator-api/examples/sklearn-clustering-simple
pdtrain bundle upload . --name "sklearn-clustering-simple" \
  --exclude "data" \
  --wait

# Upload dataset (CSV or directory). Test data is uploaded as a dataset, not bundled with code.
pdtrain dataset upload ./data --name "skl-clustering-data" --wait

# Create run (framework mode)
pdtrain run create \
  --bundle <bundle-id> \
  --dataset <dataset-id> \
  --framework sklearn \
  --framework-version py3 \
  --python-version py3 \
  --entry train.py \
  --hyperparameter n_clusters=3 \
  --hyperparameter max_iter=100

# Watch and fetch logs
pdtrain run watch <run-id>
pdtrain logs <run-id> --follow
```

## Notes
- Auto-extracts a single `.tar`, `.tgz`, or `.tar.gz` if provided as dataset.
- Falls back to synthetic data with `make_blobs` when no CSV found.
- Ignores unknown hyperparameters safely.
