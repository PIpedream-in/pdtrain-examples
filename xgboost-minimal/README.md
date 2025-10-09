# XGBoost (Minimal)

A minimal XGBoost training example for SageMaker Script Mode via pdtrain. It trains a small regressor on CSV data (expects numeric columns and optional `target`) or generates synthetic data. Saves a model and `metrics.json` to `SM_MODEL_DIR`.

- Framework: `xgboost`
- Entry point: `train.py`
- Input: CSV(s) in `dataset/` channel; uses all numeric columns, `target` if present
- Output: `model.json`, `metrics.json`

## Local Run

```bash
cd xgboost-minimal
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --data_dir ./data --model_dir ./output --num_round 50 --max_depth 4 --eta 0.1
```

Note (macOS): XGBoost CPU wheels use OpenMP. If you see an error about `libomp.dylib`, install it via Homebrew: `brew install libomp`.

## Run with pdtrain CLI

```bash
# Install and configure CLI (once)
pip install pdtrain
pdtrain configure

# Upload bundle (exclude local data from code bundle)
cd xgboost-minimal
pdtrain bundle upload . --name "xgboost-minimal" \
  --exclude "data" \
  --wait

# Upload dataset (CSV or directory). Test data is uploaded as a dataset, not bundled with code.
pdtrain dataset upload ./data --name "xgb-reg-data" --wait

# Create run (framework mode)
pdtrain run create \
  --bundle <bundle-id> \
  --dataset <dataset-id> \
  --framework xgboost \
  --framework-version 1.7-1 \
  --entry train.py \
  --hyperparameter num_round=50 \
  --hyperparameter max_depth=4 \
  --hyperparameter eta=0.1 \
  --hyperparameter subsample=0.8 \
  --submit --wait

# Watch and logs
pdtrain run watch <run-id>
pdtrain logs <run-id> --follow
```

## Notes
- Auto-extracts a single `.tar`, `.tgz`, or `.tar.gz` if provided as dataset.
- If no `target` column found, uses the last numeric column as target; otherwise generates synthetic data.
- Ignores unknown hyperparameters safely.
- The SageMaker XGBoost image expects Script Mode when `--entry train.py` is provided.
