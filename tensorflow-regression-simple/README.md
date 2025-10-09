# TensorFlow Regression (Simple)

A minimal TensorFlow Keras regression example for SageMaker Script Mode via pdtrain. Trains a tiny MLP on CSV data (expects numeric columns and optional `target`) or generates synthetic data. Saves a SavedModel and `metrics.json` to `SM_MODEL_DIR`.

- Framework: `tensorflow`
- Entry point: `train.py`
- Input: CSV(s) in `dataset/` channel; uses all numeric columns, `target` if present
- Output: `saved_model/`, `metrics.json`

## Local Run

```bash
cd tensorflow-regression-simple
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --data_dir ./data --model_dir ./output --epochs 3 --batch_size 32
```

## Run with pdtrain CLI

```bash
# Install and configure CLI (once)
pip install pdtrain
pdtrain configure

# Upload bundle (exclude local data from code bundle)
cd tensorflow-regression-simple
pdtrain bundle upload . --name "tf-regression-simple" \
  --exclude "data" \
  --wait

# Upload dataset (CSV or directory). Test data is uploaded as a dataset, not bundled with code.
pdtrain dataset upload ./data --name "tf-reg-data" --wait

# Create run (framework mode)
pdtrain run create \
  --bundle <bundle-id> \
  --dataset <dataset-id> \
  --framework tensorflow \
  --framework-version 2.13.0 \
  --python-version py310 \
  --entry train.py \
  --hyperparameter epochs=3 \
  --hyperparameter batch_size=32 \
  --hyperparameter learning_rate=0.001 \
  --submit --wait

# Watch and logs
pdtrain run watch <run-id>
pdtrain logs <run-id> --follow
```

## Notes
- Auto-extracts a single `.tar`, `.tgz`, or `.tar.gz` if provided as dataset.
- If no `target` column found, uses the last numeric column as target; otherwise generates synthetic data.
- Ignores unknown hyperparameters safely.
