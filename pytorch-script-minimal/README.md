# PyTorch Script Mode (Minimal)

A minimal PyTorch training example designed for script mode execution on SageMaker via the pdtrain CLI. It trains a tiny CNN on an image folder dataset (class-per-subfolder) or generates synthetic data if no dataset is provided. Artifacts are saved to `SM_MODEL_DIR` (`/opt/ml/model`).

- Framework: `pytorch`
- Entry point: `train.py`
- Inputs: Image dataset organized as `dataset/<class>/*.jpg|png|...`
- Outputs: `model.pth`, `model_info.json`, `metrics.json`

## Local Run

```bash
# From repo root
cd orchestrator-api/examples/pytorch-script-minimal

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Use example test data from the repo
python train.py \
  --data_dir ../test_data \
  --model_dir ./output \
  --epochs 2 \
  --batch_size 16
```

## Run with pdtrain CLI

```bash
# 1) Install and configure CLI (one time)
cd /Users/anish/git/codex-api/pdtrain
pip install -e .
pdtrain configure

# 2) Upload bundle (this example directory)
cd /Users/anish/git/codex-api/orchestrator-api/examples/pytorch-script-minimal
pdtrain bundle upload . --name "pytorch-script-minimal" \
  --exclude "data" \
  --exclude "test_data" \
  --wait

# 3) Upload dataset (use provided test images)
# Note: Keep test data out of the code bundle; upload as a dataset.
pdtrain dataset upload ../test_data --name "script-minimal-data" --wait

# 4) Create run (framework mode)
# Copy IDs from `pdtrain bundle list` and `pdtrain dataset list`
pdtrain run create \
  --bundle <bundle-id> \
  --dataset <dataset-id> \
  --framework pytorch \
  --framework-version 2.2.0 \
  --python-version py310 \
  --entry train.py \
  --hyperparameter epochs=2 \
  --hyperparameter batch_size=16 \
  --hyperparameter learning_rate=0.001

# 5) Monitor and fetch results
pdtrain run watch <run-id>
pdtrain logs <run-id> --follow
pdtrain artifacts download <run-id> --output ./results/
```

## Bundle Contents

- `train.py`: Minimal training script. Reads `SM_CHANNEL_DATASET` when available, else `--data_dir`. Saves artifacts to `SM_MODEL_DIR`.
- `requirements.txt`: Only PyTorch and Pillow.

## Notes

- Dataset layout must be: `<root>/<class>/*.jpg|png|bmp|tif`.
- When no dataset path is found, the script generates a small synthetic dataset to complete quickly.
- Intended as a simple, fast sanity check for end-to-end CLI flow.
