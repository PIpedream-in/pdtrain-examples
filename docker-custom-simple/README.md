# Docker Custom (Simple)

A minimal custom Docker container training example for SageMaker via pdtrain. Uses scikit-learn LinearRegression on CSV data (or synthetic) and saves artifacts to `SM_MODEL_DIR`.

- Mode: `docker` (Custom Container)
- Entry point: `train.py` (set as container ENTRYPOINT)
- Input: CSV(s) in `dataset/` channel; uses all numeric columns, `target` if present
- Output: `model.pkl`, `metrics.json`

## Build and Test Locally

```bash
cd docker-custom-simple

# Build image
docker build -t docker-custom-simple .

# Prepare local dirs
mkdir -p ./data ./output
# Put a CSV under ./data or skip to use synthetic data

# Run container locally (single line)
docker run --rm -v $(pwd)/data:/opt/ml/input/data/dataset -v $(pwd)/output:/opt/ml/model docker-custom-simple --fit_intercept true --normalize false
```

## Push to ECR (example)

```bash

docker tag docker-custom-simple:latest <your-public-dockerhub-username>/docker-custom-simple:latest
docker push <your-public-dockerhub-username>/docker-custom-simple:latest
```

## Run with pdtrain (Docker Mode)

```bash
# Install and configure CLI
pip install pdtrain
pdtrain configure

# Create run using custom image (no bundle needed)
pdtrain run create \
  --image <your-public-dockerhub-username>/docker-custom-simple:latest \
  --entry train.py \
  --param fit_intercept=true \
  --param normalize=false \
  --dataset <dataset-id>

pdtrain run watch <run-id>
pdtrain logs <run-id> --follow
```

## Contents
- `Dockerfile`: Minimal Python slim image
- `requirements.txt`: scikit-learn, pandas, numpy, joblib
- `train.py`: Reads dataset from `SM_CHANNEL_DATASET` or `--data_dir`, writes artifacts to `SM_MODEL_DIR`
