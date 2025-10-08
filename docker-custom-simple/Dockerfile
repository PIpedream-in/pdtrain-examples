FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/ml/code

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY train.py ./

# SageMaker default channels and model dir
ENV SM_MODEL_DIR=/opt/ml/model \
    SM_CHANNEL_DATASET=/opt/ml/input/data/dataset

ENTRYPOINT ["python", "train.py"]
