# SageMaker Training Examples

This directory contains comprehensive examples for training machine learning models on SageMaker using the Codex API. Each example demonstrates best practices for different frameworks and use cases.

## Available Examples

### 1. PyTorch Image Classification
**Location**: `pytorch-image-classification/`
**Framework**: PyTorch
**Use Case**: Image classification with ResNet
**Features**:
- Loads image datasets from SageMaker channels
- Trains ResNet-18 for classification
- Supports data augmentation and preprocessing
- Saves model in PyTorch format
- Includes comprehensive error handling

### 2. TensorFlow Regression
**Location**: `tensorflow-regression/`
**Framework**: TensorFlow/Keras
**Use Case**: Neural network regression
**Features**:
- Loads CSV data from SageMaker channels
- Trains configurable neural network
- Supports multiple evaluation metrics
- Saves model in SavedModel format
- Includes training history visualization

### 3. Scikit-learn Clustering
**Location**: `sklearn-clustering/`
**Framework**: Scikit-learn
**Use Case**: Unsupervised clustering
**Features**:
- Supports multiple clustering algorithms (KMeans, DBSCAN, Agglomerative)
- Loads CSV data from SageMaker channels
- Comprehensive evaluation metrics
- Cluster visualization plots
- Saves model and cluster assignments

### 4. Docker Custom Container
**Location**: `docker-custom/`
**Framework**: Any (Custom Docker)
**Use Case**: Custom container training
**Features**:
- Demonstrates custom Docker container usage
- Random Forest regression example
- Shows both Script Mode and Docker Mode
- Includes Dockerfile and build instructions
- Flexible framework support

## Quick Start

### 1. Choose an Example

Select the example that best matches your use case:
- **Image Classification**: Use PyTorch example
- **Neural Network Regression**: Use TensorFlow example
- **Clustering**: Use Scikit-learn example
- **Custom Framework**: Use Docker example

### 2. Test Locally

```bash
# Navigate to the example directory
cd examples/pytorch-image-classification/

# Install dependencies
pip install -r requirements.txt

# Run locally with test data
python train.py --data_dir ./test_data --model_dir ./output --epochs 5
```

### 3. Package and Upload

```bash
# Create bundle (avoid including local test data)
tar -czf example.tar.gz train.py requirements.txt README.md

# Upload using API client
python3 scripts/e2e_client.py example.tar.gz --entry train.py

Or, using the pdtrain CLI with exclusions for local test data:

```bash
# From within the example directory
pdtrain bundle upload . --name "my-example" \
  --exclude "data" \
  --exclude "test_data" \
  --wait

# Upload your dataset separately
pdtrain dataset upload ./data --name "my-dataset" --wait
```
```

### 4. Create and Submit Run

```bash
# Create run
curl -X POST "http://localhost:8000/v1/runs" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "execution_mode": "script",
    "bundle_id": "BUNDLE_ID",
    "framework": "pytorch",
    "framework_version": "1.13.1",
    "python_version": "py38",
    "hyperparameters": {
      "epochs": 10,
      "batch_size": 32,
      "learning_rate": 0.001
    },
    "entry": "train.py",
    "inputs": [
      {
        "type": "dataset",
        "dataset_id": "DATASET_ID",
        "version": 1
      }
    ]
  }'

# Submit to SageMaker
curl -X POST "http://localhost:8000/v1/runs/RUN_ID/submit" \
  -H "x-api-key: YOUR_API_KEY"
```

## Example Structure

Each example follows a consistent structure:

```
example-name/
├── train.py              # Main training script
├── requirements.txt      # Python dependencies
├── README.md            # Detailed documentation
└── (optional files)     # Additional configuration
```

### Key Components

1. **`train.py`**: Main training script with:
   - Command-line argument parsing
   - Data loading from SageMaker channels
   - Model training and evaluation
   - Model saving to `SM_MODEL_DIR`
   - Comprehensive error handling and logging

2. **`requirements.txt`**: Python dependencies with:
   - Framework-specific packages
   - Data processing libraries
   - Visualization tools
   - Utility libraries

3. **`README.md`**: Comprehensive documentation with:
   - What the example does
   - How to test locally
   - How to package and upload
   - Expected inputs and outputs
   - Hyperparameter descriptions
   - Troubleshooting guide

## Best Practices

### 1. Data Handling
- Always load data from `/opt/ml/input/data/dataset/` (SageMaker default)
- Handle missing values gracefully
- Use appropriate data preprocessing
- Validate data format and structure

### 2. Model Training
- Accept hyperparameters via command-line arguments
- Use appropriate logging for debugging
- Implement proper error handling
- Save model checkpoints for long training runs

### 3. Model Saving
- Save models to `/opt/ml/model/` (SageMaker default)
- Include model metadata and configuration
- Save preprocessing objects (scalers, encoders)
- Provide model loading instructions

### 4. Error Handling
- Use try-catch blocks for critical operations
- Log detailed error messages
- Provide meaningful error responses
- Handle edge cases gracefully

### 5. Logging
- Use structured logging with timestamps
- Log hyperparameters and configuration
- Log training progress and metrics
- Log model saving confirmation

## Framework-Specific Notes

### PyTorch
- Use `torch.save()` for model state dicts
- Include model architecture information
- Handle GPU/CPU device selection
- Use appropriate data loaders

### TensorFlow
- Use `model.save()` for SavedModel format
- Include model metadata and metrics
- Use callbacks for monitoring
- Handle eager execution properly

### Scikit-learn
- Use `joblib.dump()` for model persistence
- Include feature names and metadata
- Save preprocessing objects
- Provide prediction examples

### Custom Docker
- Create efficient Dockerfiles
- Use multi-stage builds for smaller images
- Include all necessary dependencies
- Test containers locally before deployment

## Troubleshooting

### Common Issues

1. **Data Loading Errors**
   - Check file paths and formats
   - Verify data structure
   - Handle missing values

2. **Memory Issues**
   - Reduce batch size or model size
   - Use data sampling
   - Optimize data loading

3. **Model Saving Errors**
   - Check directory permissions
   - Verify model format
   - Include all necessary files

4. **Hyperparameter Issues**
   - Validate parameter types
   - Check parameter ranges
   - Provide sensible defaults

### Getting Help

1. Check the example-specific README
2. Review the logs for error messages
3. Test locally before uploading
4. Verify SageMaker permissions
5. Check CloudWatch logs for detailed errors

## Contributing

To add a new example:

1. Create a new directory with the example name
2. Follow the established structure
3. Include comprehensive documentation
4. Test locally and with SageMaker
5. Update this README with the new example

## License

These examples are provided as-is for educational and reference purposes. Modify them as needed for your specific use cases.
