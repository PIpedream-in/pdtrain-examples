# SageMaker Dependencies Guide

## Issue

When using SageMaker scikit-learn containers, **DO NOT** upgrade pre-installed packages (pandas, numpy, scikit-learn) in `requirements.txt`. This causes dependency conflicts.

## What Happened

The original `requirements.txt` had:
```
pandas>=1.5.0
numpy>=1.23.0
```

SageMaker scikit-learn container has:
```
pandas==1.1.3
numpy==1.24.1
scikit-learn==1.2.1
```

When pip installed pandas 2.3.3, it broke numpy compatibility:
```
ImportError: numpy.core.multiarray failed to import
```

## Solution

**Keep requirements.txt minimal** - only add packages NOT in the base image:

```python
# ✅ CORRECT - Empty or only new packages
matplotlib>=3.5.0  # OK - not in base image

# ❌ WRONG - Don't upgrade base packages
pandas>=2.0.0
numpy>=1.25.0
scikit-learn>=1.3.0
```

## Pre-installed Packages in SageMaker Containers

### scikit-learn 1.2-1 Container
- scikit-learn==1.2.1
- pandas==1.1.3
- numpy==1.24.1
- scipy==1.8.0
- joblib==1.5.1

### Best Practice

1. **Check container versions first**: https://github.com/aws/sagemaker-scikit-learn-container
2. **Only add new packages** your code needs
3. **Test locally** with the same package versions
4. **If you need newer versions**, consider using Docker Mode with a custom image

## Testing Locally

To test with the same environment SageMaker uses:

```bash
# Use the same image SageMaker uses
docker run -it 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3 bash

# Inside container, test your code
python -c "import pandas; print(pandas.__version__)"  # 1.1.3
```
