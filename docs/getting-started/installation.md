# 📦 Installation Guide

> Get KDP up and running in your environment quickly and easily

## Quick Installation

For most users, installing KDP is as simple as:

```bash
pip install keras-data-processor
```

This will install KDP and all its required dependencies.

## Installation Methods

### Using pip (Recommended)

```bash
# Basic installation
pip install keras-data-processor

# Install with optional dependencies
pip install "keras-data-processor[all]"
```

### Using Poetry

```bash
# Add to your project
poetry add keras-data-processor

# With extras
poetry add "keras-data-processor[all]"
```

### From Source

```bash
# Clone the repository
git clone https://github.com/piotrlaczkowski/keras-data-processor.git
cd keras-data-processor

# Install using pip
pip install -e .

# Or using poetry
poetry install
```

## Dependencies

KDP requires the following core dependencies:

- Python 3.7+
- TensorFlow 2.5+
- NumPy 1.19+
- Pandas 1.2+

## Optional Dependencies

Depending on your use case, you might want to install these optional dependencies:

| Package | Purpose | Install Command |
|---------|---------|----------------|
| `scikit-learn` | For additional preprocessing capabilities | `pip install "keras-data-processor[sklearn]"` |
| `plotly` | For visualization utilities | `pip install "keras-data-processor[viz]"` |
| `xgboost` | For integration with XGBoost | `pip install "keras-data-processor[xgboost]"` |
| All extras | Complete installation | `pip install "keras-data-processor[all]"` |

## GPU Support

KDP leverages TensorFlow's GPU support. To enable GPU acceleration:

1. Install TensorFlow with GPU support
2. Ensure you have the appropriate CUDA and cuDNN versions installed

```bash
# Install TensorFlow with GPU support
pip install tensorflow-gpu
```

## Verifying Your Installation

You can verify your installation by running:

```python
import kdp

# Check version
print(f"KDP version: {kdp.__version__}")

# Basic functionality test
from kdp import PreprocessingModel, FeatureType
features = {"test": FeatureType.FLOAT}
model = PreprocessingModel(features_specs=features)
print("Installation successful!")
```

## Troubleshooting

### Common Issues

1. **Dependency Conflicts**

   If you encounter dependency conflicts, try creating a fresh virtual environment:

   ```bash
   python -m venv kdp-env
   source kdp-env/bin/activate  # On Windows: kdp-env\Scripts\activate
   pip install keras-data-processor
   ```

2. **TensorFlow Installation Problems**

   If you have issues with TensorFlow installation, refer to the [official TensorFlow installation guide](https://www.tensorflow.org/install).

3. **Version Compatibility**

   Check version compatibility if you encounter errors:

   ```bash
   pip list | grep -E "tensorflow|keras|numpy|pandas|kdp"
   ```

## Next Steps

- [Quick Start Guide](quick-start.md) - Learn the basics of KDP
- [Architecture Overview](architecture.md) - Understand KDP's components
- [Feature Processing](../features/overview.md) - Explore KDP's feature processing capabilities

---

<div class="prev-next">
  <a href="../index.md" class="prev">← Home</a>
  <a href="quick-start.md" class="next">Quick Start →</a>
</div>

<style>
.prev-next {
  display: flex;
  justify-content: space-between;
  margin-top: 40px;
}
.prev-next a {
  padding: 10px 15px;
  background-color: #f1f1f1;
  border-radius: 5px;
  text-decoration: none;
  color: #333;
}
.prev-next a:hover {
  background-color: #ddd;
}
</style>
