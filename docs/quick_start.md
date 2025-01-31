# 🚀 Quick Start Guide

## 📦 Installation

```bash
pip install keras-data-processor
```

## 🎯 Basic Usage

### 1️⃣ Define Your Features

```python
from kdp.processor import PreprocessingModel
from kdp.features import NumericalFeature, CategoricalFeature

# Define features
features = {
    "age": NumericalFeature(),
    "income": NumericalFeature(scaling="standard"),
    "occupation": CategoricalFeature(embedding_dim=32),
    "education": CategoricalFeature(embedding_dim=16)
}
```

### 2️⃣ Create Preprocessing Model

```python
# Initialize the model
model = PreprocessingModel(
    features=features,
    tabular_attention=True,  # Enable attention mechanism
    feature_selection=True   # Enable feature selection
)
```


## 🔗 Useful Links

- [📚 Full Documentation](https://kdp.readthedocs.io)
- [💻 GitHub Repository](https://github.com/piotrlaczkowski/keras-data-processor)
- [🐛 Issue Tracker](https://github.com/piotrlaczkowski/keras-data-processor/issues)
