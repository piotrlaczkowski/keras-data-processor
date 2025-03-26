# 🚀 Quick Start Guide

## 📦 Installation

```bash
pip install keras-data-processor
```

## 🎯 Basic Usage

### 1️⃣ Define Your Features

```python
from kdp.processor import PreprocessingModel
from kdp.features import FeatureType

# Define features
features_specs = {
    # Numerical features
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    # Categorical features
    "occupation": FeatureType.STRING_CATEGORICAL,
    "education": FeatureType.INTEGER_CATEGORICAL,
    # Text features
    "description": FeatureType.TEXT,
    # Date features
    "signup_date": FeatureType.DATE
}
```

### 2️⃣ Create Preprocessing Model

```python
# Initialize the model
preprocessor = PreprocessingModel(
    path_data="data/my_data.csv",            # Path to your data
    features_specs=features_specs,           # Feature specifications
    use_distribution_aware=True,             # Enable distribution-aware encoding
    tabular_attention=True,                  # Enable attention mechanism
    feature_selection_placement="ALL_FEATURES" # Enable feature selection
)

# Build the preprocessing model
result = preprocessor.build_preprocessor()
model = result["model"]
```

### 3️⃣ Use the Model

```python
# Save the model
preprocessor.save_model("preprocessor_model.keras")

# For inference
import tensorflow as tf
input_data = {...}  # Dictionary of feature inputs
processed_data = model(input_data)

# Integrate with your model
outputs = tf.keras.layers.Dense(1)(processed_data)
full_model = tf.keras.Model(inputs=model.inputs, outputs=outputs)
```

## 💡 Advanced Features

### ✨ Distribution-Aware Encoding

```python
# Enable automatic distribution detection and transformation
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    use_distribution_aware=True,
    distribution_aware_bins=1000  # More bins for finer-grained encoding
)
```

### 🌐 Global Numerical Embedding

```python
# Process all numerical features through a unified embedding
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    use_global_numerical_embedding=True,
    global_embedding_dim=16,
    global_pooling="average"  # Options: "average", "max", "concat"
)
```

### 👁️ Tabular Attention

```python
# Add attention mechanisms for tabular data
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    tabular_attention=True,
    tabular_attention_heads=4,
    tabular_attention_dim=64,
    tabular_attention_placement="MULTI_RESOLUTION"  # Options: "ALL_FEATURES", "NUMERIC", "CATEGORICAL", "MULTI_RESOLUTION"
)
```

### 🧩 Transformer Blocks

```python
# Add transformer blocks for complex feature interactions
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    transfo_nr_blocks=2,
    transfo_nr_heads=4,
    transfo_placement="ALL_FEATURES"
)
```

## 🔗 Useful Links

- [📚 Complete Documentation](index.md)
- [📊 Defining Features](features.md)
- [📈 Distribution-Aware Encoding](distribution_aware_encoder.md)
- [👁️ Tabular Attention](tabular_attention.md)
- [🌐 Advanced Examples](complex_examples.md)
