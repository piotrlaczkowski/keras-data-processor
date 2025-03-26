# Advanced Numerical Embeddings in KDP

Keras Data Processor (KDP) now provides advanced numerical embedding techniques to better capture complex numerical relationships in your data. This release introduces two embedding approaches:

---

## NumericalEmbedding

**Purpose:**
Processes individual numerical features with tailored embedding layers. This layer performs adaptive binning, applies MLP transformations per feature, and can incorporate dropout and batch normalization.

**Key Parameters:**
- **`embedding_dim`**: Dimension for each feature's embedding.
- **`mlp_hidden_units`**: Number of hidden units in the MLP applied to each feature.
- **`num_bins`**: Number of bins used for discretizing continuous inputs.
- **`init_min` and `init_max`**: Initialization boundaries for binning.
- **`dropout_rate`**: Dropout rate for regularization.
- **`use_batch_norm`**: Flag to apply batch normalization.

**Usage Example:**
```python
from kdp.layers.numerical_embedding_layer import NumericalEmbedding
import tensorflow as tf

layer = NumericalEmbedding(
    embedding_dim=8,
    mlp_hidden_units=16,
    num_bins=10,
    init_min=[-3.0, -2.0, -4.0],
    init_max=[3.0, 2.0, 4.0],
    dropout_rate=0.1,
    use_batch_norm=True,
)

# Input shape: (batch_size, num_features)
x = tf.random.normal((32, 3))
# Output shape: (32, 3, 8)
output = layer(x, training=False)
```

---

## GlobalNumericalEmbedding

**Purpose:**
Combines a set of numerical features into a single, compact representation. It does so by applying an internal advanced numerical embedding on the concatenated input and then performing a global pooling over all features.

**Key Parameters (prefixed with `global_`):**
- **`global_embedding_dim`**: Global embedding dimension (final pooled vector size).
- **`global_mlp_hidden_units`**: Hidden units in the global MLP.
- **`global_num_bins`**: Number of bins for discretization.
- **`global_init_min` and `global_init_max`**: Global initialization boundaries.
- **`global_dropout_rate`**: Dropout rate.
- **`global_use_batch_norm`**: Whether to apply batch normalization.
- **`global_pooling`**: Pooling method to use ("average" or "max").

**Usage Example:**
```python
from kdp.layers.global_numerical_embedding_layer import GlobalNumericalEmbedding
import tensorflow as tf

global_layer = GlobalNumericalEmbedding(
    global_embedding_dim=8,
    global_mlp_hidden_units=16,
    global_num_bins=10,
    global_init_min=[-3.0, -2.0],
    global_init_max=[3.0, 2.0],
    global_dropout_rate=0.1,
    global_use_batch_norm=True,
    global_pooling="average"
)

# Input shape: (batch_size, num_features)
x = tf.random.normal((32, 3))
# Global output shape: (32, 8)
global_output = global_layer(x, training=False)
```

---

## When to Use Which?

- **NumericalEmbedding:**
  Use this when you need to process each numerical feature individually, preserving their distinct characteristics via per-feature embeddings.

- **GlobalNumericalEmbedding:**
  Choose this option when you want to merge multiple numerical features into a unified global embedding using a pooling mechanism. This is particularly useful when the overall interaction across features is more important than the individual feature details.

## Advanced Configuration

Both layers offer additional parameters to fine-tune the embedding process. You can adjust dropout rates, batch normalization, and binning strategies to best suit your data. For more detailed information, please refer to the API documentation.

---

This document highlights the key differences and usage examples for the new advanced numerical embeddings available in KDP.

# 🌐 Global Numerical Embedding

## 📚 Overview

Global Numerical Embedding is a powerful technique for processing numerical features collectively rather than individually. It transforms batches of numerical features through a unified embedding approach, capturing relationships across the entire numerical feature space.

## 🔑 Key Benefits

- **Cross-Feature Learning**: Captures relationships between different numerical features
- **Unified Representation**: Creates a consistent embedding space for all numerical data
- **Dimensionality Control**: Transforms variable numbers of features into fixed-size embeddings
- **Performance Enhancement**: Typically improves performance on complex tabular datasets

## 💻 Usage

### Basic Configuration

Enable Global Numerical Embedding by setting the appropriate parameters in your `PreprocessingModel`:

```python
from kdp.processor import PreprocessingModel
from kdp.features import FeatureType

# Define features
features_specs = {
    "feature1": FeatureType.FLOAT_NORMALIZED,
    "feature2": FeatureType.FLOAT_NORMALIZED,
    "feature3": FeatureType.FLOAT_RESCALED,
    # more numerical features...
}

# Initialize with Global Numerical Embedding
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    use_global_numerical_embedding=True,  # Enable the feature
    global_embedding_dim=16,              # Output dimension per feature
    global_pooling="average"              # Pooling strategy
)

# Build the model
result = preprocessor.build_preprocessor()
```

### Advanced Configuration

Fine-tune Global Numerical Embedding with additional parameters:

```python
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    use_global_numerical_embedding=True,
    global_embedding_dim=32,           # Embedding dimension size
    global_mlp_hidden_units=64,        # Units in the MLP layer
    global_num_bins=20,                # Number of bins for discretization
    global_init_min=-3.0,              # Minimum initialization bound
    global_init_max=3.0,               # Maximum initialization bound
    global_dropout_rate=0.2,           # Dropout rate for regularization
    global_use_batch_norm=True,        # Whether to use batch normalization
    global_pooling="concat"            # Pooling strategy
)
```

## ⚙️ Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `global_embedding_dim` | int | 8 | Dimension of each feature embedding |
| `global_mlp_hidden_units` | int | 16 | Number of units in the MLP layer |
| `global_num_bins` | int | 10 | Number of bins for discretization |
| `global_init_min` | float | -3.0 | Minimum initialization bound |
| `global_init_max` | float | 3.0 | Maximum initialization bound |
| `global_dropout_rate` | float | 0.1 | Dropout rate for regularization |
| `global_use_batch_norm` | bool | True | Whether to use batch normalization |
| `global_pooling` | str | "average" | Pooling strategy ("average", "max", or "concat") |

## 🧩 Architecture

The Global Numerical Embedding layer processes numerical features through several steps:

1. **Normalization**: Numerical features are normalized to a standard range
2. **Binning**: Features are discretized into bins
3. **Embedding**: Each bin is mapped to a learned embedding vector
4. **MLP Processing**: A small MLP network processes each embedded feature
5. **Pooling**: Features are aggregated using the specified pooling strategy
6. **Output**: A fixed-size embedding representing all numerical features

```
┌─────────────┐     ┌───────────┐     ┌────────────┐     ┌─────────┐
│ Numerical   │     │ Discretize│     │  Embedding │     │   MLP   │
│ Features    │────▶│  to Bins  │────▶│   Lookup   │────▶│ Network │
└─────────────┘     └───────────┘     └────────────┘     └────┬────┘
                                                              │
                                                              ▼
┌─────────────┐     ┌───────────┐                       ┌─────────┐
│   Output    │     │  Pooling  │                       │ Feature │
│  Embedding  │◀────│ Operation │◀──────────────────────│ Vectors │
└─────────────┘     └───────────┘                       └─────────┘
```

## 🧠 Pooling Strategies

The `global_pooling` parameter controls how feature embeddings are combined:

- **"average"**: Compute the mean across all feature embeddings (default)
- **"max"**: Take the maximum value for each dimension across all embeddings
- **"concat"**: Concatenate all feature embeddings (increases output dimension)

## 🚀 When to Use

Global Numerical Embedding is particularly effective when:

- Your dataset has many numerical features (>5)
- Features have complex relationships with each other
- You want to reduce the dimensionality of your numerical features
- You're working with tabular data where feature interactions matter

## 📊 Comparison with Individual Processing

| Aspect | Global Embedding | Individual Processing |
|--------|------------------|----------------------|
| Feature Interactions | Captures cross-feature relationships | Processes each feature independently |
| Output Dimension | Fixed size regardless of input features | Scales with number of features |
| Parameter Efficiency | Shares parameters across features | Separate parameters per feature |
| Computational Cost | Higher for few features, more efficient for many | Linear with feature count |
| Model Performance | Often better for complex datasets | Simpler, may miss interactions |

## 🔍 Implementation Details

The Global Numerical Embedding implementation is based on the `GlobalNumericalEmbedding` layer:

```python
# Sample internal implementation (simplified)
class GlobalNumericalEmbedding(tf.keras.layers.Layer):
    def __init__(self, global_embedding_dim=8, global_pooling="average", **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = global_embedding_dim
        self.pooling = global_pooling
        # ...additional initialization...

    def build(self, input_shape):
        # Create embeddings, MLP layers, etc.

    def call(self, inputs):
        # 1. Discretize numerical inputs
        # 2. Apply embedding lookup
        # 3. Process through MLP
        # 4. Apply pooling
        # 5. Return transformed features
```

## 📝 Examples

### Basic Example

```python
# Simple example with default parameters
preprocessor = PreprocessingModel(
    features_specs={"feature1": FeatureType.FLOAT, "feature2": FeatureType.FLOAT},
    use_global_numerical_embedding=True
)
```

### Advanced Example with Custom Pooling

```python
# Using concatenation pooling for maximum information preservation
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    use_global_numerical_embedding=True,
    global_embedding_dim=16,
    global_pooling="concat",  # Will concatenate all feature embeddings
    global_dropout_rate=0.2   # Increased regularization
)
```

### Combined with Other Advanced Features

```python
# Combining Global Numerical Embedding with other advanced features
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    # Global Numerical Embedding
    use_global_numerical_embedding=True,
    global_embedding_dim=16,
    # Distribution-Aware Encoding
    use_distribution_aware=True,
    # Tabular Attention
    tabular_attention=True,
    tabular_attention_placement="MULTI_RESOLUTION"
)
```
