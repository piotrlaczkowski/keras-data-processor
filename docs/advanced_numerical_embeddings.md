# Advanced Numerical Embeddings in KDP

Keras Data Processor (KDP) now provides advanced numerical embedding techniques to better capture complex numerical relationships in your data. This release introduces two embedding approaches:

---

## AdvancedNumericalEmbedding

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
from kdp.custom_layers import AdvancedNumericalEmbedding
import tensorflow as tf

layer = AdvancedNumericalEmbedding(
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

## GlobalAdvancedNumericalEmbedding

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
from kdp.custom_layers import GlobalAdvancedNumericalEmbedding
import tensorflow as tf

global_layer = GlobalAdvancedNumericalEmbedding(
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

- **AdvancedNumericalEmbedding:**
  Use this when you need to process each numerical feature individually, preserving their distinct characteristics via per-feature embeddings.

- **GlobalAdvancedNumericalEmbedding:**
  Choose this option when you want to merge multiple numerical features into a unified global embedding using a pooling mechanism. This is particularly useful when the overall interaction across features is more important than the individual feature details.

## Advanced Configuration

Both layers offer additional parameters to fine-tune the embed­ding process. You can adjust dropout rates, batch normalization, and binning strategies to best suit your data. For more detailed information, please refer to the API documentation.

---

This document highlights the key differences and usage examples for the new advanced numerical embeddings available in KDP.
