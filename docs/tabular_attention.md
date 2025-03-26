# 🎯 Tabular Attention in KDP

## 📚 Overview

KDP includes powerful attention mechanisms for tabular data processing:

1. 🔄 **Standard TabularAttention**: Uniform feature processing
2. 🎛️ **MultiResolutionTabularAttention**: Type-specific feature processing

### 🔄 Standard TabularAttention
The TabularAttention layer applies attention uniformly across all features, capturing:

* 🔗 Dependencies between features for each sample
* 📊 Dependencies between samples for each feature

### 🎛️ MultiResolutionTabularAttention
The MultiResolutionTabularAttention implements a hierarchical attention mechanism:

* 📈 **Numerical Features**: Full-resolution attention preserving precise numerical relationships
* 🏷️ **Categorical Features**: Embedding-based attention capturing categorical patterns
* 🔄 **Cross-Feature Attention**: Hierarchical attention between numerical and categorical features

## 💻 Usage Examples

### Standard TabularAttention

```python
from kdp.processor import PreprocessingModel, TabularAttentionPlacementOptions

model = PreprocessingModel(
    # ... other parameters ...
    tabular_attention=True,
    tabular_attention_heads=4,
    tabular_attention_dim=64,
    tabular_attention_dropout=0.1,
    tabular_attention_placement=TabularAttentionPlacementOptions.ALL_FEATURES.value,
)
```

![Standard TabularAttention](imgs/attention_example_standard.png)

### Categorical Tabular Attention

```python
from kdp.processor import PreprocessingModel, TabularAttentionPlacementOptions

model = PreprocessingModel(
    # ... other parameters ...
    tabular_attention=True,
    tabular_attention_heads=4,
    tabular_attention_dim=64,
    tabular_attention_dropout=0.1,
    tabular_attention_embedding_dim=32,  # Dimension for categorical embeddings
    tabular_attention_placement=TabularAttentionPlacementOptions.CATEGORICAL.value,
)
```

![Categorical TabularAttention](imgs/attention_example_categorical.png)

### Multi-Resolution TabularAttention

```python
from kdp.processor import PreprocessingModel, TabularAttentionPlacementOptions

model = PreprocessingModel(
    # ... other parameters ...
    tabular_attention=True,
    tabular_attention_heads=4,
    tabular_attention_dim=64,
    tabular_attention_dropout=0.1,
    tabular_attention_embedding_dim=32,  # Dimension for categorical embeddings
    tabular_attention_placement=TabularAttentionPlacementOptions.MULTI_RESOLUTION.value,
)
```

![Multi-Resolution TabularAttention](imgs/attention_example_multi_resolution.png)

## ⚙️ Configuration Options

### Core Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `tabular_attention` | bool | Enable/disable attention mechanisms |
| `tabular_attention_heads` | int | Number of attention heads |
| `tabular_attention_dim` | int | Dimension of the attention model |
| `tabular_attention_dropout` | float | Dropout rate for regularization |

### 🎯 Placement Options
Choose where to apply attention using `tabular_attention_placement`:

* `ALL_FEATURES`: Apply uniform attention to all features
* `NUMERIC`: Apply only to numeric features
* `CATEGORICAL`: Apply only to categorical features
* `MULTI_RESOLUTION`: Use type-specific attention mechanisms
* `NONE`: Disable attention

### 🎛️ Multi-Resolution Settings
* `tabular_attention_embedding_dim`: Dimension for categorical embeddings in multi-resolution mode

## 🔍 How It Works

### Standard TabularAttention Architecture
1. 🔄 **Self-Attention**: Applied uniformly across all features
2. 📊 **Layer Normalization**: Stabilizes learning
3. 🧮 **Feed-forward Network**: Processes attention outputs

### MultiResolutionTabularAttention Architecture
1. 📈 **Numerical Processing**:
   - Full-resolution self-attention
   - Preserves numerical precision
   - Captures complex numerical relationships

2. 🏷️ **Categorical Processing**:
   - Embedding-based attention
   - Lower-dimensional representations
   - Captures categorical patterns efficiently

3. 🔄 **Cross-Feature Integration**:
   - Hierarchical attention between feature types
   - Numerical features attend to categorical features
   - Preserves type-specific characteristics while enabling interaction

## 📈 Best Practices

### When to Use Standard TabularAttention
- Data has uniform feature importance
- Features are of similar scales
- Memory usage is a concern

### When to Use MultiResolutionTabularAttention
- Mixed numerical and categorical features
- Different feature types have different importance
- Need to preserve type-specific characteristics
- Complex interactions between feature types

### Configuration Tips
1. **Attention Heads**:
   - Start with 4-8 heads
   - Increase for complex relationships
   - Monitor computational cost

2. **Dimensions**:
   - `tabular_attention_dim`: Based on feature complexity
   - `tabular_attention_embedding_dim`: Usually smaller than main dimension
   - Balance between expressiveness and efficiency

3. **Dropout**:
   - Start with 0.1
   - Increase if overfitting
   - Monitor validation performance

## 🤖 Advanced Usage

### Custom Layer Integration

```python
from kdp.layers.multi_resolution_tabular_attention_layer import MultiResolutionTabularAttention
import tensorflow as tf

# Create custom model with multi-resolution attention
numerical_inputs = tf.keras.Input(shape=(num_numerical, numerical_dim))
categorical_inputs = tf.keras.Input(shape=(num_categorical, categorical_dim))

attention_layer = MultiResolutionTabularAttention(
    num_heads=4,
    d_model=64,
    embedding_dim=32,
    dropout_rate=0.1
)

num_attended, cat_attended = attention_layer(numerical_inputs, categorical_inputs)
combined = tf.keras.layers.Concatenate(axis=1)([num_attended, cat_attended])
outputs = tf.keras.layers.Dense(1)(combined)

model = tf.keras.Model(
    inputs=[numerical_inputs, categorical_inputs],
    outputs=outputs
)
```

### Layer Factory Usage

```python
from kdp.layers_factory import PreprocessorLayerFactory

attention_layer = PreprocessorLayerFactory.multi_resolution_attention_layer(
    num_heads=4,
    d_model=64,
    embedding_dim=32,
    dropout_rate=0.1,
    name="custom_multi_attention"
)
```

## 📊 Performance Considerations

1. **Memory Usage**:
   - MultiResolutionTabularAttention is more memory-efficient for categorical features
   - Uses lower-dimensional embeddings for categorical data
   - Consider batch size when using multiple attention heads

2. **Computational Cost**:
   - Standard TabularAttention: O(n²) for n features
   - MultiResolutionTabularAttention: O(n_num² + n_cat²) for numerical and categorical features
   - Balance between resolution and performance

3. **Training Tips**:
   - Start with smaller dimensions and increase if needed
   - Monitor memory usage and training time
   - Use gradient clipping to stabilize training

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) - Attention for tabular data
- [Heterogeneous Graph Attention Network](https://arxiv.org/abs/1903.07293) - Multi-type attention mechanisms
