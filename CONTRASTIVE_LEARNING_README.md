# 🧠 Self-Supervised Contrastive Learning for KDP

**Enhance your tabular data preprocessing with self-supervised contrastive learning inspired by ReConTab!**

This feature adds a powerful self-supervised learning stage to KDP that learns robust, invariant representations of your features through contrastive learning. It's particularly effective for improving downstream task performance when you have limited labeled data.

## 🎯 Overview

The contrastive learning module implements an **asymmetric autoencoder** with regularization that:

1. **Selects salient features** through a feature selection network
2. **Creates robust embeddings** through contrastive learning with InfoNCE loss
3. **Ensures invariance** to noise through data augmentation and regularization
4. **Learns from unlabeled data** using self-supervised learning principles

## ✨ Key Features

- 🎯 **Self-Supervised Learning**: Learn from unlabeled data using contrastive learning
- 🔄 **Multi-View Learning**: Creates two augmented views for contrastive learning
- 🎲 **Data Augmentation**: Gaussian noise and random masking for robust representations
- 🧠 **Asymmetric Autoencoder**: Feature selection with reconstruction for regularization
- ⚙️ **Flexible Placement**: Apply to specific feature types or all features
- 🔧 **Highly Configurable**: 15+ parameters for fine-tuning
- 🚀 **Production Ready**: Seamlessly integrated with existing KDP pipelines

## 🚀 Quick Start

### Basic Usage

```python
from kdp import PreprocessingModel, ContrastiveLearningPlacementOptions, FeatureType

# Define your features
features_specs = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    "occupation": FeatureType.STRING_CATEGORICAL,
    "description": FeatureType.TEXT
}

# Create preprocessor with contrastive learning
preprocessor = PreprocessingModel(
    path_data="data/my_data.csv",
    features_specs=features_specs,
    # Enable contrastive learning
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
    contrastive_embedding_dim=64
)

# Build and use the preprocessor
result = preprocessor.build_preprocessor()
model = result["model"]
processed_features = model(input_data)
```

### Advanced Configuration

```python
from kdp import PreprocessingModel, ContrastiveLearningPlacementOptions, FeatureType

# Advanced contrastive learning configuration
preprocessor = PreprocessingModel(
    path_data="data/my_data.csv",
    features_specs=features_specs,
    
    # Enable contrastive learning
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
    
    # Architecture configuration
    contrastive_embedding_dim=128,
    contrastive_projection_dim=64,
    contrastive_feature_selection_units=256,
    contrastive_feature_selection_dropout=0.3,
    
    # Loss weights
    contrastive_temperature=0.07,
    contrastive_weight=1.0,
    contrastive_reconstruction_weight=0.1,
    contrastive_regularization_weight=0.01,
    
    # Normalization and augmentation
    contrastive_use_batch_norm=True,
    contrastive_use_layer_norm=True,
    contrastive_augmentation_strength=0.15
)
```

## 📊 Placement Options

You can control where contrastive learning is applied using the `contrastive_learning_placement` parameter:

```python
from kdp import ContrastiveLearningPlacementOptions

# Apply to different feature types
options = {
    "none": ContrastiveLearningPlacementOptions.NONE.value,  # Disabled
    "numeric": ContrastiveLearningPlacementOptions.NUMERIC.value,  # Only numeric features
    "categorical": ContrastiveLearningPlacementOptions.CATEGORICAL.value,  # Only categorical features
    "text": ContrastiveLearningPlacementOptions.TEXT.value,  # Only text features
    "date": ContrastiveLearningPlacementOptions.DATE.value,  # Only date features
    "all_features": ContrastiveLearningPlacementOptions.ALL_FEATURES.value  # All features
}
```

### Example: Selective Application

```python
# Apply contrastive learning only to numeric features
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
    contrastive_embedding_dim=64
)

# Apply to all features for maximum learning
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
    contrastive_embedding_dim=64
)
```

## 🔧 Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_contrastive_learning` | bool | `False` | Enable/disable contrastive learning |
| `contrastive_learning_placement` | str | `"none"` | Where to apply contrastive learning |
| `contrastive_embedding_dim` | int | `64` | Dimension of final embeddings |
| `contrastive_projection_dim` | int | `32` | Dimension of projection head |

### Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contrastive_feature_selection_units` | int | `128` | Units in feature selection layers |
| `contrastive_feature_selection_dropout` | float | `0.2` | Dropout rate for feature selection |
| `contrastive_use_batch_norm` | bool | `True` | Use batch normalization |
| `contrastive_use_layer_norm` | bool | `True` | Use layer normalization |

### Loss Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contrastive_temperature` | float | `0.1` | Temperature for contrastive loss |
| `contrastive_weight` | float | `1.0` | Weight for contrastive loss |
| `contrastive_reconstruction_weight` | float | `0.1` | Weight for reconstruction loss |
| `contrastive_regularization_weight` | float | `0.01` | Weight for regularization loss |

### Augmentation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contrastive_augmentation_strength` | float | `0.1` | Strength of data augmentation |

## 🏗️ Architecture Details

### Asymmetric Autoencoder

The contrastive learning layer uses an asymmetric autoencoder structure:

```
Input → Feature Selector → Embedding Network → Projection Head
                ↓
        Feature Reconstructor → Reconstruction Loss
```

- **Feature Selector**: Learns to select salient features
- **Embedding Network**: Creates robust embeddings
- **Projection Head**: Projects embeddings for contrastive learning
- **Feature Reconstructor**: Reconstructs input for regularization

### Contrastive Learning Process

1. **Data Augmentation**: Creates two augmented views of input data
2. **Feature Selection**: Processes both views through feature selector
3. **Embedding Creation**: Generates embeddings for both views
4. **Contrastive Loss**: Computes InfoNCE loss between embeddings
5. **Reconstruction**: Reconstructs original input for regularization
6. **Total Loss**: Combines contrastive, reconstruction, and regularization losses

### Loss Components

```python
total_loss = (
    contrastive_weight * contrastive_loss +
    reconstruction_weight * reconstruction_loss +
    regularization_weight * regularization_loss
)
```

## 🔄 Integration with Existing Features

Contrastive learning integrates seamlessly with all existing KDP features:

### Feature Selection

```python
# Works with feature selection
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    use_contrastive_learning=True,
    feature_selection_placement="numeric",  # Existing feature
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value
)
```

### Transformer Blocks

```python
# Works with transformer blocks
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    use_contrastive_learning=True,
    transfo_nr_blocks=2,  # Existing feature
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value
)
```

### Tabular Attention

```python
# Works with tabular attention
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    use_contrastive_learning=True,
    tabular_attention=True,  # Existing feature
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value
)
```

### Feature MoE

```python
# Works with feature-wise mixture of experts
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    use_contrastive_learning=True,
    feature_moe=True,  # Existing feature
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value
)
```

## 📈 Training and Inference

### Training Mode

During training, the layer:
- Creates two augmented views of input data
- Computes contrastive loss between views
- Computes reconstruction loss
- Computes regularization loss
- Returns embeddings and loss dictionary

```python
# Training mode (default)
embeddings, losses = contrastive_layer(inputs, training=True)
print(losses)
# Output: {
#     'contrastive_loss': tensor(...),
#     'reconstruction_loss': tensor(...),
#     'regularization_loss': tensor(...),
#     'total_loss': tensor(...)
# }
```

### Inference Mode

During inference, the layer:
- Processes input through feature selector and embedding network
- Returns only the embeddings (no losses)

```python
# Inference mode
embeddings = contrastive_layer(inputs, training=False)
# embeddings shape: [batch_size, embedding_dim]
```

## 💾 Model Persistence

Contrastive learning layers are fully serializable and can be saved/loaded with your models:

```python
# Save model with contrastive learning
model.save("model_with_contrastive_learning.keras")

# Load model with contrastive learning
loaded_model = tf.keras.models.load_model("model_with_contrastive_learning.keras")
```

## 🎯 Best Practices

### When to Use Contrastive Learning

- **Limited labeled data**: When you have more unlabeled than labeled data
- **Domain adaptation**: When source and target domains differ
- **Robust representations**: When you need features invariant to noise
- **Transfer learning**: When you want to pretrain on unlabeled data

### Recommended Configurations

#### For Small Datasets (< 10K samples)
```python
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
    contrastive_embedding_dim=32,
    contrastive_projection_dim=16,
    contrastive_feature_selection_units=64,
    contrastive_augmentation_strength=0.05
)
```

#### For Medium Datasets (10K - 100K samples)
```python
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
    contrastive_embedding_dim=64,
    contrastive_projection_dim=32,
    contrastive_feature_selection_units=128,
    contrastive_augmentation_strength=0.1
)
```

#### For Large Datasets (> 100K samples)
```python
preprocessor = PreprocessingModel(
    features_specs=features_specs,
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
    contrastive_embedding_dim=128,
    contrastive_projection_dim=64,
    contrastive_feature_selection_units=256,
    contrastive_augmentation_strength=0.15,
    contrastive_temperature=0.07
)
```

### Performance Tips

1. **Start with numeric features**: Apply to numeric features first, then expand
2. **Monitor losses**: Track contrastive, reconstruction, and regularization losses
3. **Adjust temperature**: Lower temperature (0.05-0.1) for better contrastive learning
4. **Tune augmentation**: Stronger augmentation for more robust representations
5. **Use appropriate embedding dimensions**: Larger for complex datasets

## 🔍 Monitoring and Debugging

### Accessing Loss Metrics

```python
# Access loss metrics from the layer
contrastive_layer = model.get_layer("contrastive_learning_feature1")
print(f"Contrastive Loss: {contrastive_layer.contrastive_loss_metric.result()}")
print(f"Reconstruction Loss: {contrastive_layer.reconstruction_loss_metric.result()}")
print(f"Regularization Loss: {contrastive_layer.regularization_loss_metric.result()}")
```

### Custom Callbacks

```python
class ContrastiveLearningCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Access contrastive learning losses
        for layer in self.model.layers:
            if hasattr(layer, 'contrastive_loss_metric'):
                print(f"Epoch {epoch} - Contrastive Loss: {layer.contrastive_loss_metric.result()}")
```

## 🧪 Testing

Run the comprehensive test suite to verify functionality:

```bash
# Run structure tests (no TensorFlow required)
python test_contrastive_learning_structure.py

# Run full tests (requires TensorFlow)
python -m pytest test/layers/test_contrastive_learning_layer.py -v
python -m pytest test/test_contrastive_learning_integration.py -v
```

## 📚 References

This implementation is inspired by:

- **ReConTab**: Self-supervised contrastive learning for tabular data
- **SimCLR**: A simple framework for contrastive learning of visual representations
- **InfoNCE**: Representation learning with contrastive predictive coding

## 🤝 Contributing

Contributions to improve the contrastive learning functionality are welcome! Please see the main [Contributing Guide](docs/contributing.md) for details.

## 📄 License

This feature is part of KDP and follows the same license terms. See the main [LICENSE](LICENSE) file for details.