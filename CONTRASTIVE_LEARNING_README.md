# Self-Supervised Contrastive Pretraining for KDP

This document describes the implementation of self-supervised contrastive pretraining inspired by ReConTab, integrated into the Keras Data Processor (KDP) framework.

## Overview

The contrastive learning implementation provides an asymmetric autoencoder with regularization that selects salient features and a contrastive loss that distills robust, invariant embeddings. This feature can be activated and deactivated as needed, making it a flexible addition to the KDP pipeline.

## Key Features

### 🎯 **Self-Supervised Learning**
- **Asymmetric Autoencoder**: Feature selection network that learns to identify salient features
- **Contrastive Loss**: InfoNCE-based loss for learning robust representations
- **Reconstruction Loss**: Ensures feature preservation during encoding
- **Regularization**: L1/L2 regularization for sparsity and smoothness

### 🔧 **Configurable Architecture**
- **Embedding Dimensions**: Customizable embedding and projection dimensions
- **Feature Selection**: Configurable network architecture for feature selection
- **Normalization**: Optional batch and layer normalization
- **Data Augmentation**: Gaussian noise and random masking for contrastive learning

### 🎛️ **Flexible Placement**
- **Feature-Specific**: Apply to numeric, categorical, text, or date features
- **All Features**: Apply contrastive learning to all feature types
- **Selective**: Choose which feature types to apply contrastive learning to

### ⚡ **Performance Optimized**
- **Optional Feature**: Disabled by default, no performance impact when not used
- **Efficient Implementation**: Optimized for both training and inference
- **Memory Efficient**: Minimal memory overhead when enabled

## Architecture

### Core Components

1. **Feature Selector Network**
   - Dense layers with ReLU activation
   - Dropout for regularization
   - Outputs selected features

2. **Feature Reconstructor Network**
   - Reconstructs original features from selected features
   - Used for reconstruction loss computation

3. **Embedding Network**
   - Creates final embeddings from selected features
   - Configurable architecture

4. **Projection Head**
   - Projects embeddings for contrastive learning
   - Used only during training

5. **Contrastive Learning Components**
   - Data augmentation (noise + masking)
   - InfoNCE loss computation
   - Multi-view learning with two augmented views

### Loss Functions

- **Contrastive Loss**: InfoNCE loss for learning invariant representations
- **Reconstruction Loss**: MSE loss for feature reconstruction
- **Regularization Loss**: L1/L2 regularization for sparsity

## Usage

### Basic Usage

```python
from kdp import PreprocessingModel, ContrastiveLearningPlacementOptions
from kdp.features import NumericalFeature, FeatureType

# Create model with contrastive learning
model = PreprocessingModel(
    features_specs={
        "numeric_feature": NumericalFeature(
            name="numeric_feature",
            feature_type=FeatureType.FLOAT_NORMALIZED
        )
    },
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
    contrastive_embedding_dim=64
)

# Build preprocessor
preprocessor = model.build_preprocessor()
```

### Advanced Configuration

```python
model = PreprocessingModel(
    features_specs={
        "numeric_feature": NumericalFeature(
            name="numeric_feature",
            feature_type=FeatureType.FLOAT_NORMALIZED
        ),
        "categorical_feature": CategoricalFeature(
            name="categorical_feature",
            feature_type=FeatureType.CATEGORICAL
        )
    },
    # Enable contrastive learning
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
    
    # Architecture configuration
    contrastive_embedding_dim=128,
    contrastive_projection_dim=64,
    contrastive_feature_selection_units=256,
    contrastive_feature_selection_dropout=0.3,
    
    # Loss weights
    contrastive_temperature=0.1,
    contrastive_weight=1.0,
    contrastive_reconstruction_weight=0.1,
    contrastive_regularization_weight=0.01,
    
    # Normalization options
    contrastive_use_batch_norm=True,
    contrastive_use_layer_norm=True,
    
    # Augmentation strength
    contrastive_augmentation_strength=0.1
)
```

### Placement Options

```python
from kdp import ContrastiveLearningPlacementOptions

# Apply to specific feature types
ContrastiveLearningPlacementOptions.NUMERIC.value      # Only numeric features
ContrastiveLearningPlacementOptions.CATEGORICAL.value  # Only categorical features
ContrastiveLearningPlacementOptions.TEXT.value         # Only text features
ContrastiveLearningPlacementOptions.DATE.value         # Only date features

# Apply to all features
ContrastiveLearningPlacementOptions.ALL_FEATURES.value

# Disable contrastive learning
ContrastiveLearningPlacementOptions.NONE.value
```

## Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_contrastive_learning` | bool | False | Enable/disable contrastive learning |
| `contrastive_learning_placement` | str | "none" | Where to apply contrastive learning |
| `contrastive_embedding_dim` | int | 64 | Dimension of final embeddings |
| `contrastive_projection_dim` | int | 32 | Dimension of projection head |

### Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contrastive_feature_selection_units` | int | 128 | Units in feature selection layers |
| `contrastive_feature_selection_dropout` | float | 0.2 | Dropout rate for feature selection |
| `contrastive_use_batch_norm` | bool | True | Use batch normalization |
| `contrastive_use_layer_norm` | bool | True | Use layer normalization |

### Loss Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contrastive_temperature` | float | 0.1 | Temperature for contrastive loss |
| `contrastive_weight` | float | 1.0 | Weight for contrastive loss |
| `contrastive_reconstruction_weight` | float | 0.1 | Weight for reconstruction loss |
| `contrastive_regularization_weight` | float | 0.01 | Weight for regularization loss |

### Augmentation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contrastive_augmentation_strength` | float | 0.1 | Strength of data augmentation |

## Integration with Existing Features

### Feature Selection
Contrastive learning works seamlessly with existing feature selection:
```python
model = PreprocessingModel(
    # ... features ...
    feature_selection_placement="numeric",
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value
)
```

### Transformer Blocks
Compatible with transformer blocks:
```python
model = PreprocessingModel(
    # ... features ...
    transfo_nr_blocks=2,
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.CATEGORICAL.value
)
```

### Tabular Attention
Works with tabular attention:
```python
model = PreprocessingModel(
    # ... features ...
    tabular_attention=True,
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value
)
```

### Feature MoE
Compatible with feature mixture of experts:
```python
model = PreprocessingModel(
    # ... features ...
    use_feature_moe=True,
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value
)
```

## Training and Inference

### Training Mode
During training, the contrastive learning layer:
1. Creates two augmented views of the input
2. Processes both views through the feature selector
3. Computes embeddings and projections
4. Calculates contrastive, reconstruction, and regularization losses
5. Returns embeddings and loss dictionary

### Inference Mode
During inference, the layer:
1. Processes input through feature selector
2. Returns embeddings only (no losses computed)
3. No data augmentation applied

## Model Persistence

Models with contrastive learning can be saved and loaded:
```python
# Save model
model.save_model("path/to/model")

# Load model
loaded_model, preprocessor = PreprocessingModel.load_model("path/to/model")

# Contrastive learning settings are preserved
assert loaded_model.use_contrastive_learning is True
assert loaded_model.contrastive_embedding_dim == 64
```

## Performance Considerations

### Memory Usage
- **Disabled**: No additional memory overhead
- **Enabled**: Additional memory for contrastive learning components
- **Scales with**: Embedding dimensions and batch size

### Computational Cost
- **Training**: ~2x forward passes due to two augmented views
- **Inference**: Single forward pass, minimal overhead
- **Optimized**: Efficient implementation with minimal computational cost

### Recommendations
- Start with default parameters for most use cases
- Increase embedding dimensions for complex datasets
- Adjust loss weights based on task requirements
- Monitor training metrics for optimal performance

## Examples

### Simple Example
```python
from kdp import PreprocessingModel, ContrastiveLearningPlacementOptions
from kdp.features import NumericalFeature, FeatureType

# Basic setup
model = PreprocessingModel(
    features_specs={
        "feature1": NumericalFeature(
            name="feature1",
            feature_type=FeatureType.FLOAT_NORMALIZED
        )
    },
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value
)

preprocessor = model.build_preprocessor()
```

### Advanced Example
```python
# Complex setup with multiple features
model = PreprocessingModel(
    features_specs={
        "numeric": NumericalFeature(name="numeric", feature_type=FeatureType.FLOAT_NORMALIZED),
        "categorical": CategoricalFeature(name="categorical", feature_type=FeatureType.CATEGORICAL),
        "text": TextFeature(name="text", feature_type=FeatureType.TEXT),
    },
    # Contrastive learning
    use_contrastive_learning=True,
    contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
    contrastive_embedding_dim=128,
    contrastive_projection_dim=64,
    
    # Other features
    feature_selection_placement="all_features",
    tabular_attention=True,
    transfo_nr_blocks=2,
)

preprocessor = model.build_preprocessor()
```

## Testing

Comprehensive tests are included to ensure functionality:

```bash
# Run layer tests
python -m pytest test/layers/test_contrastive_learning_layer.py

# Run integration tests
python -m pytest test/test_contrastive_learning_integration.py

# Run simple test script
python test_contrastive_learning.py
```

## Backward Compatibility

The contrastive learning implementation is fully backward compatible:
- **Default behavior**: Contrastive learning is disabled
- **Existing code**: Works without modification
- **Optional feature**: Can be enabled/disabled as needed
- **No breaking changes**: All existing functionality preserved

## Future Enhancements

Potential future improvements:
- **Advanced Augmentations**: More sophisticated data augmentation strategies
- **Multi-Modal Support**: Support for different data modalities
- **Adaptive Loss Weights**: Dynamic loss weight adjustment
- **Distributed Training**: Support for distributed contrastive learning
- **Custom Loss Functions**: User-defined contrastive loss functions

## Contributing

When contributing to the contrastive learning implementation:
1. Follow existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure backward compatibility
5. Test with various feature types and configurations

## References

This implementation is inspired by:
- **ReConTab**: Self-supervised contrastive learning for tabular data
- **InfoNCE**: Contrastive learning with noise-contrastive estimation
- **SimCLR**: Simple framework for contrastive learning of visual representations

## Support

For questions or issues with the contrastive learning implementation:
1. Check the test files for usage examples
2. Review the integration tests for common patterns
3. Ensure all dependencies are properly installed
4. Verify configuration parameters are correct