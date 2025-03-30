# Distribution-Aware Encoder

## Overview

The **Distribution-Aware Encoder** is an advanced preprocessing layer that automatically detects and handles various types of data distributions. It applies specialized transformations to improve model performance while preserving the statistical properties of the data. Built on pure TensorFlow operations without dependencies on TensorFlow Probability, it's lightweight and easy to deploy.

## Key Features

### 1. Automatic Distribution Detection
- Uses statistical moments (mean, variance, skewness, kurtosis) to identify distribution types
- Employs histogram analysis for multimodality detection
- Performs autocorrelation analysis for periodic pattern detection
- Adapts to data characteristics during training

### 2. Intelligent Transformations
- Applies distribution-specific transformations automatically
- Handles 16 different distribution types with specialized approaches
- Adds Fourier features (sin/cos) for periodic data
- Special handling for sparse data and zero values

### 3. Flexible Output Options
- Optional projection to fixed embedding dimension
- Distribution-specific embeddings can be added to outputs
- Automatic feature expansion for periodic data

### 4. Production-Ready Implementation
- Graph mode compatible for TensorFlow's static graph execution
- No dependencies on TensorFlow Probability for easier deployment
- Serialization support for model saving and loading

## Distribution Types Supported

The encoder automatically detects and handles these distribution types:

1. **Normal Distribution**
   - For standard normally distributed data
   - Detection: Skewness < 0.5, Kurtosis ≈ 3.0

2. **Heavy-Tailed Distribution**
   - For data with heavier tails than normal
   - Detection: Kurtosis > 4.0

3. **Multimodal Distribution**
   - For data with multiple peaks
   - Detection: Multiple significant peaks in histogram

4. **Uniform Distribution**
   - For evenly distributed data between bounds
   - Detection: Bounded between 0 and 1

5. **Exponential Distribution**
   - For data with exponential decay
   - Detection: Positive values with skewness > 1.0

6. **Log-Normal Distribution**
   - For data that is normal after log transform
   - Detection: Positive values with skewness > 2.0

7. **Discrete Distribution**
   - For data with finite distinct values
   - Detection: Low unique value ratio (< 0.1)

8. **Periodic Distribution**
   - For data with cyclic patterns
   - Detection: Significant peaks in autocorrelation

9. **Sparse Distribution**
   - For data with many zeros
   - Detection: Zero ratio > 0.5

10. **Beta Distribution**
    - For bounded data between 0 and 1 with shape parameters
    - Detection: Bounded between 0 and 1 with skewness > 0.5

11. **Gamma Distribution**
    - For positive, right-skewed data
    - Detection: Positive values with mild skewness (> 0.5)

12. **Poisson Distribution**
    - For count data
    - Handled implicitly through other transformations

13. **Cauchy Distribution**
    - For extremely heavy-tailed data
    - Detection: Very high kurtosis (> 10.0)

14. **Zero-Inflated Distribution**
    - For data with excess zeros
    - Detection: Moderate zero ratio (0.3-0.5)

15. **Bounded Distribution**
    - For data with known bounds
    - Handled implicitly through other transformations

16. **Ordinal Distribution**
    - For ordered categorical data
    - Handled similarly to discrete distributions

## Usage

### Basic Usage

The Distribution-Aware Encoder works seamlessly with numerical features. Enable it by setting `use_distribution_aware=True` in the `PreprocessingModel`.

```python
from kdp.processor import PreprocessingModel
from kdp.features import NumericalFeature

# Define features
features = {
    "feature1": NumericalFeature(),
    "feature2": NumericalFeature(),
    # etc.
}

# Initialize the model with distribution-aware encoding
model = PreprocessingModel(
    features=features,
    use_distribution_aware=True
)
```

### Manual Usage with Specific Distribution

You can specify a preferred distribution type for specific features:

```python
from kdp.processor import PreprocessingModel
from kdp.features import NumericalFeature, FeatureType
from kdp.layers.distribution_aware_encoder_layer import DistributionType

# Define features with specific distribution preferences
features = {
    "feature1": NumericalFeature(
        name="feature1",
        feature_type=FeatureType.FLOAT_NORMALIZED
    ),
    "feature2": NumericalFeature(
        name="feature2",
        feature_type=FeatureType.FLOAT_RESCALED,
        prefered_distribution=DistributionType.LOG_NORMAL  # Specify preferred distribution
    )
    # etc.
}

# Initialize the model
model = PreprocessingModel(
    features=features,
    use_distribution_aware=True
)
```

### Direct Layer Usage

You can also use the layer directly in your Keras models:

```python
import tensorflow as tf
from kdp.layers import DistributionAwareEncoder

# Creating a model with automatic distribution detection
inputs = tf.keras.Input(shape=(10,))
encoded = DistributionAwareEncoder(embedding_dim=16)(inputs)
outputs = tf.keras.layers.Dense(1)(encoded)
model = tf.keras.Model(inputs, outputs)

# Save and load model with custom objects
model.save("my_model.keras")
custom_objects = DistributionAwareEncoder.get_custom_objects()
loaded_model = tf.keras.models.load_model("my_model", custom_objects=custom_objects)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| embedding_dim | int or None | None | Output dimension for feature projection. If specified, a Dense layer projects the transformed features to this dimension. |
| epsilon | float | 1e-6 | Small value to prevent numerical issues. |
| detect_periodicity | bool | True | If True, checks for and handles periodic patterns by adding sin/cos features. |
| handle_sparsity | bool | True | If True, applies special handling for sparse data (many zeros). |
| auto_detect | bool | True | If True, automatically detects distribution type during training. |
| distribution_type | str | "unknown" | Specific distribution type to use if auto_detect is False. |
| transform_type | str | "auto" | Type of transformation to apply via DistributionTransformLayer. |
| add_distribution_embedding | bool | False | If True, adds a learned embedding for the detected distribution type. |
| trainable | bool | True | Whether the layer is trainable. |

## Output Dimensions

The output dimensions depend on the configuration:

- **Base case**: Same shape as input
- **With periodic features**: Input dimension × 3 (original + sin + cos features)
- **With embedding_dim**: (batch_size, embedding_dim)
- **With distribution_embedding**: Output has 8 additional dimensions

## Implementation Details

### 1. Distribution Detection Process

The encoder uses statistical moments and specialized tests to detect the distribution type:

```python
# Calculate basic statistics
mean = tf.reduce_mean(x)
variance = tf.math.reduce_variance(x)
std = tf.sqrt(variance + epsilon)

# Standardize for higher moments
x_std = (x - mean) / (std + epsilon)

# Calculate skewness and kurtosis
skewness = tf.reduce_mean(tf.pow(x_std, 3))
kurtosis = tf.reduce_mean(tf.pow(x_std, 4))

# Check for zeros and sparsity
zero_ratio = tf.reduce_mean(tf.cast(tf.abs(x) < epsilon, tf.float32))

# Check for discreteness
unique_ratio = tf.size(tf.unique(tf.reshape(x, [-1]))[0]) / tf.size(x)

# Score each distribution type and select the best match
```

### 2. Periodic Data Handling

For data with detected periodicity, the encoder adds Fourier features:

```python
# Normalize to [-π, π] range
normalized = (x - mean) / (std + epsilon) * π

# Generate Fourier features
sin_feature = tf.sin(frequency * normalized + phase)
cos_feature = tf.cos(frequency * normalized + phase)

# Combine with original data
transformed = tf.concat([x, sin_feature, cos_feature], axis=-1)
```

### 3. Model Serialization

When saving models containing the DistributionAwareEncoder:

```python
from kdp.layers import DistributionAwareEncoder, get_custom_objects

# Save the model
model.save("my_model.keras")

# Load the model with custom objects
custom_objects = get_custom_objects()
loaded_model = tf.keras.models.load_model("my_model", custom_objects=custom_objects)
```
```python
# Statistical moments
mean = tf.reduce_mean(inputs)
variance = tf.math.reduce_variance(inputs)
skewness = compute_skewness(inputs)
kurtosis = compute_kurtosis(inputs)

# Distribution tests
is_normal = test_normality(inputs)
is_multimodal = detect_multimodality(inputs)
is_periodic = check_periodicity(inputs)
```

### 3. Adaptive Parameters
```python
self.boundaries = self.add_weight(
    name="boundaries",
    shape=(num_bins - 1,),
    initializer="zeros",
    trainable=adaptive_binning
)

self.mixture_weights = self.add_weight(
    name="mixture_weights",
    shape=(mixture_components,),
    initializer="ones",
    trainable=True
)
```

## Best Practices

1. **Data Preparation**
   - Clean obvious outliers
   - Handle missing values
   - Ensure numeric data types

2. **Configuration**
   - Enable periodicity detection for time-related features
   - Use adaptive binning for changing distributions
   - Adjust mixture components based on complexity

3. **Performance**
   - Use appropriate batch sizes
   - Enable caching when possible
   - Monitor transformation times

4. **Monitoring**
   - Check distribution detection accuracy
   - Validate transformation quality
   - Watch for numerical instabilities

## Integration with Preprocessing Pipeline

The DistributionAwareEncoder is integrated into the numeric feature processing pipeline:

1. **Feature Statistics Collection**
   - Basic statistics (mean, variance)
   - Distribution characteristics
   - Sparsity patterns

2. **Automatic Distribution Detection**
   - Statistical tests
   - Pattern recognition
   - Threshold-based decisions

3. **Dynamic Transformation**
   - Distribution-specific handling
   - Adaptive parameter adjustment
   - Quality monitoring

## Performance Considerations

### Memory Usage
- Adaptive binning weights: O(num_bins)
- GMM parameters: O(mixture_components)
- Periodic components: O(1)

### Computational Complexity
- Distribution detection: O(n)
- Transformation: O(n)
- GMM fitting: O(n * mixture_components)

## Testing and Validation

For information on how we test and validate the Distribution-Aware Encoder, see the [Distribution-Aware Encoder Testing](distribution_aware_encoder_testing.md) documentation.

## Example Usage in Preprocessing Pipeline

```python
# Example with automatic distribution detection
from kdp.processor import PreprocessingModel
from kdp.features import NumericalFeature, FeatureType

# Define features
features = {
    # Default automatic distribution detection
    "basic_float": NumericalFeature(
        name="basic_float",
        feature_type=FeatureType.FLOAT,
    ),

    # Manually setting a gamma distribution
    "rescaled_float": NumericalFeature(
        name="rescaled_float",
        feature_type=FeatureType.FLOAT_RESCALED,
        scale=2.0,
        prefered_distribution="gamma"
    ),
}

# Create preprocessing model with distribution-aware encoding
ppr = PreprocessingModel(
    path_data="sample_data.csv",
    features_specs=features,
    features_stats_path="features_stats.json",
    overwrite_stats=True,
    output_mode="concat",
    use_distribution_aware=True
)

# Build the preprocessor
result = ppr.build_preprocessor()
```
