# Distribution-Aware Encoder

## Overview
The Distribution-Aware Encoder is an advanced preprocessing layer that automatically detects and handles various types of data distributions. It uses TensorFlow Probability (tfp) for accurate modeling and applies specialized transformations while preserving the statistical properties of the data.

## Features

### Distribution Types Supported
1. **Normal Distribution**
   - For standard normally distributed data
   - Handled via z-score normalization
   - Detection: Kurtosis ≈ 3.0, Skewness ≈ 0

2. **Heavy-Tailed Distribution**
   - For data with heavier tails than normal
   - Handled via Student's t-distribution
   - Detection: Kurtosis > 3.5

3. **Multimodal Distribution**
   - For data with multiple peaks
   - Handled via Gaussian Mixture Models
   - Detection: KDE-based peak detection

4. **Uniform Distribution**
   - For evenly distributed data
   - Handled via min-max scaling
   - Detection: Kurtosis ≈ 1.8

5. **Exponential Distribution**
   - For data with exponential decay
   - Handled via rate-based transformation
   - Detection: Skewness ≈ 2.0

6. **Log-Normal Distribution**
   - For data that is normal after log transform
   - Handled via logarithmic transformation
   - Detection: Log-transformed kurtosis ≈ 3.0

7. **Discrete Distribution**
   - For data with finite distinct values
   - Handled via empirical CDF-based encoding
   - Detection: Unique values analysis

8. **Periodic Distribution**
   - For data with cyclic patterns
   - Handled via Fourier features (sin/cos)
   - Detection: Autocorrelation analysis

9. **Sparse Distribution**
   - For data with many zeros
   - Handled via separate zero/non-zero transformations
   - Detection: Zero ratio analysis

10. **Beta Distribution**
    - For bounded data between 0 and 1
    - Handled via beta CDF transformation
    - Detection: Value range and shape analysis

11. **Gamma Distribution**
    - For positive, right-skewed data
    - Handled via gamma CDF transformation
    - Detection: Positive support and skewness

12. **Poisson Distribution**
    - For count data
    - Handled via rate parameter estimation
    - Detection: Integer values and variance≈mean

14. **Cauchy Distribution**
    - For extremely heavy-tailed data
    - Handled via robust location-scale estimation
    - Detection: Undefined moments

15. **Zero-Inflated Distribution**
    - For data with excess zeros
    - Handled via mixture model approach
    - Detection: Zero proportion analysis

## Usage

### Basic Usage

The capability only works with numerical features!

```python
from kdp.processor import PreprocessingModel
from kdp.features import NumericalFeature

# Define features
features = {
    # Numerical features
    "feature1": NumericalFeature(),
    "feature2": NumericalFeature(),
    # etc ..
}

# Initialize the model
model = PreprocessingModel( # here
    features=features,
    use_distribution_aware=True
)
```

### Manual Usage

```python
from kdp.processor import PreprocessingModel
from kdp.features import NumericalFeature

# Define features
features = {
    # Numerical features
    # Numerical features
    "feature1": NumericalFeature(
        name="feature1",
        feature_type=FeatureType.FLOAT_NORMALIZED
    ),
    "feature2": NumericalFeature(
        name="feature2",
        feature_type=FeatureType.FLOAT_RESCALED
        prefered_distribution="log_normal" # here we could specify a prefered distribution (normal, periodic, etc)
    )
    # etc ..
}

# Initialize the model
model = PreprocessingModel( # here
    features=features,
    use_distribution_aware=True,
    distribution_aware_bins=1000, # 1000 is the default value, but you can change it for finer data
)
```

### Advanced Configuration
```python
encoder = DistributionAwareEncoder(
    num_bins=1000,
    epsilon=1e-6,
    detect_periodicity=True,
    handle_sparsity=True,
    adaptive_binning=True,
    mixture_components=3,
    trainable=True
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_bins | int | 1000 | Number of bins for quantile encoding |
| epsilon | float | 1e-6 | Small value for numerical stability |
| detect_periodicity | bool | True | Enable periodic pattern detection | Remove this parameter when having multimodal functions/distributions
| handle_sparsity | bool | True | Enable special handling for sparse data |
| adaptive_binning | bool | True | Enable adaptive bin boundaries |
| mixture_components | int | 3 | Number of components for mixture models |
| trainable | bool | True | Whether parameters are trainable |

## Key Features

### 1. Automatic Distribution Detection
- Uses statistical moments and tests
- Employs KDE for multimodality detection
- Handles mixed distributions via ensemble approach

### 2. Adaptive Transformations
- Learns optimal parameters during training
- Adjusts to data distribution changes
- Handles complex periodic patterns

### 3. Fourier Feature Generation
- Automatic frequency detection
- Multiple harmonic components
- Phase-aware transformations

### 4. Robust Handling
- Special treatment for zeros
- Outlier-resistant transformations
- Numerical stability safeguards

## Implementation Details

### 1. Periodic Data Handling
```python
# Normalize to [-π, π] range
normalized = inputs * π / scale
# Generate Fourier features
features = [
    sin(freq * normalized + phase),
    cos(freq * normalized + phase)
]
# Add harmonics if multimodal
if is_multimodal:
    for h in [2, 3, 4]:
        features.extend([
            sin(h * freq * normalized + phase),
            cos(h * freq * normalized + phase)
        ])
```

### 2. Distribution Detection
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

## Best Practices

1. **Data Preparation**
   - Clean outliers if not meaningful
   - Handle missing values before encoding
   - Ensure numeric data type

2. **Configuration**
   - Start with default parameters
   - Adjust based on data characteristics
   - Monitor distribution detection results

3. **Performance Optimization**
   - Use appropriate batch sizes
   - Enable caching for repeated processing
   - Adjust mixture components based on data

### Distribution Detection
```python
# Access distribution information
dist_info = encoder._estimate_distribution(inputs)
print(f"Detected distribution: {dist_info['type']}")
print(f"Statistics: {dist_info['stats']}")
```

### Transformation Quality
```python
# Monitor transformed output statistics
transformed = encoder(inputs)
print(f"Output mean: {tf.reduce_mean(transformed)}")
print(f"Output variance: {tf.math.reduce_variance(transformed)}")
```
