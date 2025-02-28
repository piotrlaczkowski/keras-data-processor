# Testing the Distribution-Aware Encoder

## Overview

The `DistributionAwareEncoder` is a sophisticated layer that automatically detects and handles various data distributions. To ensure its reliability, we've implemented comprehensive testing that verifies its functionality across different distribution types.

## Key Improvements

We've made several improvements to the `DistributionAwareEncoder` class:

1. **Fixed Multimodality Detection**: Corrected the implementation of the `_detect_multimodality` method to properly handle peak detection and periodicity checking.

2. **Enhanced Discrete Distribution Handling**: Improved the `_handle_discrete` method to work reliably in both eager and graph execution modes, replacing the `StaticHashTable` approach with a more compatible implementation.

3. **Graph Mode Compatibility**: Ensured all methods work correctly in TensorFlow's graph execution mode, which is essential for production deployment.

## Testing Strategy

Our testing approach for the `DistributionAwareEncoder` includes:

### 1. Distribution-Specific Tests

We test each supported distribution type individually:

- **Normal Distribution**: Verifies correct handling of normally distributed data
- **Heavy-Tailed Distribution**: Tests Student's t-distribution handling
- **Multimodal Distribution**: Checks detection and transformation of bimodal data
- **Uniform Distribution**: Validates uniform distribution handling
- **Discrete Distribution**: Tests handling of data with finite distinct values
- **Sparse Distribution**: Verifies special handling for data with many zeros
- **Periodic Distribution**: Tests detection and transformation of cyclic patterns

### 2. Graph Mode Compatibility Test

We verify that the encoder works correctly in TensorFlow's graph execution mode by:

1. Creating a simple model with the encoder
2. Compiling the model
3. Training it for one epoch
4. Verifying no errors occur during graph compilation and execution

## Sample Test Code

Here's an example of how we test the `DistributionAwareEncoder`:

```python
import numpy as np
import pytest
import tensorflow as tf

from kdp.custom_layers import DistributionAwareEncoder, DistributionType

@pytest.fixture
def encoder():
    """Create a DistributionAwareEncoder instance for testing."""
    return DistributionAwareEncoder(num_bins=10, detect_periodicity=True, handle_sparsity=True)

def test_normal_distribution(encoder):
    """Test that normal distribution is correctly identified and transformed."""
    # Generate normal distribution data
    np.random.seed(42)
    data = np.random.normal(0, 1, (100, 1))

    # Transform the data
    transformed = encoder(data)

    # Check that the output is finite and in a reasonable range
    assert np.all(np.isfinite(transformed))
    assert -2.0 <= np.min(transformed) <= 2.0
    assert -2.0 <= np.max(transformed) <= 2.0
```

## Running the Tests

To run the tests, use the following command:

```bash
poetry run pytest tests/test_distribution_encoder.py -v
```

## Best Practices for Using Distribution-Aware Encoder

1. **Data Preparation**:
   - Clean obvious outliers if they're not meaningful
   - Handle missing values before encoding
   - Ensure numeric data type

2. **Configuration**:
   - Start with default parameters
   - Adjust based on your data characteristics
   - Monitor distribution detection results

3. **Performance Optimization**:
   - Use appropriate batch sizes
   - Enable caching for repeated processing
   - Adjust mixture components based on data complexity

4. **Distribution Monitoring**:
   - For debugging, you can access the detected distribution:
     ```python
     # Access distribution information
     dist_info = encoder._estimate_distribution(inputs)
     print(f"Detected distribution: {dist_info['type']}")
     ```

## Integration with Preprocessing Pipeline

The `DistributionAwareEncoder` is fully integrated into the KDP preprocessing pipeline. To use it, simply enable it in your `PreprocessingModel`:

```python
from kdp.processor import PreprocessingModel
from kdp.features import NumericalFeature, FeatureType

# Define features
features = {
    "feature1": NumericalFeature(
        name="feature1",
        feature_type=FeatureType.FLOAT_NORMALIZED
    ),
    "feature2": NumericalFeature(
        name="feature2",
        feature_type=FeatureType.FLOAT_RESCALED,
        prefered_distribution="log_normal"  # Manually specify distribution if needed
    )
}

# Initialize the model with distribution-aware encoding
model = PreprocessingModel(
    features=features,
    use_distribution_aware=True,
    distribution_aware_bins=1000  # Adjust bin count for finer data resolution
)
