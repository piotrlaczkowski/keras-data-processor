# Feature Selection in Keras Data Processor

The Keras Data Processor includes a sophisticated feature selection mechanism based on the Gated Residual Variable Selection Network (GRVSN) architecture. This document explains the components, usage, and benefits of this feature.

## Overview

The feature selection mechanism uses a combination of gated units and residual networks to automatically learn the importance of different features in your data. It can be applied to both numeric and categorical features, either independently or together.

## Components

### 1. GatedLinearUnit

The `GatedLinearUnit` is the basic building block that implements a gated activation function:

```python
gl = GatedLinearUnit(units=64)
x = tf.random.normal((32, 100))
y = gl(x)
```

Key features:
- Applies a linear transformation followed by a sigmoid gate
- Selectively filters input data based on learned weights
- Helps control information flow through the network

### 2. GatedResidualNetwork

The `GatedResidualNetwork` combines gated linear units with residual connections:

```python
grn = GatedResidualNetwork(units=64, dropout_rate=0.2)
x = tf.random.normal((32, 100))
y = grn(x)
```

Key features:
- Uses ELU activation for non-linearity
- Includes dropout for regularization
- Adds residual connections to help with gradient flow
- Applies layer normalization for stability

### 3. VariableSelection

The `VariableSelection` layer is the main feature selection component:

```python
vs = VariableSelection(nr_features=3, units=64, dropout_rate=0.2)
x1 = tf.random.normal((32, 100))
x2 = tf.random.normal((32, 200))
x3 = tf.random.normal((32, 300))
selected_features, weights = vs([x1, x2, x3])
```

Key features:
- Processes each feature independently using GRNs
- Calculates feature importance weights using softmax
- Returns both selected features and their weights
- Supports different input dimensions for each feature

## Usage in Preprocessing Model

### Configuration

Configure feature selection in your preprocessing model:

```python
model = PreprocessingModel(
    # ... other parameters ...
    feature_selection_placement="all_features",  # or "numeric" or "categorical"
    feature_selection_units=64,
    feature_selection_dropout=0.2
)
```

### Placement Options

The `FeatureSelectionPlacementOptions` enum provides several options for where to apply feature selection:

1. `NONE`: Disable feature selection
2. `NUMERIC`: Apply only to numeric features
3. `CATEGORICAL`: Apply only to categorical features
4. `ALL_FEATURES`: Apply to all features

### Accessing Feature Weights

After processing, feature weights are available in the `processed_features` dictionary:

```python
# Process your data
processed = model.transform(data)

# Access feature weights
numeric_weights = processed["numeric_feature_weights"]
categorical_weights = processed["categorical_feature_weights"]
```

## Benefits

1. **Automatic Feature Selection**: The model learns which features are most important for your task.
2. **Interpretability**: Feature weights provide insights into feature importance.
3. **Improved Performance**: By focusing on relevant features, the model can achieve better performance.
4. **Regularization**: Dropout and residual connections help prevent overfitting.
5. **Flexibility**: Can be applied to different feature types and combinations.

## Integration with Other Features

The feature selection mechanism integrates seamlessly with other preprocessing components:

1. **Transformer Blocks**: Can be used before or after transformer blocks
2. **Tabular Attention**: Complements tabular attention by focusing on important features
3. **Custom Preprocessors**: Works with any custom preprocessing steps

## Example

Here's a complete example of using feature selection:

```python
from kdp.processor import PreprocessingModel
from kdp.features import NumericalFeature, CategoricalFeature

# Define features
features = {
    "numeric_1": NumericalFeature(
        name="numeric_1",
        feature_type=FeatureType.FLOAT_NORMALIZED
    ),
    "numeric_2": NumericalFeature(
        name="numeric_2",
        feature_type=FeatureType.FLOAT_NORMALIZED
    ),
    "category_1": CategoricalFeature(
        name="category_1",
        feature_type=FeatureType.STRING_CATEGORICAL
    )
}

# Create model with feature selection
model = PreprocessingModel(
    # ... other parameters ...
    features_specs=features,
    feature_selection_placement="all_features", # or "numeric" or "categorical"
    feature_selection_units=64,
    feature_selection_dropout=0.2
)

# Build and use the model
preprocessor = model.build_preprocessor()
processed_data = model.transform(data) # data can be pd.DataFrame, python Dict, or tf.data.Dataset

# Analyze feature importance
for feature_name in features:
    weights = processed_data[f"{feature_name}_weights"]
    print(f"Feature {feature_name} importance: {weights.mean()}")
```

## Testing

The feature selection components include comprehensive unit tests that verify:

1. Output shapes and types
2. Gating mechanism behavior
3. Residual connections
4. Dropout behavior
5. Feature weight properties
6. Serialization/deserialization

Run the tests using:
```bash
python -m pytest test/test_feature_selection.py -v
```
